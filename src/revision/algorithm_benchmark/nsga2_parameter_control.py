"""Matched NSGA-II population/generation control experiment for Condition 2.

This script is deliberately separate from ``nsga2_optimization.py``.  It
imports the copied source problem without modifying it, reuses the trained
surrogates, objectives, constraints, economics and decision-variable bounds,
and varies only population size and generation count.

Default design
--------------
* fixed runtime features: 2025-06-15 08:00
* configurations: (300, 300), (300, 600), (600, 300), (900, 200)
* repeat seeds: 11, 23, 42
* operators: FloatRandomSampling, SBX(prob=0.9, eta=15), and
  PM(prob=1.0, prob_var=1/n_var, eta=20)

The 300 x 300 run is a lower-budget control (90,000 evaluations).  The other
three configurations have the same nominal 180,000-evaluation budget, so they
form the primary population-versus-generation comparison.

Safety gate
-----------
The optimizer runs only when ``--execute`` is supplied.  Without it the
script prints the registered design.  Completed run directories are reused,
which makes interrupted experiments resumable without deleting results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "nsga2_optimization.py"
OUTPUT_ROOT = ROOT / "nsga2_parameter_control"

RUNTIME = {"year": 2025, "month": 6, "day": 15, "hour": 8}
VARIABLE_NAMES = ["Cu_in", "TB", "IB", "QB", "t"]
OBJECTIVE_NAMES = ["Cu_out", "negative_As_out", "E_total", "negative_Net_profit"]
CONSTRAINT_NAMES = ["G1_Cu", "G2_As", "G3_J", "G4_V", "G5_ratio"]
DEFAULT_SEEDS = (11, 23, 42)
FEASIBILITY_TOLERANCE = 1.0e-8
NORMALIZED_HV_REFERENCE = np.full(4, 1.10, dtype=float)


@dataclass(frozen=True)
class Configuration:
    population_size: int
    generations: int

    @property
    def config_id(self) -> str:
        return f"pop{self.population_size}_gen{self.generations}"

    @property
    def nominal_evaluations(self) -> int:
        return self.population_size * self.generations


CONFIGURATIONS = (
    Configuration(300, 300),
    Configuration(300, 600),
    Configuration(600, 300),
    Configuration(900, 200),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the matched Condition-2 NSGA-II parameter control"
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
        help="Repeat seeds (default: 11 23 42)",
    )
    parser.add_argument(
        "--only", choices=[item.config_id for item in CONFIGURATIONS], nargs="+",
        help="Optionally run only selected registered configurations",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Run missing optimizer jobs and then calculate normalized indicators",
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Recalculate indicators from completed result files without optimizing",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_module() -> ModuleType:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    spec = importlib.util.spec_from_file_location("esrl_cmo_nsga2_parameter_source", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import copied source: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_problem() -> tuple[ModuleType, object]:
    module = load_source_module()
    module.surrogate = module.SurrogateModel()
    # Restrict tree-model prediction to one worker so optimizer repeats do not
    # oversubscribe the workstation or change the algorithmic comparison.
    for target_models in module.surrogate._models.values():
        for model in target_models.values():
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1
    problem = module.FourStageOptProblem(dict(RUNTIME))
    if problem.n_obj != 4 or problem.n_ieq_constr != 5 or problem.n_var != 5:
        raise RuntimeError(
            "Copied Condition-2 problem no longer has the expected 5 variables, "
            "4 objectives and 5 inequality constraints"
        )
    return module, problem


def build_algorithm(problem: object, configuration: Configuration) -> NSGA2:
    return NSGA2(
        pop_size=configuration.population_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, n_offsprings=2),
        mutation=PM(prob=1.0, prob_var=1.0 / problem.n_var, eta=20),
        eliminate_duplicates=True,
    )


def feasible_nondominated_population(result: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if result.pop is None:
        return np.empty((0, 4)), np.empty((0, 5)), np.empty((0, 5))
    f = np.asarray(result.pop.get("F"), dtype=float)
    x = np.asarray(result.pop.get("X"), dtype=float)
    g = np.asarray(result.pop.get("G"), dtype=float)
    finite = (
        np.isfinite(f).all(axis=1)
        & np.isfinite(x).all(axis=1)
        & np.isfinite(g).all(axis=1)
    )
    feasible = finite & (g <= FEASIBILITY_TOLERANCE).all(axis=1)
    f, x, g = f[feasible], x[feasible], g[feasible]
    if len(f):
        indices = NonDominatedSorting().do(f, only_non_dominated_front=True)
        f, x, g = f[indices], x[indices], g[indices]
    return f, x, g


def pareto_table(f: np.ndarray, x: np.ndarray, g: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for index in range(len(f)):
        row: dict[str, float | int] = {"Solution_ID": index + 1}
        row.update({name: float(x[index, j]) for j, name in enumerate(VARIABLE_NAMES)})
        row.update({
            "Cu_out": float(f[index, 0]),
            "As_out": float(-f[index, 1]),
            "E_total": float(f[index, 2]),
            "Net_profit": float(-f[index, 3]),
        })
        row.update({name: float(g[index, j]) for j, name in enumerate(CONSTRAINT_NAMES)})
        rows.append(row)
    return pd.DataFrame(rows, columns=[
        "Solution_ID", *VARIABLE_NAMES,
        "Cu_out", "As_out", "E_total", "Net_profit", *CONSTRAINT_NAMES,
    ])


def objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    required = {"Cu_out", "As_out", "E_total", "Net_profit"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing objective columns: {sorted(missing)}")
    return np.column_stack([
        frame["Cu_out"].to_numpy(float),
        -frame["As_out"].to_numpy(float),
        frame["E_total"].to_numpy(float),
        -frame["Net_profit"].to_numpy(float),
    ])


def run_one(
    module: ModuleType,
    problem: object,
    configuration: Configuration,
    seed: int,
    output_root: Path,
) -> dict:
    run_dir = output_root / configuration.config_id / f"seed_{seed}"
    pareto_path = run_dir / "pareto.csv"
    metadata_path = run_dir / "run_metadata.json"
    if pareto_path.exists() and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("status") == "success":
            print(f"[reuse] {configuration.config_id}, seed={seed}")
            return metadata

    run_dir.mkdir(parents=True, exist_ok=True)
    algorithm = build_algorithm(problem, configuration)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    result = minimize(
        problem,
        algorithm,
        ("n_gen", configuration.generations),
        seed=seed,
        verbose=False,
        save_history=False,
    )
    elapsed = time.perf_counter() - started
    f, x, g = feasible_nondominated_population(result)
    table = pareto_table(f, x, g)
    table.to_csv(pareto_path, index=False, encoding="utf-8-sig")

    final_f = np.asarray(result.pop.get("F"), dtype=float)
    final_g = np.asarray(result.pop.get("G"), dtype=float)
    finite = np.isfinite(final_f).all(axis=1) & np.isfinite(final_g).all(axis=1)
    feasible = finite & (final_g <= FEASIBILITY_TOLERANCE).all(axis=1)
    metadata = {
        "status": "success",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at.isoformat(),
        "source_problem": str(SOURCE),
        "source_problem_sha256": file_sha256(SOURCE),
        "condition": "Condition 2",
        "problem_class": "FourStageOptProblem",
        "runtime_features": dict(RUNTIME),
        "population_size": configuration.population_size,
        "generations": configuration.generations,
        "nominal_evaluations": configuration.nominal_evaluations,
        "actual_evaluations": int(result.algorithm.evaluator.n_eval),
        "seed": int(seed),
        "runtime_seconds": float(elapsed),
        "n_final_population": int(len(final_f)),
        "n_final_feasible": int(feasible.sum()),
        "final_population_feasibility_rate": float(feasible.mean()),
        "n_final_feasible_nondominated": int(len(table)),
        "n_variables": int(problem.n_var),
        "n_objectives": int(problem.n_obj),
        "n_inequality_constraints": int(problem.n_ieq_constr),
        "objective_minimization_order": OBJECTIVE_NAMES,
        "constraint_order": CONSTRAINT_NAMES,
        "decision_variable_order": VARIABLE_NAMES,
        "xl": np.asarray(problem.xl, dtype=float).tolist(),
        "xu": np.asarray(problem.xu, dtype=float).tolist(),
        "constraint_parameters": dict(module.PARAMS),
        "sampling": "FloatRandomSampling()",
        "crossover": "SBX(prob=0.9, eta=15, n_offsprings=2)",
        "mutation": f"PM(prob=1.0, prob_var=1/{problem.n_var}, eta=20)",
        "mutation_probability_semantics": (
            "all offspring are mutation candidates; each variable has probability 1/n_var"
        ),
        "eliminate_duplicates": True,
        "feasibility_tolerance": FEASIBILITY_TOLERANCE,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "pymoo", "scikit-learn", "joblib")
        },
        "pareto_file": str(pareto_path),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(
        f"[done] {configuration.config_id}, seed={seed}: "
        f"n_eval={metadata['actual_evaluations']}, "
        f"feasible={metadata['final_population_feasibility_rate']:.3f}, "
        f"ND={metadata['n_final_feasible_nondominated']}, "
        f"time={elapsed:.1f}s"
    )
    return metadata


def collect_metadata(output_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(output_root.glob("pop*_gen*/seed_*/run_metadata.json")):
        with path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["metadata_file"] = str(path)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).sort_values(["nominal_evaluations", "population_size", "seed"])
    serializable = frame.copy()
    for column in serializable.columns:
        if serializable[column].map(lambda value: isinstance(value, (dict, list))).any():
            serializable[column] = serializable[column].map(
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list)) else value
            )
    serializable.to_csv(output_root / "run_metadata.csv", index=False, encoding="utf-8-sig")
    return frame


def spacing(normalized_objectives: np.ndarray) -> float:
    points = np.unique(np.asarray(normalized_objectives, dtype=float), axis=0)
    if len(points) < 2:
        return float("nan")
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    return float(np.std(nearest, ddof=1))


def discover_completed_runs(output_root: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(output_root.glob("pop*_gen*/seed_*/pareto.csv")):
        metadata_path = path.with_name("run_metadata.json")
        if not metadata_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        frame = pd.read_csv(path)
        if metadata.get("status") != "success" or frame.empty:
            continue
        records.append({
            "path": path,
            "metadata": metadata,
            "frame": frame,
            "f": objective_matrix(frame),
        })
    return records


def analyze(output_root: Path) -> None:
    records = discover_completed_runs(output_root)
    if not records:
        raise FileNotFoundError(f"No completed non-empty runs under {output_root}")

    pooled = np.vstack([record["f"] for record in records])
    pooled = np.unique(pooled, axis=0)
    ideal = pooled.min(axis=0)
    nadir = pooled.max(axis=0)
    raw_span = nadir - ideal
    span = np.where(raw_span > 0.0, raw_span, 1.0)
    pooled_normalized = (pooled - ideal) / span
    reference_indices = NonDominatedSorting().do(
        pooled_normalized, only_non_dominated_front=True
    )
    reference_normalized = pooled_normalized[reference_indices]
    reference_raw = pooled[reference_indices]

    hv_indicator = HV(ref_point=NORMALIZED_HV_REFERENCE)
    igd_indicator = IGD(reference_normalized)
    igd_plus_indicator = IGDPlus(reference_normalized)
    metric_rows: list[dict] = []
    for record in records:
        metadata = record["metadata"]
        normalized = (np.unique(record["f"], axis=0) - ideal) / span
        metric_rows.append({
            "config_id": f"pop{metadata['population_size']}_gen{metadata['generations']}",
            "population_size": metadata["population_size"],
            "generations": metadata["generations"],
            "nominal_evaluations": metadata["nominal_evaluations"],
            "actual_evaluations": metadata["actual_evaluations"],
            "seed": metadata["seed"],
            "runtime_seconds": metadata["runtime_seconds"],
            "final_population_feasibility_rate": metadata["final_population_feasibility_rate"],
            "n_final_feasible_nondominated": len(normalized),
            "normalized_HV": float(hv_indicator(normalized)),
            "normalized_IGD": float(igd_indicator(normalized)),
            "normalized_IGD_plus": float(igd_plus_indicator(normalized)),
            "normalized_spacing": spacing(normalized),
            "source_file": str(record["path"]),
            "normalization_ideal": json.dumps(ideal.tolist()),
            "normalization_nadir": json.dumps(nadir.tolist()),
            "normalized_HV_reference": json.dumps(NORMALIZED_HV_REFERENCE.tolist()),
            "empirical_reference_size": len(reference_normalized),
        })

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["nominal_evaluations", "population_size", "generations", "seed"]
    )
    summary = metrics.groupby(
        ["config_id", "population_size", "generations", "nominal_evaluations"],
        sort=True,
    ).agg(
        n_seeds=("seed", "nunique"),
        actual_evaluations_mean=("actual_evaluations", "mean"),
        runtime_seconds_mean=("runtime_seconds", "mean"),
        runtime_seconds_sd=("runtime_seconds", "std"),
        feasibility_rate_mean=("final_population_feasibility_rate", "mean"),
        feasibility_rate_sd=("final_population_feasibility_rate", "std"),
        front_size_mean=("n_final_feasible_nondominated", "mean"),
        front_size_sd=("n_final_feasible_nondominated", "std"),
        normalized_HV_mean=("normalized_HV", "mean"),
        normalized_HV_sd=("normalized_HV", "std"),
        normalized_IGD_mean=("normalized_IGD", "mean"),
        normalized_IGD_sd=("normalized_IGD", "std"),
        normalized_IGD_plus_mean=("normalized_IGD_plus", "mean"),
        normalized_IGD_plus_sd=("normalized_IGD_plus", "std"),
        normalized_spacing_mean=("normalized_spacing", "mean"),
        normalized_spacing_sd=("normalized_spacing", "std"),
    ).reset_index()
    summary["budget_group"] = np.where(
        summary["nominal_evaluations"] == 180_000,
        "matched 180000-evaluation comparison",
        "lower-budget control",
    )

    metrics.to_csv(output_root / "indicator_by_run.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_root / "indicator_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(reference_raw, columns=OBJECTIVE_NAMES).to_csv(
        output_root / "pooled_reference_front_minimization_raw.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(
        reference_normalized,
        columns=[f"{name}_normalized" for name in OBJECTIVE_NAMES],
    ).to_csv(
        output_root / "pooled_reference_front_normalized.csv",
        index=False, encoding="utf-8-sig",
    )
    with (output_root / "indicator_method.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "objective_orientation": "all four columns are minimized",
            "objective_order": OBJECTIVE_NAMES,
            "normalization_scope": "joint across all completed Condition-2 parameter-control runs",
            "normalization_formula": "(F - pooled ideal) / (pooled nadir - pooled ideal)",
            "pooled_ideal": ideal.tolist(),
            "pooled_nadir": nadir.tolist(),
            "zero_span_objectives": [
                OBJECTIVE_NAMES[index] for index, value in enumerate(raw_span) if value == 0.0
            ],
            "empirical_reference_front": "nondominated union of all normalized completed-run fronts",
            "empirical_reference_size": int(len(reference_normalized)),
            "hypervolume_reference_point": NORMALIZED_HV_REFERENCE.tolist(),
            "indicators": {
                "normalized_HV": "larger is better",
                "normalized_IGD": "smaller is better",
                "normalized_IGD_plus": "smaller is better",
                "normalized_spacing": "sample standard deviation of nearest-neighbour distances; smaller is more even",
            },
            "comparison_rule": (
                "Use pop300_gen600, pop600_gen300 and pop900_gen200 for the primary "
                "matched-budget comparison. Treat pop300_gen300 as a lower-budget control."
            ),
            "reference_limitation": (
                "The pooled front is an empirical reference, not a known true Pareto front; "
                "therefore indicators support relative algorithm-setting comparison only."
            ),
        }, handle, ensure_ascii=False, indent=2)
    print(f"[analysis] wrote normalized indicators for {len(metrics)} completed runs")


def print_plan(args: argparse.Namespace) -> None:
    selected = [
        item for item in CONFIGURATIONS
        if args.only is None or item.config_id in set(args.only)
    ]
    print(json.dumps({
        "status": "NOT RUN - review mode",
        "source_problem": str(SOURCE),
        "condition": "Condition 2 / FourStageOptProblem",
        "runtime_features": RUNTIME,
        "configurations": [
            {
                "config_id": item.config_id,
                "population_size": item.population_size,
                "generations": item.generations,
                "nominal_evaluations": item.nominal_evaluations,
            }
            for item in selected
        ],
        "seeds": list(dict.fromkeys(args.seeds)),
        "operators": {
            "sampling": "FloatRandomSampling()",
            "crossover": "SBX(prob=0.9, eta=15, n_offsprings=2)",
            "mutation": "PM(prob=1.0, prob_var=1/n_var, eta=20)",
        },
        "outputs": str(args.output_dir),
        "analysis": (
            "joint per-objective normalization; normalized HV with reference "
            "[1.1,1.1,1.1,1.1]; IGD/IGD+ to pooled empirical nondominated front; "
            "nearest-neighbour spacing in normalized objective space"
        ),
        "execution_gate": "Supply --execute to run missing jobs or --analyze-only for completed jobs.",
    }, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.execute and args.analyze_only:
        raise ValueError("Choose either --execute or --analyze-only, not both")
    if not args.execute and not args.analyze_only:
        print_plan(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.execute:
        selected = [
            item for item in CONFIGURATIONS
            if args.only is None or item.config_id in set(args.only)
        ]
        seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
        if not selected or not seeds:
            raise ValueError("At least one registered configuration and one seed are required")
        module, problem = load_problem()
        for configuration in selected:
            for seed in seeds:
                run_one(module, problem, configuration, seed, args.output_dir)
        collect_metadata(args.output_dir)
    else:
        collect_metadata(args.output_dir)
    analyze(args.output_dir)


if __name__ == "__main__":
    main()
