"""NSGA-III and C-TAEA extensions for the existing constrained MOEA problem.

The existing ``nsga2_optimization.py`` remains unchanged.  This module imports
and reuses its three Problem classes, surrogate-model wrapper, four objective
functions, five constraints, economic equations, runtime features, and exact
decision-variable bounds.  Only the evolutionary algorithm is replaced.

Safety gate
-----------
Running the file without ``--execute`` only prints the experiment plan.  An
optimization starts only when the user explicitly supplies ``--execute``.
This makes it possible to review the new code without accidentally launching
the computational experiment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "nsga2_optimization.py"
OUTPUT_ROOT = ROOT / "constrained_moea_results"

ALGORITHM_LABELS = {
    "nsga2": "NSGA-II (matched rerun)",
    "nsga3": "NSGA-III",
    "ctaea": "C-TAEA",
}
CONDITION_DEFS = {
    1: ("ThreeStageOptProblem", "Condition 1", ["Cu_in", "TA", "IA", "QA", "t"]),
    2: ("FourStageOptProblem", "Condition 2", ["Cu_in", "TB", "IB", "QB", "t"]),
    3: ("SerialOptProblem", "Condition 3",
        ["Cu_in", "TA", "IA", "QA", "TB", "IB", "QB", "t"]),
}

# Decision domains used by the archived ESRL-CMO policy.  The default remains
# the source NSGA-II domain; this optional domain is used only for the direct
# method comparison requested during revision.
ESRLCMO_BOUNDS = {
    1: (
        [29.0, 48.0, 8000.0, 111.0, 2.0],
        [55.0, 65.0, 27000.0, 123.0, 8.0],
    ),
    2: (
        [29.0, 55.0, 8000.0, 113.0, 2.0],
        [55.0, 65.0, 27000.0, 123.0, 8.0],
    ),
    3: (
        [29.0, 40.0, 8000.0, 111.0, 40.0, 8000.0, 113.0, 2.0],
        [55.0, 65.0, 27000.0, 123.0, 65.0, 27000.0, 123.0, 8.0],
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review or run NSGA-III/C-TAEA on the unchanged ESRL-CMO problem"
    )
    parser.add_argument("--algorithm", choices=["nsga2", "nsga3", "ctaea", "both"], default="both")
    parser.add_argument("--condition", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)
    parser.add_argument("--day", type=int)
    parser.add_argument("--hour", type=int)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--bounds-domain", choices=["source", "esrlcmo"], default="source",
        help="Use the source NSGA-II bounds or the archived ESRL-CMO policy bounds",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        help="Optional repeat seeds; omission reuses NSGA2_PARAMS['seed'] only",
    )
    parser.add_argument("--population-size", type=int,
                        help="Optional controlled population-size override")
    parser.add_argument("--generations", type=int,
                        help="Optional controlled generation-count override")
    parser.add_argument(
        "--include-matched-nsga2", action="store_true",
        help="Also rerun NSGA-II under the same pymoo version, budget and operators",
    )
    parser.add_argument(
        "--save-history", action="store_true",
        help="Retain every generation in memory (off by default)",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually run the optimizers. Without this flag, only the plan is printed.",
    )
    return parser.parse_args()


def load_source_module() -> ModuleType:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    spec = importlib.util.spec_from_file_location("esrl_cmo_nsga2_source", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_from_args(args: argparse.Namespace) -> dict[str, int]:
    supplied = {"year": args.year, "month": args.month,
                "day": args.day, "hour": args.hour}
    if any(value is None for value in supplied.values()):
        raise ValueError(
            "A fixed --year --month --day --hour is required for a reproducible matched run"
        )
    return {
        key: int(value) for key, value in supplied.items()
    }


def build_algorithm(name: str, problem, pop_size: int, seed: int):
    # Operators are explicit and identical across the two added methods and the
    # optional matched NSGA-II rerun.  The historical source NSGA-II relied on
    # pymoo defaults, so cross-version comparisons must use the matched rerun.
    common = {
        "sampling": FloatRandomSampling(),
        "crossover": SBX(prob=0.9, eta=15, n_offsprings=2),
        "mutation": PM(prob=1.0, prob_var=1.0 / problem.n_var, eta=20),
        "eliminate_duplicates": True,
    }
    if name == "nsga2":
        return NSGA2(pop_size=pop_size, **common), None
    ref_dirs = get_reference_directions("energy", problem.n_obj, pop_size, seed=seed)
    if len(ref_dirs) != pop_size:
        raise RuntimeError(f"Expected {pop_size} reference directions, got {len(ref_dirs)}")
    if name == "nsga3":
        return NSGA3(ref_dirs=ref_dirs, pop_size=pop_size, **common), ref_dirs
    if name == "ctaea":
        # C-TAEA derives its population size from the number of reference directions.
        return CTAEA(ref_dirs=ref_dirs, **common), ref_dirs
    raise KeyError(name)


def final_feasible_nondominated(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if result.pop is None:
        return np.empty((0, 4)), np.empty((0, 0)), np.empty((0, 5))
    f = np.asarray(result.pop.get("F"), dtype=float)
    x = np.asarray(result.pop.get("X"), dtype=float)
    g_raw = result.pop.get("G")
    g = np.asarray(g_raw, dtype=float) if g_raw is not None else np.empty((len(f), 0))
    finite = np.isfinite(f).all(axis=1) & np.isfinite(x).all(axis=1)
    feasible = finite & ((g <= 0).all(axis=1) if g.size else True)
    f, x, g = f[feasible], x[feasible], g[feasible]
    if len(f):
        nd = NonDominatedSorting().do(f, only_non_dominated_front=True)
        f, x, g = f[nd], x[nd], g[nd]
    return f, x, g


def result_table(f: np.ndarray, x: np.ndarray, g: np.ndarray,
                 algorithm_label: str, condition_label: str,
                 variable_names: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for i in range(len(f)):
        row = {
            "Method": algorithm_label,
            "Condition": condition_label,
            "Solution_ID": i + 1,
        }
        row.update({name: float(x[i, j]) for j, name in enumerate(variable_names)})
        row.update({
            "Cu_out": float(f[i, 0]),
            "As_out": float(-f[i, 1]),
            "E_total": float(f[i, 2]),
            "Net_profit": float(-f[i, 3]),
        })
        if g.size:
            row.update({name: float(g[i, j]) for j, name in enumerate(
                ["G1_Cu", "G2_As", "G3_J", "G4_V", "G5_ratio"]
            )})
        rows.append(row)
    columns = [
        "Method", "Condition", "Solution_ID", *variable_names,
        "Cu_out", "As_out", "E_total", "Net_profit",
        "G1_Cu", "G2_As", "G3_J", "G4_V", "G5_ratio",
    ]
    return pd.DataFrame(rows, columns=columns)


def run(args: argparse.Namespace) -> None:
    module = load_source_module()
    params = dict(module.NSGA2_PARAMS)
    pop_size = int(args.population_size if args.population_size is not None else params["pop_size"])
    n_gen = int(args.generations if args.generations is not None else params["n_gen"])
    seeds = list(dict.fromkeys(
        args.seeds if args.seeds is not None else [int(params["seed"])]
    ))
    if not seeds:
        raise ValueError("At least one seed is required")
    runtime = runtime_from_args(args)

    # Reuse the exact trained surrogates and expose them through the source
    # module's existing global, as required by its Problem._evaluate methods.
    module.surrogate = module.SurrogateModel()
    for target_models in module.surrogate._models.values():
        for model in target_models.values():
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1

    algorithms = list(("nsga3", "ctaea") if args.algorithm == "both" else (args.algorithm,))
    if args.include_matched_nsga2:
        algorithms.insert(0, "nsga2")
    conditions = (1, 2, 3) if args.condition == "all" else (int(args.condition),)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows: list[dict] = []

    for condition in conditions:
        class_name, condition_label, variable_names = CONDITION_DEFS[condition]
        problem_class = getattr(module, class_name)
        problem = problem_class(runtime)
        if args.bounds_domain == "esrlcmo":
            xl, xu = ESRLCMO_BOUNDS[condition]
            problem.xl = np.asarray(xl, dtype=float)
            problem.xu = np.asarray(xu, dtype=float)
        for algorithm_name in algorithms:
            for seed in seeds:
                algorithm, ref_dirs = build_algorithm(algorithm_name, problem, pop_size, seed)
                label = ALGORITHM_LABELS[algorithm_name]
                started = time.perf_counter()
                result = minimize(
                    problem,
                    algorithm,
                    ("n_gen", n_gen),
                    seed=seed,
                    verbose=False,
                    save_history=args.save_history,
                )
                runtime_seconds = time.perf_counter() - started
                f, x, g = final_feasible_nondominated(result)
                final_g = np.asarray(result.pop.get("G"), dtype=float)
                final_feasible_rate = float((final_g <= 1.0e-8).all(axis=1).mean())
                table = result_table(f, x, g, label, condition_label, variable_names)
                run_dir = args.output_dir / algorithm_name / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                table.to_csv(run_dir / f"pareto_{algorithm_name}_condition{condition}.csv",
                             index=False, encoding="utf-8-sig")
                metadata_rows.append({
                "algorithm": label,
                "condition": condition_label,
                "source_problem": str(SOURCE),
                "population_size": pop_size,
                "generations": n_gen,
                "seed": seed,
                "runtime_seconds": runtime_seconds,
                "n_evaluations": int(result.algorithm.evaluator.n_eval),
                "n_final_feasible_nondominated": len(table),
                "final_population_feasibility_rate": final_feasible_rate,
                "reference_direction_method": "energy" if ref_dirs is not None else "not applicable",
                "n_reference_directions": len(ref_dirs) if ref_dirs is not None else 0,
                "sampling": "FloatRandomSampling",
                "crossover": "SBX(prob=0.9, eta=15, n_offsprings=2)",
                "mutation": f"PM(prob=1.0, prob_var=1/{problem.n_var}, eta=20)",
                "operator_profile": "controlled common operators; not canonical C-TAEA defaults",
                "bounds_domain": args.bounds_domain,
                "runtime": json.dumps(runtime),
                "xl": json.dumps(problem.xl.tolist()),
                "xu": json.dumps(problem.xu.tolist()),
                "constraints": json.dumps(module.PARAMS),
                "save_history": args.save_history,
                })
    metadata_path = args.output_dir / "run_metadata.csv"
    new_metadata = pd.DataFrame(metadata_rows)
    if metadata_path.exists():
        previous = pd.read_csv(metadata_path)
        new_metadata = pd.concat([previous, new_metadata], ignore_index=True)
        new_metadata = new_metadata.drop_duplicates(
            subset=["algorithm", "condition", "seed"], keep="last"
        )
    new_metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")


def print_plan(args: argparse.Namespace) -> None:
    plan = {
        "status": "NOT RUN - code-generation/review mode",
        "source_problem": str(SOURCE),
        "algorithms_added": ["NSGA-III", "C-TAEA"],
        "optional_control": "Use --include-matched-nsga2 for a same-version NSGA-II control",
        "seeds": args.seeds if args.seeds is not None else "NSGA2_PARAMS['seed']",
        "conditions": args.condition,
        "inheritance": [
            "three original Problem classes",
            "four original objective equations",
            "five original g<=0 constraints",
            "original decision-variable ranges",
            "original surrogate models and economic equations",
            "NSGA2_PARAMS population size and generations",
        ],
        "execution_gate": "Supply --execute only after the experiment plan is approved.",
        "runtime_requirement": "A fixed --year --month --day --hour is mandatory with --execute",
        "operator_profile": "Controlled common sampling/SBX/PM operators across methods; not canonical C-TAEA defaults",
        "planned_output": str(args.output_dir),
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if not args.execute:
        print_plan(args)
        return
    run(args)


if __name__ == "__main__":
    main()
