"""Fair comparison of RVEA, matched NSGA-II, and archived ESRL-CMO sets.

The script re-evaluates every retained decision vector with the same surrogate
cascade, fixed time features, objective equations, economic equation, and five
constraints.  It then restricts all methods to the intersection of their
decision domains, removes infeasible and dominated points, selects an equal
number of representative points per method within each condition, and computes
jointly normalized HV, IGD+, and spacing.

No analysis is performed unless ``--execute`` is supplied.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "nsga2_optimization.py"
DEFAULT_INPUT = ROOT / "result" / "rvea_common_matched_20260901"
DEFAULT_OUTPUT = ROOT / "result" / "rvea_esrl_comparison_20260901"
FEASIBILITY_TOLERANCE = 1.0e-8
DOMAIN_TOLERANCE = 1.0e-9
FIXED_RUNTIME = {"year": 2025, "month": 6, "day": 15, "hour": 8}
DEFAULT_ANALYSIS_SEED = 20260901
DEFAULT_SUBSAMPLES = 500

CONDITIONS = {
    1: {
        "class": "ThreeStageOptProblem",
        "variables": ["Cu_in", "TA", "IA", "QA", "t"],
        "xl": [29.0, 48.0, 8000.0, 111.0, 2.0],
        "xu": [55.0, 65.0, 27000.0, 123.0, 8.0],
        "archive": ROOT / "result" / "outputs_condition1" / "pareto_ppo_condition1.csv",
    },
    2: {
        "class": "FourStageOptProblem",
        "variables": ["Cu_in", "TB", "IB", "QB", "t"],
        "xl": [29.0, 55.0, 8000.0, 113.0, 2.0],
        "xu": [55.0, 65.0, 27000.0, 122.0, 8.0],
        "archive": ROOT / "result" / "outputs_condition2" / "pareto_ppo_condition2.csv",
    },
    3: {
        "class": "SerialOptProblem",
        "variables": ["Cu_in", "TA", "IA", "QA", "TB", "IB", "QB", "t"],
        "xl": [29.0, 40.0, 10000.0, 113.0, 40.0, 10000.0, 114.0, 2.0],
        "xu": [55.0, 65.0, 22000.0, 122.0, 65.0, 24000.0, 122.0, 8.0],
        "archive": ROOT / "result" / "outputs_condition3" / "pareto_ppo_condition3.csv",
    },
}

METHOD_ORDER = {"ESRL-CMO": 0, "NSGA-II": 1, "RVEA": 2}


@dataclass
class Record:
    condition: int
    method: str
    seed: int | None
    source_file: Path
    source_count: int
    common_domain_count: int
    feasible_count: int
    f: np.ndarray
    x: np.ndarray
    g: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--subsamples", type=int, default=DEFAULT_SUBSAMPLES)
    parser.add_argument("--analysis-seed", type=int, default=DEFAULT_ANALYSIS_SEED)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def load_source_module():
    spec = importlib.util.spec_from_file_location("rvea_common_problem", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.surrogate = module.SurrogateModel()
    for target_models in module.surrogate._models.values():
        for model in target_models.values():
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1
    return module


def nondominated(f: np.ndarray) -> np.ndarray:
    return NonDominatedSorting().do(f, only_non_dominated_front=True)


def evaluate_frame(module, condition: int, frame: pd.DataFrame,
                   method: str, seed: int | None, source_file: Path) -> Record:
    cfg = CONDITIONS[condition]
    missing = set(cfg["variables"]) - set(frame.columns)
    if missing:
        raise ValueError(f"{source_file} is missing variables: {sorted(missing)}")
    x_all = frame[cfg["variables"]].to_numpy(float)
    xl = np.asarray(cfg["xl"], dtype=float)
    xu = np.asarray(cfg["xu"], dtype=float)
    in_domain = (
        np.isfinite(x_all).all(axis=1)
        & (x_all >= xl - DOMAIN_TOLERANCE).all(axis=1)
        & (x_all <= xu + DOMAIN_TOLERANCE).all(axis=1)
    )
    x = x_all[in_domain]
    problem = getattr(module, cfg["class"])(FIXED_RUNTIME)
    problem.xl = xl.copy()
    problem.xu = xu.copy()
    out: dict[str, np.ndarray] = {}
    problem._evaluate(x, out)
    f = np.asarray(out["F"], dtype=float)
    g = np.asarray(out["G"], dtype=float)
    finite = np.isfinite(f).all(axis=1) & np.isfinite(g).all(axis=1)
    feasible = finite & (g <= FEASIBILITY_TOLERANCE).all(axis=1)
    f, x, g = f[feasible], x[feasible], g[feasible]
    if len(f):
        _, unique_idx = np.unique(f, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        f, x, g = f[unique_idx], x[unique_idx], g[unique_idx]
        keep = nondominated(f)
        f, x, g = f[keep], x[keep], g[keep]
    return Record(
        condition=condition,
        method=method,
        seed=seed,
        source_file=source_file,
        source_count=len(frame),
        common_domain_count=int(in_domain.sum()),
        feasible_count=int(feasible.sum()),
        f=f,
        x=x,
        g=g,
    )


def discover_records(module, input_dir: Path) -> list[Record]:
    records: list[Record] = []
    for method_dir, method in (("nsga2", "NSGA-II"), ("rvea", "RVEA")):
        for path in sorted((input_dir / method_dir).glob("seed_*/pareto_*_condition*.csv")):
            seed = int(path.parent.name.removeprefix("seed_"))
            condition = int(path.stem.rsplit("condition", 1)[1])
            records.append(evaluate_frame(
                module, condition, pd.read_csv(path), method, seed, path
            ))
    for condition, cfg in CONDITIONS.items():
        path = cfg["archive"]
        records.append(evaluate_frame(
            module, condition, pd.read_csv(path), "ESRL-CMO", None, path
        ))
    expected = {(c, m) for c in CONDITIONS for m in METHOD_ORDER}
    found = {(r.condition, r.method) for r in records}
    if expected != found:
        raise RuntimeError(f"Missing method/condition combinations: {sorted(expected - found)}")
    if any(len(record.f) == 0 for record in records):
        empty = [(r.condition, r.method, r.seed) for r in records if len(r.f) == 0]
        raise RuntimeError(f"No feasible nondominated points for: {empty}")
    expected_seeds = {11, 23, 42}
    for condition in CONDITIONS:
        for method in ("NSGA-II", "RVEA"):
            found_seeds = {
                int(record.seed) for record in records
                if record.condition == condition and record.method == method
            }
            if found_seeds != expected_seeds:
                raise RuntimeError(
                    f"{method}, Condition {condition}: expected seeds {sorted(expected_seeds)}, "
                    f"found {sorted(found_seeds)}"
                )
    return records


def farthest_point_subset(z: np.ndarray, k: int) -> np.ndarray:
    """Deterministic extreme-preserving maximin subset in normalized space."""
    if len(z) <= k:
        return np.arange(len(z), dtype=int)
    selected: list[int] = []
    for objective in range(z.shape[1]):
        idx = int(np.argmin(z[:, objective]))
        if idx not in selected:
            selected.append(idx)
    min_distance = np.full(len(z), np.inf)
    if selected:
        min_distance = np.min(
            np.linalg.norm(z[:, None, :] - z[np.asarray(selected)][None, :, :], axis=2),
            axis=1,
        )
        min_distance[np.asarray(selected)] = -np.inf
    while len(selected) < k:
        idx = int(np.argmax(min_distance))
        selected.append(idx)
        candidate_distance = np.linalg.norm(z - z[idx], axis=1)
        min_distance = np.minimum(min_distance, candidate_distance)
        min_distance[np.asarray(selected)] = -np.inf
    return np.asarray(selected, dtype=int)


def resampled_metrics(z: np.ndarray, reference: np.ndarray, k: int,
                      n_subsamples: int, rng: np.random.Generator) -> dict[str, float]:
    """Cardinality-controlled indicators using uniform sampling without replacement."""
    hv = HV(ref_point=np.full(4, 1.1))
    spacing_indicator = SpacingIndicator()
    # Precompute the IGD+ distance from each pooled-reference point to every
    # candidate.  This avoids rebuilding the full pairwise tensor 500 times.
    igd_plus_distance = np.sqrt(
        np.square(np.maximum(z[None, :, :] - reference[:, None, :], 0.0)).sum(axis=2)
    )
    hv_values: list[float] = []
    igd_values: list[float] = []
    spacing_values: list[float] = []
    repeats = 1 if len(z) == k else n_subsamples
    for _ in range(repeats):
        idx = np.arange(len(z)) if len(z) == k else rng.choice(len(z), size=k, replace=False)
        sample = z[idx]
        hv_values.append(float(hv(sample)))
        igd_values.append(float(np.min(igd_plus_distance[:, idx], axis=1).mean()))
        spacing_values.append(float(spacing_indicator(sample)))
    result: dict[str, float] = {}
    for name, values in (
        ("HV", hv_values), ("IGD_plus", igd_values), ("spacing", spacing_values)
    ):
        result[f"normalized_{name}"] = float(np.mean(values))
        result[f"subsample_{name}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        result[f"subsample_{name}_q025"] = float(np.quantile(values, 0.025))
        result[f"subsample_{name}_q975"] = float(np.quantile(values, 0.975))
    result["n_subsamples"] = repeats
    return result


def metric_analysis(records: list[Record], target_size: int, metadata: pd.DataFrame,
                    n_subsamples: int, analysis_seed: int):
    metric_rows: list[dict] = []
    selected_rows: list[dict] = []
    audit_rows: list[dict] = []
    reference_rows: list[pd.DataFrame] = []

    for condition in CONDITIONS:
        subset = [record for record in records if record.condition == condition]
        pooled = np.vstack([record.f for record in subset])
        ideal = pooled.min(axis=0)
        nadir = pooled.max(axis=0)
        span = nadir - ideal
        if np.any(span <= 1.0e-12):
            raise RuntimeError(
                f"Condition {condition} contains a non-identifiable objective span: {span}"
            )
        pooled_norm = np.unique((pooled - ideal) / span, axis=0)
        reference = pooled_norm[nondominated(pooled_norm)]
        reference_rows.append(pd.DataFrame(reference, columns=[
            "Cu_out_norm", "negative_As_out_norm", "E_total_norm", "negative_Net_profit_norm"
        ]).assign(condition=f"Condition {condition}"))
        k = min(target_size, min(len(record.f) for record in subset))
        for record in subset:
            normalized = (record.f - ideal) / span
            representative_idx = farthest_point_subset(normalized, k)
            method_offset = METHOD_ORDER[record.method] * 100
            seed_offset = int(record.seed) if record.seed is not None else 0
            rng = np.random.default_rng(
                analysis_seed + condition * 1000 + method_offset + seed_offset
            )
            sampled = resampled_metrics(
                normalized, reference, k, n_subsamples, rng
            )
            selected_idx = representative_idx
            selected_f = record.f[selected_idx]
            selected_x = record.x[selected_idx]
            selected_g = record.g[selected_idx]
            selected_z = normalized[selected_idx]

            meta_match = metadata[
                (metadata.get("condition", pd.Series(dtype=str)) == f"Condition {condition}")
                & (metadata.get("seed", pd.Series(dtype=float)) == record.seed)
            ] if record.seed is not None else pd.DataFrame()
            if record.method == "RVEA" and not meta_match.empty:
                meta_match = meta_match[meta_match["algorithm"] == "RVEA"]
            elif record.method == "NSGA-II" and not meta_match.empty:
                meta_match = meta_match[meta_match["algorithm"].str.startswith("NSGA-II")]
            runtime_seconds = float(meta_match.iloc[0]["runtime_seconds"]) if len(meta_match) == 1 else np.nan
            n_evaluations = float(meta_match.iloc[0]["n_evaluations"]) if len(meta_match) == 1 else np.nan
            final_feasibility = (
                float(meta_match.iloc[0]["final_population_feasibility_rate"])
                if len(meta_match) == 1 else record.feasible_count / max(record.common_domain_count, 1)
            )

            metric_rows.append({
                "condition": f"Condition {condition}",
                "method": record.method,
                "seed": record.seed if record.seed is not None else "archive",
                "source_count": record.source_count,
                "common_domain_count": record.common_domain_count,
                "feasible_count": record.feasible_count,
                "full_nondominated_count": len(record.f),
                "comparable_set_size": k,
                "final_feasibility_rate": final_feasibility,
                **sampled,
                "runtime_seconds": runtime_seconds,
                "n_evaluations": n_evaluations,
                "Cu_out_mean_g_L": float(selected_f[:, 0].mean()),
                "As_out_mean_g_L": float((-selected_f[:, 1]).mean()),
                "E_total_mean_kWh": float(selected_f[:, 2].mean()),
                "Net_profit_mean_1e4_CNY_batch": float((-selected_f[:, 3]).mean() / 1.0e4),
                "normalization_ideal": json.dumps(ideal.tolist()),
                "normalization_nadir": json.dumps(nadir.tolist()),
                "indicator_cardinality_method": "uniform sampling without replacement",
                "source_file": str(record.source_file),
            })
            audit_rows.append({
                "condition": f"Condition {condition}",
                "method": record.method,
                "seed": record.seed if record.seed is not None else "archive",
                "source_count": record.source_count,
                "common_domain_count": record.common_domain_count,
                "feasible_count": record.feasible_count,
                "full_nondominated_count": len(record.f),
                "comparable_set_size": k,
            })
            cfg = CONDITIONS[condition]
            for local_id, (x, f, g) in enumerate(zip(selected_x, selected_f, selected_g), start=1):
                row = {
                    "condition": f"Condition {condition}",
                    "method": record.method,
                    "seed": record.seed if record.seed is not None else "archive",
                    "representative_id": local_id,
                }
                row.update({name: float(value) for name, value in zip(cfg["variables"], x)})
                row.update({
                    "Cu_out": float(f[0]),
                    "As_out": float(-f[1]),
                    "E_total": float(f[2]),
                    "Net_profit_CNY": float(-f[3]),
                    **{f"G{i + 1}": float(value) for i, value in enumerate(g)},
                })
                selected_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    metrics["method_order"] = metrics["method"].map(METHOD_ORDER)
    metrics = metrics.sort_values(["condition", "method_order", "seed"]).drop(columns="method_order")
    selected = pd.DataFrame(selected_rows)
    audit = pd.DataFrame(audit_rows)
    reference_frame = pd.concat(reference_rows, ignore_index=True)
    return metrics, selected, audit, reference_frame


def mean_sd(series: pd.Series, digits: int = 4) -> str:
    values = series.dropna().astype(float)
    if len(values) == 0:
        return "not recorded"
    if len(values) == 1:
        return f"{values.iloc[0]:.{digits}f}"
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"


def build_summaries(metrics: pd.DataFrame):
    summary_rows: list[dict] = []
    table_rows: list[dict] = []
    for (condition, method), group in metrics.groupby(["condition", "method"], sort=False):
        row = {
            "condition": condition,
            "method": method,
            "n_runs": len(group),
            "comparable_set_size": int(group["comparable_set_size"].iloc[0]),
            "feasibility_rate_mean": group["final_feasibility_rate"].mean(),
            "feasibility_rate_sd": group["final_feasibility_rate"].std(ddof=1),
            "HV_mean": group["normalized_HV"].mean(),
            "HV_sd": group["normalized_HV"].std(ddof=1),
            "IGD_plus_mean": group["normalized_IGD_plus"].mean(),
            "IGD_plus_sd": group["normalized_IGD_plus"].std(ddof=1),
            "spacing_mean": group["normalized_spacing"].mean(),
            "spacing_sd": group["normalized_spacing"].std(ddof=1),
            "runtime_seconds_mean": group["runtime_seconds"].mean(),
            "runtime_seconds_sd": group["runtime_seconds"].std(ddof=1),
            "Cu_out_mean": group["Cu_out_mean_g_L"].mean(),
            "As_out_mean": group["As_out_mean_g_L"].mean(),
            "E_total_mean": group["E_total_mean_kWh"].mean(),
            "Net_profit_mean_1e4_CNY": group["Net_profit_mean_1e4_CNY_batch"].mean(),
        }
        summary_rows.append(row)
        table_rows.append({
            "Condition": condition,
            "Method": method,
            "Runs / archive": "1 archive" if method == "ESRL-CMO" else f"{len(group)} runs",
            "Feasible ND points before sampling": mean_sd(group["full_nondominated_count"], 1),
            "Comparable set size": int(group["comparable_set_size"].iloc[0]),
            "Feasibility assessment": (
                f"Archive {group['final_feasibility_rate'].iloc[0]:.4f}"
                if method == "ESRL-CMO" else mean_sd(group["final_feasibility_rate"], 4)
            ),
            "Normalized HV ↑": mean_sd(group["normalized_HV"], 4),
            "Normalized IGD+ ↓": mean_sd(group["normalized_IGD_plus"], 4),
            "Normalized spacing ↓": mean_sd(group["normalized_spacing"], 4),
        })
    summary = pd.DataFrame(summary_rows)
    summary["method_order"] = summary["method"].map(METHOD_ORDER)
    summary = summary.sort_values(["condition", "method_order"]).drop(columns="method_order")
    table = pd.DataFrame(table_rows)
    table["method_order"] = table["Method"].map(METHOD_ORDER)
    table = table.sort_values(["Condition", "method_order"]).drop(columns="method_order")
    return summary, table


def execute(args: argparse.Namespace) -> None:
    if args.target_size < 2:
        raise ValueError("--target-size must be at least 2")
    if args.subsamples < 1:
        raise ValueError("--subsamples must be positive")
    metadata_path = args.input_dir / "run_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Matched optimizer metadata is missing; the full run is incomplete: {metadata_path}"
        )
    module = load_source_module()
    records = discover_records(module, args.input_dir)
    metadata = pd.read_csv(metadata_path)
    metrics, selected, audit, reference = metric_analysis(
        records, args.target_size, metadata, args.subsamples, args.analysis_seed
    )
    summary, table_s25 = build_summaries(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "indicator_by_run.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "indicator_summary.csv", index=False, encoding="utf-8-sig")
    table_s25.to_csv(args.output_dir / "table_s25_algorithm_comparison.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(args.output_dir / "representative_solutions.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(args.output_dir / "candidate_filter_audit.csv", index=False, encoding="utf-8-sig")
    reference.to_csv(args.output_dir / "pooled_reference_front_normalized.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(args.output_dir / "RVEA_ESRLCMO_NSGAII_comparison.xlsx", engine="openpyxl") as writer:
        table_s25.to_excel(writer, sheet_name="Table_S25", index=False)
        summary.to_excel(writer, sheet_name="Indicator_Summary", index=False)
        metrics.to_excel(writer, sheet_name="Indicator_By_Run", index=False)
        audit.to_excel(writer, sheet_name="Filter_Audit", index=False)
        selected.to_excel(writer, sheet_name="Representative_Solutions", index=False)
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps({
        "fixed_runtime": FIXED_RUNTIME,
        "feasibility_tolerance": FEASIBILITY_TOLERANCE,
        "domain_tolerance": DOMAIN_TOLERANCE,
        "common_domains": {str(k): {"xl": v["xl"], "xu": v["xu"]} for k, v in CONDITIONS.items()},
        "target_size_cap": args.target_size,
        "subsamples": args.subsamples,
        "analysis_seed": args.analysis_seed,
        "actual_comparable_set_size": {
            condition: int(group["comparable_set_size"].iloc[0])
            for condition, group in metrics.groupby("condition")
        },
        "objectives": ["Cu_out", "-As_out", "E_total", "-Net_profit_CNY"],
        "normalization": "joint ideal/nadir per condition across all full feasible nondominated sets",
        "reference_front": "pooled nondominated union of all full jointly normalized sets",
        "indicator_cardinality_control": "uniform random sampling without replacement; per-run indicators are means across subsamples",
        "representative_selection": "deterministic extreme-preserving maximin subset, used only for exported representative solutions",
        "spacing_definition": "pymoo SpacingIndicator; city-block nearest-neighbor distance; population denominator n",
        "HV_reference_point": [1.1, 1.1, 1.1, 1.1],
        "RVEA_and_NSGAII_repeats": [11, 23, 42],
        "ESRL_CMO_repeat_status": "one archived policy-derived candidate set; no seed SD available",
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def print_plan(args: argparse.Namespace) -> None:
    print(json.dumps({
        "status": "NOT RUN - analysis gate",
        "matched_input": str(args.input_dir),
        "output": str(args.output_dir),
        "fixed_runtime": FIXED_RUNTIME,
        "target_size_cap": args.target_size,
        "subsamples": args.subsamples,
        "analysis_seed": args.analysis_seed,
        "methods": ["ESRL-CMO archive", "matched NSGA-II", "RVEA"],
        "steps": [
            "common-domain filtering",
            "common surrogate/objective/constraint re-evaluation",
            "feasible nondominated filtering",
            "equal-cardinality uniform subsampling per condition",
            "pooled joint normalization and HV/IGD+/spacing",
        ],
    }, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.execute:
        execute(args)
    else:
        print_plan(args)


if __name__ == "__main__":
    main()
