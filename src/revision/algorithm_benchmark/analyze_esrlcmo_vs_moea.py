"""Compare archived ESRL-CMO candidates with fresh constrained-MOEA runs.

This analysis does not modify the source optimizers. It harmonizes objective
orientation and profit units, optionally restricts every method to the shared
decision domain, selects an equal-size objective-space subset, and evaluates
HV, IGD+, spacing, and pairwise coverage using a pooled empirical reference
front for each condition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "result"
MOEA_ROOT = ROOT / "constrained_moea_tuned_benchmark"
OUTPUT_ROOT = ROOT / "esrlcmo_vs_moea"
ALGORITHMS = ("nsga2", "nsga3", "ctaea")
LABELS = {
    "esrlcmo": "ESRL-CMO",
    "nsga2": "NSGA-II",
    "nsga3": "NSGA-III",
    "ctaea": "C-TAEA",
}
SEEDS = (11, 23, 42, 57, 89)
HV_REFERENCE = np.full(4, 1.10, dtype=float)

# Intersections of the implemented PPO and evolutionary decision bounds.
COMMON_BOUNDS = {
    1: {
        "Cu_in": (29.0, 55.0), "TA": (48.0, 65.0),
        "IA": (8000.0, 27000.0), "QA": (111.0, 123.0), "t": (2.0, 8.0),
    },
    2: {
        "Cu_in": (29.0, 55.0), "TB": (55.0, 65.0),
        "IB": (8000.0, 27000.0), "QB": (113.0, 122.0), "t": (2.0, 8.0),
    },
    3: {
        "Cu_in": (29.0, 55.0), "TA": (40.0, 65.0),
        "IA": (10000.0, 22000.0), "QA": (113.0, 122.0),
        "TB": (40.0, 65.0), "IB": (10000.0, 24000.0),
        "QB": (114.0, 122.0), "t": (2.0, 8.0),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--domain", choices=("common", "native"), default="common")
    parser.add_argument("--moea-root", type=Path, default=MOEA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def objective_matrix(frame: pd.DataFrame, method: str) -> np.ndarray:
    profit = frame["Net_profit"].to_numpy(float)
    if method == "esrlcmo":
        profit = profit * 10000.0
    return np.column_stack([
        frame["Cu_out"].to_numpy(float),
        -frame["As_out"].to_numpy(float),
        frame["E_total"].to_numpy(float),
        -profit,
    ])


def common_domain_mask(frame: pd.DataFrame, condition: int) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for column, (low, high) in COMMON_BOUNDS[condition].items():
        values = frame[column].to_numpy(float)
        mask &= np.isfinite(values) & (values >= low - 1.0e-9) & (values <= high + 1.0e-9)
    return mask


def nondominated(values: np.ndarray) -> np.ndarray:
    values = np.unique(np.asarray(values, dtype=float), axis=0)
    if not len(values):
        return values
    indices = NonDominatedSorting().do(values, only_non_dominated_front=True)
    return values[indices]


def maximin_subset(values: np.ndarray, size: int) -> np.ndarray:
    values = np.unique(np.asarray(values, dtype=float), axis=0)
    if len(values) <= size:
        return values
    selected: list[int] = []
    for objective in range(values.shape[1]):
        candidate = int(np.argmin(values[:, objective]))
        if candidate not in selected:
            selected.append(candidate)
    min_distance = np.full(len(values), np.inf)
    for index in selected:
        min_distance = np.minimum(min_distance, np.linalg.norm(values - values[index], axis=1))
    min_distance[selected] = -np.inf
    while len(selected) < size:
        index = int(np.argmax(min_distance))
        selected.append(index)
        min_distance = np.minimum(min_distance, np.linalg.norm(values - values[index], axis=1))
        min_distance[selected] = -np.inf
    return values[selected]


def spacing(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.std(distances.min(axis=1), ddof=1))


def coverage(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of B weakly dominated by at least one point in A."""
    if not len(b):
        return float("nan")
    dominated = np.zeros(len(b), dtype=bool)
    for point in a:
        dominated |= np.all(point <= b + 1.0e-12, axis=1) & np.any(point < b - 1.0e-12, axis=1)
    return float(dominated.mean())


def load_records(condition: int, domain: str, moea_root: Path) -> list[dict]:
    records: list[dict] = []
    esrl_path = RESULT_ROOT / f"outputs_condition{condition}" / f"pareto_ppo_condition{condition}.csv"
    esrl = pd.read_csv(esrl_path)
    records.append({"method": "esrlcmo", "seed": 42, "path": esrl_path, "frame": esrl})
    for method in ALGORITHMS:
        for seed in SEEDS:
            path = moea_root / method / f"seed_{seed}" / f"pareto_{method}_condition{condition}.csv"
            records.append({"method": method, "seed": seed, "path": path, "frame": pd.read_csv(path)})

    for record in records:
        frame = record["frame"]
        finite = np.isfinite(objective_matrix(frame, record["method"])).all(axis=1)
        if domain == "common":
            finite &= common_domain_mask(frame, condition)
        record["n_original"] = int(len(frame))
        record["n_domain"] = int(finite.sum())
        record["f_raw"] = objective_matrix(frame.loc[finite].copy(), record["method"])
        record["f_nd"] = nondominated(record["f_raw"])
    return records


def analyze_condition(condition: int, target_size: int, domain: str, moea_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = load_records(condition, domain, moea_root)
    available = [record["f_nd"] for record in records if len(record["f_nd"])]
    pooled = np.vstack(available)
    ideal, nadir = pooled.min(axis=0), pooled.max(axis=0)
    span = np.where(nadir > ideal, nadir - ideal, 1.0)
    pooled_norm = (pooled - ideal) / span
    reference = nondominated(pooled_norm)
    hv = HV(ref_point=HV_REFERENCE)
    igd_plus = IGDPlus(reference)

    metric_rows: list[dict] = []
    selected_by_key: dict[tuple[str, int], np.ndarray] = {}
    audit_rows: list[dict] = []
    for record in records:
        norm = (record["f_nd"] - ideal) / span if len(record["f_nd"]) else record["f_nd"]
        selected = maximin_subset(norm, target_size)
        key = (record["method"], int(record["seed"]))
        selected_by_key[key] = selected
        audit_rows.append({
            "condition": condition,
            "method": LABELS[record["method"]],
            "seed": record["seed"],
            "n_original": record["n_original"],
            "n_in_domain": record["n_domain"],
            "n_nondominated": len(record["f_nd"]),
            "n_selected": len(selected),
            "source_file": str(record["path"]),
        })
        metric_rows.append({
            "condition": f"Condition {condition}",
            "method": LABELS[record["method"]],
            "seed": record["seed"],
            "n_selected": len(selected),
            "HV": float(hv(selected)),
            "IGD_plus": float(igd_plus(selected)),
            "spacing": spacing(selected),
        })

    coverage_rows: list[dict] = []
    esrl = selected_by_key[("esrlcmo", 42)]
    for method in ALGORITHMS:
        for seed in SEEDS:
            other = selected_by_key[(method, seed)]
            coverage_rows.append({
                "condition": f"Condition {condition}",
                "comparator": LABELS[method],
                "seed": seed,
                "C_ESRLCMO_over_comparator": coverage(esrl, other),
                "C_comparator_over_ESRLCMO": coverage(other, esrl),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(coverage_rows), pd.DataFrame(audit_rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (condition, method), part in metrics.groupby(["condition", "method"], sort=True):
        rows.append({
            "condition": condition,
            "method": method,
            "n_runs": len(part),
            "HV_mean": part.HV.mean(),
            "HV_sd": part.HV.std(ddof=1) if len(part) > 1 else np.nan,
            "IGD_plus_mean": part.IGD_plus.mean(),
            "IGD_plus_sd": part.IGD_plus.std(ddof=1) if len(part) > 1 else np.nan,
            "spacing_mean": part.spacing.mean(),
            "spacing_sd": part.spacing.std(ddof=1) if len(part) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_all, coverage_all, audit_all = [], [], []
    for condition in (1, 2, 3):
        metrics, coverage_table, audit = analyze_condition(
            condition, args.target_size, args.domain, args.moea_root
        )
        metrics_all.append(metrics)
        coverage_all.append(coverage_table)
        audit_all.append(audit)
    metrics = pd.concat(metrics_all, ignore_index=True)
    coverage_table = pd.concat(coverage_all, ignore_index=True)
    audit = pd.concat(audit_all, ignore_index=True)
    summary = summarize(metrics)
    coverage_summary = coverage_table.groupby(["condition", "comparator"], sort=True).agg(
        C_ESRLCMO_mean=("C_ESRLCMO_over_comparator", "mean"),
        C_ESRLCMO_sd=("C_ESRLCMO_over_comparator", "std"),
        C_comparator_mean=("C_comparator_over_ESRLCMO", "mean"),
        C_comparator_sd=("C_comparator_over_ESRLCMO", "std"),
    ).reset_index()

    suffix = f"{args.domain}_n{args.target_size}"
    metrics.to_csv(args.output_dir / f"indicator_by_run_{suffix}.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / f"indicator_summary_{suffix}.csv", index=False, encoding="utf-8-sig")
    coverage_table.to_csv(args.output_dir / f"coverage_by_run_{suffix}.csv", index=False, encoding="utf-8-sig")
    coverage_summary.to_csv(args.output_dir / f"coverage_summary_{suffix}.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(args.output_dir / f"candidate_audit_{suffix}.csv", index=False, encoding="utf-8-sig")
    with (args.output_dir / f"method_{suffix}.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "domain": args.domain,
            "target_size": args.target_size,
            "objective_orientation": ["Cu_out", "-As_out", "E_total", "-Net_profit_CNY"],
            "ESRLCMO_profit_conversion": "Net_profit in archived PPO CSV multiplied by 10000",
            "normalization": "pooled ideal/nadir within each condition after domain filtering",
            "reference_front": "nondominated union of all included method/run fronts",
            "selection": "deterministic objective-space maximin subset initialized with objective extremes",
            "HV_reference": HV_REFERENCE.tolist(),
            "moea_root": str(args.moea_root),
            "common_bounds": COMMON_BOUNDS if args.domain == "common" else None,
            "limitation": "ESRL-CMO is one archived policy output; evolutionary methods have five seeds.",
        }, handle, ensure_ascii=False, indent=2)
    print(summary.to_string(index=False))
    print("\nCoverage\n", coverage_summary.to_string(index=False))
    print("\nAudit\n", audit.to_string(index=False))


if __name__ == "__main__":
    main()
