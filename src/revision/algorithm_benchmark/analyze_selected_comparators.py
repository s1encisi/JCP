"""Compare archived ESRL-CMO with existing NSGA-II, RVEA, and C-TAEA fronts.

No optimizer is rerun. The script harmonizes objective orientation and profit
units, selects equal-size subsets, and recomputes condition-wise HV, IGD+,
spacing, feasibility, and pairwise coverage from the existing result files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "result"
FORMAL_ROOT = ROOT / "esrlcmo_domain_benchmark"
SCREEN_ROOT = ROOT.parents[3] / "moea_algorithm_screening"
OUTPUT_ROOT = ROOT / "selected_comparator_analysis"
HV_REFERENCE = np.full(4, 1.10, dtype=float)
TARGET_SIZE = 38

METHODS = {
    "esrlcmo": {
        "label": "ESRL-CMO",
        "seeds": (42,),
        "evaluations": "Archived policy front",
    },
    "nsga2": {
        "label": "NSGA-II",
        "seeds": (11, 23, 42, 57, 89),
        "evaluations": "180,000",
    },
    "rvea": {
        "label": "RVEA",
        "seeds": (11, 42, 89),
        "evaluations": "10,000",
    },
    "ctaea": {
        "label": "C-TAEA",
        "seeds": (11, 23, 42, 57, 89),
        "evaluations": "180,000",
    },
}


def objective_matrix(frame: pd.DataFrame, method: str) -> np.ndarray:
    profit = frame["Net_profit"].to_numpy(float)
    if method == "esrlcmo":
        profit = profit * 10000.0
    return np.column_stack(
        [
            frame["Cu_out"].to_numpy(float),
            -frame["As_out"].to_numpy(float),
            frame["E_total"].to_numpy(float),
            -profit,
        ]
    )


def nondominated(values: np.ndarray) -> np.ndarray:
    values = np.unique(np.asarray(values, dtype=float), axis=0)
    if not len(values):
        return values
    return values[
        NonDominatedSorting().do(values, only_non_dominated_front=True)
    ]


def maximin_subset(values: np.ndarray, size: int) -> np.ndarray:
    values = np.unique(np.asarray(values, dtype=float), axis=0)
    if len(values) <= size:
        return values
    selected: list[int] = []
    for objective in range(values.shape[1]):
        candidate = int(np.argmin(values[:, objective]))
        if candidate not in selected:
            selected.append(candidate)
    nearest = np.full(len(values), np.inf)
    for index in selected:
        nearest = np.minimum(nearest, np.linalg.norm(values - values[index], axis=1))
    nearest[selected] = -np.inf
    while len(selected) < size:
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(nearest, np.linalg.norm(values - values[index], axis=1))
        nearest[selected] = -np.inf
    return values[selected]


def spacing(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.std(distances.min(axis=1), ddof=1))


def coverage(a: np.ndarray, b: np.ndarray) -> float:
    dominated = np.zeros(len(b), dtype=bool)
    for point in a:
        dominated |= np.all(point <= b + 1.0e-12, axis=1) & np.any(
            point < b - 1.0e-12, axis=1
        )
    return float(dominated.mean())


def source_path(method: str, seed: int, condition: int) -> Path:
    if method == "esrlcmo":
        return (
            RESULT_ROOT
            / f"outputs_condition{condition}"
            / f"pareto_ppo_condition{condition}.csv"
        )
    root = SCREEN_ROOT if method == "rvea" else FORMAL_ROOT
    return (
        root
        / method
        / f"seed_{seed}"
        / f"pareto_{method}_condition{condition}.csv"
    )


def rvea_feasibility(condition: int, seed: int) -> float:
    metadata = pd.read_csv(SCREEN_ROOT / "eligible_run_metadata.csv")
    row = metadata[
        (metadata["algorithm"] == "rvea")
        & (metadata["condition"] == f"Condition {condition}")
        & (metadata["seed"] == seed)
    ].iloc[0]
    return float(row["final_population_feasibility_rate"])


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict] = []
    all_coverage: list[dict] = []
    all_audit: list[dict] = []
    reference_rows: list[pd.DataFrame] = []

    for condition in (1, 2, 3):
        records: list[dict] = []
        for method, info in METHODS.items():
            for seed in info["seeds"]:
                path = source_path(method, seed, condition)
                frame = pd.read_csv(path)
                raw = objective_matrix(frame, method)
                raw = raw[np.isfinite(raw).all(axis=1)]
                nd = nondominated(raw)
                records.append(
                    {
                        "method": method,
                        "seed": seed,
                        "path": path,
                        "n_original": len(frame),
                        "raw": raw,
                        "nd": nd,
                    }
                )

        pooled = np.vstack([record["nd"] for record in records])
        ideal, nadir = pooled.min(axis=0), pooled.max(axis=0)
        span = np.where(nadir > ideal, nadir - ideal, 1.0)
        pooled_norm = (pooled - ideal) / span
        reference = nondominated(pooled_norm)
        reference_rows.append(
            pd.DataFrame(
                reference,
                columns=["Cu_out_norm", "negative_As_out_norm", "E_total_norm", "negative_Net_profit_norm"],
            ).assign(condition=f"Condition {condition}")
        )
        hv = HV(ref_point=HV_REFERENCE)
        igd_plus = IGDPlus(reference)
        selected: dict[tuple[str, int], np.ndarray] = {}

        for record in records:
            normalized = (record["nd"] - ideal) / span
            subset = maximin_subset(normalized, TARGET_SIZE)
            key = (record["method"], record["seed"])
            selected[key] = subset
            feasibility = (
                rvea_feasibility(condition, record["seed"])
                if record["method"] == "rvea"
                else 1.0
            )
            all_metrics.append(
                {
                    "condition": f"Condition {condition}",
                    "method": METHODS[record["method"]]["label"],
                    "seed": record["seed"],
                    "evaluations": METHODS[record["method"]]["evaluations"],
                    "n_selected": len(subset),
                    "HV": float(hv(subset)),
                    "IGD_plus": float(igd_plus(subset)),
                    "spacing": spacing(subset),
                    "feasible_fraction": feasibility,
                }
            )
            all_audit.append(
                {
                    "condition": f"Condition {condition}",
                    "method": METHODS[record["method"]]["label"],
                    "seed": record["seed"],
                    "n_original": record["n_original"],
                    "n_nondominated": len(record["nd"]),
                    "n_selected": len(subset),
                    "source_file": str(record["path"]),
                }
            )

        esrl = selected[("esrlcmo", 42)]
        for method in ("nsga2", "rvea", "ctaea"):
            for seed in METHODS[method]["seeds"]:
                other = selected[(method, seed)]
                all_coverage.append(
                    {
                        "condition": f"Condition {condition}",
                        "comparator": METHODS[method]["label"],
                        "seed": seed,
                        "C_ESRLCMO_over_comparator": coverage(esrl, other),
                        "C_comparator_over_ESRLCMO": coverage(other, esrl),
                    }
                )

    metrics = pd.DataFrame(all_metrics)
    summary = (
        metrics.groupby(["condition", "method", "evaluations"], sort=True)
        .agg(
            n_runs=("seed", "size"),
            HV_mean=("HV", "mean"),
            HV_sd=("HV", "std"),
            IGD_plus_mean=("IGD_plus", "mean"),
            IGD_plus_sd=("IGD_plus", "std"),
            spacing_mean=("spacing", "mean"),
            spacing_sd=("spacing", "std"),
            feasible_fraction_mean=("feasible_fraction", "mean"),
            selected_size_mean=("n_selected", "mean"),
        )
        .reset_index()
    )
    coverage_by_run = pd.DataFrame(all_coverage)
    coverage_summary = (
        coverage_by_run.groupby(["condition", "comparator"], sort=True)
        .agg(
            C_ESRLCMO_mean=("C_ESRLCMO_over_comparator", "mean"),
            C_ESRLCMO_sd=("C_ESRLCMO_over_comparator", "std"),
            C_comparator_mean=("C_comparator_over_ESRLCMO", "mean"),
            C_comparator_sd=("C_comparator_over_ESRLCMO", "std"),
        )
        .reset_index()
    )

    metrics.to_csv(OUTPUT_ROOT / "indicator_by_run.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_ROOT / "indicator_summary.csv", index=False, encoding="utf-8-sig")
    coverage_by_run.to_csv(OUTPUT_ROOT / "coverage_by_run.csv", index=False, encoding="utf-8-sig")
    coverage_summary.to_csv(OUTPUT_ROOT / "coverage_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_audit).to_csv(OUTPUT_ROOT / "candidate_audit.csv", index=False, encoding="utf-8-sig")
    pd.concat(reference_rows, ignore_index=True).to_csv(
        OUTPUT_ROOT / "pooled_reference_front_normalized.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT_ROOT / "method.json").write_text(
        json.dumps(
            {
                "optimization_rerun": False,
                "methods": METHODS,
                "target_size": TARGET_SIZE,
                "objective_orientation": ["Cu_out", "-As_out", "E_total", "-Net_profit_CNY"],
                "normalization": "joint condition-wise min-max across all included existing fronts",
                "reference_front": "pooled empirical nondominated union",
                "HV_reference": HV_REFERENCE.tolist(),
                "selection": "deterministic maximin subset initialized with four objective extremes",
                "scope": "end-point result-set comparison; unequal evaluation budgets are reported explicitly",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print("\nCoverage\n", coverage_summary.to_string(index=False))


if __name__ == "__main__":
    main()
