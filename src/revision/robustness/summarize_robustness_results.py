"""Audit and summarize completed robustness stress-test outputs.

This script does not rerun either optimizer and does not alter the archived
experiment outputs.  It re-evaluates only the deterministic, already-selected
recommendations at the unperturbed baseline so that perturbation results can
be reported against an explicit baseline.  All output files are written next
to this script.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from robustness_common import OriginalEvaluators, STATE_COLUMNS


ROOT = Path(__file__).resolve().parent
MEASUREMENT_ROOT = ROOT / "measurement_execution_results"
FEED_ROOT = ROOT / "feed_disturbance_results"
RUNTIME = {"year": 2025, "month": 6, "day": 15, "hour": 8}
METHODS = ("ESRL-CMO", "NSGA-II")
CONDITIONS = tuple(f"Condition {i}" for i in (1, 2, 3))


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _finite_missing(frame: pd.DataFrame) -> int:
    numeric = frame.select_dtypes(include=[np.number])
    return int((~np.isfinite(numeric.to_numpy(float))).sum())


def _evaluate_baseline(selected: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # The original PPO module prints its configuration whenever a condition is
    # selected; suppress that diagnostic output in this read-only summary.
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = OriginalEvaluators(RUNTIME)
        rows = []
        evaluated_cache = {}
        for condition_number, condition in enumerate(CONDITIONS, start=1):
            for method in METHODS:
                subset = selected.loc[
                    (selected["condition"] == condition)
                    & (selected["method"] == method)
                ].copy()
                states = subset[STATE_COLUMNS[condition_number]].to_numpy(float)
                evaluated = evaluator.evaluate(method, condition_number, states)
                evaluated_cache[(condition, method)] = evaluated
                row = {
                    "condition": condition,
                    "method": method,
                    "n_recommendations": len(states),
                    "baseline_feasibility_rate": float(np.mean(evaluated["feasible"])),
                    "baseline_Cu_out_mean_g_per_L": float(np.mean(evaluated["Cu_out"])),
                    "baseline_As_out_mean_g_per_L": float(np.mean(evaluated["As_out"])),
                    "baseline_E_total_mean_kWh": float(np.mean(evaluated["E_total"])),
                    "baseline_Net_profit_mean_10k_CNY": float(
                        np.mean(evaluated["Net_profit_10k_CNY"])
                    ),
                    "feasibility_tolerance": "each cost < 1e-3"
                    if method == "ESRL-CMO" else "each g <= 1e-8",
                }
                for i in range(5):
                    row[f"baseline_C{i + 1}_violation_rate"] = float(
                        np.mean(evaluated["constraint_violated"][:, i])
                    )
                rows.append(row)
    return pd.DataFrame(rows), evaluated_cache


def _stat(group: pd.DataFrame, column: str, suffix: str) -> dict[str, float]:
    values = group[column].to_numpy(float)
    return {
        f"{suffix}_mean": float(np.mean(values)),
        f"{suffix}_sd": float(np.std(values, ddof=1)),
        f"{suffix}_min": float(np.min(values)),
        f"{suffix}_max": float(np.max(values)),
    }


def _measurement_summary(frame: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    highest = frame.loc[
        np.isclose(frame["noise_sigma_fraction_of_original_range"], 0.05)
    ].copy()
    rows = []
    for keys, group in highest.groupby(["scenario", "condition", "method"], sort=True):
        scenario, condition, method = keys
        row = {
            "scenario": scenario,
            "condition": condition,
            "method": method,
            "noise_sigma_fraction_of_original_range": 0.05,
            "n_replicates": int(len(group)),
            "n_recommendations_per_replicate": int(group["n_recommendations"].iloc[0]),
            "perturbed_variables": str(group["perturbed_variables"].iloc[0]),
        }
        for column, suffix in (
            ("feasibility_rate", "feasibility_rate"),
            ("Cu_out_mean_absolute_shift", "Cu_out_abs_change_g_per_L"),
            ("As_out_mean_absolute_shift", "As_out_abs_change_g_per_L"),
            ("E_total_mean_absolute_shift", "E_total_abs_change_kWh"),
            ("Net_profit_10k_CNY_mean_shift", "Net_profit_signed_change_10k_CNY"),
            ("Net_profit_10k_CNY_mean_absolute_shift", "Net_profit_abs_change_10k_CNY"),
            ("clipping_rate", "clipping_rate"),
        ):
            row.update(_stat(group, column, suffix))
        for i in range(5):
            row[f"C{i + 1}_violation_rate_mean"] = float(group[f"C{i + 1}_violation_rate"].mean())
        base = baseline.loc[
            (baseline["condition"] == condition) & (baseline["method"] == method)
        ].iloc[0]
        row["baseline_feasibility_rate"] = float(base["baseline_feasibility_rate"])
        rows.append(row)
    return pd.DataFrame(rows)


def _feed_summary(frame: pd.DataFrame, steps: pd.DataFrame,
                  baseline: pd.DataFrame) -> pd.DataFrame:
    baseline_index = baseline.set_index(["condition", "method"])
    rows = []
    for keys, group in frame.groupby(["condition", "method", "mode"], sort=True):
        condition, method, mode = keys
        row = {
            "condition": condition,
            "method": method,
            "mode": mode,
            "n_replicates": int(len(group)),
            "n_recommendations_per_replicate": int(group["n_recommendations"].iloc[0]),
            "steps_per_replicate": int(group["steps"].iloc[0]),
            "historical_Cu_in_sd_g_per_L": float(group["historical_Cu_in_sd_g_per_L"].iloc[0]),
        }
        for column, suffix in (
            ("feasibility_rate", "feasibility_rate"),
            ("Cu_out_mean_absolute_shift", "Cu_out_abs_change_g_per_L"),
            ("As_out_mean_absolute_shift", "As_out_abs_change_g_per_L"),
            ("E_total_mean_absolute_shift", "E_total_abs_change_kWh"),
            ("Net_profit_10k_CNY_mean_shift", "Net_profit_signed_change_10k_CNY"),
            ("Net_profit_10k_CNY_mean_absolute_shift", "Net_profit_abs_change_10k_CNY"),
            ("Cu_in_clipping_rate", "Cu_in_clipping_rate"),
        ):
            row.update(_stat(group, column, suffix))

        base = baseline_index.loc[(condition, method)]
        row["baseline_feasibility_rate"] = float(base["baseline_feasibility_rate"])
        step_group = steps.loc[
            (steps["condition"] == condition)
            & (steps["method"] == method)
            & (steps["mode"] == mode)
        ].copy()
        row["Cu_in_disturbance_min_g_per_L"] = float(step_group["Cu_in_disturbance_g_per_L"].min())
        row["Cu_in_disturbance_max_g_per_L"] = float(step_group["Cu_in_disturbance_g_per_L"].max())
        row["minimum_step_feasibility_rate"] = float(step_group["feasibility_rate"].min())

        for metric, base_column, suffix in (
            ("mean_Cu_out", "baseline_Cu_out_mean_g_per_L", "step_peak_abs_Cu_out_change_g_per_L"),
            ("mean_As_out", "baseline_As_out_mean_g_per_L", "step_peak_abs_As_out_change_g_per_L"),
            ("mean_E_total", "baseline_E_total_mean_kWh", "step_peak_abs_E_total_change_kWh"),
            ("mean_Net_profit_10k_CNY", "baseline_Net_profit_mean_10k_CNY", "step_peak_abs_Net_profit_change_10k_CNY"),
        ):
            changes = step_group[metric].to_numpy(float) - float(base[base_column])
            row[suffix] = float(np.max(np.abs(changes)))
            if metric == "mean_Net_profit_10k_CNY":
                row["step_Net_profit_change_min_10k_CNY"] = float(np.min(changes))
                row["step_Net_profit_change_max_10k_CNY"] = float(np.max(changes))
        rows.append(row)
    return pd.DataFrame(rows)


def _worst_feed_modes(feed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, method), group in feed.groupby(["condition", "method"], sort=True):
        row = {"condition": condition, "method": method}
        criteria = {
            "lowest_mean_feasibility": ("feasibility_rate_mean", "min"),
            "lowest_single_step_feasibility": ("minimum_step_feasibility_rate", "min"),
            "largest_mean_abs_Cu_change": ("Cu_out_abs_change_g_per_L_mean", "max"),
            "largest_mean_abs_As_change": ("As_out_abs_change_g_per_L_mean", "max"),
            "largest_mean_abs_energy_change": ("E_total_abs_change_kWh_mean", "max"),
            "largest_peak_abs_profit_change": ("step_peak_abs_Net_profit_change_10k_CNY", "max"),
        }
        for label, (column, direction) in criteria.items():
            idx = group[column].idxmin() if direction == "min" else group[column].idxmax()
            row[f"{label}_mode"] = str(feed.loc[idx, "mode"])
            row[f"{label}_value"] = float(feed.loc[idx, column])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    selected_measurement_path = MEASUREMENT_ROOT / "selected_archived_recommendations.csv"
    selected_feed_path = FEED_ROOT / "selected_archived_recommendations.csv"
    selected_measurement = _read(selected_measurement_path)
    selected_feed = _read(selected_feed_path)
    measurement = _read(MEASUREMENT_ROOT / "measurement_execution_replicate_summary.csv")
    feed = _read(FEED_ROOT / "feed_disturbance_replicate_summary.csv")
    steps = _read(FEED_ROOT / "feed_disturbance_step_summary.csv")

    if not selected_measurement.equals(selected_feed):
        raise ValueError("The two experiments did not use identical selected recommendations")
    baseline, _ = _evaluate_baseline(selected_measurement)
    measurement_paper = _measurement_summary(measurement, baseline)
    feed_paper = _feed_summary(feed, steps, baseline)
    worst_feed = _worst_feed_modes(feed_paper)

    baseline.to_csv(ROOT / "robustness_baseline_summary.csv", index=False, encoding="utf-8-sig")
    measurement_paper.to_csv(
        ROOT / "measurement_execution_paper_summary.csv", index=False, encoding="utf-8-sig"
    )
    feed_paper.to_csv(
        ROOT / "feed_disturbance_paper_summary.csv", index=False, encoding="utf-8-sig"
    )
    worst_feed.to_csv(
        ROOT / "feed_disturbance_worst_modes.csv", index=False, encoding="utf-8-sig"
    )

    expected = {
        "measurement_replicate_rows": 3 * 2 * 3 * 30 * 2,
        "feed_replicate_rows": 3 * 2 * 4 * 30,
        "feed_step_rows": 3 * 2 * 4 * 30 * 20,
        "selected_recommendation_rows": 3 * 2 * 100,
    }
    observed = {
        "measurement_replicate_rows": int(len(measurement)),
        "feed_replicate_rows": int(len(feed)),
        "feed_step_rows": int(len(steps)),
        "selected_recommendation_rows": int(len(selected_measurement)),
    }
    audit = {
        "status": "completed",
        "expected_rows": expected,
        "observed_rows": observed,
        "row_counts_match": expected == observed,
        "identical_selected_recommendations": True,
        "nonfinite_numeric_values": {
            "measurement_replicate_summary": _finite_missing(measurement),
            "feed_replicate_summary": _finite_missing(feed),
            "feed_step_summary": _finite_missing(steps),
        },
        "duplicate_key_rows": {
            "measurement": int(measurement.duplicated([
                "scenario", "condition", "method",
                "noise_sigma_fraction_of_original_range", "replicate"
            ]).sum()),
            "feed": int(feed.duplicated([
                "condition", "method", "mode", "replicate"
            ]).sum()),
            "feed_steps": int(steps.duplicated([
                "condition", "method", "mode", "replicate", "timestep"
            ]).sum()),
        },
        "random_number_audit": {
            "measurement_both_methods_share_each_cell_seed": bool(
                (measurement.groupby([
                    "scenario", "condition", "noise_sigma_fraction_of_original_range",
                    "replicate", "random_seed"
                ])["method"].nunique() == 2).all()
            ),
            "measurement_each_cell_has_30_unique_seeds": bool(
                (measurement.groupby([
                    "scenario", "condition", "noise_sigma_fraction_of_original_range"
                ])["random_seed"].nunique() == 30).all()
            ),
            "measurement_unique_seeds_across_540_cells": int(
                measurement.drop_duplicates([
                    "scenario", "condition", "noise_sigma_fraction_of_original_range",
                    "replicate"
                ])["random_seed"].nunique()
            ),
            "measurement_cross_cell_seed_collisions": 540 - int(
                measurement.drop_duplicates([
                    "scenario", "condition", "noise_sigma_fraction_of_original_range",
                    "replicate"
                ])["random_seed"].nunique()
            ),
            "feed_both_methods_share_each_sequence_seed": bool(
                (feed.groupby([
                    "condition", "mode", "replicate", "sequence_seed"
                ])["method"].nunique() == 2).all()
            ),
            "feed_unique_sequence_seeds_across_360_sequences": int(
                feed.drop_duplicates([
                    "condition", "mode", "replicate"
                ])["sequence_seed"].nunique()
            ),
            "interpretation": (
                "Measurement seeds are unique within every reported cell and shared "
                "between methods. Ninety numeric seed values recur across different "
                "scenario-condition cells because of the arithmetic seed schedule; "
                "this does not duplicate observations within a cell but creates "
                "cross-cell correlation. Feed sequence seeds are globally unique."
            ),
        },
        "units": {
            "Cu_out": "g/L",
            "As_out": "g/L",
            "E_total": "kWh per archived batch recommendation",
            "Net_profit": "10^4 CNY per archived batch recommendation",
            "Cu_in_disturbance": "g/L",
        },
        "design_boundary": (
            "Synthetic open-loop stress tests using frozen archived recommendations, "
            "frozen surrogate models, frozen constraints/economic equations, and fixed "
            "batch duration; no field validation, online adaptation, re-optimization, "
            "or optimizer/policy retraining."
        ),
        "interpretive_cautions": [
            "The 1/3/5% Gaussian levels are fractions of the original variable ranges, not calibrated sensor specifications.",
            "Clipping censors perturbations at each method's original bounds; clipping rates are reported in the paper summaries.",
            "ESRL-CMO and NSGA-II retain their original feasibility tolerances (1e-3 versus 1e-8), so method-to-method feasibility differences near a boundary must be interpreted cautiously.",
            "Feed disturbances use the historical Cu_in sample standard deviation for amplitude but remain synthetic open-loop trajectories.",
            "Condition 3 profit changes are interpretable only within method because the archived method-specific flow conventions differ.",
            "Four NSGA-II Condition 2 and two NSGA-II Condition 3 representatives are infeasible at the fixed baseline runtime; therefore perturbed feasibility must be reported together with the 0.96 and 0.98 baseline rates.",
        ],
        "output_files": [
            "robustness_baseline_summary.csv",
            "measurement_execution_paper_summary.csv",
            "feed_disturbance_paper_summary.csv",
            "feed_disturbance_worst_modes.csv",
        ],
    }
    (ROOT / "robustness_audit_summary.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
