"""Measurement-input and execution-deviation stress tests.

The test starts from deterministic representative subsets of the archived
ESRL-CMO and NSGA-II recommendations.  It does not retrain or rerun either
optimizer.  Measurement uncertainty perturbs values supplied to the offline
surrogate/constraint check; execution deviation perturbs only the implemented
temperature/current/flow settings.  Batch duration is always fixed.

No experiment runs unless ``--execute`` is supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .robustness_common import (
        OriginalEvaluators, PROJECT_ROOT, STATE_COLUMNS, load_archive,
        scenario_statistics, select_representatives, state_matrix,
    )
except ImportError:  # Direct script execution.
    from robustness_common import (
        OriginalEvaluators, PROJECT_ROOT, STATE_COLUMNS, load_archive,
        scenario_statistics, select_representatives, state_matrix,
    )


OUTPUT_ROOT = PROJECT_ROOT / "analyses" / "robustness" / "measurement_execution_results"
METHODS = ("ESRL-CMO", "NSGA-II")
NOISE_LEVELS = (0.01, 0.03, 0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measurement/execution deviation stress test")
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--n-recommendations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--month", type=int, default=6)
    parser.add_argument("--day", type=int, default=15)
    parser.add_argument("--hour", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true",
                        help="Run the experiment; omission prints the plan only")
    args = parser.parse_args()
    if args.replicates <= 0 or args.n_recommendations <= 0:
        parser.error("replicates and n-recommendations must be positive")
    return args


def perturb(states: np.ndarray, indices: list[int], low: np.ndarray,
            high: np.ndarray, level: float, z_scores: np.ndarray) -> tuple[np.ndarray, float]:
    result = states.copy()
    span = high - low
    for idx in indices:
        result[:, idx] += z_scores[:, idx] * span[idx] * level
    unclipped = result.copy()
    result = np.clip(result, low, high)
    return result, float(np.mean(result[:, indices] != unclipped[:, indices]))


def execute(args: argparse.Namespace) -> None:
    runtime = {"year": args.year, "month": args.month,
               "day": args.day, "hour": args.hour}
    evaluator = OriginalEvaluators(runtime)
    summary_rows: list[dict] = []
    selected_rows: list[pd.DataFrame] = []

    for condition in (1, 2, 3):
        # t is the final state entry. Measurement uncertainty includes Cu_in
        # and the active control readings; execution deviation excludes Cu_in.
        measurement_indices = list(range(len(STATE_COLUMNS[condition]) - 1))
        execution_indices = list(range(1, len(STATE_COLUMNS[condition]) - 1))
        method_data = {}
        for method in METHODS:
            selected = select_representatives(
                load_archive(method, condition), args.n_recommendations
            )
            selected.insert(0, "method", method)
            selected.insert(1, "condition", f"Condition {condition}")
            selected_rows.append(selected)
            states = state_matrix(selected, condition)
            method_data[method] = (states, evaluator.evaluate(method, condition, states))

        n = min(len(values[0]) for values in method_data.values())
        if any(len(values[0]) != n for values in method_data.values()):
            raise ValueError("Equal representative counts are required for common random numbers")

        for level_index, level in enumerate(NOISE_LEVELS):
            for replicate in range(args.replicates):
                # Common random numbers: both methods receive the same standard-normal draws.
                measure_seed = args.seed + 100000 * condition + 1000 * level_index + replicate
                execute_seed = args.seed + 200000 * condition + 1000 * level_index + replicate
                z_measure = np.random.default_rng(measure_seed).normal(
                    size=(n, len(STATE_COLUMNS[condition]))
                )
                z_execute = np.random.default_rng(execute_seed).normal(
                    size=(n, len(STATE_COLUMNS[condition]))
                )
                for method in METHODS:
                    states, clean = method_data[method]
                    low, high = evaluator.bounds(method, condition)
                    measured_states, measured_clip = perturb(
                        states, measurement_indices, low, high, level, z_measure
                    )
                    executed_states, executed_clip = perturb(
                        states, execution_indices, low, high, level, z_execute
                    )
                    for scenario, perturbed_states, clip_rate, seed, indices in (
                        ("measurement_input_uncertainty", measured_states,
                         measured_clip, measure_seed, measurement_indices),
                        ("execution_deviation", executed_states,
                         executed_clip, execute_seed, execution_indices),
                    ):
                        evaluated = evaluator.evaluate(method, condition, perturbed_states)
                        row = scenario_statistics(evaluated, clean)
                        row.update({
                            "scenario": scenario,
                            "condition": f"Condition {condition}",
                            "method": method,
                            "noise_sigma_fraction_of_original_range": level,
                            "replicate": replicate,
                            "random_seed": seed,
                            "n_recommendations": n,
                            "perturbed_variables": " | ".join(
                                STATE_COLUMNS[condition][i] for i in indices
                            ),
                            "clipping_rate": clip_rate,
                        })
                        summary_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    selected = pd.concat(selected_rows, ignore_index=True)
    summary.to_csv(args.output_dir / "measurement_execution_replicate_summary.csv",
                   index=False, encoding="utf-8-sig")
    selected.to_csv(args.output_dir / "selected_archived_recommendations.csv",
                    index=False, encoding="utf-8-sig")

    aggregate_columns = {
        "feasibility_rate": ["mean", "std"],
        "Cu_out_mean_absolute_shift": ["mean", "std"],
        "As_out_mean_absolute_shift": ["mean", "std"],
        "E_total_mean_absolute_shift": ["mean", "std"],
        "Net_profit_10k_CNY_mean_shift": ["mean", "std"],
    }
    aggregate = summary.groupby(
        ["scenario", "condition", "method", "noise_sigma_fraction_of_original_range"],
        sort=True,
    ).agg(aggregate_columns)
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate.to_csv(args.output_dir / "measurement_execution_aggregate_summary.csv",
                     index=False, encoding="utf-8-sig")
    metadata = {
        "status": "completed",
        "noise_levels": list(NOISE_LEVELS),
        "replicates": args.replicates,
        "representatives_per_method_condition": args.n_recommendations,
        "selection": "deterministic maximin coverage of Cu_out/-As_out/log1p(E_total)",
        "runtime_features": runtime,
        "feasibility_tolerance": {"ESRL-CMO": "each cost < 1e-3",
                                  "NSGA-II": "each g <= 1e-8"},
        "measurement_interpretation": "sensitivity of offline surrogate prediction and C1-C5 checking to uncertain measured inputs",
        "execution_interpretation": "sensitivity to deviations of implemented T/I/Q set points",
        "unchanged": ["archived recommendations", "original surrogates",
                      "method-specific economic equations", "C1-C5 constraints",
                      "original variable ranges", "batch duration"],
        "limitations": [
            "The 1/3/5% levels are generic fractions of original ranges, not instrument specifications.",
            "Independent Gaussian errors do not represent correlated bias, delay, dropout, or drift.",
            "Condition 3 absolute cross-method profit is not compared because the archived method-specific flow conventions differ.",
        ],
    }
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def print_plan(args: argparse.Namespace) -> None:
    print(json.dumps({
        "status": "NOT RUN - code-generation/review mode",
        "experiments": {
            "measurement_input_uncertainty": "perturb measured Cu_in and active control readings used by offline surrogate/C1-C5 checks",
            "execution_deviation": "perturb implemented T/I/Q settings only",
        },
        "noise_sigma_fraction_of_original_range": list(NOISE_LEVELS),
        "replicates": args.replicates,
        "representatives_per_method_condition": args.n_recommendations,
        "methods": list(METHODS),
        "duration_t": "held fixed",
        "control": "Same standard-normal draws; original method-specific evaluators, constraints and ranges retained.",
        "planned_output": str(args.output_dir),
        "execution_gate": "Supply --execute only after review.",
    }, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if not args.execute:
        print_plan(args)
        return
    execute(args)


if __name__ == "__main__":
    main()
