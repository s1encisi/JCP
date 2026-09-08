"""Controlled feed-Cu disturbance stress test for archived recommendations.

The two methods receive the same additive Cu_in disturbance sequences. Their
archived control variables are held fixed, and each perturbed state is
evaluated by that method's original surrogate/objective/constraint code. This
isolates feed disturbance from online re-optimization or policy adaptation.

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
        OriginalEvaluators, PROJECT_ROOT, load_archive, scenario_statistics,
        select_representatives, state_matrix,
    )
except ImportError:  # Direct script execution.
    from robustness_common import (
        OriginalEvaluators, PROJECT_ROOT, load_archive, scenario_statistics,
        select_representatives, state_matrix,
    )


OUTPUT_ROOT = PROJECT_ROOT / "analyses" / "robustness" / "feed_disturbance_results"
DATA_FILE = PROJECT_ROOT / "data" / "data_by_operation_mode_with_features.xlsx"
MODES = ("ar1", "step", "gradual_drift", "mixed")
METHODS = ("ESRL-CMO", "NSGA-II")
SHEETS = {1: "three_stage", 2: "four_stage", 3: "serial"}
CU_IN_COLUMN = "电积前液Cu（g/L）"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feed concentration disturbance experiment")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--n-recommendations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--month", type=int, default=6)
    parser.add_argument("--day", type=int, default=15)
    parser.add_argument("--hour", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true",
                        help="Run the experiment; omission prints the plan only")
    args = parser.parse_args()
    if args.steps <= 1 or args.replicates <= 0 or args.n_recommendations <= 0:
        parser.error("steps must exceed 1; replicates and n-recommendations must be positive")
    return args


def historical_cu_sd(condition: int) -> float:
    values = pd.to_numeric(
        pd.read_excel(DATA_FILE, sheet_name=SHEETS[condition])[CU_IN_COLUMN],
        errors="coerce",
    ).dropna()
    if len(values) < 2:
        raise ValueError(f"Insufficient Cu_in records for Condition {condition}")
    return max(float(values.std(ddof=1)), 0.25)


def generate_delta_sequence(mode: str, sd: float, n_steps: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Create a zero-centred disturbance shared by both methods."""
    if mode == "ar1":
        phi = 0.70
        values = np.zeros(n_steps, dtype=float)
        innovation_sd = sd * np.sqrt(1.0 - phi**2)
        for i in range(1, n_steps):
            values[i] = phi * values[i - 1] + rng.normal(0.0, innovation_sd)
    elif mode == "step":
        values = np.zeros(n_steps, dtype=float)
        for point in (n_steps // 3, 2 * n_steps // 3):
            values[point:] += rng.choice((-1.0, 1.0)) * rng.uniform(sd, 2.0 * sd)
    elif mode == "gradual_drift":
        values = rng.choice((-1.0, 1.0)) * np.linspace(-sd, sd, n_steps)
        values += rng.normal(0.0, 0.20 * sd, n_steps)
    elif mode == "mixed":
        values = generate_delta_sequence("ar1", sd, n_steps, rng)
        values[n_steps // 2:] += rng.choice((-1.0, 1.0)) * 1.5 * sd
    else:
        raise KeyError(mode)
    return values


def repeat_baseline(baseline: dict[str, np.ndarray], n_steps: int) -> dict[str, np.ndarray]:
    result = {}
    for name, values in baseline.items():
        result[name] = np.tile(values, (n_steps, 1)) if np.asarray(values).ndim == 2 \
            else np.tile(values, n_steps)
    return result


def execute(args: argparse.Namespace) -> None:
    runtime = {"year": args.year, "month": args.month,
               "day": args.day, "hour": args.hour}
    evaluator = OriginalEvaluators(runtime)
    summary_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    selected_rows: list[pd.DataFrame] = []

    for condition in (1, 2, 3):
        sd = historical_cu_sd(condition)
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

        for replicate in range(args.replicates):
            for mode_index, mode in enumerate(MODES):
                sequence_seed = args.seed + 10000 * condition + 100 * replicate + mode_index
                delta = generate_delta_sequence(
                    mode, sd, args.steps, np.random.default_rng(sequence_seed)
                )
                for method in METHODS:
                    base_states, clean = method_data[method]
                    low, high = evaluator.bounds(method, condition)
                    states = np.tile(base_states, (args.steps, 1))
                    raw_cu = np.concatenate([base_states[:, 0] + d for d in delta])
                    states[:, 0] = np.clip(raw_cu, low[0], high[0])
                    evaluated = evaluator.evaluate(method, condition, states)
                    stats = scenario_statistics(
                        evaluated, repeat_baseline(clean, args.steps)
                    )
                    stats.update({
                        "condition": f"Condition {condition}",
                        "method": method,
                        "mode": mode,
                        "replicate": replicate,
                        "sequence_seed": sequence_seed,
                        "n_recommendations": len(base_states),
                        "steps": args.steps,
                        "historical_Cu_in_sd_g_per_L": sd,
                        "Cu_in_clipping_rate": float(np.mean(states[:, 0] != raw_cu)),
                    })
                    summary_rows.append(stats)

                    n_rec = len(base_states)
                    for step, disturbance in enumerate(delta):
                        sl = slice(step * n_rec, (step + 1) * n_rec)
                        trajectory_rows.append({
                            "condition": f"Condition {condition}",
                            "method": method,
                            "mode": mode,
                            "replicate": replicate,
                            "timestep": step,
                            "Cu_in_disturbance_g_per_L": float(disturbance),
                            "mean_Cu_out": float(np.mean(evaluated["Cu_out"][sl])),
                            "mean_As_out": float(np.mean(evaluated["As_out"][sl])),
                            "mean_E_total": float(np.mean(evaluated["E_total"][sl])),
                            "mean_Net_profit_10k_CNY": float(
                                np.mean(evaluated["Net_profit_10k_CNY"][sl])
                            ),
                            "feasibility_rate": float(np.mean(evaluated["feasible"][sl])),
                        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    selected = pd.concat(selected_rows, ignore_index=True)
    summary.to_csv(args.output_dir / "feed_disturbance_replicate_summary.csv",
                   index=False, encoding="utf-8-sig")
    trajectories.to_csv(args.output_dir / "feed_disturbance_step_summary.csv",
                        index=False, encoding="utf-8-sig")
    selected.to_csv(args.output_dir / "selected_archived_recommendations.csv",
                    index=False, encoding="utf-8-sig")
    aggregate = summary.groupby(["condition", "method", "mode"], sort=True).agg(
        n_replicates=("replicate", "size"),
        feasibility_rate_mean=("feasibility_rate", "mean"),
        feasibility_rate_sd=("feasibility_rate", "std"),
        profit_mean_shift=("Net_profit_10k_CNY_mean_shift", "mean"),
        profit_mean_shift_sd=("Net_profit_10k_CNY_mean_shift", "std"),
        Cu_out_mean_absolute_shift=("Cu_out_mean_absolute_shift", "mean"),
        As_out_mean_absolute_shift=("As_out_mean_absolute_shift", "mean"),
        E_total_mean_absolute_shift=("E_total_mean_absolute_shift", "mean"),
    ).reset_index()
    aggregate.to_csv(args.output_dir / "feed_disturbance_aggregate_summary.csv",
                     index=False, encoding="utf-8-sig")
    metadata = {
        "status": "completed",
        "design": "fixed archived recommendations; identical additive Cu_in disturbances",
        "patterns": list(MODES),
        "steps": args.steps,
        "replicates": args.replicates,
        "representatives_per_method_condition": args.n_recommendations,
        "selection": "deterministic maximin coverage of Cu_out/-As_out/log1p(E_total)",
        "runtime_features": runtime,
        "feasibility_tolerance": {"ESRL-CMO": "each cost < 1e-3",
                                  "NSGA-II": "each g <= 1e-8"},
        "unchanged": ["original surrogates", "method-specific economic equations",
                      "C1-C5 constraints", "decision-variable bounds", "archived controls"],
        "limitations": [
            "Synthetic disturbances require plant-calibrated amplitudes before publication.",
            "This is an open-loop recommendation stress test, not online adaptation or re-optimization.",
            "Condition 3 absolute cross-method profit is not compared because the archived method-specific flow conventions differ.",
        ],
    }
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def print_plan(args: argparse.Namespace) -> None:
    print(json.dumps({
        "status": "NOT RUN - code-generation/review mode",
        "experiment": "Feed concentration disturbance",
        "patterns": list(MODES),
        "steps_per_sequence": args.steps,
        "replicates": args.replicates,
        "representatives_per_method_condition": args.n_recommendations,
        "methods": list(METHODS),
        "control": "Same additive Cu_in sequence; archived controls fixed; original method-specific evaluators retained.",
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
