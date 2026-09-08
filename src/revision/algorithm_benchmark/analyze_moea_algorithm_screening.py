"""Compute simple, jointly normalized indicators for the MOEA screening runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import moea_algorithm_screening as screening


ROOT = Path(__file__).resolve().parent
INPUT_ROOT = ROOT / "moea_algorithm_screening"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def spacing(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    return float(np.std(nearest, ddof=1))


def discover(root: Path, metadata: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    completed = metadata.loc[metadata["status"].eq("completed")].copy()
    completed = completed.drop_duplicates(
        subset=["algorithm", "condition", "seed"], keep="last"
    )
    for _, row in completed.iterrows():
        condition = int(str(row["condition"]).split()[-1])
        algorithm = str(row["algorithm"])
        seed = int(row["seed"])
        path = (
            root
            / algorithm
            / f"seed_{seed}"
            / f"pareto_{algorithm}_condition{condition}.csv"
        )
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        matrix = (
            screening.objective_matrix(frame)
            if len(frame)
            else np.empty((0, 4), dtype=float)
        )
        records.append(
            {
                "algorithm": algorithm,
                "algorithm_label": str(row["algorithm_label"]),
                "condition": condition,
                "condition_label": str(row["condition"]),
                "seed": seed,
                "path": path,
                "matrix": matrix,
                "metadata": row,
            }
        )
    return records


def execute(args: argparse.Namespace) -> None:
    metadata_path = args.input_dir / "run_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    raw_metadata = pd.read_csv(metadata_path)
    # Ignore superseded pilot methods that may remain in an interrupted output
    # directory; only methods declared by the current screening configuration
    # are eligible for the common normalization and ranking.
    metadata = raw_metadata.loc[
        raw_metadata["algorithm"].isin(screening.ALGORITHMS)
    ].copy()
    excluded = raw_metadata.loc[
        ~raw_metadata["algorithm"].isin(screening.ALGORITHMS)
    ].copy()
    records = discover(args.input_dir, metadata)
    if not records:
        raise FileNotFoundError(f"No completed result fronts under {args.input_dir}")

    metric_rows: list[dict] = []
    reference_rows: list[pd.DataFrame] = []
    for condition in sorted({record["condition"] for record in records}):
        subset = [record for record in records if record["condition"] == condition]
        nonempty = [record["matrix"] for record in subset if len(record["matrix"])]
        if not nonempty:
            continue
        pooled = np.vstack(nonempty)
        ideal = pooled.min(axis=0)
        nadir = pooled.max(axis=0)
        span = np.where(nadir > ideal, nadir - ideal, 1.0)
        pooled_norm = (pooled - ideal) / span
        nd = NonDominatedSorting().do(pooled_norm, only_non_dominated_front=True)
        reference = pooled_norm[nd]
        hv = HV(ref_point=np.full(4, 1.10))
        igd_plus = IGDPlus(reference)

        reference_rows.append(
            pd.DataFrame(
                reference,
                columns=[
                    "Cu_out_norm",
                    "negative_As_out_norm",
                    "E_total_norm",
                    "negative_Net_profit_norm",
                ],
            ).assign(condition=f"Condition {condition}")
        )

        for record in subset:
            matrix = record["matrix"]
            if len(matrix):
                normalized = (matrix - ideal) / span
                hv_value = float(hv(normalized))
                igd_value = float(igd_plus(normalized))
                spacing_value = spacing(normalized)
            else:
                hv_value = 0.0
                igd_value = float("inf")
                spacing_value = float("nan")
            md = record["metadata"]
            metric_rows.append(
                {
                    "condition": record["condition_label"],
                    "algorithm": record["algorithm"],
                    "algorithm_label": record["algorithm_label"],
                    "seed": record["seed"],
                    "n_evaluations": int(md["n_evaluations"]),
                    "runtime_seconds": float(md["runtime_seconds"]),
                    "final_population_feasibility_rate": float(
                        md["final_population_feasibility_rate"]
                    ),
                    "n_final_feasible_nondominated": len(matrix),
                    "HV": hv_value,
                    "IGD_plus": igd_value,
                    "spacing": spacing_value,
                    "constraint_handling": str(md["constraint_handling"]),
                    "source_file": str(record["path"]),
                    "normalization_ideal": json.dumps(ideal.tolist()),
                    "normalization_nadir": json.dumps(nadir.tolist()),
                }
            )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["condition", "algorithm", "seed"]
    )
    condition_summary = (
        metrics.groupby(["condition", "algorithm", "algorithm_label"], sort=True)
        .agg(
            runs=("seed", "nunique"),
            HV_mean=("HV", "mean"),
            HV_sd=("HV", "std"),
            IGD_plus_mean=("IGD_plus", "mean"),
            IGD_plus_sd=("IGD_plus", "std"),
            spacing_mean=("spacing", "mean"),
            spacing_sd=("spacing", "std"),
            feasibility_mean=("final_population_feasibility_rate", "mean"),
            feasible_ND_mean=("n_final_feasible_nondominated", "mean"),
            runtime_mean_seconds=("runtime_seconds", "mean"),
        )
        .reset_index()
    )

    overall = (
        metrics.groupby(["algorithm", "algorithm_label"], sort=True)
        .agg(
            successful_runs=("seed", "size"),
            conditions=("condition", "nunique"),
            HV_mean=("HV", "mean"),
            HV_sd=("HV", "std"),
            IGD_plus_mean=("IGD_plus", "mean"),
            IGD_plus_sd=("IGD_plus", "std"),
            spacing_mean=("spacing", "mean"),
            feasibility_mean=("final_population_feasibility_rate", "mean"),
            feasible_ND_mean=("n_final_feasible_nondominated", "mean"),
            runtime_mean_seconds=("runtime_seconds", "mean"),
            runtime_total_seconds=("runtime_seconds", "sum"),
        )
        .reset_index()
    )
    overall["HV_rank"] = overall["HV_mean"].rank(
        ascending=False, method="min"
    )
    overall["IGD_plus_rank"] = overall["IGD_plus_mean"].rank(
        ascending=True, method="min"
    )
    overall["screen_rank_score"] = (
        overall["HV_rank"] + overall["IGD_plus_rank"]
    ) / 2.0
    overall = overall.sort_values(
        ["screen_rank_score", "HV_rank", "IGD_plus_rank", "runtime_mean_seconds"]
    ).reset_index(drop=True)
    overall.insert(0, "screen_rank", np.arange(1, len(overall) + 1))

    failures = metadata.loc[metadata["status"].ne("completed")].copy()
    metrics.to_csv(args.input_dir / "indicator_by_run.csv", index=False, encoding="utf-8-sig")
    condition_summary.to_csv(
        args.input_dir / "indicator_by_condition.csv", index=False, encoding="utf-8-sig"
    )
    overall.to_csv(
        args.input_dir / "overall_screening_ranking.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(reference_rows, ignore_index=True).to_csv(
        args.input_dir / "pooled_reference_front_normalized.csv",
        index=False,
        encoding="utf-8-sig",
    )
    failures.to_csv(
        args.input_dir / "failed_runs.csv", index=False, encoding="utf-8-sig"
    )
    metadata.to_csv(
        args.input_dir / "eligible_run_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    excluded.to_csv(
        args.input_dir / "excluded_superseded_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        overall[
            [
                "screen_rank",
                "algorithm_label",
                "successful_runs",
                "HV_mean",
                "IGD_plus_mean",
                "spacing_mean",
                "feasibility_mean",
                "feasible_ND_mean",
                "runtime_mean_seconds",
            ]
        ].to_string(index=False)
    )
    print(f"\nFailed runs: {len(failures)}")


def print_plan(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            {
                "status": "NOT RUN - supply --execute",
                "input": str(args.input_dir),
                "metrics": ["jointly normalized HV", "IGD+", "spacing", "feasibility", "runtime"],
                "ranking": "mean of HV rank and IGD+ rank across all condition/seed runs",
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if not args.execute:
        print_plan(args)
        return
    execute(args)


if __name__ == "__main__":
    main()
