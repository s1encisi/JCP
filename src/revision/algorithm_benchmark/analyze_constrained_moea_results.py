"""Post-process matched NSGA-II, NSGA-III and C-TAEA runs.

The script expects outputs produced by ``constrained_moea_extensions.py``.
It builds a pooled, per-condition nondominated reference set, normalizes all
four minimization objectives jointly, and reports HV, IGD+, and spacing for
each method/seed.  No post-processing occurs unless ``--execute`` is supplied.
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
INPUT_ROOT = ROOT / "constrained_moea_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze constrained-MOEA result files")
    parser.add_argument("--input-dir", type=Path, default=INPUT_ROOT)
    parser.add_argument("--execute", action="store_true",
                        help="Calculate indicators; omission prints the plan only")
    return parser.parse_args()


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


def spacing(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    return float(np.std(nearest, ddof=1))


def discover(root: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(root.glob("*/seed_*/pareto_*_condition*.csv")):
        algorithm = path.parents[1].name
        seed = int(path.parent.name.removeprefix("seed_"))
        condition = int(path.stem.rsplit("condition", 1)[1])
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        records.append({"algorithm": algorithm, "seed": seed,
                        "condition": condition, "path": path, "frame": frame})
    if not records:
        raise FileNotFoundError(f"No result CSVs found under {root}")
    return records


def execute(args: argparse.Namespace) -> None:
    records = discover(args.input_dir)
    metric_rows: list[dict] = []
    reference_rows: list[pd.DataFrame] = []
    for condition in sorted({record["condition"] for record in records}):
        subset = [record for record in records if record["condition"] == condition]
        matrices = [objective_matrix(record["frame"]) for record in subset]
        pooled = np.vstack(matrices)
        ideal, nadir = pooled.min(axis=0), pooled.max(axis=0)
        span = np.where(nadir > ideal, nadir - ideal, 1.0)
        pooled_normalized = (pooled - ideal) / span
        nd = NonDominatedSorting().do(
            pooled_normalized, only_non_dominated_front=True
        )
        reference = pooled_normalized[nd]
        hv = HV(ref_point=np.full(4, 1.10))
        igd_plus = IGDPlus(reference)

        offset = 0
        for record, matrix in zip(subset, matrices):
            normalized = (matrix - ideal) / span
            metric_rows.append({
                "condition": f"Condition {condition}",
                "algorithm": record["algorithm"],
                "seed": record["seed"],
                "n_final_feasible_nondominated": len(matrix),
                "normalized_HV": float(hv(normalized)),
                "normalized_IGD_plus": float(igd_plus(normalized)),
                "normalized_spacing": spacing(normalized),
                "source_file": str(record["path"]),
                "normalization_ideal": json.dumps(ideal.tolist()),
                "normalization_nadir": json.dumps(nadir.tolist()),
            })
            offset += len(matrix)
        reference_rows.append(pd.DataFrame(reference, columns=[
            "Cu_out_norm", "negative_As_out_norm", "E_total_norm",
            "negative_Net_profit_norm",
        ]).assign(condition=f"Condition {condition}"))

    metrics = pd.DataFrame(metric_rows).sort_values(["condition", "algorithm", "seed"])
    summary = metrics.groupby(["condition", "algorithm"], sort=True).agg(
        n_seeds=("seed", "nunique"),
        HV_mean=("normalized_HV", "mean"),
        HV_sd=("normalized_HV", "std"),
        IGD_plus_mean=("normalized_IGD_plus", "mean"),
        IGD_plus_sd=("normalized_IGD_plus", "std"),
        spacing_mean=("normalized_spacing", "mean"),
        spacing_sd=("normalized_spacing", "std"),
    ).reset_index()
    metrics.to_csv(args.input_dir / "indicator_by_seed.csv", index=False,
                   encoding="utf-8-sig")
    summary.to_csv(args.input_dir / "indicator_summary.csv", index=False,
                   encoding="utf-8-sig")
    pd.concat(reference_rows, ignore_index=True).to_csv(
        args.input_dir / "pooled_reference_front_normalized.csv",
        index=False, encoding="utf-8-sig",
    )


def print_plan(args: argparse.Namespace) -> None:
    print(json.dumps({
        "status": "NOT RUN - code-generation/review mode",
        "input": str(args.input_dir),
        "indicators": ["normalized HV", "normalized IGD+", "normalized spacing"],
        "normalization": "joint ideal/nadir per condition across all matched runs",
        "reference_front": "pooled nondominated front per condition",
        "execution_gate": "Supply --execute only after optimizer runs exist.",
    }, indent=2))


def main() -> None:
    args = parse_args()
    if not args.execute:
        print_plan(args)
        return
    execute(args)


if __name__ == "__main__":
    main()
