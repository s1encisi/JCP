"""Post-process the new RVEA run against archived ESRL-CMO and NSGA-II results.

This script does not run any optimizer or surrogate model.  It reads the saved
solution sets, converts batch profit to one CNY convention, selects 100 rows by
the same ``np.linspace`` rule used by the archived NSGA-II export, and reports
the original unnormalised HV/Spacing protocol alongside auditable summaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.indicators.spacing import SpacingIndicator


OPT_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = OPT_ROOT / "result"
RVEA_ROOT = RESULT_ROOT / "rvea_original_nsga_20260428_1800"
OUTPUT_ROOT = RVEA_ROOT / "analysis"
HV_REFERENCE = np.array([12.0, -0.0, 1.0e6, 1.0e6], dtype=float)
N_REPORT = 100

CONDITIONS = {
    1: {
        "label": "Condition 1",
        "nsga": RESULT_ROOT / "newnsga2" / "pareto_nsga2_三段工况.csv",
        "esrl": RESULT_ROOT / "outputs_condition1" / "pareto_ppo_condition1.csv",
        "rvea": RVEA_ROOT / "rvea" / "seed_42" / "pareto_rvea_condition1.csv",
        "q_columns": ("QA",),
    },
    2: {
        "label": "Condition 2",
        "nsga": RESULT_ROOT / "newnsga2" / "pareto_nsga2_四段工况.csv",
        "esrl": RESULT_ROOT / "outputs_condition2" / "pareto_ppo_condition2.csv",
        "rvea": RVEA_ROOT / "rvea" / "seed_42" / "pareto_rvea_condition2.csv",
        "q_columns": ("QB",),
    },
    3: {
        "label": "Condition 3",
        "nsga": RESULT_ROOT / "newnsga2" / "pareto_nsga2_串联工况.csv",
        "esrl": RESULT_ROOT / "outputs_condition3" / "pareto_ppo_condition3.csv",
        "rvea": RVEA_ROOT / "rvea" / "seed_42" / "pareto_rvea_condition3.csv",
        "q_columns": ("QA", "QB"),
    },
}

METHODS = {
    "ESRL-CMO": "esrl",
    "NSGA-II": "nsga",
    "RVEA": "rvea",
}


def batch_profit_cny(frame: pd.DataFrame, q_columns: tuple[str, ...]) -> np.ndarray:
    """Recalculate profit from archived outcomes without surrogate re-evaluation."""
    if len(q_columns) == 1:
        flow = frame[q_columns[0]].to_numpy(dtype=float)
    else:
        flow = frame[list(q_columns)].mean(axis=1).to_numpy(dtype=float)
    cu_in = frame["Cu_in"].to_numpy(dtype=float)
    cu_out = frame["Cu_out"].to_numpy(dtype=float)
    energy = frame["E_total"].to_numpy(dtype=float)
    duration = frame["t"].to_numpy(dtype=float)

    recovered_cu = np.maximum(0.0, (cu_in - cu_out) * flow * duration)
    copper_return = (98.44 - 0.14 - 82.64) * recovered_cu
    avoided_treatment = 2.5 * (11.64 + cu_out) * flow * duration
    electricity_cost = 0.6 * energy
    operating_cost = 200.0 * duration
    return copper_return - avoided_treatment - electricity_cost - operating_cost


def objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            frame["Cu_out"].to_numpy(dtype=float),
            -frame["As_out"].to_numpy(dtype=float),
            frame["E_total"].to_numpy(dtype=float),
            -frame["Profit_CNY_common"].to_numpy(dtype=float),
        ]
    )


def select_report_set(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) < N_REPORT:
        raise ValueError(f"Need at least {N_REPORT} solutions, found {len(frame)}")
    positions = np.linspace(0, len(frame) - 1, N_REPORT, dtype=int)
    selected = frame.iloc[positions].copy().reset_index(drop=True)
    selected.insert(0, "Report_ID", np.arange(1, N_REPORT + 1))
    selected.insert(1, "Source_row_1based", positions + 1)
    return selected


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    f = objective_matrix(frame)
    hv = float(HV(ref_point=HV_REFERENCE)(f))
    spacing = float(SpacingIndicator()(f))
    return {
        "HV_raw_common_CNY": hv,
        "Spacing_raw_common_CNY": spacing,
        "Mean_Cu_out": float(frame["Cu_out"].mean()),
        "Std_Cu_out": float(frame["Cu_out"].std(ddof=0)),
        "Mean_As_out": float(frame["As_out"].mean()),
        "Std_As_out": float(frame["As_out"].std(ddof=0)),
        "Mean_E_total": float(frame["E_total"].mean()),
        "Std_E_total": float(frame["E_total"].std(ddof=0)),
        "Mean_Net_profit_1e4_CNY": float(frame["Profit_CNY_common"].mean() / 1.0e4),
        "Std_Net_profit_1e4_CNY": float(frame["Profit_CNY_common"].std(ddof=0) / 1.0e4),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    audit_rows: list[dict] = []
    report_frames: list[pd.DataFrame] = []

    for condition, cfg in CONDITIONS.items():
        for method, source_key in METHODS.items():
            source = Path(cfg[source_key])
            frame = pd.read_csv(source)
            frame["Profit_CNY_common"] = batch_profit_cny(frame, cfg["q_columns"])

            archived_profit = frame["Net_profit"].to_numpy(dtype=float)
            if method == "ESRL-CMO":
                archived_profit = archived_profit * 1.0e4
            profit_mae = float(
                np.mean(np.abs(frame["Profit_CNY_common"].to_numpy() - archived_profit))
            )

            report = select_report_set(frame)
            report.insert(2, "Condition_label", cfg["label"])
            report.insert(3, "Method_label", method)
            report_path = OUTPUT_ROOT / f"report100_{source_key}_condition{condition}.csv"
            report.to_csv(report_path, index=False, encoding="utf-8-sig")
            report_frames.append(report)

            row = {
                "Condition": cfg["label"],
                "Method": method,
                "Source_solution_count": len(frame),
                "Report_solution_count": len(report),
                "Source_file": str(source),
                **metrics(report),
            }
            summary_rows.append(row)
            audit_rows.append(
                {
                    "Condition": cfg["label"],
                    "Method": method,
                    "Source_solution_count": len(frame),
                    "Selected_solution_count": len(report),
                    "Profit_recalculation_MAE_CNY": profit_mae,
                    "Profit_flow_convention": (
                        cfg["q_columns"][0]
                        if len(cfg["q_columns"]) == 1
                        else "(QA + QB) / 2"
                    ),
                    "Selection_rule": "np.linspace over archived row order",
                    "Source_file": str(source),
                    "Report_file": str(report_path),
                }
            )

    summary = pd.DataFrame(summary_rows)
    audit = pd.DataFrame(audit_rows)
    summary.to_csv(
        OUTPUT_ROOT / "comparison_100point_common_units.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit.to_csv(
        OUTPUT_ROOT / "source_and_selection_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    nsga_archived = pd.read_csv(RESULT_ROOT / "newnsga2" / "quality_metrics_nsga2.csv")
    nsga_archived.to_csv(
        OUTPUT_ROOT / "archived_nsga_metrics_unchanged.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "optimizers_run_by_this_analysis": [],
        "new_optimizer_run": "RVEA only",
        "rvea_runtime_features": "2026-04-28 18:00",
        "rvea_population_size": 1200,
        "rvea_generations": 600,
        "rvea_seed": 42,
        "rvea_reference_direction_seed": 42,
        "rvea_operator_profile": "pymoo 0.6.1.5 source NSGA-II defaults",
        "archived_nsga_runtime_features_verified": "2026-04-26 14:00",
        "archive_note": (
            "NSGA-II and ESRL-CMO were read from existing archives and were not rerun. "
            "The three archives do not share identical temporal inputs; comparisons are descriptive."
        ),
        "profit_convention": (
            "Recalculated from archived Cu_in, Cu_out, E_total, duration and flow; "
            "Condition 3 uses (QA + QB) / 2 for every method."
        ),
        "selection_rule": "100 positions from np.linspace over each archived row order",
        "hv_reference_point": HV_REFERENCE.tolist(),
        "spacing": "pymoo SpacingIndicator in unnormalised minimisation objective space",
    }
    (OUTPUT_ROOT / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report_all = pd.concat(report_frames, ignore_index=True, sort=False)
    full_rvea = pd.concat(
        [
            pd.read_csv(CONDITIONS[c]["rvea"]).assign(
                Condition_label=CONDITIONS[c]["label"], Method_label="RVEA"
            )
            for c in CONDITIONS
        ],
        ignore_index=True,
        sort=False,
    )
    run_metadata = pd.read_csv(RVEA_ROOT / "run_metadata.csv")
    workbook_data = {
        "summary": summary.astype(object).where(pd.notna(summary), None).to_dict(orient="records"),
        "audit": audit.astype(object).where(pd.notna(audit), None).to_dict(orient="records"),
        "run_metadata": run_metadata.astype(object).where(pd.notna(run_metadata), None).to_dict(orient="records"),
        "report100": report_all.astype(object).where(pd.notna(report_all), None).to_dict(orient="records"),
        "full_rvea": full_rvea.astype(object).where(pd.notna(full_rvea), None).to_dict(orient="records"),
        "protocol": metadata,
    }
    (OUTPUT_ROOT / "workbook_data.json").write_text(
        json.dumps(workbook_data, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )

    print(summary.to_string(index=False))
    print(f"\nWrote analysis outputs to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
