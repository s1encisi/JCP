"""Data-grounded audit of temporal proxies and electrochemical PDP structure.

This script uses only the copied final modeling data and copied trained
ExtraTrees models.  It does not alter or retrain the source implementation.
Calendar variables are treated as proxies for slow operational changes, not as
direct electrochemical causes.  Segmented regressions summarize the trained
model's partial-dependence curves and therefore remain model-association
diagnostics rather than kinetic parameter estimates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "data_by_operation_mode_with_features.xlsx"
MODEL_ROOT = PROJECT_ROOT / "optimization" / "joblib"
OUTPUT_ROOT = PROJECT_ROOT / "analyses" / "mechanistic_audit"

DATETIME = "日期时间"
CU_IN = "电积前液Cu（g/L）"
CU_OUT = "废铜液原液罐Cu（g/L）"
AS_OUT = "废铜液原液罐As（g/L）"
TA, IA, VA, QA = "三段溶液温度℃", "三段电流强度A", "三段电压V", "三段流量m3/h"
TB, IB, VB, QB = "四段溶液温度℃", "四段电流强度A", "四段电压V", "四段流量m3/h"

CONDITIONS = {
    "three_stage": "Condition 1",
    "four_stage": "Condition 2",
    "serial": "Condition 3",
}

VOLTAGE_SPECS = {
    "Condition 1 / IA": ("three_stage", IA, MODEL_ROOT / "voltage_three_stage" / "extra_trees_three_stage.joblib"),
    "Condition 2 / IB": ("four_stage", IB, MODEL_ROOT / "voltage_four_stage" / "extra_trees_four_stage.joblib"),
    "Condition 3 / IA": ("serial", IA, MODEL_ROOT / "voltage_serial" / "extra_trees_serial.joblib"),
    "Condition 3 / IB": ("serial", IB, MODEL_ROOT / "voltage_serial" / "extra_trees_serial.joblib"),
}


def configure_plot_style() -> None:
    mpl.use("Agg")
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "legend.fontsize": 7.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
    })


def load_sheet(sheet: str) -> pd.DataFrame:
    frame = pd.read_excel(DATA_FILE, sheet_name=sheet)
    frame[DATETIME] = pd.to_datetime(frame[DATETIME], errors="raise")
    return frame


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    frame = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "group": groups}).dropna()
    if frame.empty:
        return float("nan")
    total = float(((frame["value"] - frame["value"].mean()) ** 2).sum())
    if total <= 0:
        return 0.0
    grouped = frame.groupby("group", observed=True)["value"]
    between = float(sum(len(v) * (v.mean() - frame["value"].mean()) ** 2 for _, v in grouped))
    return between / total


def temporal_proxy_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eta_rows: list[dict] = []
    annual_rows: list[dict] = []
    combined: list[pd.DataFrame] = []
    variable_map = {
        "three_stage": [CU_IN, TA, IA, VA, QA, CU_OUT, AS_OUT],
        "four_stage": [CU_IN, TB, IB, VB, QB, CU_OUT, AS_OUT],
        "serial": [CU_IN, TA, IA, VA, QA, TB, IB, VB, QB, CU_OUT, AS_OUT],
    }
    for sheet, condition in CONDITIONS.items():
        frame = load_sheet(sheet)
        part = frame[[DATETIME]].copy()
        part["condition"] = condition
        combined.append(part)
        for variable in variable_map[sheet]:
            for calendar in ("year", "month", "day"):
                eta_rows.append({
                    "condition": condition,
                    "variable": variable,
                    "calendar_feature": calendar,
                    "eta_squared": eta_squared(frame[variable], frame[calendar]),
                    "n": int(frame[[variable, calendar]].dropna().shape[0]),
                })
            years = sorted(int(y) for y in frame["year"].dropna().unique())
            if 2024 in years and 2025 in years:
                a = pd.to_numeric(frame.loc[frame["year"] == 2024, variable], errors="coerce").dropna()
                b = pd.to_numeric(frame.loc[frame["year"] == 2025, variable], errors="coerce").dropna()
                pooled = float(pd.concat([a, b]).std(ddof=1))
                annual_rows.append({
                    "condition": condition,
                    "variable": variable,
                    "n_2024": len(a),
                    "n_2025": len(b),
                    "mean_2024": float(a.mean()),
                    "mean_2025": float(b.mean()),
                    "absolute_shift_2025_minus_2024": float(b.mean() - a.mean()),
                    "standardized_shift_all_year_sd": float((b.mean() - a.mean()) / pooled) if pooled > 0 else 0.0,
                })

    timeline = pd.concat(combined, ignore_index=True).sort_values(DATETIME).reset_index(drop=True)
    gap_hours = timeline[DATETIME].diff().dt.total_seconds().div(3600)
    new_block = timeline["condition"].ne(timeline["condition"].shift()) | gap_hours.gt(2.1) | gap_hours.isna()
    timeline["block_id"] = new_block.cumsum()
    blocks = timeline.groupby("block_id", as_index=False).agg(
        condition=("condition", "first"),
        start=(DATETIME, "min"),
        end=(DATETIME, "max"),
        n_samples=(DATETIME, "size"),
    )
    blocks["duration_h_including_last_2h_interval"] = blocks["n_samples"] * 2.0
    block_summary = blocks.groupby("condition", as_index=False).agg(
        n_blocks=("block_id", "size"),
        median_duration_h=("duration_h_including_last_2h_interval", "median"),
        max_duration_h=("duration_h_including_last_2h_interval", "max"),
        short_blocks_lt_4_samples=("n_samples", lambda s: int((s < 4).sum())),
    )
    overall = pd.DataFrame([{
        "condition": "All conditions",
        "n_blocks": len(blocks),
        "median_duration_h": float(blocks["duration_h_including_last_2h_interval"].median()),
        "max_duration_h": float(blocks["duration_h_including_last_2h_interval"].max()),
        "short_blocks_lt_4_samples": int((blocks["n_samples"] < 4).sum()),
    }])
    block_summary = pd.concat([block_summary, overall], ignore_index=True)

    timeline["year_month"] = timeline[DATETIME].dt.to_period("M").astype(str)
    monthly = timeline.groupby(["year_month", "condition"], as_index=False).size()
    monthly["monthly_fraction"] = monthly["size"] / monthly.groupby("year_month")["size"].transform("sum")
    return pd.DataFrame(eta_rows), pd.DataFrame(annual_rows), blocks, pd.concat([block_summary, monthly], ignore_index=True, sort=False)


def fit_hinge(x: np.ndarray, y: np.ndarray, breakpoints: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, float]:
    columns = [np.ones_like(x), x]
    columns.extend(np.maximum(0.0, x - bp) for bp in breakpoints)
    design = np.column_stack(columns)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return coef, fitted, r2


def best_three_segment_fit(x: np.ndarray, y: np.ndarray, min_points: int = 8) -> dict:
    best: dict | None = None
    n = len(x)
    for i in range(min_points, n - 2 * min_points + 1):
        for j in range(i + min_points, n - min_points + 1):
            bp = (float(x[i]), float(x[j]))
            coef, fitted, r2 = fit_hinge(x, y, bp)
            sse = float(np.sum((y - fitted) ** 2))
            if best is None or sse < best["sse"]:
                slopes = (float(coef[1]), float(coef[1] + coef[2]), float(coef[1] + coef[2] + coef[3]))
                best = {"breakpoints": bp, "coef": coef, "fitted": fitted, "r2": r2, "sse": sse, "slopes": slopes}
    if best is None:
        raise ValueError("Insufficient grid points for three-segment fit")
    return best


def best_two_segment_fit(x: np.ndarray, y: np.ndarray, min_points: int = 10) -> dict:
    best: dict | None = None
    for i in range(min_points, len(x) - min_points + 1):
        bp = (float(x[i]),)
        coef, fitted, r2 = fit_hinge(x, y, bp)
        sse = float(np.sum((y - fitted) ** 2))
        if best is None or sse < best["sse"]:
            best = {
                "breakpoint": bp[0],
                "coef": coef,
                "fitted": fitted,
                "r2": r2,
                "sse": sse,
                "slopes": (float(coef[1]), float(coef[1] + coef[2])),
            }
    if best is None:
        raise ValueError("Insufficient grid points for two-segment fit")
    return best


def model_pdp(model, frame: pd.DataFrame, feature: str) -> tuple[np.ndarray, np.ndarray]:
    features = list(model.feature_names_in_)
    clean = frame[features].dropna().astype(float).copy()
    result = partial_dependence(model, clean, [feature], grid_resolution=60, kind="average")
    return np.asarray(result["grid_values"][0], float), np.asarray(result["average"][0], float)


def voltage_segment_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary: list[dict] = []
    curves: list[pd.DataFrame] = []
    cathode_area_total_m2 = 35 * 1.100 * 1.029 * 2
    fig, axes = plt.subplots(2, 2, figsize=(7.48, 5.4), constrained_layout=True)
    for ax, (label, (sheet, feature, model_path)) in zip(axes.flat, VOLTAGE_SPECS.items()):
        model = joblib.load(model_path)
        x_amp, y_volt = model_pdp(model, load_sheet(sheet), feature)
        x_ka = x_amp / 1000.0
        fit = best_three_segment_fit(x_ka, y_volt)
        bp1, bp2 = fit["breakpoints"]
        bounds = [(float(x_ka.min()), bp1), (bp1, bp2), (bp2, float(x_ka.max()))]
        for segment, ((lo, hi), slope) in enumerate(zip(bounds, fit["slopes"]), start=1):
            summary.append({
                "curve": label,
                "segment": segment,
                "current_low_kA": lo,
                "current_high_kA": hi,
                "current_density_low_A_m2": lo * 1000 / cathode_area_total_m2,
                "current_density_high_A_m2": hi * 1000 / cathode_area_total_m2,
                "slope_V_per_kA": slope,
                "piecewise_fit_R2": fit["r2"],
                "n_pdp_grid_points": len(x_ka),
            })
        curve = pd.DataFrame({
            "curve": label,
            "current_A": x_amp,
            "current_kA": x_ka,
            "current_density_A_m2": x_amp / cathode_area_total_m2,
            "pdp_voltage_V": y_volt,
            "piecewise_fit_voltage_V": fit["fitted"],
        })
        curves.append(curve)
        ax.plot(x_ka, y_volt, color="#1F5A99", marker="o", ms=2.2, lw=0.8, label="ExtraTrees PDP")
        ax.plot(x_ka, fit["fitted"], color="#C0602B", lw=1.25, label="3-segment fit")
        ax.axvline(bp1, color="#777777", ls="--", lw=0.7)
        ax.axvline(bp2, color="#777777", ls="--", lw=0.7)
        ax.set_title(label)
        ax.set_xlabel("Current (kA)")
        ax.set_ylabel("Partial dependence of voltage (V)")
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.savefig(OUTPUT_ROOT / "voltage_pdp_segmented_fits.png", dpi=600)
    fig.savefig(OUTPUT_ROOT / "voltage_pdp_segmented_fits.pdf")
    plt.close(fig)
    return pd.DataFrame(summary), pd.concat(curves, ignore_index=True)


def cu_as_threshold_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_sheet("four_stage")
    model = joblib.load(MODEL_ROOT / "as_four_stage" / "extra_trees_four_stage.joblib")
    x, y = model_pdp(model, frame, CU_IN)
    fit = best_two_segment_fit(x, y)
    observed_min_x = float(x[int(np.argmin(y))])
    temperature_k = float(pd.to_numeric(frame[TB], errors="coerce").median() + 273.15)
    threshold_molar = fit["breakpoint"] / 63.546
    ideal_cu_potential_v = 0.340 + (8.314462618 * temperature_k / (2 * 96485.33212)) * math.log(threshold_molar)
    low_molar = float(x.min() / 63.546)
    high_molar = float(x.max() / 63.546)
    potential_span_mv = (8.314462618 * temperature_k / (2 * 96485.33212)) * math.log(high_molar / low_molar) * 1000
    summary = pd.DataFrame([{
        "curve": "Condition 2: Cu_in -> As_out",
        "hinge_breakpoint_Cu_in_g_L": fit["breakpoint"],
        "observed_PDP_minimum_Cu_in_g_L": observed_min_x,
        "slope_below_break_g_L_As_per_g_L_Cu": fit["slopes"][0],
        "slope_above_break_g_L_As_per_g_L_Cu": fit["slopes"][1],
        "piecewise_fit_R2": fit["r2"],
        "median_electrolyte_temperature_C": temperature_k - 273.15,
        "ideal_Cu2plus_concentration_at_break_mol_L": threshold_molar,
        "ideal_Cu2plus_Cu_Nernst_potential_at_break_V_vs_SHE": ideal_cu_potential_v,
        "ideal_Cu_equilibrium_potential_span_over_PDP_grid_mV": potential_span_mv,
        "interpretation_limit": "Ideal-activity Cu2+/Cu calculation only; acidity, As speciation, and electrode potentials were not recorded.",
    }])
    curve = pd.DataFrame({
        "Cu_in_g_L": x,
        "pdp_As_out_g_L": y,
        "two_segment_fit_As_out_g_L": fit["fitted"],
    })
    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    ax.plot(x, y, color="#1F5A99", marker="o", ms=2.5, lw=0.85, label="ExtraTrees PDP")
    ax.plot(x, fit["fitted"], color="#C0602B", lw=1.25, label="2-segment fit")
    ax.axvline(fit["breakpoint"], color="#777777", ls="--", lw=0.8,
               label=f"hinge = {fit['breakpoint']:.2f} g/L")
    ax.set_xlabel(r"$Cu_{in}$ (g L$^{-1}$)")
    ax.set_ylabel(r"Partial dependence of $As_{out}$ (g L$^{-1}$)")
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT_ROOT / "cu_as_pdp_threshold_fit.png", dpi=600)
    fig.savefig(OUTPUT_ROOT / "cu_as_pdp_threshold_fit.pdf")
    plt.close(fig)
    return summary, curve


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    eta, annual, blocks, block_and_month = temporal_proxy_audit()
    voltage_summary, voltage_curves = voltage_segment_audit()
    cu_as_summary, cu_as_curve = cu_as_threshold_audit()
    eta.to_csv(OUTPUT_ROOT / "calendar_group_eta_squared.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUTPUT_ROOT / "annual_standardized_shifts.csv", index=False, encoding="utf-8-sig")
    blocks.to_csv(OUTPUT_ROOT / "operating_condition_blocks.csv", index=False, encoding="utf-8-sig")
    block_and_month.to_csv(OUTPUT_ROOT / "block_and_monthly_condition_summary.csv", index=False, encoding="utf-8-sig")
    voltage_summary.to_csv(OUTPUT_ROOT / "voltage_pdp_segment_summary.csv", index=False, encoding="utf-8-sig")
    voltage_curves.to_csv(OUTPUT_ROOT / "voltage_pdp_segment_curves.csv", index=False, encoding="utf-8-sig")
    cu_as_summary.to_csv(OUTPUT_ROOT / "cu_as_threshold_summary.csv", index=False, encoding="utf-8-sig")
    cu_as_curve.to_csv(OUTPUT_ROOT / "cu_as_threshold_curve.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "data_file": str(DATA_FILE),
        "model_root": str(MODEL_ROOT),
        "calendar_interpretation": "proxy for recorded and unrecorded slow operational changes; not a causal process input",
        "pdp_interpretation": "model association summarized by segmented regression; not direct kinetic identification",
    }
    (OUTPUT_ROOT / "analysis_scope.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Mechanistic audit completed.")
    print("\nVoltage PDP segmented fits")
    print(voltage_summary.to_string(index=False))
    print("\nCu-As threshold audit")
    print(cu_as_summary.to_string(index=False))
    print("\nLargest absolute 2025-vs-2024 standardized shifts by condition")
    ranked = annual.assign(abs_shift=annual["standardized_shift_all_year_sd"].abs()).sort_values(
        ["condition", "abs_shift"], ascending=[True, False]
    ).groupby("condition", as_index=False).head(5)
    print(ranked.drop(columns="abs_shift").to_string(index=False))


if __name__ == "__main__":
    main()
