"""Extended, reproducible sensitivity analysis for the archived recommendations.

The calculation preserves the economic equation and the method-specific flow
conventions implemented in the archived ESRL-CMO and NSGA-II results.  It does
not retrain a policy or rerun an optimizer.  The ±30% intervals are transparent
stress ranges, not confidence intervals or assertions about market volatility.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_archived_result_analysis as archived


OUT = Path(__file__).resolve().parent / "economic_sensitivity_extended"
BASE = {
    "p_Cu": 98.44,
    "c_reprocess": 0.14,
    "c_concentrate": 82.64,
    "p_e": 0.60,
    "c_op": 200.0,
    "c_As_treat": 2.50,
}
FACTORS = np.array([0.70, 0.85, 1.00, 1.15, 1.30], dtype=float)
N_JOINT_DRAWS = 5000
SEED = 20260830


def economic_components(archive: archived.Archive) -> dict[str, np.ndarray | str]:
    df = archive.data
    q, q_basis = archived.effective_flow(df, archive.method, archive.condition)
    cu_in = df["Cu_in"].to_numpy(float)
    cu_out = df["Cu_out"].to_numpy(float)
    t = df["t"].to_numpy(float)
    energy = df["E_total"].to_numpy(float)
    m_cu = np.maximum(0.0, (cu_in - cu_out) * q * t)
    as_cu_load = (archived.ECON["As_out_fixed"] + cu_out) * q * t
    return {
        "m_Cu": m_cu,
        "as_cu_load": as_cu_load,
        "energy": energy,
        "t": t,
        "q_basis": q_basis,
    }


def profit_10k(comp: dict[str, np.ndarray | str], params: dict[str, float]) -> np.ndarray:
    m_cu = np.asarray(comp["m_Cu"], dtype=float)
    load = np.asarray(comp["as_cu_load"], dtype=float)
    energy = np.asarray(comp["energy"], dtype=float)
    t = np.asarray(comp["t"], dtype=float)
    copper_margin = params["p_Cu"] - params["c_reprocess"] - params["c_concentrate"]
    value = copper_margin * m_cu
    treatment = params["c_As_treat"] * load
    electricity = params["p_e"] * energy
    operation = params["c_op"] * t
    return (value - treatment - electricity - operation) / 1.0e4


def summarise(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_profit_10k_CNY": float(np.mean(values)),
        "sd_profit_10k_CNY": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "q05_profit_10k_CNY": float(np.quantile(values, 0.05)),
        "median_profit_10k_CNY": float(np.quantile(values, 0.50)),
        "q95_profit_10k_CNY": float(np.quantile(values, 0.95)),
        "profitable_fraction": float(np.mean(values > 0.0)),
    }


def one_at_a_time(archives: list[archived.Archive]) -> pd.DataFrame:
    rows: list[dict] = []
    for arc in archives:
        comp = economic_components(arc)
        baseline = profit_10k(comp, BASE)
        stored = arc.data["profit_stored_10k"].to_numpy(float)
        max_error = float(np.max(np.abs(baseline - stored)))
        if max_error > 5.0e-4:
            raise RuntimeError(
                f"Baseline audit failed for {arc.method}, Condition {arc.condition}: {max_error}"
            )
        for name, base_value in BASE.items():
            for factor in FACTORS:
                params = dict(BASE)
                params[name] = base_value * float(factor)
                values = profit_10k(comp, params)
                row = {
                    "condition": arc.condition,
                    "method": arc.method,
                    "parameter": name,
                    "baseline_value": base_value,
                    "factor": float(factor),
                    "change_percent": float((factor - 1.0) * 100.0),
                    "scenario_value": params[name],
                    "n_archived_recommendations": len(values),
                    "q_basis": comp["q_basis"],
                    "baseline_audit_max_abs_error_10k_CNY": max_error,
                }
                row.update(summarise(values))
                row["mean_change_from_baseline_10k_CNY"] = float(
                    np.mean(values - baseline)
                )
                rows.append(row)
    return pd.DataFrame(rows)


def joint_uncertainty(archives: list[archived.Archive]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    parameter_names = list(BASE)
    factors = rng.uniform(0.70, 1.30, size=(N_JOINT_DRAWS, len(parameter_names)))
    draws = pd.DataFrame({
        "draw": np.arange(1, N_JOINT_DRAWS + 1),
        **{
            name: BASE[name] * factors[:, j]
            for j, name in enumerate(parameter_names)
        },
    })
    rows: list[dict] = []
    for arc in archives:
        comp = economic_components(arc)
        m_cu = np.asarray(comp["m_Cu"], dtype=float)[None, :]
        load = np.asarray(comp["as_cu_load"], dtype=float)[None, :]
        energy = np.asarray(comp["energy"], dtype=float)[None, :]
        t = np.asarray(comp["t"], dtype=float)[None, :]
        p = draws
        margin = (
            p["p_Cu"].to_numpy() - p["c_reprocess"].to_numpy()
            - p["c_concentrate"].to_numpy()
        )[:, None]
        profits = (
            margin * m_cu
            - p["c_As_treat"].to_numpy()[:, None] * load
            - p["p_e"].to_numpy()[:, None] * energy
            - p["c_op"].to_numpy()[:, None] * t
        ) / 1.0e4
        mean_by_draw = profits.mean(axis=1)
        profitable_by_draw = (profits > 0.0).mean(axis=1)
        for draw_idx in range(N_JOINT_DRAWS):
            rows.append({
                "condition": arc.condition,
                "method": arc.method,
                "draw": draw_idx + 1,
                "mean_profit_10k_CNY": float(mean_by_draw[draw_idx]),
                "profitable_fraction": float(profitable_by_draw[draw_idx]),
            })
    detail = pd.DataFrame(rows)
    grouped = detail.groupby(["condition", "method"], sort=True)
    summary = grouped.agg(
        n_joint_draws=("draw", "size"),
        mean_of_set_mean_profit_10k_CNY=("mean_profit_10k_CNY", "mean"),
        sd_of_set_mean_profit_10k_CNY=("mean_profit_10k_CNY", "std"),
        q05_set_mean_profit_10k_CNY=("mean_profit_10k_CNY", lambda x: x.quantile(0.05)),
        median_set_mean_profit_10k_CNY=("mean_profit_10k_CNY", "median"),
        q95_set_mean_profit_10k_CNY=("mean_profit_10k_CNY", lambda x: x.quantile(0.95)),
        probability_set_mean_profit_positive=("mean_profit_10k_CNY", lambda x: float((x > 0).mean())),
        mean_solution_profitable_fraction=("profitable_fraction", "mean"),
    ).reset_index()
    return detail, summary


def break_even_summary(archives: list[archived.Archive]) -> pd.DataFrame:
    rows: list[dict] = []
    for arc in archives:
        comp = economic_components(arc)
        m_cu = np.asarray(comp["m_Cu"], dtype=float)
        non_copper = (
            BASE["c_As_treat"] * np.asarray(comp["as_cu_load"], dtype=float)
            + BASE["p_e"] * np.asarray(comp["energy"], dtype=float)
            + BASE["c_op"] * np.asarray(comp["t"], dtype=float)
        )
        price = BASE["c_reprocess"] + BASE["c_concentrate"] + non_copper / m_cu
        rows.append({
            "condition": arc.condition,
            "method": arc.method,
            "n_archived_recommendations": len(price),
            "q_basis": comp["q_basis"],
            "mean_break_even_p_Cu_CNY_per_kg": float(np.mean(price)),
            "q05_break_even_p_Cu_CNY_per_kg": float(np.quantile(price, 0.05)),
            "median_break_even_p_Cu_CNY_per_kg": float(np.quantile(price, 0.50)),
            "q95_break_even_p_Cu_CNY_per_kg": float(np.quantile(price, 0.95)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    archives = archived.load_archives()
    oat = one_at_a_time(archives)
    joint_detail, joint_summary = joint_uncertainty(archives)
    break_even = break_even_summary(archives)
    oat.to_csv(OUT / "one_at_a_time_summary.csv", index=False, encoding="utf-8-sig")
    joint_detail.to_csv(OUT / "joint_draw_summary.csv", index=False, encoding="utf-8-sig")
    joint_summary.to_csv(OUT / "joint_uncertainty_summary.csv", index=False, encoding="utf-8-sig")
    break_even.to_csv(OUT / "break_even_copper_price_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "status": "completed",
        "analysis": "fixed archived recommendation sets; ex-post economic recalculation",
        "baseline": BASE,
        "one_at_a_time_factors": FACTORS.tolist(),
        "joint_draws": N_JOINT_DRAWS,
        "joint_seed": SEED,
        "joint_factor_distribution": "independent Uniform(0.70, 1.30) for all six coefficients",
        "limitations": [
            "Stress ranges are not confidence intervals or forecasts.",
            "Recommendations, surrogate predictions, constraints and batch duration are fixed.",
            "Condition 3 retains each archived method's different effective-flow convention; absolute cross-method profit is not interpreted.",
        ],
    }
    (OUT / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(oat.shape, joint_detail.shape, joint_summary.shape, break_even.shape)


if __name__ == "__main__":
    main()
