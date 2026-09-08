"""Archived-result analysis for ESRL-CMO and NSGA-II.

This script does not train a policy, call a surrogate model, or rerun an
optimizer.  It reads the archived feasible recommendation sets, applies only
an ex-post copper-price change to the existing economic results, and produces
the requested two-dimensional Pareto projections.

Accounting is kept identical to the archived implementations:
  * p_Cu = 98.44 CNY/kg at baseline;
  * As_out = 11.64 g/L in the treatment-cost term for both methods;
  * RC5 is a positive treatment-cost amount and is subtracted from profit;
  * all non-price terms and all process recommendations remain fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "optimization" / "result"
PRICE_OUT = PROJECT_ROOT / "analyses" / "price_sensitivity"
PROJECTION_OUT = PROJECT_ROOT / "analyses" / "pareto_projections"

BASE_PRICE = 98.44
PRICE_CHANGES = (-0.30, -0.275, -0.25, -0.225, -0.20, -0.175,
                 -0.15, -0.125, -0.10, 0.0, 0.15, 0.30)

ECON = {
    "c_reprocess": 0.14,
    "c_concentrate": 82.64,
    "p_e": 0.60,
    "c_op": 200.0,
    "c_As_treat": 2.50,
    "As_out_fixed": 11.64,
}

METHOD_ORDER = ("ESRL-CMO", "NSGA-II")
METHOD_STYLE = {
    "ESRL-CMO": {"color": "#1F5A99", "marker": "o"},
    "NSGA-II": {"color": "#C0602B", "marker": "^"},
}


@dataclass(frozen=True)
class Archive:
    method: str
    condition: int
    source: Path
    data: pd.DataFrame


def configure_plot_style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.75,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def _condition_from_columns(columns: set[str]) -> int:
    has_a = {"TA", "IA", "QA"}.issubset(columns)
    has_b = {"TB", "IB", "QB"}.issubset(columns)
    if has_a and has_b:
        return 3
    if has_a:
        return 1
    if has_b:
        return 2
    raise ValueError(f"Cannot infer condition from columns: {sorted(columns)}")


def _canonicalize(df: pd.DataFrame, method: str, condition: int,
                  source: Path) -> pd.DataFrame:
    required = {"Cu_in", "Cu_out", "As_out", "E_total", "Net_profit", "t"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")

    out = df.copy()
    if method == "NSGA-II":
        gcols = ["G1_Cu", "G2_As", "G3_J", "G4_V", "G5_ratio"]
        missing_g = set(gcols) - set(out.columns)
        if missing_g:
            raise ValueError(f"{source} is missing constraints: {sorted(missing_g)}")
        out["feasible"] = (out[gcols] <= 1e-8).all(axis=1)
        out["profit_stored_10k"] = pd.to_numeric(out["Net_profit"]) / 1e4
    else:
        if "costs_max" not in out.columns:
            raise ValueError(f"{source} is missing costs_max")
        out["feasible"] = pd.to_numeric(out["costs_max"]) < 1e-3
        out["profit_stored_10k"] = pd.to_numeric(out["Net_profit"])

    if not bool(out["feasible"].all()):
        count = int((~out["feasible"]).sum())
        raise ValueError(f"{source} contains {count} infeasible archived rows")

    out["method"] = method
    out["condition"] = condition
    out["source_file"] = str(source)
    if "Solution_ID" not in out.columns:
        out["Solution_ID"] = [f"{method}-C{condition}-{i+1:04d}" for i in range(len(out))]
    return out.reset_index(drop=True)


def load_archives() -> list[Archive]:
    archives: list[Archive] = []

    nsga_paths = sorted((RESULT_ROOT / "newnsga2").glob("pareto_nsga2_*.csv"))
    if len(nsga_paths) != 3:
        raise FileNotFoundError(f"Expected 3 NSGA-II Pareto CSV files, found {len(nsga_paths)}")
    for path in nsga_paths:
        raw = pd.read_csv(path)
        condition = _condition_from_columns(set(raw.columns))
        archives.append(Archive("NSGA-II", condition, path,
                                _canonicalize(raw, "NSGA-II", condition, path)))

    for condition in (1, 2, 3):
        path = (RESULT_ROOT / f"outputs_condition{condition}" /
                f"pareto_ppo_condition{condition}.csv")
        if not path.exists():
            raise FileNotFoundError(path)
        raw = pd.read_csv(path)
        archives.append(Archive("ESRL-CMO", condition, path,
                                _canonicalize(raw, "ESRL-CMO", condition, path)))

    archives.sort(key=lambda x: (x.condition, METHOD_ORDER.index(x.method)))
    expected = {(c, m) for c in (1, 2, 3) for m in METHOD_ORDER}
    actual = {(a.condition, a.method) for a in archives}
    if actual != expected:
        raise ValueError(f"Archive set mismatch. Expected {expected}, found {actual}")
    return archives


def effective_flow(df: pd.DataFrame, method: str, condition: int) -> tuple[np.ndarray, str]:
    """Return the flow convention used by the archived implementation."""
    if condition == 1:
        return df["QA"].to_numpy(float), "QA"
    if condition == 2:
        return df["QB"].to_numpy(float), "QB"
    if method == "NSGA-II":
        return ((df["QA"].to_numpy(float) + df["QB"].to_numpy(float)) / 2.0,
                "(QA+QB)/2 [archived NSGA-II]")
    return df["QA"].to_numpy(float), "QA [archived ESRL-CMO]"


def full_profit(df: pd.DataFrame, method: str, condition: int,
                copper_price: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    q_eff, q_basis = effective_flow(df, method, condition)
    cu_in = df["Cu_in"].to_numpy(float)
    cu_out = df["Cu_out"].to_numpy(float)
    duration = df["t"].to_numpy(float)
    energy = df["E_total"].to_numpy(float)
    m_cu = np.maximum(0.0, (cu_in - cu_out) * q_eff * duration)
    r_cu = (copper_price - ECON["c_reprocess"] - ECON["c_concentrate"]) * m_cu
    rc5 = ECON["c_As_treat"] * (ECON["As_out_fixed"] + cu_out) * q_eff * duration
    profit = (r_cu - rc5 - ECON["p_e"] * energy - ECON["c_op"] * duration) / 1e4
    return profit, m_cu, rc5, q_basis


def run_price_sensitivity(archives: list[Archive]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    PRICE_OUT.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict] = []
    detail_rows: list[pd.DataFrame] = []

    for archive in archives:
        df = archive.data
        recomputed, m_cu, rc5, q_basis = full_profit(
            df, archive.method, archive.condition, BASE_PRICE)
        stored = df["profit_stored_10k"].to_numpy(float)
        abs_err = np.abs(recomputed - stored)
        audit_rows.append({
            "method": archive.method,
            "condition": archive.condition,
            "n_solutions": len(df),
            "q_basis": q_basis,
            "max_abs_baseline_profit_error_10k_CNY": float(abs_err.max()),
            "mean_abs_baseline_profit_error_10k_CNY": float(abs_err.mean()),
            "source_file": str(archive.source),
        })
        if float(abs_err.max()) > 5e-4:
            raise ValueError(
                f"Baseline profit audit failed for {archive.method}, Condition "
                f"{archive.condition}: max error={abs_err.max():.6g}"
            )

        base = df[["Solution_ID", "Cu_in", "Cu_out", "As_out", "E_total", "t",
                   "profit_stored_10k", "method", "condition", "source_file"]].copy()
        base["m_Cu_kg"] = m_cu
        base["RC5_treatment_cost_CNY"] = rc5
        base["q_basis"] = q_basis
        for change in PRICE_CHANGES:
            scenario = base.copy()
            price = BASE_PRICE * (1.0 + change)
            scenario["copper_price_change_fraction"] = change
            scenario["copper_price_CNY_per_kg"] = price
            # Only p_Cu changes. This exact incremental form preserves every
            # other term already stored in the archived baseline result.
            scenario["profit_repriced_10k_CNY"] = (
                stored + (price - BASE_PRICE) * m_cu / 1e4
            )
            scenario["loss_from_baseline_10k_CNY"] = (
                stored - scenario["profit_repriced_10k_CNY"].to_numpy(float)
            )
            detail_rows.append(scenario)

    audit = pd.DataFrame(audit_rows).sort_values(["condition", "method"])
    detail = pd.concat(detail_rows, ignore_index=True)
    grouped = detail.groupby(
        ["condition", "method", "copper_price_change_fraction", "copper_price_CNY_per_kg"],
        sort=True,
    )
    summary = grouped["profit_repriced_10k_CNY"].agg(
        n_solutions="size",
        mean_profit_10k_CNY="mean",
        median_profit_10k_CNY="median",
        sd_profit_10k_CNY="std",
        min_profit_10k_CNY="min",
        q25_profit_10k_CNY=lambda x: x.quantile(0.25),
        q75_profit_10k_CNY=lambda x: x.quantile(0.75),
        max_profit_10k_CNY="max",
    ).reset_index()
    profitable = grouped["profit_repriced_10k_CNY"].apply(lambda x: float((x > 0).mean()))
    loss = grouped["loss_from_baseline_10k_CNY"].mean()
    summary["profitable_fraction"] = profitable.to_numpy()
    summary["mean_loss_from_baseline_10k_CNY"] = loss.to_numpy()
    summary["analysis_type"] = "fixed archived solution-set ex-post repricing"
    summary["As_out_fixed_in_RC5_g_per_L"] = ECON["As_out_fixed"]

    detail.to_csv(PRICE_OUT / "price_sensitivity_solution_level.csv", index=False,
                  encoding="utf-8-sig")
    summary.to_csv(PRICE_OUT / "price_sensitivity_summary.csv", index=False,
                   encoding="utf-8-sig")
    audit.to_csv(PRICE_OUT / "economic_formula_audit.csv", index=False,
                 encoding="utf-8-sig")
    plot_price_sensitivity(summary)
    return summary, detail, audit


def _clean_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_price_sensitivity(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.45), constrained_layout=True)
    for condition, ax in zip((1, 2, 3), axes):
        sub = summary[summary["condition"] == condition]
        for method in METHOD_ORDER:
            values = sub[sub["method"] == method].sort_values("copper_price_change_fraction")
            x = values["copper_price_change_fraction"].to_numpy(float) * 100
            y = values["mean_profit_10k_CNY"].to_numpy(float)
            q25 = values["q25_profit_10k_CNY"].to_numpy(float)
            q75 = values["q75_profit_10k_CNY"].to_numpy(float)
            style = METHOD_STYLE[method]
            ax.plot(x, y, color=style["color"], marker=style["marker"], markersize=3.2,
                    linewidth=1.15, label=method)
            ax.fill_between(x, q25, q75, color=style["color"], alpha=0.10, linewidth=0)
        ax.axhline(0, color="#333333", linewidth=0.75)
        ax.axvline(0, color="#777777", linewidth=0.65, linestyle="--")
        ax.set_title(f"Condition {condition}")
        ax.set_xlabel("Copper-price change (%)")
        if condition == 1:
            ax.set_ylabel(r"Mean net profit ($10^4$ CNY batch$^{-1}$)")
        _clean_axis(ax)
    axes[0].legend(frameon=False, loc="upper left")
    fig.savefig(PRICE_OUT / "copper_price_sensitivity_all_conditions.png", dpi=600)
    fig.savefig(PRICE_OUT / "copper_price_sensitivity_all_conditions.pdf")
    fig.savefig(PRICE_OUT / "copper_price_sensitivity_all_conditions.svg")
    plt.close(fig)

    # Table S20 context: the six severe downturns for Condition 2.
    c2 = summary[
        (summary["condition"] == 2)
        & (summary["copper_price_change_fraction"].between(-0.3001, -0.1749))
    ].copy()
    changes = sorted(c2["copper_price_change_fraction"].unique())
    fig, ax = plt.subplots(figsize=(5.9, 3.25), constrained_layout=True)
    x = np.arange(len(changes))
    width = 0.36
    for offset, method in zip((-width / 2, width / 2), METHOD_ORDER):
        vals = (c2[c2["method"] == method]
                .set_index("copper_price_change_fraction")
                .loc[changes, "mean_profit_10k_CNY"].to_numpy(float))
        style = METHOD_STYLE[method]
        ax.bar(x + offset, vals, width=width, color=style["color"], alpha=0.88,
               label=method, edgecolor="white", linewidth=0.4,
               hatch="//" if method == "NSGA-II" else None)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, [f"{100*v:g}%" for v in changes])
    ax.set_xlabel("Copper-price change")
    ax.set_ylabel(r"Mean net profit ($10^4$ CNY batch$^{-1}$)")
    ax.set_title("Condition 2: ex-post repricing of fixed recommendation sets")
    ax.legend(frameon=False)
    _clean_axis(ax)
    fig.savefig(PRICE_OUT / "condition2_downturn_repricing.png", dpi=600)
    fig.savefig(PRICE_OUT / "condition2_downturn_repricing.pdf")
    fig.savefig(PRICE_OUT / "condition2_downturn_repricing.svg")
    plt.close(fig)


def _projection_long_table(archives: list[Archive]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for archive in archives:
        df = archive.data.copy()
        part = df[["Solution_ID", "Cu_out", "As_out", "E_total",
                   "profit_stored_10k", "feasible", "source_file"]].copy()
        part = part.rename(columns={"profit_stored_10k": "Net_profit_10k_CNY"})
        part["method"] = archive.method
        part["condition"] = archive.condition
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def run_projections(archives: list[Archive]) -> pd.DataFrame:
    PROJECTION_OUT.mkdir(parents=True, exist_ok=True)
    long = _projection_long_table(archives)
    long.to_csv(PROJECTION_OUT / "pareto_projection_data.csv", index=False,
                encoding="utf-8-sig")

    configure_plot_style()
    pairs = (
        ("Cu_out", "As_out", r"$Cu_{out}$ (g L$^{-1}$)", r"$As_{out}$ (g L$^{-1}$)"),
        ("Cu_out", "E_total", r"$Cu_{out}$ (g L$^{-1}$)", r"$E_{total}$ (kWh batch$^{-1}$)"),
        ("As_out", "E_total", r"$As_{out}$ (g L$^{-1}$)",
         r"$E_{total}$ (kWh batch$^{-1}$)"),
    )

    fig, axes = plt.subplots(3, 3, figsize=(7.48, 7.15), constrained_layout=True)
    for i, condition in enumerate((1, 2, 3)):
        for j, (xcol, ycol, xlabel, ylabel) in enumerate(pairs):
            ax = axes[i, j]
            for method in METHOD_ORDER:
                values = long[(long["condition"] == condition) & (long["method"] == method)]
                style = METHOD_STYLE[method]
                if method == "ESRL-CMO":
                    ax.scatter(values[xcol], values[ycol], s=10, marker=style["marker"],
                               facecolors="none", edgecolors=style["color"], linewidths=0.55,
                               alpha=0.58, label=method)
                else:
                    ax.scatter(values[xcol], values[ycol], s=11, marker=style["marker"],
                               color=style["color"], edgecolors="none", alpha=0.40,
                               label=method)
            if i == 2:
                ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(("Cu-As", "Cu-energy", "As-energy")[j] + " projection")
            ax.text(0.03, 0.96, f"Condition {condition}", transform=ax.transAxes,
                    ha="left", va="top", fontweight="bold", fontsize=8.5)
            _clean_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(PROJECTION_OUT / "all_conditions_three_2D_projections.png", dpi=600)
    fig.savefig(PROJECTION_OUT / "all_conditions_three_2D_projections.pdf")
    fig.savefig(PROJECTION_OUT / "all_conditions_three_2D_projections.svg")
    plt.close(fig)

    # SI Fig. S10: the two conditions whose projections are not already shown
    # in main-text Fig. 6d-f.  This keeps the supplementary panel focused and
    # avoids duplicating Condition 2.
    fig, axes = plt.subplots(2, 3, figsize=(7.48, 5.0), constrained_layout=True)
    panel_letters = iter("abcdef")
    for i, condition in enumerate((1, 3)):
        for j, (xcol, ycol, xlabel, ylabel) in enumerate(pairs):
            ax = axes[i, j]
            for method in METHOD_ORDER:
                values = long[(long["condition"] == condition) & (long["method"] == method)]
                style = METHOD_STYLE[method]
                ax.scatter(
                    values[xcol], values[ycol], s=11, marker=style["marker"],
                    facecolors="none" if method == "ESRL-CMO" else style["color"],
                    edgecolors=style["color"] if method == "ESRL-CMO" else "none",
                    linewidths=0.55, alpha=0.58 if method == "ESRL-CMO" else 0.40,
                    label=method,
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.text(0.02, 1.02, f"({next(panel_letters)}) Condition {condition}",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontweight="bold", fontsize=8.5)
            _clean_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(PROJECTION_OUT / "figS10_conditions1_3_projections.png", dpi=600)
    fig.savefig(PROJECTION_OUT / "figS10_conditions1_3_projections.pdf")
    fig.savefig(PROJECTION_OUT / "figS10_conditions1_3_projections.svg")
    plt.close(fig)

    for condition in (1, 2, 3):
        fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.5), constrained_layout=True)
        for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, pairs):
            for method in METHOD_ORDER:
                values = long[(long["condition"] == condition) & (long["method"] == method)]
                style = METHOD_STYLE[method]
                ax.scatter(values[xcol], values[ycol], s=12, marker=style["marker"],
                           facecolors="none" if method == "ESRL-CMO" else style["color"],
                           edgecolors=style["color"] if method == "ESRL-CMO" else "none",
                           linewidths=0.55, alpha=0.58 if method == "ESRL-CMO" else 0.40,
                           label=method)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            _clean_axis(ax)
        axes[0].legend(frameon=False)
        fig.savefig(PROJECTION_OUT / f"condition{condition}_three_2D_projections.png", dpi=600)
        fig.savefig(PROJECTION_OUT / f"condition{condition}_three_2D_projections.pdf")
        fig.savefig(PROJECTION_OUT / f"condition{condition}_three_2D_projections.svg")
        plt.close(fig)
    return long


def main() -> None:
    configure_plot_style()
    archives = load_archives()
    summary, detail, audit = run_price_sensitivity(archives)
    projections = run_projections(archives)
    print("Archived-result analysis completed.")
    print(f"  Price summary rows: {len(summary)}")
    print(f"  Solution-scenario rows: {len(detail)}")
    print(f"  Projection rows: {len(projections)}")
    print(f"  Maximum baseline audit error: "
          f"{audit['max_abs_baseline_profit_error_10k_CNY'].max():.3e} (10^4 CNY)")


if __name__ == "__main__":
    main()
