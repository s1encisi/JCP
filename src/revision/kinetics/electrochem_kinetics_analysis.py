"""Electrochemical-kinetics analysis of the ExtraTrees voltage-current PDP curves.

Reads the archived partial-dependence data (modeling/interpretability_outputs/pdp)
read-only, converts current to current density, and fits the activation (Tafel),
ohmic (linear) and concentration (mass-transfer) polarization regimes. Also
extracts the As_out vs Cu_in and As_out vs VB PDP curves for the Cu-As
co-deposition discussion. Results are written to the code_run folder only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openpyxl

ESRL = Path(__file__).resolve().parents[2] / 'esrlcmo_project'
PDP = ESRL / "modeling" / "interpretability_outputs" / "pdp"
OUT = Path(__file__).resolve().parent / "electrochem_kinetics_results"
OUT.mkdir(parents=True, exist_ok=True)

# Geometry from the original nsga2_optimization.py PARAMS
N_CATHODE = 35          # cathodes per cell
A_CATHODE = 1.100 * 1.029 * 2.0   # cathode plate area, both faces (m2)
N_CELL = 16             # cells per unit
AREA = N_CATHODE * A_CATHODE       # total active area per unit (m2)


def load_pdp(path: Path, sheet: str) -> tuple[np.ndarray, np.ndarray]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    xs, ys = [], []
    first = True
    for row in ws.iter_rows(values_only=True):
        if first:
            first = False
            continue
        x, y = row[0], row[1]
        if x is not None and y is not None:
            try:
                xs.append(float(x))
                ys.append(float(y))
            except (TypeError, ValueError):
                pass
    wb.close()
    return np.asarray(xs), np.asarray(ys)


def linreg(x, y):
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef[0], coef[1]  # slope, intercept


def r2(x, y):
    if len(x) < 3:
        return float("nan")
    sse = np.sum((y - np.polyval(np.polyfit(x, y, 1), x)) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return 1.0 - sse / sst if sst > 0 else float("nan")


def fit_regions(I, V, name):
    """Fit V vs J (ohmic slope) and V vs ln J (Tafel slope) over three regimes."""
    J = I / AREA  # A/m2
    lnJ = np.log(J)
    out = {
        "name": name,
        "n_points": int(len(I)),
        "I_min_A": float(I.min()),
        "I_max_A": float(I.max()),
        "J_min": float(J.min()),
        "J_max": float(J.max()),
    }
    # --- activation / low-current regime: V vs ln J (Tafel-like) ---
    # use the first third of the current range where the curve rises from its floor
    n = len(I)
    lo = int(n * 0.0)
    hi = int(n * 0.35) + 1
    a_slope, a_int = linreg(lnJ[lo:hi], V[lo:hi])
    out["activation_I_range_A"] = [float(I[lo]), float(I[hi - 1])]
    out["activation_J_range"] = [float(J[lo]), float(J[hi - 1])]
    out["activation_tafel_slope_V_per_decade"] = float(a_slope * np.log(10))
    out["activation_R2_V_vs_lnJ"] = float(r2(lnJ[lo:hi], V[lo:hi]))
    # --- ohmic / mid-current regime: V vs J linear ---
    # use the middle third
    lo = int(n * 0.40)
    hi = int(n * 0.70) + 1
    o_slope, o_int = linreg(J[lo:hi], V[lo:hi])
    out["ohmic_I_range_A"] = [float(I[lo]), float(I[hi - 1])]
    out["ohmic_J_range"] = [float(J[lo]), float(J[hi - 1])]
    out["ohmic_slope_V_per_A_m2"] = float(o_slope)
    out["ohmic_slope_mV_per_A_m2"] = float(o_slope * 1e3)
    out["ohmic_R2_V_vs_J"] = float(r2(J[lo:hi], V[lo:hi]))
    # --- concentration / high-current regime: identify steepening ---
    lo = int(n * 0.75)
    hi = n
    c_slope, c_int = linreg(J[lo:hi], V[lo:hi])
    out["concentration_I_range_A"] = [float(I[lo]), float(I[-1])]
    out["concentration_slope_V_per_A_m2"] = float(c_slope)
    out["concentration_slope_mV_per_A_m2"] = float(c_slope * 1e3)
    out["concentration_R2_V_vs_J"] = float(r2(J[lo:hi], V[lo:hi]))
    # voltage per cell at min and max
    out["V_per_cell_min"] = float(V.min() / N_CELL)
    out["V_per_cell_max"] = float(V.max() / N_CELL)
    # total voltage change and per-cell change
    out["dV_total_V"] = float(V.max() - V.min())
    out["dV_per_cell_V"] = float((V.max() - V.min()) / N_CELL)
    return out


def main():
    results = {}
    for cond, feat in [("Condition1", "IA"), ("Condition2", "IB"),
                       ("Condition3", "IA"), ("Condition3", "IB")]:
        I, V = load_pdp(PDP / f"pdp_voltage_{cond}.xlsx", feat)
        results[f"{cond}_{feat}"] = fit_regions(I, V, f"{cond} {feat}")

    # Cu-As co-deposition: As_out vs Cu_in (Condition 2) and As_out vs VB
    for cond, feat in [("Condition2", "Cu_in"), ("Condition2", "VB"),
                       ("Condition1", "QA"), ("Condition3", "QB")]:
        x, y = load_pdp(PDP / f"pdp_as_{cond}.xlsx", feat)
        key = f"As_out_vs_{feat}_{cond}"
        imin = int(np.argmin(y))
        imax = int(np.argmax(y))
        results[key] = {
            "name": f"As_out vs {feat} ({cond})",
            "n_points": int(len(x)),
            "x_min": float(x.min()),
            "x_max": float(x.max()),
            "y_min": float(y.min()),
            "y_max": float(y.max()),
            "x_at_y_min": float(x[imin]),
            "y_at_x_min": float(y[imin]),
            "x_at_y_max": float(x[imax]),
            "total_span": float(y.max() - y.min()),
            "full_curve_x": x.tolist(),
            "full_curve_y": y.tolist(),
        }

    (OUT / "electrochem_kinetics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    # concise console report
    print(f"{'curve':24s} {'I range (A)':>16s} {'act Tafel(V/dec)':>16s} "
          f"{'ohm(mV/A/m2)':>13s} {'conc(mV/A/m2)':>14s} {'dV/cell':>8s}")
    for k in ["Condition1_IA", "Condition2_IB", "Condition3_IA", "Condition3_IB"]:
        r = results[k]
        print(f"{k:24s} {r['I_min_A']:7.0f}-{r['I_max_A']:<7.0f} "
              f"{r['activation_tafel_slope_V_per_decade']:16.3f} "
              f"{r['ohmic_slope_mV_per_A_m2']:13.3f} "
              f"{r['concentration_slope_mV_per_A_m2']:14.3f} "
              f"{r['dV_per_cell_V']:8.3f}")
    print()
    print("Cu-As co-deposition:")
    for k in ["As_out_vs_Cu_in_Condition2", "As_out_vs_VB_Condition2"]:
        r = results[k]
        print(f"  {r['name']}: y_min={r['y_min']:.3f} at x={r['x_at_y_min']:.3f}, "
              f"span={r['total_span']:.3f}")
    print("\nAREA (m2/unit) =", AREA, "  J = I/", AREA)
    print("J_max check: I=27000 -> J=", 27000 / AREA)


if __name__ == "__main__":
    main()
