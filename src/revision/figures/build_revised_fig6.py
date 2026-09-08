"""Replace Fig. 6i with verified ex-post copper-price sensitivity results.

Panels a-h are preserved pixel-for-pixel from the original manuscript asset.
Only the lower-right panel is replaced.  Source tables and the exact composite
rectangle are recorded in the accompanying manifest.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
SOURCE_FIG = PROJECT / "revision_work" / "qa" / "manuscript_media" / "word" / "media" / "image7.tiff"
SOURCE_DATA = ROOT / "analyses" / "economic_sensitivity_extended" / "one_at_a_time_summary.csv"
OUT = ROOT / "analyses" / "figure_updates"
COMPOSITE_RECT = (700, 570, 1054, 866)  # left, top, right, bottom in source pixels


def price_data() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE_DATA)
    selected = frame[
        (frame["condition"] == 2)
        & (frame["parameter"] == "p_Cu")
        & (frame["method"].isin(["ESRL-CMO", "NSGA-II"]))
    ].copy()
    return selected.sort_values(["method", "change_percent"])


def draw_panel(width_px: int, height_px: int, dpi: int = 220) -> Image.Image:
    frame = price_data()
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 6.2,
        "axes.labelsize": 6.4,
        "axes.titlesize": 6.6,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 5.4,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    styles = {
        "ESRL-CMO": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
        "NSGA-II": {"color": "#D55E00", "marker": "s", "linestyle": "--"},
    }
    for method in ("ESRL-CMO", "NSGA-II"):
        part = frame[frame["method"] == method]
        x = part["change_percent"].to_numpy(float)
        y = part["mean_profit_10k_CNY"].to_numpy(float)
        low = part["q05_profit_10k_CNY"].to_numpy(float)
        high = part["q95_profit_10k_CNY"].to_numpy(float)
        style = styles[method]
        ax.fill_between(x, low, high, color=style["color"], alpha=0.10, linewidth=0)
        ax.plot(x, y, label=method, linewidth=1.05, markersize=2.6, **style)
    ax.axhline(0.0, color="#555555", linewidth=0.55)
    ax.set_xlim(-32, 32)
    ax.set_xticks([-30, -15, 0, 15, 30])
    ax.set_xlabel("Copper-price change (%)", labelpad=1.5)
    ax.set_ylabel(r"Net profit ($10^4$ CNY/batch)", labelpad=1.0)
    ax.set_title("(i) Ex-post repricing", loc="left", pad=2.0, fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", ncols=1, handlelength=1.8)
    fig.subplots_adjust(left=0.25, right=0.985, bottom=0.21, top=0.88)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor="white", transparent=False)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def standalone() -> None:
    frame = price_data()
    with mpl.rc_context({
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }):
        fig, ax = plt.subplots(figsize=(89 / 25.4, 62 / 25.4), layout="constrained")
        styles = {
            "ESRL-CMO": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
            "NSGA-II": {"color": "#D55E00", "marker": "s", "linestyle": "--"},
        }
        for method in ("ESRL-CMO", "NSGA-II"):
            part = frame[frame["method"] == method]
            x = part["change_percent"].to_numpy(float)
            y = part["mean_profit_10k_CNY"].to_numpy(float)
            low = part["q05_profit_10k_CNY"].to_numpy(float)
            high = part["q95_profit_10k_CNY"].to_numpy(float)
            style = styles[method]
            ax.fill_between(x, low, high, color=style["color"], alpha=0.10, linewidth=0)
            ax.plot(x, y, label=method, linewidth=1.3, markersize=3.5, **style)
        ax.axhline(0, color="#555555", linewidth=0.7)
        ax.set(
            xlim=(-32, 32),
            xticks=[-30, -15, 0, 15, 30],
            xlabel="Copper-price change (%)",
            ylabel=r"Net profit ($10^4$ CNY batch$^{-1}$)",
        )
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper left")
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(OUT / f"figure6i_price_sensitivity.{suffix}", dpi=600, facecolor="white")
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = Image.open(SOURCE_FIG).convert("RGB")
    left, top, right, bottom = COMPOSITE_RECT
    panel = draw_panel(right - left, bottom - top)
    revised = base.copy()
    revised.paste(panel, (left, top))
    revised.save(OUT / "revised_fig6_preview.png", dpi=(220, 220))
    standalone()
    manifest = {
        "source_figure": str(SOURCE_FIG),
        "source_data": str(SOURCE_DATA),
        "preserved_panels": "a-h",
        "replaced_panel": "i",
        "composite_rectangle_pixels": COMPOSITE_RECT,
        "panel_estimator": "mean across each archived recommendation set",
        "band": "5th-95th percentile across archived recommendations",
        "analysis": "fixed archived recommendation sets; ex-post repricing only",
        "condition": 2,
    }
    (OUT / "revised_fig6_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
