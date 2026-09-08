"""Replace two unsupported graphical-abstract panels with verified analyses.

All surrounding artwork is preserved pixel-for-pixel.  The upper-right model
panel reports the leakage-controlled random/chronological validation, and the
lower-centre panel reports ex-post repricing of fixed archived recommendations.
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
SOURCE = PROJECT / "revision_work" / "qa" / "manuscript_media" / "word" / "media" / "image1.tiff"
TEMPORAL = ROOT / "analyses" / "strict_nested_temporal_validation" / "results" / "strict_nested_80_20_pooled_metrics.csv"
ECONOMIC = ROOT / "analyses" / "economic_sensitivity_extended" / "one_at_a_time_summary.csv"
OUT = ROOT / "analyses" / "figure_updates"

TEMPORAL_RECT = (2096, 42, 2605, 449)
ECONOMIC_RECT = (1096, 1024, 1591, 1492)


def _image_from_figure(fig: plt.Figure, dpi: int) -> Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor="white", transparent=False)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def temporal_panel(width: int, height: int, dpi: int = 220) -> Image.Image:
    frame = pd.read_csv(TEMPORAL)
    order = ["Cu_out", "As_out", "Voltage"]
    labels = [r"$Cu_{out}$", r"$As_{out}$", "Voltage"]
    random = [float(frame[(frame.target == t) & (frame.split == "random_80_20")].iloc[0].R2) for t in order]
    chrono = [float(frame[(frame.target == t) & (frame.split == "chronological_80_20")].iloc[0].R2) for t in order]
    with mpl.rc_context({
        "font.family": "Arial", "font.size": 7.0, "axes.labelsize": 7.2,
        "axes.titlesize": 7.5, "xtick.labelsize": 6.6, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2, "axes.linewidth": 0.65,
    }):
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        x = np.arange(3)
        bar_width = 0.34
        ax.bar(x - bar_width / 2, random, bar_width, color="#0072B2", label="Random")
        ax.bar(x + bar_width / 2, chrono, bar_width, color="#D55E00", label="Chronological")
        ax.axhline(0, color="#555555", linewidth=0.55)
        ax.set(xticks=x, xticklabels=labels, ylabel=r"Test $R^2$", ylim=(-1.7, 1.0))
        ax.set_title("Leakage-controlled 80/20 validation", loc="left", pad=2, fontweight="bold")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.4, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower left", ncols=1)
        fig.subplots_adjust(left=0.17, right=0.98, bottom=0.19, top=0.87)
        return _image_from_figure(fig, dpi)


def economic_panel(width: int, height: int, dpi: int = 220) -> Image.Image:
    frame = pd.read_csv(ECONOMIC)
    frame = frame[(frame.condition == 2) & (frame.parameter == "p_Cu")]
    styles = {
        "ESRL-CMO": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
        "NSGA-II": {"color": "#D55E00", "marker": "s", "linestyle": "--"},
    }
    with mpl.rc_context({
        "font.family": "Arial", "font.size": 7.0, "axes.labelsize": 7.2,
        "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2, "axes.linewidth": 0.65,
    }):
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        for method in ("ESRL-CMO", "NSGA-II"):
            part = frame[frame.method == method].sort_values("change_percent")
            x = part.change_percent.to_numpy(float)
            y = part.mean_profit_10k_CNY.to_numpy(float)
            low = part.q05_profit_10k_CNY.to_numpy(float)
            high = part.q95_profit_10k_CNY.to_numpy(float)
            style = styles[method]
            ax.fill_between(x, low, high, color=style["color"], alpha=0.10, linewidth=0)
            ax.plot(x, y, label=method, linewidth=1.0, markersize=2.5, **style)
        ax.axhline(0, color="#555555", linewidth=0.55)
        ax.set(xlim=(-32, 32), xticks=[-30, -15, 0, 15, 30],
               xlabel="Copper-price change (%)", ylabel=r"Profit ($10^4$ CNY/batch)")
        ax.set_title("Ex-post repricing, Condition 2", loc="left", pad=2, fontweight="bold")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.4, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper left")
        fig.subplots_adjust(left=0.22, right=0.985, bottom=0.20, top=0.87)
        return _image_from_figure(fig, dpi)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = Image.open(SOURCE).convert("RGB")
    revised = base.copy()
    for rect, builder in ((TEMPORAL_RECT, temporal_panel), (ECONOMIC_RECT, economic_panel)):
        left, top, right, bottom = rect
        revised.paste(builder(right - left, bottom - top), (left, top))
    revised.save(OUT / "revised_graphical_abstract_preview.png", dpi=(220, 220))
    manifest = {
        "source_figure": str(SOURCE),
        "temporal_source": str(TEMPORAL),
        "economic_source": str(ECONOMIC),
        "replaced_rectangles_pixels": {
            "random_split_prediction_panel": TEMPORAL_RECT,
            "price_downturn_bar_panel": ECONOMIC_RECT,
        },
        "preserved_artwork": "all pixels outside the two recorded rectangles",
        "economic_interpretation": "fixed archived recommendations; ex-post repricing only",
    }
    (OUT / "revised_graphical_abstract_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
