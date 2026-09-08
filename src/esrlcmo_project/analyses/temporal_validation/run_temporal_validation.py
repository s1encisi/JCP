"""Controlled random-versus-chronological validation of the nine ExtraTrees models.

The final preprocessed three-sheet dataset, feature sets, targets, 90/10 holdout
fraction, saved ExtraTrees hyperparameters, estimator random states, missing-row
handling, and metrics are held fixed.  Exact duplicate model rows are removed
once before either split so that identical records cannot occur on both sides
of a random holdout.  The experimental factor is then the split rule:

* random_90_10: the original ``train_test_split`` implementation;
* chronological_90_10: the earliest 90% is used for training and the latest
  10% for testing after sorting by ``日期时间``.

This is a temporal holdout diagnostic.  It does not turn ExtraTrees into an
autoregressive sequence model and it does not re-run the upstream cleaning or
feature-engineering pipeline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "data_by_operation_mode_with_features.xlsx"
CONFIG_ROOT = PROJECT_ROOT / "modeling" / "outputs"
OUTPUT_ROOT = PROJECT_ROOT / "analyses" / "temporal_validation"
MODEL_ROOT = OUTPUT_ROOT / "models"

DATETIME_COL = "日期时间"
TEST_FRACTION = 0.10
SPLITS = ("random_90_10", "chronological_90_10")
CONDITION_LABEL = {
    "three_stage": "Condition 1",
    "four_stage": "Condition 2",
    "serial": "Condition 3",
}


@dataclass(frozen=True)
class ModelSpec:
    target_key: str
    target_label: str
    sheet: str
    features: tuple[str, ...]
    target_column: str
    config_file: Path
    unit: str
    derive_serial_voltage: bool = False


COMMON_TIME = ("year", "month", "day", "hour")
CU_TARGET = "废铜液原液罐Cu（g/L）"
AS_TARGET = "废铜液原液罐As（g/L）"
CU_IN = "电积前液Cu（g/L）"
TA, IA, VA, QA = "三段溶液温度℃", "三段电流强度A", "三段电压V", "三段流量m3/h"
TB, IB, VB, QB = "四段溶液温度℃", "四段电流强度A", "四段电压V", "四段流量m3/h"
VAVG = "三四段平均电压V"


def model_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for key, label, target, config, unit in (
        ("Cu_out", "Cu$_{out}$", CU_TARGET, CONFIG_ROOT / "cu_config.json", "g/L"),
        ("As_out", "As$_{out}$", AS_TARGET, CONFIG_ROOT / "as_config.json", "g/L"),
    ):
        specs.extend([
            ModelSpec(key, label, "three_stage",
                      (CU_IN, TA, IA, QA, VA, *COMMON_TIME, "three_stage_power"),
                      target, config, unit),
            ModelSpec(key, label, "four_stage",
                      (CU_IN, TB, IB, QB, VB, *COMMON_TIME, "four_stage_power"),
                      target, config, unit),
            ModelSpec(key, label, "serial",
                      (CU_IN, TA, IA, QA, VA, TB, IB, QB, VB, *COMMON_TIME, "total_power"),
                      target, config, unit),
        ])
    voltage_config = CONFIG_ROOT / "voltage_config.json"
    specs.extend([
        ModelSpec("Voltage", "Voltage", "three_stage",
                  (CU_IN, TA, IA, QA, *COMMON_TIME), VA, voltage_config, "V"),
        ModelSpec("Voltage", "Voltage", "four_stage",
                  (CU_IN, TB, IB, QB, *COMMON_TIME), VB, voltage_config, "V"),
        ModelSpec("Voltage", "Voltage", "serial",
                  (CU_IN, TA, IA, QA, TB, IB, QB, *COMMON_TIME), VAVG,
                  voltage_config, "V", derive_serial_voltage=True),
    ])
    return specs


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


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_percent": float(
            np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
        ),
    }


def load_model_frame(spec: ModelSpec) -> pd.DataFrame:
    frame = pd.read_excel(DATA_FILE, sheet_name=spec.sheet)
    if spec.derive_serial_voltage:
        frame[VAVG] = (pd.to_numeric(frame[VA], errors="coerce")
                       + pd.to_numeric(frame[VB], errors="coerce")) / 2.0
    columns = [DATETIME_COL, *spec.features, spec.target_column]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{spec.sheet}/{spec.target_key} missing columns: {sorted(missing)}")
    frame = frame[columns].dropna(subset=[*spec.features, spec.target_column]).copy()
    frame[DATETIME_COL] = pd.to_datetime(frame[DATETIME_COL], errors="coerce")
    if frame[DATETIME_COL].isna().any():
        raise ValueError(f"{spec.sheet}/{spec.target_key} contains invalid timestamps")
    n_before = len(frame)
    frame = frame.drop_duplicates(subset=columns, keep="first").reset_index(drop=True)
    frame.attrs["n_exact_duplicate_rows_removed"] = n_before - len(frame)
    return frame


def load_saved_settings(spec: ModelSpec) -> tuple[int, dict]:
    payload = json.loads(spec.config_file.read_text(encoding="utf-8"))
    cfg = payload[spec.sheet]
    seed = int(cfg["BEST_RANDOM_STATE"])
    allowed = {"n_estimators", "max_depth", "min_samples_split",
               "min_samples_leaf", "max_features"}
    params = {k: v for k, v in cfg["BEST_PARAMS"].items() if k in allowed}
    return seed, params


def split_frame(frame: pd.DataFrame, spec: ModelSpec, split: str,
                seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split == "random_90_10":
        train_idx, test_idx = train_test_split(
            np.arange(len(frame)), test_size=TEST_FRACTION, random_state=seed
        )
        return frame.iloc[train_idx].copy(), frame.iloc[test_idx].copy()
    if split == "chronological_90_10":
        ordered = frame.sort_values(DATETIME_COL, kind="mergesort").reset_index(drop=True)
        n_test = math.ceil(len(ordered) * TEST_FRACTION)
        cut = len(ordered) - n_test
        if cut <= 0:
            raise ValueError(f"Not enough rows for {spec.sheet}/{spec.target_key}")
        return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
    raise ValueError(split)


def run_validation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    spec_rows: list[dict] = []

    for spec in model_specs():
        frame = load_model_frame(spec)
        seed, params = load_saved_settings(spec)
        spec_rows.append({
            "condition": CONDITION_LABEL[spec.sheet],
            "sheet": spec.sheet,
            "target": spec.target_key,
            "target_column": spec.target_column,
            "features": " | ".join(spec.features),
            "n_total_after_original_dropna_rule_and_exact_deduplication": len(frame),
            "n_exact_duplicate_rows_removed": frame.attrs[
                "n_exact_duplicate_rows_removed"
            ],
            "test_fraction": TEST_FRACTION,
            "estimator_random_state": seed,
            **params,
            "config_file": str(spec.config_file),
            "data_file": str(DATA_FILE),
        })

        for split in SPLITS:
            train, test = split_frame(frame, spec, split, seed)
            model = ExtraTreesRegressor(random_state=seed, **params)
            model.fit(train[list(spec.features)], train[spec.target_column].to_numpy(float))
            pred_train = model.predict(train[list(spec.features)])
            pred_test = model.predict(test[list(spec.features)])
            train_metrics = metrics(train[spec.target_column].to_numpy(float), pred_train)
            test_metrics = metrics(test[spec.target_column].to_numpy(float), pred_test)

            metric_rows.append({
                "condition": CONDITION_LABEL[spec.sheet],
                "sheet": spec.sheet,
                "target": spec.target_key,
                "target_label": spec.target_label,
                "unit": spec.unit,
                "split": split,
                "n_total": len(frame),
                "n_train": len(train),
                "n_test": len(test),
                "train_start": train[DATETIME_COL].min(),
                "train_end": train[DATETIME_COL].max(),
                "test_start": test[DATETIME_COL].min(),
                "test_end": test[DATETIME_COL].max(),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            })

            prediction = test[[DATETIME_COL, spec.target_column]].copy()
            prediction = prediction.rename(columns={spec.target_column: "observed"})
            prediction["predicted"] = pred_test
            prediction["residual"] = prediction["observed"] - prediction["predicted"]
            prediction["condition"] = CONDITION_LABEL[spec.sheet]
            prediction["sheet"] = spec.sheet
            prediction["target"] = spec.target_key
            prediction["unit"] = spec.unit
            prediction["split"] = split
            prediction_rows.append(prediction)

            model_path = MODEL_ROOT / f"{spec.target_key}_{spec.sheet}_{split}.joblib"
            joblib.dump(model, model_path)

    metrics_df = pd.DataFrame(metric_rows).sort_values(["target", "condition", "split"])
    predictions_df = pd.concat(prediction_rows, ignore_index=True).sort_values(
        ["target", "condition", "split", DATETIME_COL]
    )
    specs_df = pd.DataFrame(spec_rows).sort_values(["target", "condition"])

    metrics_df.to_csv(OUTPUT_ROOT / "temporal_validation_metrics.csv", index=False,
                      encoding="utf-8-sig")
    predictions_df.to_csv(OUTPUT_ROOT / "temporal_validation_predictions.csv", index=False,
                          encoding="utf-8-sig")
    specs_df.to_csv(OUTPUT_ROOT / "temporal_validation_model_specs.csv", index=False,
                    encoding="utf-8-sig")
    return metrics_df, predictions_df, specs_df


def pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (target, split), group in predictions.groupby(["target", "split"], sort=True):
        values = metrics(group["observed"].to_numpy(float), group["predicted"].to_numpy(float))
        rows.append({"condition": "Pooled", "target": target, "split": split,
                     "n_test": len(group), **values})
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_ROOT / "temporal_validation_pooled_metrics.csv", index=False,
                  encoding="utf-8-sig")
    return result


def _clean_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figures(metrics_df: pd.DataFrame, predictions_df: pd.DataFrame) -> None:
    configure_plot_style()
    target_order = ("Cu_out", "As_out", "Voltage")
    condition_order = ("Condition 1", "Condition 2", "Condition 3")
    colors = {"random_90_10": "#1F5A99", "chronological_90_10": "#C0602B"}
    labels = {"random_90_10": "Random 90/10", "chronological_90_10": "Chronological 90/10"}

    fig, axes = plt.subplots(3, 3, figsize=(7.48, 7.15), constrained_layout=True)
    for i, condition in enumerate(condition_order):
        for j, target in enumerate(target_order):
            ax = axes[i, j]
            sub = predictions_df[
                (predictions_df["condition"] == condition)
                & (predictions_df["target"] == target)
            ]
            lo = float(min(sub["observed"].min(), sub["predicted"].min()))
            hi = float(max(sub["observed"].max(), sub["predicted"].max()))
            pad = max((hi - lo) * 0.04, 1e-6)
            for split in SPLITS:
                values = sub[sub["split"] == split]
                ax.scatter(values["observed"], values["predicted"], s=10,
                           facecolors="none", edgecolors=colors[split], linewidths=0.55,
                           alpha=0.60, label=labels[split])
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#333333",
                    linestyle="--", linewidth=0.75)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            if i == 2:
                ax.set_xlabel("Observed")
            if j == 0:
                ax.set_ylabel("Predicted")
            if i == 0:
                ax.set_title({"Cu_out": r"$Cu_{out}$", "As_out": r"$As_{out}$",
                              "Voltage": "Voltage"}[target])
            ax.text(0.03, 0.96, condition, transform=ax.transAxes, ha="left", va="top",
                    fontsize=8.3, fontweight="bold")
            _clean_axis(ax)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(OUTPUT_ROOT / "random_vs_chronological_predictions.png", dpi=600)
    fig.savefig(OUTPUT_ROOT / "random_vs_chronological_predictions.pdf")
    fig.savefig(OUTPUT_ROOT / "random_vs_chronological_predictions.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.55), constrained_layout=True, sharey=True)
    x = np.arange(len(condition_order))
    width = 0.36
    for ax, target in zip(axes, target_order):
        sub = metrics_df[metrics_df["target"] == target]
        for offset, split in zip((-width / 2, width / 2), SPLITS):
            values = (sub[sub["split"] == split].set_index("condition")
                      .loc[list(condition_order), "test_R2"].to_numpy(float))
            ax.bar(x + offset, values, width=width, color=colors[split], alpha=0.88,
                   label=labels[split], edgecolor="white", linewidth=0.4,
                   hatch="//" if split == "chronological_90_10" else None)
        ax.axhline(0, color="#333333", linewidth=0.75)
        ax.set_xticks(x, ["C1", "C2", "C3"])
        ax.set_xlabel("Operating condition")
        ax.set_title({"Cu_out": r"$Cu_{out}$", "As_out": r"$As_{out}$",
                      "Voltage": "Voltage"}[target])
        _clean_axis(ax)
    axes[0].set_ylabel(r"Test $R^2$")
    axes[0].legend(frameon=False)
    fig.savefig(OUTPUT_ROOT / "random_vs_chronological_r2.png", dpi=600)
    fig.savefig(OUTPUT_ROOT / "random_vs_chronological_r2.pdf")
    fig.savefig(OUTPUT_ROOT / "random_vs_chronological_r2.svg")
    plt.close(fig)


def main() -> None:
    configure_plot_style()
    metrics_df, predictions_df, _ = run_validation()
    pooled = pooled_metrics(predictions_df)
    make_figures(metrics_df, predictions_df)
    print("Temporal validation completed.")
    print(metrics_df[["condition", "target", "split", "n_train", "n_test",
                      "test_R2", "test_RMSE", "test_MAE", "test_MAPE_percent"]]
          .to_string(index=False))
    print("\nPooled test metrics")
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
