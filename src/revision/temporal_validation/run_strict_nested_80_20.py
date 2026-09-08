"""Leakage-controlled random versus chronological 80/20 validation.

This script evaluates the nine ExtraTrees surrogate models using the copied
project data only.  For every operating condition and target, it creates an
outer 80/20 holdout.  Hyperparameters are then selected exclusively inside
the outer training partition:

* chronological_80_20: chronological outer holdout and five-fold
  ``TimeSeriesSplit`` on the chronologically ordered outer training data;
* random_80_20: seeded random outer holdout and shuffled five-fold ``KFold``
  on the outer training data.

The same fixed, predeclared six-candidate ExtraTrees search set is used for
all nine models and both split rules.  The outer test set is never used for
imputation, duplicate removal in the training set, feature selection, model
selection, or hyperparameter tuning.  Exact duplicates are removed within
the training and test partitions separately and are audited explicitly.
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "data_by_operation_mode_with_features.xlsx"
OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
MODEL_ROOT = OUTPUT_ROOT / "models"

DATETIME_COL = "日期时间"
TEST_FRACTION = 0.20
N_CV_SPLITS = 5
SPLIT_SEED = 20260830
ESTIMATOR_SEED = 20260830
SPLITS = ("random_80_20", "chronological_80_20")
CONDITION_LABEL = {
    "three_stage": "Condition 1",
    "four_stage": "Condition 2",
    "serial": "Condition 3",
}

# Declared before any model fitting.  These candidates span tree count,
# depth, feature subsampling, and leaf/split regularisation while keeping the
# search computationally reproducible and identical for every model.
PARAM_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "n_estimators": 150,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": 0.60,
    },
    {
        "n_estimators": 200,
        "max_depth": 16,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 0.80,
    },
    {
        "n_estimators": 250,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.00,
    },
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.00,
    },
    {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "max_features": 0.80,
    },
    {
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": 0.80,
    },
)


@dataclass(frozen=True)
class ModelSpec:
    target_key: str
    sheet: str
    features: tuple[str, ...]
    target_column: str
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
    for target_key, target_column, unit in (
        ("Cu_out", CU_TARGET, "g/L"),
        ("As_out", AS_TARGET, "g/L"),
    ):
        specs.extend(
            [
                ModelSpec(
                    target_key,
                    "three_stage",
                    (CU_IN, TA, IA, QA, VA, *COMMON_TIME, "three_stage_power"),
                    target_column,
                    unit,
                ),
                ModelSpec(
                    target_key,
                    "four_stage",
                    (CU_IN, TB, IB, QB, VB, *COMMON_TIME, "four_stage_power"),
                    target_column,
                    unit,
                ),
                ModelSpec(
                    target_key,
                    "serial",
                    (
                        CU_IN,
                        TA,
                        IA,
                        QA,
                        VA,
                        TB,
                        IB,
                        QB,
                        VB,
                        *COMMON_TIME,
                        "total_power",
                    ),
                    target_column,
                    unit,
                ),
            ]
        )
    specs.extend(
        [
            ModelSpec(
                "Voltage",
                "three_stage",
                (CU_IN, TA, IA, QA, *COMMON_TIME),
                VA,
                "V",
            ),
            ModelSpec(
                "Voltage",
                "four_stage",
                (CU_IN, TB, IB, QB, *COMMON_TIME),
                VB,
                "V",
            ),
            ModelSpec(
                "Voltage",
                "serial",
                (CU_IN, TA, IA, QA, TB, IB, QB, *COMMON_TIME),
                VAVG,
                "V",
                derive_serial_voltage=True,
            ),
        ]
    )
    return specs


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialise {type(value)!r}")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    denominator = np.maximum(np.abs(y_true), np.finfo(float).eps)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE_percent": float(np.mean(np.abs(y_true - y_pred) / denominator) * 100.0),
    }


def sheet_audit() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sheet in CONDITION_LABEL:
        raw = pd.read_excel(DATA_FILE, sheet_name=sheet)
        parsed_time = pd.to_datetime(raw[DATETIME_COL], errors="coerce")
        rows.append(
            {
                "condition": CONDITION_LABEL[sheet],
                "sheet": sheet,
                "n_rows": len(raw),
                "n_columns": len(raw.columns),
                "n_missing_cells_all_columns": int(raw.isna().sum().sum()),
                "n_rows_with_missing_all_columns": int(raw.isna().any(axis=1).sum()),
                "n_exact_duplicates_all_columns": int(raw.duplicated(keep="first").sum()),
                "n_duplicate_timestamps": int(parsed_time.duplicated(keep="first").sum()),
                "n_invalid_timestamps": int(parsed_time.isna().sum()),
                "time_start": parsed_time.min(),
                "time_end": parsed_time.max(),
            }
        )
    return pd.DataFrame(rows)


def load_model_frame(spec: ModelSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_excel(DATA_FILE, sheet_name=spec.sheet)
    if spec.derive_serial_voltage:
        raw[VAVG] = (
            pd.to_numeric(raw[VA], errors="coerce")
            + pd.to_numeric(raw[VB], errors="coerce")
        ) / 2.0
    required = [DATETIME_COL, *spec.features, spec.target_column]
    missing_columns = sorted(set(required) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"{spec.sheet}/{spec.target_key}: missing {missing_columns}")

    frame = raw[required].copy()
    original_time = frame[DATETIME_COL].copy()
    frame[DATETIME_COL] = pd.to_datetime(original_time, errors="coerce")
    invalid_time = int((original_time.notna() & frame[DATETIME_COL].isna()).sum())
    if frame[DATETIME_COL].isna().any():
        raise ValueError(
            f"{spec.sheet}/{spec.target_key}: {frame[DATETIME_COL].isna().sum()} "
            "missing or invalid timestamps prevent leakage-safe chronological splitting"
        )

    invalid_numeric = 0
    for column in [*spec.features, spec.target_column]:
        original = frame[column].copy()
        converted = pd.to_numeric(original, errors="coerce")
        invalid_numeric += int((original.notna() & converted.isna()).sum())
        frame[column] = converted

    audit = {
        "condition": CONDITION_LABEL[spec.sheet],
        "sheet": spec.sheet,
        "target": spec.target_key,
        "n_rows_before_outer_split": len(frame),
        "n_required_columns": len(required),
        "n_missing_feature_cells_before_split": int(frame[list(spec.features)].isna().sum().sum()),
        "n_rows_with_missing_features_before_split": int(
            frame[list(spec.features)].isna().any(axis=1).sum()
        ),
        "n_missing_targets_before_split": int(frame[spec.target_column].isna().sum()),
        "n_invalid_numeric_cells_from_nonmissing_values": invalid_numeric,
        "n_invalid_timestamps": invalid_time,
        "n_exact_duplicates_required_columns_before_split": int(
            frame.duplicated(subset=required, keep="first").sum()
        ),
    }
    return frame, audit


def outer_split(frame: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split == "random_80_20":
        # Group exact records so a duplicate pair can never straddle the
        # random outer holdout.  This grouping is fixed before any fitting and
        # does not use test-set outcomes for model or parameter selection.
        exact_record_groups = pd.util.hash_pandas_object(
            frame, index=False
        ).astype("uint64")
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_FRACTION,
            random_state=SPLIT_SEED,
        )
        train_index, test_index = next(
            splitter.split(frame, groups=exact_record_groups)
        )
        return frame.iloc[train_index].copy(), frame.iloc[test_index].copy()

    if split == "chronological_80_20":
        ordered = frame.sort_values(DATETIME_COL, kind="mergesort").reset_index(drop=True)
        cut = int(math.floor(len(ordered) * (1.0 - TEST_FRACTION)))
        if cut <= 0 or cut >= len(ordered):
            raise ValueError("Not enough rows for chronological 80/20 split")
        # Keep equal timestamps on one side of the holdout boundary.  Moving
        # the cut left prevents the same acquisition time from entering both.
        boundary_time = ordered.loc[cut, DATETIME_COL]
        while cut > 0 and ordered.loc[cut - 1, DATETIME_COL] == boundary_time:
            cut -= 1
        if cut <= 0:
            raise ValueError("Timestamp ties consume the chronological training partition")
        return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()

    raise ValueError(split)


def prepare_outer_partitions(
    frame: pd.DataFrame, spec: ModelSpec, split: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train, test = outer_split(frame, split)
    required = [DATETIME_COL, *spec.features, spec.target_column]
    n_train_before = len(train)
    n_test_before = len(test)

    train_target_missing = int(train[spec.target_column].isna().sum())
    test_target_missing = int(test[spec.target_column].isna().sum())
    train = train.loc[train[spec.target_column].notna()].copy()
    test = test.loc[test[spec.target_column].notna()].copy()

    train_dup = int(train.duplicated(subset=required, keep="first").sum())
    test_dup = int(test.duplicated(subset=required, keep="first").sum())
    train = train.drop_duplicates(subset=required, keep="first").copy()
    test = test.drop_duplicates(subset=required, keep="first").copy()

    train_hash = pd.util.hash_pandas_object(train[required], index=False)
    test_hash = pd.util.hash_pandas_object(test[required], index=False)
    train_hash_set = set(train_hash.astype("uint64").tolist())
    cross_overlap = int(test_hash.astype("uint64").isin(train_hash_set).sum())

    if split == "chronological_80_20":
        train = train.sort_values(DATETIME_COL, kind="mergesort").reset_index(drop=True)
        test = test.sort_values(DATETIME_COL, kind="mergesort").reset_index(drop=True)
        if train[DATETIME_COL].max() >= test[DATETIME_COL].min():
            raise AssertionError("Chronological train/test time ranges overlap")
    else:
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

    if len(train) < N_CV_SPLITS + 2 or len(test) < 2:
        raise ValueError(f"Too few usable rows for {spec.sheet}/{spec.target_key}/{split}")

    audit = {
        "condition": CONDITION_LABEL[spec.sheet],
        "sheet": spec.sheet,
        "target": spec.target_key,
        "split": split,
        "n_train_before_target_filter_and_within_partition_dedup": n_train_before,
        "n_test_before_target_filter_and_within_partition_dedup": n_test_before,
        "n_train_missing_targets_excluded": train_target_missing,
        "n_test_missing_targets_excluded": test_target_missing,
        "n_train_exact_duplicates_removed_within_partition": train_dup,
        "n_test_exact_duplicates_removed_within_partition": test_dup,
        "n_train_final": len(train),
        "n_test_final": len(test),
        "actual_test_fraction_after_target_filter_and_dedup": len(test) / (len(train) + len(test)),
        "n_train_missing_feature_cells": int(train[list(spec.features)].isna().sum().sum()),
        "n_test_missing_feature_cells": int(test[list(spec.features)].isna().sum().sum()),
        "n_exact_test_rows_also_present_in_train": cross_overlap,
        "train_start": train[DATETIME_COL].min(),
        "train_end": train[DATETIME_COL].max(),
        "test_start": test[DATETIME_COL].min(),
        "test_end": test[DATETIME_COL].max(),
    }
    return train, test, audit


def make_estimator(params: dict[str, Any], needs_imputation: bool) -> Any:
    model = ExtraTreesRegressor(
        **params,
        random_state=ESTIMATOR_SEED,
        n_jobs=-1,
        bootstrap=False,
    )
    if needs_imputation:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )
    return model


def cv_iterator(split: str, n_rows: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    placeholder = np.empty((n_rows, 1))
    if split == "chronological_80_20":
        return TimeSeriesSplit(n_splits=N_CV_SPLITS).split(placeholder)
    return KFold(
        n_splits=N_CV_SPLITS,
        shuffle=True,
        random_state=SPLIT_SEED,
    ).split(placeholder)


def tune_inside_training(
    train: pd.DataFrame,
    spec: ModelSpec,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    features = list(spec.features)
    X = train[features]
    y = train[spec.target_column].to_numpy(float)
    needs_imputation = bool(X.isna().any().any())
    cv_pairs = list(cv_iterator(split, len(train)))

    selection_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for candidate_index, params in enumerate(PARAM_CANDIDATES, start=1):
        fold_rmse: list[float] = []
        fold_mae: list[float] = []
        fold_r2: list[float] = []
        for fold_index, (fit_index, validation_index) in enumerate(cv_pairs, start=1):
            estimator = make_estimator(params, needs_imputation)
            estimator.fit(X.iloc[fit_index], y[fit_index])
            prediction = estimator.predict(X.iloc[validation_index])
            values = regression_metrics(y[validation_index], prediction)
            fold_rmse.append(values["RMSE"])
            fold_mae.append(values["MAE"])
            fold_r2.append(values["R2"])
            if candidate_index == 1:
                fit_times = train.iloc[fit_index][DATETIME_COL]
                validation_times = train.iloc[validation_index][DATETIME_COL]
                fold_rows.append(
                    {
                        "condition": CONDITION_LABEL[spec.sheet],
                        "sheet": spec.sheet,
                        "target": spec.target_key,
                        "split": split,
                        "cv_strategy": "TimeSeriesSplit" if split.startswith("chronological") else "KFold",
                        "fold": fold_index,
                        "n_fit": len(fit_index),
                        "n_validation": len(validation_index),
                        "fit_time_start": fit_times.min(),
                        "fit_time_end": fit_times.max(),
                        "validation_time_start": validation_times.min(),
                        "validation_time_end": validation_times.max(),
                    }
                )

        selection_rows.append(
            {
                "condition": CONDITION_LABEL[spec.sheet],
                "sheet": spec.sheet,
                "target": spec.target_key,
                "split": split,
                "candidate_index": candidate_index,
                **params,
                "mean_validation_RMSE": float(np.mean(fold_rmse)),
                "std_validation_RMSE": float(np.std(fold_rmse, ddof=1)),
                "mean_validation_MAE": float(np.mean(fold_mae)),
                "std_validation_MAE": float(np.std(fold_mae, ddof=1)),
                "mean_validation_R2": float(np.mean(fold_r2)),
                "std_validation_R2": float(np.std(fold_r2, ddof=1)),
                "n_cv_splits": N_CV_SPLITS,
                "cv_strategy": "TimeSeriesSplit" if split.startswith("chronological") else "KFold",
                "imputation_fitted_inside_each_fold": needs_imputation,
            }
        )

    best = min(
        selection_rows,
        key=lambda row: (row["mean_validation_RMSE"], row["candidate_index"]),
    )
    selected_params = {
        key: best[key]
        for key in (
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
        )
    }
    for row in selection_rows:
        row["selected"] = row["candidate_index"] == best["candidate_index"]
    return selected_params, selection_rows, fold_rows


def run() -> dict[str, pd.DataFrame]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    sheet_audit_df = sheet_audit()
    model_input_audit_rows: list[dict[str, Any]] = []
    partition_audit_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for spec in model_specs():
        frame, input_audit = load_model_frame(spec)
        model_input_audit_rows.append(input_audit)
        for split in SPLITS:
            train, test, partition_audit = prepare_outer_partitions(frame, spec, split)
            partition_audit_rows.append(partition_audit)

            selected_params, selection, folds = tune_inside_training(train, spec, split)
            selection_rows.extend(selection)
            fold_rows.extend(folds)

            needs_imputation = bool(train[list(spec.features)].isna().any().any())
            final_estimator = make_estimator(selected_params, needs_imputation)
            fit_started = time.perf_counter()
            final_estimator.fit(
                train[list(spec.features)],
                train[spec.target_column].to_numpy(float),
            )
            fit_seconds = time.perf_counter() - fit_started
            predict_started = time.perf_counter()
            train_prediction = final_estimator.predict(train[list(spec.features)])
            test_prediction = final_estimator.predict(test[list(spec.features)])
            predict_seconds = time.perf_counter() - predict_started

            train_values = regression_metrics(
                train[spec.target_column].to_numpy(float), train_prediction
            )
            test_values = regression_metrics(
                test[spec.target_column].to_numpy(float), test_prediction
            )
            selected_candidate = next(
                row for row in selection if bool(row["selected"])
            )
            metric_rows.append(
                {
                    "condition": CONDITION_LABEL[spec.sheet],
                    "sheet": spec.sheet,
                    "target": spec.target_key,
                    "unit": spec.unit,
                    "split": split,
                    "n_train": len(train),
                    "n_test": len(test),
                    "train_start": train[DATETIME_COL].min(),
                    "train_end": train[DATETIME_COL].max(),
                    "test_start": test[DATETIME_COL].min(),
                    "test_end": test[DATETIME_COL].max(),
                    "selected_candidate_index": selected_candidate["candidate_index"],
                    **selected_params,
                    "inner_cv_mean_RMSE": selected_candidate["mean_validation_RMSE"],
                    "inner_cv_std_RMSE": selected_candidate["std_validation_RMSE"],
                    "final_fit_seconds": fit_seconds,
                    "final_train_and_test_prediction_seconds": predict_seconds,
                    **{f"train_{key}": value for key, value in train_values.items()},
                    **{f"test_{key}": value for key, value in test_values.items()},
                }
            )

            prediction_frame = test[[DATETIME_COL, spec.target_column]].copy()
            prediction_frame = prediction_frame.rename(
                columns={spec.target_column: "observed"}
            )
            prediction_frame["predicted"] = test_prediction
            prediction_frame["residual"] = (
                prediction_frame["observed"] - prediction_frame["predicted"]
            )
            prediction_frame["condition"] = CONDITION_LABEL[spec.sheet]
            prediction_frame["sheet"] = spec.sheet
            prediction_frame["target"] = spec.target_key
            prediction_frame["unit"] = spec.unit
            prediction_frame["split"] = split
            prediction_frames.append(prediction_frame)

            model_path = MODEL_ROOT / f"{spec.target_key}_{spec.sheet}_{split}.joblib"
            joblib.dump(final_estimator, model_path)

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["target", "condition", "split"]
    )
    predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["target", "condition", "split", DATETIME_COL]
    )
    pooled_rows: list[dict[str, Any]] = []
    for (target, split), group in predictions_df.groupby(["target", "split"], sort=True):
        values = regression_metrics(
            group["observed"].to_numpy(float), group["predicted"].to_numpy(float)
        )
        pooled_rows.append(
            {
                "condition": "Pooled",
                "target": target,
                "split": split,
                "n_test": len(group),
                **values,
            }
        )
    pooled_df = pd.DataFrame(pooled_rows).sort_values(["target", "split"])

    outputs = {
        "sheet_audit": sheet_audit_df,
        "model_input_audit": pd.DataFrame(model_input_audit_rows).sort_values(
            ["target", "condition"]
        ),
        "partition_audit": pd.DataFrame(partition_audit_rows).sort_values(
            ["target", "condition", "split"]
        ),
        "model_selection": pd.DataFrame(selection_rows).sort_values(
            ["target", "condition", "split", "candidate_index"]
        ),
        "cv_folds": pd.DataFrame(fold_rows).sort_values(
            ["target", "condition", "split", "fold"]
        ),
        "metrics": metrics_df,
        "pooled_metrics": pooled_df,
        "predictions": predictions_df,
    }
    for name, dataframe in outputs.items():
        dataframe.to_csv(
            OUTPUT_ROOT / f"strict_nested_80_20_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    manifest = {
        "analysis": "Strict nested random-versus-chronological 80/20 ExtraTrees validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_file": str(DATA_FILE.resolve()),
        "output_root": str(OUTPUT_ROOT.resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "outer_test_fraction_nominal": TEST_FRACTION,
        "split_seed": SPLIT_SEED,
        "estimator_seed": ESTIMATOR_SEED,
        "inner_cv_splits": N_CV_SPLITS,
        "predeclared_parameter_candidates": list(PARAM_CANDIDATES),
        "selection_criterion": "minimum mean inner-validation RMSE; candidate order breaks exact ties",
        "data_handling": {
            "outer_test_isolation": (
                "The outer test set is not used for imputation, duplicate removal in the training "
                "partition, feature selection, parameter selection, or model fitting."
            ),
            "duplicates": (
                "Exact duplicates over timestamp, model features, and target are removed separately "
                "within each outer training and test partition. For the random holdout, exact records "
                "are assigned as a group so no exact duplicate can straddle the boundary; "
                "cross-partition overlap is audited."
            ),
            "missing_features": (
                "If required feature values are missing, median imputation is fitted inside each "
                "inner fold and then on the complete outer training partition only. No imputer is "
                "constructed when the required feature matrix is complete."
            ),
            "missing_targets": (
                "Rows without an observed target cannot be scored and are excluded separately after "
                "the outer split; counts are reported."
            ),
            "outlier_processing": "No outlier detection, removal, or clipping is applied.",
            "feature_selection": "No feature selection is applied; manuscript feature sets are fixed.",
        },
        "limitations": [
            (
                "The chronological holdout diagnoses forward generalisation under the observed "
                "2024-2025 operating history; it is not an independent plant, campaign, or prospective trial."
            ),
            (
                "Calendar variables and unrecorded maintenance, assay, electrolyte, feed, and operating-policy "
                "changes can be confounded, so performance differences do not identify a causal mechanism."
            ),
            (
                "MAPE is included for continuity but is sensitive to small denominators; R2, RMSE, and MAE "
                "should be interpreted jointly."
            ),
            (
                "The fixed six-candidate grid is a controlled model-selection set rather than proof of a "
                "globally optimal ExtraTrees configuration."
            ),
        ],
        "output_files": {name: f"strict_nested_80_20_{name}.csv" for name in outputs},
    }
    (OUTPUT_ROOT / "strict_nested_80_20_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "strict_nested_80_20_results.json").write_text(
        json.dumps(
            {
                "nine_model_metrics": metrics_df.to_dict(orient="records"),
                "pooled_metrics": pooled_df.to_dict(orient="records"),
                "selected_parameters": metrics_df[
                    [
                        "condition",
                        "sheet",
                        "target",
                        "split",
                        "selected_candidate_index",
                        "n_estimators",
                        "max_depth",
                        "min_samples_split",
                        "min_samples_leaf",
                        "max_features",
                        "inner_cv_mean_RMSE",
                        "inner_cv_std_RMSE",
                    ]
                ].to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return outputs


def main() -> None:
    started = time.perf_counter()
    outputs = run()
    elapsed = time.perf_counter() - started
    print(f"Completed strict nested 80/20 validation in {elapsed:.1f} s")
    columns = [
        "condition",
        "target",
        "split",
        "n_train",
        "n_test",
        "selected_candidate_index",
        "test_R2",
        "test_RMSE",
        "test_MAE",
        "test_MAPE_percent",
    ]
    print(outputs["metrics"][columns].to_string(index=False))
    print("\nPooled test metrics")
    print(outputs["pooled_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
