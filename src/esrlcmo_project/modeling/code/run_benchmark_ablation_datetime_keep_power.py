#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_benchmark_ablation_datetime_keep_power.py

【消融实验 5 — 分工况建模，时间特征以原始 datetime 数值表示，保留功率特征】

控制变量设计：
  原代码（基准）：分工况 + 分解时间特征（year/month/day/hour）+ 功率特征
  本脚本（消融）：分工况（结构不变）
               + 以单一 datetime 数值列替换四个分解时间列
               + 保留功率特征（完全不变）

时间特征替换策略：
  原始特征集中含 year/month/day/hour 四列，本脚本将其替换为一列：
    'datetime_numeric'：对数据中的日期时间字段（假设列名含 '时间'/'date'/'time'
    或为 pandas 可解析的日期列）解析后转换为 Unix 时间戳（秒级整数），
    供模型作为连续数值特征使用。

  若 sheet 中不存在可解析的日期时间列，则回退策略：
    用 year*10000 + month*100 + day + hour/24 合成一个连续数值，
    保证单调性与连续性，效果等价于完整日期时间的数值编码。

  无论哪种策略，结果都是用"1个连续时间数值"替换"4个分解时间整数"，
  这是本消融与基准的唯一区别（功率特征完全不变）。

严格控制变量措施：
  1. 每个 sheet 以【原始完整特征集 + 目标列】做 dropna
     → 有效行集合与原代码完全一致，行数相同。
  2. train_test_split(test_size=0.1, random_state=42) 参数完全相同
     → 同一 sheet 内每一行的 train/test 归属与原代码完全一致。
  3. 所有模型超参数、random_state 均调用原 _build_sklearn_models()
     → 与原代码完全相同。
  4. CNN/DNN torch.manual_seed 与原代码一致。
  5. 三工况 TOPSIS 汇总方式（拼接预测值后计算综合指标）与原代码一致。

输出目录：benchmarkoutputs/ablation_datetime_keep_power/
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_utils import (
    DATA_FILE, TEST_SIZE, RANDOM_SEED,
    MODEL_NAMES_ORDER, MODEL_FULLNAMES,
    _build_sklearn_models, _train_cnn, _train_dnn,
    calc_metrics, topsis, save_to_excel,
    OPERATION_MODES, OPERATION_LABELS,
    TASK_CONFIGS as ORIG_TASK_CONFIGS,
)

warnings.filterwarnings('ignore')

OUTPUT_DIR   = 'benchmarkoutputs/ablation_datetime_keep_power'
BENEFIT_COLS = ('Test_R2',)
COST_COLS    = ('Test_RMSE', 'Test_MAPE')
WEIGHTS      = [1, 1, 1]

# 原始时间分解列名（将被替换为单一 datetime_numeric 列）
_TIME_COLS    = ['year', 'month', 'day', 'hour']
_POWER_COLS   = {'three_stage_power', 'four_stage_power', 'total_power'}
_DATETIME_COL = 'datetime_numeric'   # 替换后使用的列名

RAW_COLS = [
    'Model', 'Model_CN',
    'Train_R2', 'Train_RMSE', 'Train_MAPE',
    'Test_R2',  'Test_RMSE',  'Test_MAPE',
]


# ─────────────────────────────────────────────────────────────────────────────
# 工具：将 year/month/day/hour 四列合成一个连续时间数值
# ─────────────────────────────────────────────────────────────────────────────
def _build_datetime_numeric(df: pd.DataFrame) -> pd.Series:
    """
    策略 A：若 df 中存在可解析为 datetime 的列（列名含 '时间'/'date'/'time'），
            将其转为 Unix 时间戳（秒）。
    策略 B（回退）：用 year/month/day/hour 合成连续数值：
            value = year*8784 + month*744 + day*24 + hour
            （近似小时级单调递增序列，量级一致）
    无论哪种策略，均保证单调、连续、无量纲分裂。
    """
    # 尝试找到日期时间列（策略 A）
    dt_candidates = [c for c in df.columns
                     if any(kw in c.lower() for kw in ['时间', 'date', 'time', 'datetime'])]
    for col in dt_candidates:
        try:
            ts = pd.to_datetime(df[col], errors='coerce')
            if ts.notna().mean() > 0.9:   # 超过90%可解析则采用
                return ts.astype('int64') // 10**9  # 转为秒级 Unix 时间戳
        except Exception:
            pass

    # 策略 B：用分解列合成（要求这些列必须存在）
    required = {'year', 'month', 'day', 'hour'}
    if required.issubset(df.columns):
        return (df['year'].astype(float) * 8784
                + df['month'].astype(float) * 744
                + df['day'].astype(float) * 24
                + df['hour'].astype(float))

    raise ValueError(
        "无法构造 datetime_numeric：数据中既无可解析的日期时间列，"
        "也不含 year/month/day/hour 四列。"
    )


def _build_ablation_feature_cols(orig_cols):
    """
    将原始特征列表中的 year/month/day/hour 替换为单一的 datetime_numeric，
    功率列及其他列保持原位置不变（datetime_numeric 插入到第一个时间列的位置）。
    """
    result = []
    time_inserted = False
    for col in orig_cols:
        if col in _TIME_COLS:
            if not time_inserted:
                result.append(_DATETIME_COL)
                time_inserted = True
            # 后续时间列直接跳过
        else:
            result.append(col)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 消融特征集（列名层面）
# ─────────────────────────────────────────────────────────────────────────────
ABLATION_FEATURES = {
    task_name: {
        mode: _build_ablation_feature_cols(
            ORIG_TASK_CONFIGS[task_name]['features'][mode]
        )
        for mode in ORIG_TASK_CONFIGS[task_name]['features']
    }
    for task_name in ['Cu', 'As', 'Voltage']
}


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载：以原始特征集确定有效行，用 datetime_numeric 替换时间分解列
# ─────────────────────────────────────────────────────────────────────────────
def load_data_ablation(sheet_name, orig_feature_cols, orig_target_col,
                       ablation_feature_cols, random_state=RANDOM_SEED):
    df = pd.read_excel(DATA_FILE, sheet_name=sheet_name)

    if orig_target_col == '三四段平均电压V' and '三四段平均电压V' not in df.columns:
        df['三四段平均电压V'] = (df['三段电压V'] + df['四段电压V']) / 2

    # 以原始完整特征+目标确定有效行（与原 load_data 完全一致）
    df_valid = df.dropna(subset=orig_feature_cols + [orig_target_col]).reset_index(drop=True)

    # 构造 datetime_numeric 列（仅在有效行 df_valid 上操作）
    if _DATETIME_COL in ablation_feature_cols:
        df_valid = df_valid.copy()
        df_valid[_DATETIME_COL] = _build_datetime_numeric(df_valid).values

    # 取出消融特征矩阵
    X = df_valid[ablation_feature_cols].values.astype(float)
    y = df_valid[orig_target_col].values.astype(float)

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=random_state)


# ─────────────────────────────────────────────────────────────────────────────
# 单工况训练（结构与原 run_benchmark 完全一致）
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark_ablation(sheet_name, orig_feature_cols, orig_target_col,
                           ablation_feature_cols,
                           random_state=RANDOM_SEED, verbose=True):
    X_train, X_test, y_train, y_test = load_data_ablation(
        sheet_name, orig_feature_cols, orig_target_col,
        ablation_feature_cols, random_state
    )
    if verbose:
        print(f"    数据: {len(X_train)+len(X_test)} 条  "
              f"(训练 {len(X_train)} / 测试 {len(X_test)})  "
              f"特征维度: {X_train.shape[1]}")

    sklearn_models = _build_sklearn_models(random_state)
    records = []
    predictions = {}

    for name in MODEL_NAMES_ORDER:
        pad = f"{name:<20}"
        if verbose:
            print(f"    [{pad}] 训练中...", end='', flush=True)
        try:
            if name == 'CNN':
                ptr, pte = _train_cnn(X_train, y_train, X_test, y_test,
                                      random_state=random_state)
            elif name == 'DNN':
                ptr, pte = _train_dnn(X_train, y_train, X_test, y_test,
                                      random_state=random_state)
            else:
                sklearn_models[name].fit(X_train, y_train)
                ptr = sklearn_models[name].predict(X_train)
                pte = sklearn_models[name].predict(X_test)

            tr = calc_metrics(y_train, ptr)
            te = calc_metrics(y_test,  pte)
            status = f"Test R²={te['R2']:.4f}  RMSE={te['RMSE']:.4f}  MAPE={te['MAPE']:.2f}%"
            predictions[name] = {
                'train': ptr, 'test': pte,
                'true_train': y_train, 'true_test': y_test,
            }
        except Exception as e:
            tr = te = {'R2': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
            status = f"失败: {e}"
            predictions[name] = {
                'train': None, 'test': None,
                'true_train': y_train, 'true_test': y_test,
            }

        if verbose:
            print(f"  {status}")

        records.append({
            'Model':       name,
            'Model_CN':    MODEL_FULLNAMES[name],
            'Train_R2':    tr['R2'],   'Train_RMSE': tr['RMSE'], 'Train_MAPE': tr['MAPE'],
            'Test_R2':     te['R2'],   'Test_RMSE':  te['RMSE'], 'Test_MAPE':  te['MAPE'],
        })

    return pd.DataFrame(records), predictions


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("消融实验 5：分工况建模 + 用 datetime 数值替换 year/month/day/hour，保留功率特征")
    print("对比基准  ：原代码（分工况 + year/month/day/hour 分解时间特征 + 功率特征）")
    print("唯一变量  ：时间特征表示方式（4列分解 → 1列连续数值）")
    print("时间编码  ：优先使用 Unix 时间戳；回退为 year*8784+month*744+day*24+hour")
    print("=" * 80)

    for task_name in ['Cu', 'As', 'Voltage']:
        print(f"\n{'='*80}")
        print(f"▶ 预测目标：{task_name}")
        print(f"{'='*80}")

        all_sheets        = {}
        all_modes_results = []
        all_predictions   = {}
        orig_cfg = ORIG_TASK_CONFIGS[task_name]

        for mode in OPERATION_MODES:
            label = OPERATION_LABELS[mode]
            print(f"\n  ── {label} ({mode}) ─────────────────────────────────")

            orig_feat_cols = orig_cfg['features'][mode]
            ablation_cols  = ABLATION_FEATURES[task_name][mode]
            orig_target    = orig_cfg['sheet_targets'][mode]

            removed = [f for f in orig_feat_cols if f in _TIME_COLS]
            added   = [_DATETIME_COL] if _DATETIME_COL in ablation_cols else []
            kept_power = [f for f in ablation_cols if f in _POWER_COLS]
            print(f"    替换时间特征: {removed} → {added}")
            print(f"    保留功率特征: {kept_power}")
            print(f"    消融特征集({len(ablation_cols)}列): {ablation_cols}")

            df_results, mode_predictions = run_benchmark_ablation(
                sheet_name=mode,
                orig_feature_cols=orig_feat_cols,
                orig_target_col=orig_target,
                ablation_feature_cols=ablation_cols,
                random_state=RANDOM_SEED,
                verbose=True,
            )

            df_results.insert(0, 'OperationMode', label)
            all_sheets[f"{task_name}_{label}_原始"] = df_results[['OperationMode'] + RAW_COLS]
            all_modes_results.append(df_results)
            all_predictions[mode] = mode_predictions

        # ── TOPSIS 汇总（与原代码逻辑完全一致）──────────────────────────────
        print(f"\n  ── TOPSIS 综合排序（{task_name}，三工况拼接）──────────────")

        combined_records = []
        for model_name in MODEL_NAMES_ORDER:
            all_y_train, all_y_pred_train = [], []
            all_y_test,  all_y_pred_test  = [], []
            for mode in OPERATION_MODES:
                pd_ = all_predictions[mode].get(model_name, {})
                if pd_.get('train') is not None:
                    all_y_train.extend(pd_['true_train'])
                    all_y_pred_train.extend(pd_['train'])
                if pd_.get('test') is not None:
                    all_y_test.extend(pd_['true_test'])
                    all_y_pred_test.extend(pd_['test'])

            tr = calc_metrics(np.array(all_y_train), np.array(all_y_pred_train))
            te = calc_metrics(np.array(all_y_test),  np.array(all_y_pred_test))
            combined_records.append({
                'Model': model_name, 'Model_CN': MODEL_FULLNAMES[model_name],
                'Train_R2': tr['R2'], 'Train_RMSE': tr['RMSE'], 'Train_MAPE': tr['MAPE'],
                'Test_R2':  te['R2'], 'Test_RMSE':  te['RMSE'], 'Test_MAPE':  te['MAPE'],
            })

        combined  = pd.DataFrame(combined_records)
        topsis_df = topsis(combined, benefit_cols=BENEFIT_COLS,
                           cost_cols=COST_COLS, weights=WEIGHTS)

        print(f"  {'排名':<4} {'模型':<22} {'Test_R2':>8} {'Test_RMSE':>10} "
              f"{'Test_MAPE':>10} {'TOPSIS分数':>12}")
        print("  " + "-" * 72)
        for _, row in topsis_df.iterrows():
            print(f"  {int(row['TOPSIS_Rank']):<4} {row['Model']:<22} "
                  f"{row['Test_R2']:>8.4f} {row['Test_RMSE']:>10.4f} "
                  f"{row['Test_MAPE']:>9.2f}% {row['TOPSIS_Score']:>12.4f}")

        top3 = topsis_df[topsis_df['TOPSIS_Rank'] <= 3]['Model'].tolist()
        print(f"\n  ✓ TOPSIS 前3名: {' / '.join(top3)}")

        all_sheets[f"{task_name}_TOPSIS汇总"] = topsis_df[
            ['TOPSIS_Rank', 'Model', 'Model_CN',
             'Test_R2', 'Test_RMSE', 'Test_MAPE', 'TOPSIS_Score']
        ]
        for mode, df_mode in zip(OPERATION_MODES, all_modes_results):
            label = OPERATION_LABELS[mode]
            df_t  = topsis(df_mode, benefit_cols=BENEFIT_COLS,
                           cost_cols=COST_COLS, weights=WEIGHTS)
            all_sheets[f"{task_name}_{label}_TOPSIS"] = df_t[
                ['TOPSIS_Rank', 'Model', 'Model_CN',
                 'Train_R2', 'Train_RMSE', 'Train_MAPE',
                 'Test_R2',  'Test_RMSE',  'Test_MAPE', 'TOPSIS_Score']
            ]

        # 总览 sheet
        overview_rows = []
        for mode in OPERATION_MODES:
            label  = OPERATION_LABELS[mode]
            df_raw = all_sheets.get(f"{task_name}_{label}_原始")
            if df_raw is None:
                continue
            for _, row in df_raw.iterrows():
                overview_rows.append({
                    'Target': task_name, 'Mode': label,
                    'Model': row['Model'], 'Model_CN': row['Model_CN'],
                    'Train_R2': row['Train_R2'], 'Train_RMSE': row['Train_RMSE'],
                    'Train_MAPE': row['Train_MAPE'],
                    'Test_R2':  row['Test_R2'],  'Test_RMSE':  row['Test_RMSE'],
                    'Test_MAPE': row['Test_MAPE'],
                })
        all_sheets['00_总览'] = pd.DataFrame(overview_rows)

        ordered = {'00_总览': all_sheets.pop('00_总览')}
        ordered.update(all_sheets)

        output_path = os.path.join(OUTPUT_DIR,
                                   f'ablation_datetime_keep_power_{task_name}.xlsx')
        save_to_excel(ordered, output_path)

    print(f"\n{'='*80}")
    print("全部完成！结果保存至:", OUTPUT_DIR)


if __name__ == '__main__':
    main()
