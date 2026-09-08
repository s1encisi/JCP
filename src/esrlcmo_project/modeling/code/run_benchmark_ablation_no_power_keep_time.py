#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_benchmark_ablation_no_power_keep_time.py

【消融实验 4 — 分工况建模，去除功率特征，保留时间特征（year/month/day/hour）】

控制变量设计：
  原代码（基准）：分工况 + year/month/day/hour + 功率特征
  本脚本（消融）：分工况（结构不变）+ 保留 year/month/day/hour + 去掉功率特征

严格控制变量措施：
  1. 每个 sheet 以【原始完整特征集 + 目标列】做 dropna
     → 有效行集合与原代码完全一致，行数相同。
  2. train_test_split(test_size=0.1, random_state=42) 参数完全相同
     → 同一 sheet 内每一行的 train/test 归属与原代码完全一致。
  3. 所有模型超参数、random_state 均调用原 _build_sklearn_models()
     → 与原代码完全相同。
  4. CNN/DNN torch.manual_seed 与原代码一致。
  5. 三工况 TOPSIS 汇总方式（拼接预测值后计算综合指标）与原代码一致。

  唯一改变的变量：输入特征去掉功率列（*_power），时间列原样保留。

功率特征定义：
  three_stage 工况：three_stage_power
  four_stage  工况：four_stage_power
  serial      工况：total_power

输出目录：benchmarkoutputs/ablation_no_power_keep_time/
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

OUTPUT_DIR   = 'benchmarkoutputs/ablation_no_power_keep_time'
BENEFIT_COLS = ('Test_R2',)
COST_COLS    = ('Test_RMSE', 'Test_MAPE')
WEIGHTS      = [1, 1, 1]

# ─────────────────────────────────────────────────────────────────────────────
# 消融特征集：仅剔除功率列，时间列原样保留
# ─────────────────────────────────────────────────────────────────────────────
_REMOVE_POWER = {'three_stage_power', 'four_stage_power', 'total_power'}

ABLATION_FEATURES = {
    task_name: {
        mode: [f for f in ORIG_TASK_CONFIGS[task_name]['features'][mode]
               if f not in _REMOVE_POWER]
        for mode in ORIG_TASK_CONFIGS[task_name]['features']
    }
    for task_name in ['Cu', 'As', 'Voltage']
}

RAW_COLS = [
    'Model', 'Model_CN',
    'Train_R2', 'Train_RMSE', 'Train_MAPE',
    'Test_R2',  'Test_RMSE',  'Test_MAPE',
]


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载：以原始特征集确定有效行，仅替换输入特征列
# ─────────────────────────────────────────────────────────────────────────────
def load_data_ablation(sheet_name, orig_feature_cols, orig_target_col,
                       ablation_feature_cols, random_state=RANDOM_SEED):
    df = pd.read_excel(DATA_FILE, sheet_name=sheet_name)

    if orig_target_col == '三四段平均电压V' and '三四段平均电压V' not in df.columns:
        df['三四段平均电压V'] = (df['三段电压V'] + df['四段电压V']) / 2

    # 以原始完整特征+目标确定有效行（与原 load_data 完全一致）
    df_valid = df.dropna(subset=orig_feature_cols + [orig_target_col]).reset_index(drop=True)

    # 仅替换输入特征列
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
    print("消融实验 4：分工况建模 + 保留 year/month/day/hour，去除功率特征")
    print("对比基准  ：原代码（分工况 + 含时间/功率特征）")
    print("唯一变量  ：输入特征去掉功率列（three_stage_power/four_stage_power/total_power）")
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

            removed = [f for f in orig_feat_cols if f not in ablation_cols]
            kept_time = [f for f in ablation_cols if f in {'year', 'month', 'day', 'hour'}]
            print(f"    去除特征 {len(removed)} 列: {removed}")
            print(f"    保留时间特征: {kept_time}")

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
                                   f'ablation_no_power_keep_time_{task_name}.xlsx')
        save_to_excel(ordered, output_path)

    print(f"\n{'='*80}")
    print("全部完成！结果保存至:", OUTPUT_DIR)


if __name__ == '__main__':
    main()
