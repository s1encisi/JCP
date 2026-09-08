#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_benchmark_ablation_merge_conditions.py

【消融实验 2 — 不分工况建模，保留完整特征集（含时间/功率特征）】

控制变量设计：
  原代码（基准）：分工况建模 + 完整特征集（含时间/功率）
  本脚本（消融）：不分工况（三个 sheet 数据合并）+ 完整特征集（与原代码完全相同）

严格控制变量措施：
  1. 每个 sheet 以【原始完整特征集 + 目标列】做 dropna
     → 有效行集合与原代码完全一致，行数相同。
  2. train_test_split(test_size=0.1, random_state=42) 参数完全相同
     → 同一 sheet 内每一行的 train/test 归属与原代码完全一致。
  3. 三个 sheet 各自完成 dropna + split 后，将训练集拼接为一个统一训练集，
     测试集同理，然后在这个合并数据集上训练单一模型。
     （等价于原代码分工况训练后看三工况共享模型的性能）
  4. 所有模型超参数、random_state 均调用原 _build_sklearn_models()
     → 与原代码完全相同。
  5. CNN/DNN torch.manual_seed 与原代码一致。

  注意：三工况特征列数不同（three_stage/four_stage 比 serial 少若干列）。
  处理策略：采用"最大特征集对齐"——以 serial 工况的特征集为超集，
  three_stage/four_stage 缺失的列用 0 填充，并在特征名称列表中明确标注。
  这是在不改变其他任何条件下、合并异构工况数据的最中性处理方式。

  唯一改变的变量：三工况数据合并为一个数据集后训练单一模型（不分工况）。

输出目录：benchmarkoutputs/ablation_merge_conditions/
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

OUTPUT_DIR   = 'benchmarkoutputs/ablation_merge_conditions'
BENEFIT_COLS = ('Test_R2',)
COST_COLS    = ('Test_RMSE', 'Test_MAPE')
WEIGHTS      = [1, 1, 1]

RAW_COLS = [
    'Model', 'Model_CN',
    'Train_R2', 'Train_RMSE', 'Train_MAPE',
    'Test_R2',  'Test_RMSE',  'Test_MAPE',
]


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载：以原始完整特征集确定有效行，返回对齐后的 X, y（含划分）
# ─────────────────────────────────────────────────────────────────────────────
def load_data_for_merge(sheet_name, feature_cols, target_col,
                        random_state=RANDOM_SEED):
    """
    复现原 load_data() 的行筛选与划分，返回划分后的 X_train/X_test/y_train/y_test，
    以及用于拼接的特征列名列表（供后续对齐）。
    """
    df = pd.read_excel(DATA_FILE, sheet_name=sheet_name)

    if target_col == '三四段平均电压V' and '三四段平均电压V' not in df.columns:
        df['三四段平均电压V'] = (df['三段电压V'] + df['四段电压V']) / 2

    df_valid = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    X = df_valid[feature_cols].values.astype(float)
    y = df_valid[target_col].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, feature_cols


def build_merged_dataset(task_name, random_state=RANDOM_SEED):
    """
    对三个工况分别做 load_data_for_merge，然后对齐特征维度并合并。

    特征对齐策略：
      - 收集三个工况的特征列名，取并集（保持确定顺序）
      - 每个工况缺失的列用 0 填充
      - 这样合并后的特征矩阵维度统一，模型可以直接训练
    """
    orig_cfg = ORIG_TASK_CONFIGS[task_name]

    per_mode = {}
    for mode in OPERATION_MODES:
        feat_cols  = orig_cfg['features'][mode]
        target_col = orig_cfg['sheet_targets'][mode]
        Xtr, Xte, ytr, yte, cols = load_data_for_merge(
            sheet_name=mode,
            feature_cols=feat_cols,
            target_col=target_col,
            random_state=random_state,
        )
        per_mode[mode] = {'Xtr': Xtr, 'Xte': Xte,
                          'ytr': ytr, 'yte': yte, 'cols': list(cols)}

    # 建立统一特征列顺序（取三个工况特征列的有序并集）
    seen = {}
    for mode in OPERATION_MODES:
        for col in per_mode[mode]['cols']:
            if col not in seen:
                seen[col] = len(seen)
    all_feature_cols = list(seen.keys())
    total_dim = len(all_feature_cols)

    def align(X_arr, src_cols, all_cols):
        """将 X_arr（行×len(src_cols)）扩展到 行×len(all_cols)，缺失列补0"""
        col_idx = {c: i for i, c in enumerate(src_cols)}
        X_new = np.zeros((X_arr.shape[0], len(all_cols)), dtype=float)
        for j, col in enumerate(all_cols):
            if col in col_idx:
                X_new[:, j] = X_arr[:, col_idx[col]]
            # 否则保持 0
        return X_new

    X_train_list, y_train_list = [], []
    X_test_list,  y_test_list  = [], []
    for mode in OPERATION_MODES:
        d = per_mode[mode]
        X_train_list.append(align(d['Xtr'], d['cols'], all_feature_cols))
        X_test_list.append( align(d['Xte'], d['cols'], all_feature_cols))
        y_train_list.append(d['ytr'])
        y_test_list.append( d['yte'])

    X_train = np.vstack(X_train_list)
    X_test  = np.vstack(X_test_list)
    y_train = np.concatenate(y_train_list)
    y_test  = np.concatenate(y_test_list)

    return X_train, X_test, y_train, y_test, all_feature_cols, per_mode


# ─────────────────────────────────────────────────────────────────────────────
# 训练（在合并数据集上，参数与原代码完全一致）
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark_merged(task_name, random_state=RANDOM_SEED, verbose=True):
    X_train, X_test, y_train, y_test, feat_cols, per_mode = \
        build_merged_dataset(task_name, random_state)

    if verbose:
        print(f"    合并后数据: 训练 {len(X_train)} / 测试 {len(X_test)}, "
              f"特征维度 {X_train.shape[1]}")
        for mode in OPERATION_MODES:
            d = per_mode[mode]
            print(f"      {OPERATION_LABELS[mode]}: 训练 {len(d['ytr'])} / "
                  f"测试 {len(d['yte'])}，原始特征 {len(d['cols'])} 列")

    sklearn_models = _build_sklearn_models(random_state)
    records = []

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
        except Exception as e:
            tr = te = {'R2': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
            status = f"失败: {e}"

        if verbose:
            print(f"  {status}")

        records.append({
            'Model':       name,
            'Model_CN':    MODEL_FULLNAMES[name],
            'Train_R2':    tr['R2'],   'Train_RMSE': tr['RMSE'], 'Train_MAPE': tr['MAPE'],
            'Test_R2':     te['R2'],   'Test_RMSE':  te['RMSE'], 'Test_MAPE':  te['MAPE'],
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────
def main(random_state=RANDOM_SEED):
    print("=" * 80)
    print("消融实验 2：不分工况建模（三工况数据合并）+ 保留完整特征集")
    print("对比基准  ：原代码（分工况 + 完整特征集）")
    print("唯一变量  ：三工况数据合并为一个数据集后训练单一模型")
    print("特征对齐  ：three_stage/four_stage 缺失的 serial 专属列用 0 填充")
    print(f"随机种子  ：{random_state}")
    print("=" * 80)

    all_sheets    = {}
    overview_rows = []

    for task_name in ['Cu', 'As', 'Voltage']:
        print(f"\n{'='*80}")
        print(f"▶ 预测目标：{task_name}")
        print(f"{'='*80}")

        df_results = run_benchmark_merged(task_name, random_state=random_state, verbose=True)

        all_sheets[f"{task_name}_原始"] = df_results[RAW_COLS]

        topsis_df = topsis(df_results, benefit_cols=BENEFIT_COLS,
                           cost_cols=COST_COLS, weights=WEIGHTS)

        print(f"\n  TOPSIS综合排序（{task_name}，不分工况）")
        print(f"  {'排名':<4} {'模型':<22} {'Test_R2':>8} {'Test_RMSE':>10} "
              f"{'Test_MAPE':>10} {'TOPSIS分数':>12}")
        print("  " + "-" * 72)
        for _, row in topsis_df.iterrows():
            print(f"  {int(row['TOPSIS_Rank']):<4} {row['Model']:<22} "
                  f"{row['Test_R2']:>8.4f} {row['Test_RMSE']:>10.4f} "
                  f"{row['Test_MAPE']:>9.2f}% {row['TOPSIS_Score']:>12.4f}")

        top3 = topsis_df[topsis_df['TOPSIS_Rank'] <= 3]['Model'].tolist()
        print(f"\n  ✓ TOPSIS 前3名: {' / '.join(top3)}")

        all_sheets[f"{task_name}_TOPSIS"] = topsis_df[
            ['TOPSIS_Rank', 'Model', 'Model_CN',
             'Train_R2', 'Train_RMSE', 'Train_MAPE',
             'Test_R2',  'Test_RMSE',  'Test_MAPE', 'TOPSIS_Score']
        ]

        for _, row in df_results.iterrows():
            overview_rows.append({
                'Target':     task_name,
                'Condition':  '不分工况-完整特征',
                'Model':      row['Model'],
                'Model_CN':   row['Model_CN'],
                'Train_R2':   row['Train_R2'],
                'Train_RMSE': row['Train_RMSE'],
                'Train_MAPE': row['Train_MAPE'],
                'Test_R2':    row['Test_R2'],
                'Test_RMSE':  row['Test_RMSE'],
                'Test_MAPE':  row['Test_MAPE'],
            })

    ordered = {'00_总览': pd.DataFrame(overview_rows)}
    ordered.update(all_sheets)

    output_path = os.path.join(OUTPUT_DIR, 'ablation_merge_conditions.xlsx')
    save_to_excel(ordered, output_path)
    print(f"\n{'='*80}")
    print("全部完成！结果保存至:", output_path)


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='设置随机种子值')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='随机种子值 (默认: %d)' % RANDOM_SEED)
    args = parser.parse_args()
    main(random_state=args.seed)
