#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_benchmark_with_seeds.py

使用三个不同的随机种子值对铜、砷和电压进行预测，并将结果保存到Excel文件中。
"""

import os
import pandas as pd
import numpy as np
from benchmark_utils import (
    TASK_CONFIGS, OPERATION_MODES, OPERATION_LABELS, MODEL_FULLNAMES, MODEL_NAMES_ORDER,
    run_benchmark, topsis, save_to_excel, calc_metrics
)

# 定义三个不同的随机种子值
RANDOM_SEEDS = [315, 460, 170]

# 输出目录
OUTPUT_DIR = 'benchmarkoutputs/seeds'

# TOPSIS 指标配置（与 benchmark_run.py 保持一致）
BENEFIT_COLS = ('Test_R2',)
COST_COLS    = ('Test_RMSE', 'Test_MAPE')
WEIGHTS      = [1, 1, 1]  # 等权重

# 输出列顺序定义
RAW_COLS = [
    'Model', 'Model_CN',
    'Train_R2', 'Train_RMSE', 'Train_MAPE',
    'Test_R2',  'Test_RMSE',  'Test_MAPE',
]
TOPSIS_COLS = RAW_COLS + ['TOPSIS_Score', 'TOPSIS_Rank']


def main():
    print("开始使用三个不同的随机种子值进行预测...\n")
    
    # 为每个随机种子运行对应的预测目标
    seed_target_mapping = {
        177: 'Cu',
        500: 'As',
        170: 'Voltage'
    }
    
    for seed, task_name in seed_target_mapping.items():
        print(f"\n{'='*80}")
        print(f"随机种子: {seed} | 预测目标: {task_name}")
        print(f"{'='*80}")
        
        # 存储所有结果
        all_sheets = {}   # { sheet_name: DataFrame }
        
        # 获取当前任务配置
        task_config = TASK_CONFIGS[task_name]
        print(f"\n{'='*80}")
        print(f"▶ 预测目标：{task_name}")
        print(f"{'='*80}")
        
        # 用于后续 TOPSIS 汇总的全工况结果列表
        all_modes_results = []
        
        # 收集预测值用于后续合并计算
        all_predictions = {}
        
        # 对每种运行模式
        for mode in OPERATION_MODES:
            label = OPERATION_LABELS[mode]
            print(f"\n  ── {label} ({mode}) ─────────────────────────────────")
            
            # 获取特征列和目标列
            feature_cols = task_config['features'][mode]
            target_col = task_config['sheet_targets'][mode]
            
            # 运行基准测试
            df_results, mode_predictions = run_benchmark(
                sheet_name=mode,
                feature_cols=feature_cols,
                target_col=target_col,
                random_state=seed,
                verbose=True
            )
            
            # 添加工况标识列，方便汇总 sheet 区分
            df_results.insert(0, 'OperationMode', label)
            
            # 保存原始指标 sheet（每工况独立）
            sheet_key = f"{task_name}_{label}_原始"
            all_sheets[sheet_key] = df_results[['OperationMode'] + RAW_COLS]
            
            # 收集至全工况列表（去掉 OperationMode 列以便合并 TOPSIS）
            all_modes_results.append(df_results)
            
            # 收集预测值用于后续合并计算
            all_predictions[mode] = mode_predictions
        
        # ── TOPSIS 分析：汇总三个工况后对14模型综合排序 ──────────────────────
        # 策略：将三工况的训练集和测试集分别拼接后计算指标
        print(f"\n  ── TOPSIS 综合排序（{task_name}，三工况拼接）──────────────")
        
        # 为每个模型计算拼接后的指标
        combined_records = []
        for model_name in MODEL_NAMES_ORDER:
            # 收集所有工况的训练数据
            all_y_train = []
            all_y_pred_train = []
            all_y_test = []
            all_y_pred_test = []
            
            for mode in OPERATION_MODES:
                if mode in all_predictions and model_name in all_predictions[mode]:
                    pred_data = all_predictions[mode][model_name]
                    if pred_data['train'] is not None:
                        all_y_train.extend(pred_data['true_train'])
                        all_y_pred_train.extend(pred_data['train'])
                    if pred_data['test'] is not None:
                        all_y_test.extend(pred_data['true_test'])
                        all_y_pred_test.extend(pred_data['test'])
            
            # 计算拼接后的指标
            train_metrics = calc_metrics(np.array(all_y_train), np.array(all_y_pred_train))
            test_metrics = calc_metrics(np.array(all_y_test), np.array(all_y_pred_test))
            
            combined_records.append({
                'Model': model_name,
                'Model_CN': MODEL_FULLNAMES[model_name],
                'Train_R2': train_metrics['R2'],
                'Train_RMSE': train_metrics['RMSE'],
                'Train_MAPE': train_metrics['MAPE'],
                'Test_R2': test_metrics['R2'],
                'Test_RMSE': test_metrics['RMSE'],
                'Test_MAPE': test_metrics['MAPE'],
            })
        
        combined = pd.DataFrame(combined_records)
        
        topsis_df = topsis(
            combined,
            benefit_cols=BENEFIT_COLS,
            cost_cols=COST_COLS,
            weights=WEIGHTS,
        )
        
        # 打印排名
        print(f"  {'排名':<4} {'模型':<22} {'Test_R2':>8} {'Test_RMSE':>10} "
              f"{'Test_MAPE':>10} {'TOPSIS分数':>12}")
        print("  " + "-" * 72)
        for _, row in topsis_df.iterrows():
            print(f"  {int(row['TOPSIS_Rank']):<4} {row['Model']:<22} "
                  f"{row['Test_R2']:>8.4f} {row['Test_RMSE']:>10.4f} "
                  f"{row['Test_MAPE']:>9.2f}% {row['TOPSIS_Score']:>12.4f}")
        
        top3 = topsis_df[topsis_df['TOPSIS_Rank'] <= 3]['Model'].tolist()
        print(f"\n  ✓ TOPSIS 前3名（进入下一阶段）: {' / '.join(top3)}")
        
        topsis_sheet_key = f"{task_name}_TOPSIS汇总"
        all_sheets[topsis_sheet_key] = topsis_df[['TOPSIS_Rank', 'Model', 'Model_CN',
                                                    'Test_R2', 'Test_RMSE', 'Test_MAPE',
                                                    'TOPSIS_Score']]
        
        # ── 各工况独立 TOPSIS（补充细节）────────────────────────────────────
        for mode, df_mode in zip(OPERATION_MODES, all_modes_results):
            label = OPERATION_LABELS[mode]
            df_topsis_mode = topsis(
                df_mode,
                benefit_cols=BENEFIT_COLS,
                cost_cols=COST_COLS,
                weights=WEIGHTS,
            )
            sheet_key = f"{task_name}_{label}_TOPSIS"
            all_sheets[sheet_key] = df_topsis_mode[
                ['TOPSIS_Rank', 'Model', 'Model_CN',
                 'Train_R2', 'Train_RMSE', 'Train_MAPE',
                 'Test_R2',  'Test_RMSE',  'Test_MAPE',
                 'TOPSIS_Score']
            ]
        
        # ─────────────────────────────────────────────────────────────────────────
        # 汇总所有工况的完整对比（一个总览 sheet）
        # ─────────────────────────────────────────────────────────────────────────
        print(f"\n{'='*80}")
        print("生成总览 sheet ……")
        overview_rows = []
        for mode in OPERATION_MODES:
            label  = OPERATION_LABELS[mode]
            sheet_key_raw = f"{task_name}_{label}_原始"
            df_raw = all_sheets.get(sheet_key_raw)
            if df_raw is None:
                continue
            for _, row in df_raw.iterrows():
                overview_rows.append({
                    'Target':    task_name,
                    'Mode':      label,
                    'Model':     row['Model'],
                    'Model_CN':  row['Model_CN'],
                    'Train_R2':  row['Train_R2'],
                    'Train_RMSE':row['Train_RMSE'],
                    'Train_MAPE':row['Train_MAPE'],
                    'Test_R2':   row['Test_R2'],
                    'Test_RMSE': row['Test_RMSE'],
                    'Test_MAPE': row['Test_MAPE'],
                })
        all_sheets['00_总览'] = pd.DataFrame(overview_rows)
        
        # ─────────────────────────────────────────────────────────────────────────
        # 写入 Excel
        # ─────────────────────────────────────────────────────────────────────────
        output_path = os.path.join(OUTPUT_DIR, f'benchmark_results_{task_name}_seed{seed}.xlsx')
        
        # 调整 sheet 顺序：总览放第一
        ordered_sheets = {}
        if '00_总览' in all_sheets:
            ordered_sheets['00_总览'] = all_sheets.pop('00_总览')
        ordered_sheets.update(all_sheets)
        
        save_to_excel(ordered_sheets, output_path)
    
    print(f"\n{'='*80}")
    print("所有随机种子的预测已完成！")
    print(f"结果已保存到: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
