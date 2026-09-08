"""
废铜液原液罐Cu（g/L）预测模型 —— 整合版 v3
- 数据集: data_by_operation_mode_with_features.xlsx
- 包含三个独立模型：
  1. three_stage模型：仅使用三段数据
     特征：剔除电压，增加 year month day hour three_stage_power
  2. four_stage模型：仅使用四段数据（四段三段建模）
     特征：剔除电压，增加 year month day hour four_stage_power
  3. serial模型：使用串联数据
     特征：剔除电压，增加 year month day hour total_power
- 每个模型保持独立配置和训练流程
- 每次运行后自动将最优超参数和随机种子写回配置表
"""
from pathlib import Path
from model_utils import setup_matplotlib, load_config, save_config, train_model

# 设置matplotlib
setup_matplotlib()

# 配置文件路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_OUTPUT_ROOT = PROJECT_ROOT / 'modeling' / 'outputs'
CONFIG_FILE = str(MODEL_OUTPUT_ROOT / 'cu_config.json')

# ============================================================================
# 全局配置表
# ============================================================================

# three_stage 模型配置
# 特征：包含三段电压V，增加 year month day hour three_stage_power
THREE_STAGE_CONFIG = {
    'USE_OPTIMIZED_RANDOM_STATE': False,   # True=自动优化 / False=使用下方固定值
    'MANUAL_RANDOM_STATE': 177,            # USE_OPTIMIZED_RANDOM_STATE=False 时生效
    'USE_BAYESIAN_OPTIMIZATION': False,     # True=贝叶斯优化超参数
    'OUTPUT_DIR': str(MODEL_OUTPUT_ROOT / 'cu_three_stage'),
    'SHEET_NAME': 'three_stage',
    'MODEL_TAG': 'three_stage',
    'FEATURE_COLS': [
        '电积前液Cu（g/L）',
        '三段溶液温度℃',
        '三段电流强度A',
        '三段流量m3/h',
        '三段电压V',
        'year', 'month', 'day', 'hour',
        'three_stage_power'
    ],
    'PREDICTION_TARGET': '废铜液原液罐Cu（g/L）',
    # ── 最优超参数（每次运行后自动更新，USE_BAYESIAN_OPTIMIZATION=False 时直接使用） ──
    'BEST_RANDOM_STATE': 177,
    'BEST_PARAMS': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 0.8,
    },
}

# four_stage 模型配置（四段三段建模）
# 特征：包含四段电压V，增加 year month day hour four_stage_power
FOUR_STAGE_CONFIG = {
    'USE_OPTIMIZED_RANDOM_STATE': False,
    'MANUAL_RANDOM_STATE': 274,
    'USE_BAYESIAN_OPTIMIZATION': False,
    'OUTPUT_DIR': str(MODEL_OUTPUT_ROOT / 'cu_four_stage'),
    'SHEET_NAME': 'four_stage',
    'MODEL_TAG': 'four_stage',
    'FEATURE_COLS': [
        '电积前液Cu（g/L）',
        '四段溶液温度℃',
        '四段电流强度A',
        '四段流量m3/h',
        '四段电压V',
        'year', 'month', 'day', 'hour',
        'four_stage_power'
    ],
    'PREDICTION_TARGET': '废铜液原液罐Cu（g/L）',
    'BEST_RANDOM_STATE': 330,
    'BEST_PARAMS': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 0.8,
    },
}

# serial 模型配置（串联建模）
# 特征：包含三段电压V和四段电压V，增加 year month day hour total_power
SERIAL_CONFIG = {
    'USE_OPTIMIZED_RANDOM_STATE': False,
    'MANUAL_RANDOM_STATE': 166,
    'USE_BAYESIAN_OPTIMIZATION': False,
    'OUTPUT_DIR': str(MODEL_OUTPUT_ROOT / 'cu_serial'),
    'SHEET_NAME': 'serial',
    'MODEL_TAG': 'serial',
    'FEATURE_COLS': [
        '电积前液Cu（g/L）',
        # 三段特征（包含三段电压V）
        '三段溶液温度℃', '三段电流强度A', '三段流量m3/h', '三段电压V',
        # 四段特征（包含四段电压V）
        '四段溶液温度℃', '四段电流强度A', '四段流量m3/h', '四段电压V',
        # 时间与功率特征
        'year', 'month', 'day', 'hour',
        'total_power'
    ],
    'PREDICTION_TARGET': '废铜液原液罐Cu（g/L）',
    'BEST_RANDOM_STATE': 255,
    'BEST_PARAMS': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 0.8,
    },
}
# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("=" * 120)
    print("废铜液原液罐Cu（g/L）预测模型 —— 整合版 v3")
    print("数据集: data_by_operation_mode_with_features.xlsx")
    print("特征变化: 包含电压特征，增加时间特征(year/month/day/hour)和功率特征")
    print("=" * 120)

    # 加载之前的最优配置
    saved_config = load_config(CONFIG_FILE)
    if saved_config:
        print("\n已加载之前的最优配置:")
        for model_name, config in saved_config.items():
            print(f"  {model_name}:")
            print(f"    最佳随机种子: {config.get('BEST_RANDOM_STATE')}")
            print(f"    最优超参数: {config.get('BEST_PARAMS')}")

        # 更新全局配置表
        if 'three_stage' in saved_config:
            THREE_STAGE_CONFIG['BEST_RANDOM_STATE'] = saved_config['three_stage'].get('BEST_RANDOM_STATE')
            THREE_STAGE_CONFIG['BEST_PARAMS'] = saved_config['three_stage'].get('BEST_PARAMS', THREE_STAGE_CONFIG['BEST_PARAMS'])
        if 'four_stage' in saved_config:
            FOUR_STAGE_CONFIG['BEST_RANDOM_STATE'] = saved_config['four_stage'].get('BEST_RANDOM_STATE')
            FOUR_STAGE_CONFIG['BEST_PARAMS'] = saved_config['four_stage'].get('BEST_PARAMS', FOUR_STAGE_CONFIG['BEST_PARAMS'])
        if 'serial' in saved_config:
            SERIAL_CONFIG['BEST_RANDOM_STATE'] = saved_config['serial'].get('BEST_RANDOM_STATE')
            SERIAL_CONFIG['BEST_PARAMS'] = saved_config['serial'].get('BEST_PARAMS', SERIAL_CONFIG['BEST_PARAMS'])

    # 训练三个模型（每个模型训练后自动将最优配置写回配置表）
    three_stage_train_metrics, three_stage_test_metrics, three_stage_best_rs = train_model(THREE_STAGE_CONFIG, unit='g/L')
    four_stage_train_metrics, four_stage_test_metrics, four_stage_best_rs = train_model(FOUR_STAGE_CONFIG, unit='g/L')
    serial_train_metrics, serial_test_metrics, serial_best_rs = train_model(SERIAL_CONFIG, unit='g/L')

    print("\n" + "=" * 120)
    print("所有模型训练完成！")
    print("=" * 120)

    # ── 各工况单独指标与最优配置 ───────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("各工况单独指标与最优配置")
    print("=" * 120)

    # 定义工况配置映射
    configs = [
        ('三段', THREE_STAGE_CONFIG, three_stage_train_metrics, three_stage_test_metrics, three_stage_best_rs),
        ('四段', FOUR_STAGE_CONFIG, four_stage_train_metrics, four_stage_test_metrics, four_stage_best_rs),
        ('串联', SERIAL_CONFIG, serial_train_metrics, serial_test_metrics, serial_best_rs)
    ]

    for name, config, train_metrics, test_metrics, best_rs in configs:
        print(f"\n{name}工况:")
        print("-" * 80)
        print(f"最佳随机种子: {best_rs}")
        print("最优超参数:")
        for k, v in config['BEST_PARAMS'].items():
            print(f"  {k}: {v}")
        print("\n训练集指标:")
        print(f"  R²:   {train_metrics['R2']:.4f}")
        print(f"  RMSE: {train_metrics['RMSE']:.4f}")
        print(f"  MAPE: {train_metrics['MAPE']:.2f}%")
        print("测试集指标:")
        print(f"  R²:   {test_metrics['R2']:.4f}")
        print(f"  RMSE: {test_metrics['RMSE']:.4f}")
        print(f"  MAPE: {test_metrics['MAPE']:.2f}%")
        print("-" * 80)

    # ── 三工况整体指标汇总 ─────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("三个工况整体指标汇总")
    print("=" * 120)

    tag_map = {
        'three_stage': ('outputs/cu_three_stage', 'prediction_results_three_stage_train.xlsx',
                        'prediction_results_three_stage_test.xlsx'),
        'four_stage':  ('outputs/cu_four_stage',  'prediction_results_four_stage_train.xlsx',
                        'prediction_results_four_stage_test.xlsx'),
        'serial':      ('outputs/cu_serial',       'prediction_results_serial_train.xlsx',
                        'prediction_results_serial_test.xlsx'),
    }

    try:
        import pandas as pd
        import numpy as np
        from sklearn.metrics import r2_score, mean_squared_error
        
        y_train_all, y_pred_train_all = [], []
        y_test_all,  y_pred_test_all  = [], []

        for tag, (out_dir, train_file, test_file) in tag_map.items():
            tr = pd.read_excel(f'{out_dir}/{train_file}')
            te = pd.read_excel(f'{out_dir}/{test_file}')
            col = '废铜液原液罐Cu（g/L）'
            y_train_all.extend(tr[col].values);      y_pred_train_all.extend(tr['预测值'].values)
            y_test_all.extend(te[col].values);       y_pred_test_all.extend(te['预测值'].values)

        y_train_all      = np.array(y_train_all)
        y_pred_train_all = np.array(y_pred_train_all)
        y_test_all       = np.array(y_test_all)
        y_pred_test_all  = np.array(y_pred_test_all)

        def _print_overall(y_true, y_pred, label):
            r2   = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
            print(f"\n{label}集整体指标:")
            print(f"  R²:   {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAPE: {mape:.2f}%")

        print("\n训练集整体指标:\n" + "-" * 80)
        _print_overall(y_train_all, y_pred_train_all, "训练")

        print("\n测试集整体指标:\n" + "-" * 80)
        _print_overall(y_test_all, y_pred_test_all, "测试")

        print("\n" + "=" * 120)

    except Exception as e:
        print(f"读取文件时出错: {e}")
        print("请确保所有模型已成功训练并生成了预测结果文件")
        print("=" * 120)

    # 保存当前的最优配置
    config_data = {
        'three_stage': {
            'BEST_RANDOM_STATE': THREE_STAGE_CONFIG['BEST_RANDOM_STATE'],
            'BEST_PARAMS': THREE_STAGE_CONFIG['BEST_PARAMS']
        },
        'four_stage': {
            'BEST_RANDOM_STATE': FOUR_STAGE_CONFIG['BEST_RANDOM_STATE'],
            'BEST_PARAMS': FOUR_STAGE_CONFIG['BEST_PARAMS']
        },
        'serial': {
            'BEST_RANDOM_STATE': SERIAL_CONFIG['BEST_RANDOM_STATE'],
            'BEST_PARAMS': SERIAL_CONFIG['BEST_PARAMS']
        }
    }
    save_config(CONFIG_FILE, config_data)
    
    # 将指标输出到Excel文件
    try:
        import pandas as pd
        import os
        
        # 创建输出目录
        output_dir = 'outputs/cu_metrics'
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备各工况的指标数据
        工况指标 = []
        for name, config, train_metrics, test_metrics, best_rs in configs:
            工况指标.append({
                '工况': name,
                '最佳随机种子': best_rs,
                '训练集R²': train_metrics['R2'],
                '训练集RMSE': train_metrics['RMSE'],
                '训练集MAPE': train_metrics['MAPE'],
                '测试集R²': test_metrics['R2'],
                '测试集RMSE': test_metrics['RMSE'],
                '测试集MAPE': test_metrics['MAPE']
            })
        
        # 准备整体指标数据
        整体指标 = []
        if 'y_train_all' in locals() and 'y_pred_train_all' in locals():
            r2_train = r2_score(y_train_all, y_pred_train_all)
            rmse_train = np.sqrt(mean_squared_error(y_train_all, y_pred_train_all))
            mape_train = np.mean(np.abs((y_train_all - y_pred_train_all) / (np.abs(y_train_all) + 1e-8))) * 100
            
            r2_test = r2_score(y_test_all, y_pred_test_all)
            rmse_test = np.sqrt(mean_squared_error(y_test_all, y_pred_test_all))
            mape_test = np.mean(np.abs((y_test_all - y_pred_test_all) / (np.abs(y_test_all) + 1e-8))) * 100
            
            整体指标.append({
                '数据集': '训练集',
                'R²': r2_train,
                'RMSE': rmse_train,
                'MAPE': mape_train
            })
            整体指标.append({
                '数据集': '测试集',
                'R²': r2_test,
                'RMSE': rmse_test,
                'MAPE': mape_test
            })
        
        # 创建DataFrame
        df_工况指标 = pd.DataFrame(工况指标)
        df_整体指标 = pd.DataFrame(整体指标)
        
        # 输出到Excel文件
        excel_path = f'{output_dir}/cu_metrics_summary.xlsx'
        with pd.ExcelWriter(excel_path) as writer:
            df_工况指标.to_excel(writer, sheet_name='各工况指标', index=False)
            df_整体指标.to_excel(writer, sheet_name='整体指标', index=False)
        
        print(f"\n指标已成功输出到Excel文件: {excel_path}")
        
    except Exception as e:
        print(f"输出Excel文件时出错: {e}")
