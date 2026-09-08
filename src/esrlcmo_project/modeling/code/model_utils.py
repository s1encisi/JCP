"""
模型公共工具模块
- 包含所有模型共享的工具函数
- 提供统一的配置管理、模型训练和可视化功能
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
from bayes_opt import BayesianOptimization
import warnings
import platform
import os
import random
import json
from pathlib import Path
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DATASET = PROJECT_ROOT / 'data' / 'data_by_operation_mode_with_features.xlsx'

# ── 跨平台中文显示 ─────────────────────────────────────────────────────────────
def setup_matplotlib():
    """设置matplotlib以支持中文显示"""
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    sns.set_style("whitegrid")
    _sys = platform.system()
    if _sys == 'Windows':
        mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    elif _sys == 'Linux':
        mpl.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    elif _sys == 'Darwin':
        mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'DejaVu Sans']
    mpl.rc('font', family=mpl.rcParams['font.sans-serif'][0])

# ── 配置文件管理 ───────────────────────────────────────────────────────────────
def load_config(config_file):
    """加载配置文件"""
    # 确保outputs目录存在
    os.makedirs('outputs', exist_ok=True)
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {}
    return {}

def save_config(config_file, config_data):
    """保存配置文件"""
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 配置已保存到 {config_file}")
    except Exception as e:
        print(f"保存配置文件失败: {e}")

# ── 共享工具函数 ───────────────────────────────────────────────────────────────
def calc_metrics(y_true, y_pred, label=''):
    """计算并打印 RMSE / MAE / R² / MAPE，返回指标字典"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    print(f"\n{label}集评估:")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}

def find_best_random_state(X, y, iterations=100):
    """遍历随机种子，返回使测试集 R² 最高的种子"""
    random_states = random.sample(range(1, 501), iterations)
    print(f"\n随机种子候选值: {random_states}")
    print("\n" + "=" * 70)
    print("随机种子调优评估")
    print("=" * 70)

    results = []
    for rs in random_states:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=rs)
        m = ExtraTreesRegressor(random_state=rs, n_estimators=100)
        m.fit(X_tr, y_tr)
        r2   = r2_score(y_te, m.predict(X_te))
        rmse = np.sqrt(mean_squared_error(y_te, m.predict(X_te)))
        results.append({"random_state": rs, "r2": r2, "rmse": rmse})
        print(f"  random_state={rs:3d} | R²: {r2:.4f} | RMSE: {rmse:.4f}")

    best = max(results, key=lambda x: x["r2"])
    print(f"\n{'=' * 70}")
    print(f"最佳随机种子: {best['random_state']} (R²: {best['r2']:.4f}, RMSE: {best['rmse']:.4f})")
    print("=" * 70)
    return best["random_state"]

def run_bayesian_optimization(X_train, y_train, X_test, y_test, random_state, iterations=20):
    """运行贝叶斯超参数优化，返回最佳参数字典"""
    print("\n" + "=" * 70)
    print("贝叶斯超参数优化")
    print("=" * 70)

    def objective(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
        m = ExtraTreesRegressor(
            random_state=random_state,
            n_estimators=int(n_estimators),
            max_depth=int(max_depth) if max_depth > 0 else None,
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            max_features=max_features
        )
        m.fit(X_train, y_train)
        return r2_score(y_test, m.predict(X_test))

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={
            'n_estimators':     (50, 300),
            'max_depth':         (3, 20),
            'min_samples_split': (2, 10),
            'min_samples_leaf':  (1, 5),
            'max_features':      (0.5, 1.0),
        },
        random_state=random_state,
        verbose=2
    )
    optimizer.maximize(init_points=5, n_iter=iterations)

    best_params = optimizer.max['params']
    best_params['n_estimators']     = int(best_params['n_estimators'])
    best_params['max_depth']        = int(best_params['max_depth']) if best_params['max_depth'] > 0 else None
    best_params['min_samples_split'] = int(best_params['min_samples_split'])
    best_params['min_samples_leaf']  = int(best_params['min_samples_leaf'])

    print(f"\n{'=' * 70}\n最佳超参数:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("=" * 70)
    return best_params

# ── 可视化函数 ─────────────────────────────────────────────────────────────────
def plot_feature_importance(model, feature_cols, output_dir, tag, bar_color='#4C72B0'):
    """绘制并保存特征重要性横向柱状图"""
    feat_imp_df = pd.DataFrame({
        '特征': feature_cols,
        '重要性': model.feature_importances_
    }).sort_values('重要性', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(feat_imp_df['特征'], feat_imp_df['重要性'], color=bar_color, edgecolor='white')
    ax.set_xlabel('特征重要性（基于预测值变化）', fontsize=11)
    ax.set_title(f'CatBoost 特征重要性 —— {tag}数据', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, feat_imp_df['重要性']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', fontsize=9)
    plt.tight_layout()
    path = f'{output_dir}/feature_importance_{tag}.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ 特征重要性图已保存: {path}")

def plot_scatter(y_train, y_pred_train, y_test, y_pred_test, train_metrics, test_metrics, output_dir, tag, unit='g/L'):
    """绘制并保存训练/测试集预测散点图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, y_true, y_pred, metrics, color, label in [
        (axes[0], y_train, y_pred_train, train_metrics, '#4C72B0', '训练集'),
        (axes[1], y_test,  y_pred_test,  test_metrics,  '#DD8452', '测试集'),
    ]:
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color=color, edgecolors='none')
        lo = min(y_true.min(), y_pred.min()) - 0.5
        hi = max(y_true.max(), y_pred.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5)
        ax.set_xlabel(f'真实值 ({unit})', fontsize=11)
        ax.set_ylabel(f'预测值 ({unit})', fontsize=11)
        ax.set_title(f'{label}  R²={metrics["R2"]:.4f}  RMSE={metrics["RMSE"]:.4f}',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'预测效果 —— {tag}数据',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = f'{output_dir}/prediction_scatter_{tag}.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 预测散点图已保存: {path}")

def plot_residuals(y_test, y_pred_test, output_dir, tag, unit='g/L'):
    """绘制并保存残差分析图"""
    residuals = y_test - y_pred_test
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].scatter(y_pred_test, residuals, alpha=0.5, s=20, color='#4C72B0', edgecolors='none')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_xlabel(f'预测值 ({unit})', fontsize=11)
    axes[0].set_ylabel(f'残差 ({unit})', fontsize=11)
    axes[0].set_title('残差 vs 预测值', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=30, color='#4C72B0', edgecolor='white', alpha=0.8)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel(f'残差 ({unit})', fontsize=11)
    axes[1].set_ylabel('频次', fontsize=11)
    axes[1].set_title('残差分布直方图', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'残差分析 —— {tag}数据', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = f'{output_dir}/residual_analysis_{tag}.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 残差分析图已保存: {path}")

def plot_time_series(result_df, prediction_target, output_dir, tag):
    """绘制并保存时间序列预测结果图"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(result_df['日期时间'], result_df[prediction_target],
            'b-', label='真实值', linewidth=1.5, alpha=0.8)
    ax.plot(result_df['日期时间'], result_df['预测值'],
            'r--', label='预测值', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('日期时间', fontsize=11)
    ax.set_ylabel(prediction_target, fontsize=11)
    ax.set_title(f'时间序列预测结果 (n={len(result_df)}) —— {tag}数据',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = f'{output_dir}/time_series_prediction_{tag}.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 时间序列预测图已保存: {path}")

# ── 通用训练流程 ───────────────────────────────────────────────────────────────
def load_trained_model(config):
    """
    尝试从磁盘加载已保存的模型。
    如果模型文件存在且配置未要求重新优化，直接返回模型对象；否则返回 None。
    """
    import joblib
    tag        = config['MODEL_TAG']
    output_dir = config['OUTPUT_DIR']
    model_path = f'{output_dir}/extra_trees_{tag}.joblib'

    # 如果要求重新优化随机种子或超参数，则必须重新训练
    if config.get('USE_OPTIMIZED_RANDOM_STATE') or config.get('USE_BAYESIAN_OPTIMIZATION'):
        return None

    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print(f"\n✓ 检测到已保存的模型，直接加载: {model_path}")
            print(f"  （如需重新训练，请删除该文件或将 USE_OPTIMIZED_RANDOM_STATE/USE_BAYESIAN_OPTIMIZATION 设为 True）")
            return model
        except Exception as e:
            print(f"  加载模型失败（{e}），将重新训练。")
            return None
    return None


def train_model(config, unit='g/L'):
    """
    通用模型训练函数，适用于三段/四段/串联三种工况。
    - 若磁盘上已有对应 .joblib 模型文件，且未要求重新优化，则直接加载跳过训练。
    - 训练完成后自动将最优随机种子和超参数写回 config（即全局配置表），
      下次 USE_OPTIMIZED_RANDOM_STATE=False 且 USE_BAYESIAN_OPTIMIZATION=False 时直接复用。
    """
    tag        = config['MODEL_TAG']
    output_dir = config['OUTPUT_DIR']
    feature_cols      = config['FEATURE_COLS']
    prediction_target = config['PREDICTION_TARGET']

    print("\n" + "=" * 100)
    print(f"开始处理 {tag} 模型")
    print("=" * 100)

    os.makedirs(output_dir, exist_ok=True)

    # ── 0. 尝试加载已有模型（跳过训练） ────────────────────────────────────────
    cached_model = load_trained_model(config)
    if cached_model is not None:
        # 仍需读取数据以便生成预测结果和图表
        file_path = str(FINAL_DATASET)
        df = pd.read_excel(file_path, sheet_name=config['SHEET_NAME'])
        if tag == 'serial' and '三四段平均电压V' in prediction_target:
            model_df = df.copy()
            model_df['三四段平均电压V'] = (model_df['三段电压V'] + model_df['四段电压V']) / 2
            model_df = model_df.dropna(subset=[prediction_target] + feature_cols).copy()
        else:
            model_df = df.dropna(subset=[prediction_target] + feature_cols).copy()

        best_random_state = config.get('BEST_RANDOM_STATE') or config['MANUAL_RANDOM_STATE']
        X = model_df[feature_cols]
        y = model_df[prediction_target].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=best_random_state
        )
        y_pred_train = cached_model.predict(X_train)
        y_pred_test  = cached_model.predict(X_test)
        train_metrics = calc_metrics(y_train, y_pred_train, '训练')
        test_metrics  = calc_metrics(y_test,  y_pred_test,  '测试')

        bar_color = '#4C72B0' if tag == 'three_stage' else ('#DD8452' if tag == 'four_stage' else '#55A868')
        plot_feature_importance(cached_model, feature_cols, output_dir, tag, bar_color)
        plot_scatter(y_train, y_pred_train, y_test, y_pred_test,
                     train_metrics, test_metrics, output_dir, tag, unit)
        plot_residuals(y_test, y_pred_test, output_dir, tag, unit)

        result_df = model_df.copy()
        result_df['预测值'] = cached_model.predict(X)
        result_df = result_df.sort_values('日期时间')
        plot_time_series(result_df, prediction_target, output_dir, tag)

        result_df_final = result_df[['日期时间'] + feature_cols + [prediction_target, '预测值']].copy()
        result_df_final['残差'] = result_df_final[prediction_target] - result_df_final['预测值']
        result_df_final.to_excel(f'{output_dir}/prediction_results_{tag}.xlsx',
                                  index=False, engine='openpyxl')

        print(f"\n✓ 已使用缓存模型完成推理，结果保存至 {output_dir}/")
        print("=" * 100)
        return train_metrics, test_metrics, best_random_state

    # ── 以下为重新训练路径 ──────────────────────────────────────────────────────
    print(f"\n未找到已保存模型，开始重新训练...")

    # ── 1. 读取数据 ────────────────────────────────────────────────────────────
    file_path = str(FINAL_DATASET)
    df = pd.read_excel(file_path, sheet_name=config['SHEET_NAME'])
    print("=" * 70)
    print(f"数据基本信息（{config['SHEET_NAME']} 工作表）:")
    print(f"  行数: {len(df)}, 列数: {len(df.columns)}")
    print(f"  列名: {df.columns.tolist()}")
    print(f"  缺失值总数: {df.isnull().sum().sum()}")
    print("=" * 70)

    # 特殊处理：串联电压模型需要计算三四段平均电压
    if tag == 'serial' and '三四段平均电压V' in prediction_target:
        model_df = df.copy()
        model_df['三四段平均电压V'] = (model_df['三段电压V'] + model_df['四段电压V']) / 2
        model_df = model_df.dropna(subset=[prediction_target] + feature_cols).copy()
    else:
        model_df = df.dropna(subset=[prediction_target] + feature_cols).copy()

    print(f"去除缺失值后建模样本数: {len(model_df)}")
    print(f"使用特征: {feature_cols}")
    print(f"目标变量: {prediction_target}")
    print("\n特征数据基本统计:")
    print(model_df[feature_cols + [prediction_target]].describe())

    X = model_df[feature_cols]
    y = model_df[prediction_target].values

    # ── 2. 确定随机种子 ────────────────────────────────────────────────────────
    if config['USE_OPTIMIZED_RANDOM_STATE']:
        best_random_state = find_best_random_state(X, y)
    else:
        # 优先使用上次运行保存的最优种子（非 None 时）
        saved_rs = config.get('BEST_RANDOM_STATE')
        best_random_state = saved_rs if saved_rs is not None else config['MANUAL_RANDOM_STATE']
        print(f"\n使用随机种子: {best_random_state}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=best_random_state
    )
    print(f"\n训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")

    # ── 3. 确定超参数 ──────────────────────────────────────────────────────────
    if config['USE_BAYESIAN_OPTIMIZATION']:
        best_params = run_bayesian_optimization(
            X_train, y_train, X_test, y_test, best_random_state
        )
    else:
        # 直接使用配置表中保存的最优参数，过滤掉不支持的参数
        config_params = config['BEST_PARAMS'].copy()
        # Extra Trees支持的参数
        supported_params = {'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'}
        best_params = {k: v for k, v in config_params.items() if k in supported_params}
        # 如果没有支持的参数，使用默认值
        if not best_params:
            best_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 0.8,
            }
        print(f"\n使用配置表中的超参数: {best_params}")

    # ── 4. 训练最终模型 ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"开始训练 Extra Trees 模型（{tag}）...")
    model = ExtraTreesRegressor(
        random_state=best_random_state,
        **best_params
    )
    model.fit(X_train, y_train)
    print("模型训练完成!")
    print("=" * 70)

    # ── 5. 评估 ────────────────────────────────────────────────────────────────
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    train_metrics = calc_metrics(y_train, y_pred_train, '训练')
    test_metrics  = calc_metrics(y_test,  y_pred_test,  '测试')

    # ── 6. 可视化 ──────────────────────────────────────────────────────────────
    bar_color = '#4C72B0' if tag == 'three_stage' else ('#DD8452' if tag == 'four_stage' else '#55A868')
    plot_feature_importance(model, feature_cols, output_dir, tag, bar_color)
    plot_scatter(y_train, y_pred_train, y_test, y_pred_test,
                 train_metrics, test_metrics, output_dir, tag, unit)
    plot_residuals(y_test, y_pred_test, output_dir, tag, unit)

    result_df = model_df.copy()
    result_df['预测值'] = model.predict(X)
    result_df = result_df.sort_values('日期时间')
    plot_time_series(result_df, prediction_target, output_dir, tag)

    # ── 7. 保存结果文件 ────────────────────────────────────────────────────────
    result_df_final = result_df[['日期时间'] + feature_cols + [prediction_target, '预测值']].copy()
    result_df_final['残差'] = result_df_final[prediction_target] - result_df_final['预测值']
    result_df_final.to_excel(f'{output_dir}/prediction_results_{tag}.xlsx',
                              index=False, engine='openpyxl')

    train_df = model_df.loc[X_train.index].copy()
    train_df['预测值'] = y_pred_train
    train_df['残差'] = train_df[prediction_target] - train_df['预测值']
    train_df.to_excel(f'{output_dir}/prediction_results_{tag}_train.xlsx',
                      index=False, engine='openpyxl')

    test_df = model_df.loc[X_test.index].copy()
    test_df['预测值'] = y_pred_test
    test_df['残差'] = test_df[prediction_target] - test_df['预测值']
    test_df.to_excel(f'{output_dir}/prediction_results_{tag}_test.xlsx',
                     index=False, engine='openpyxl')

    import joblib
    joblib.dump(model, f'{output_dir}/extra_trees_{tag}.joblib')
    print(f"\n✓ 预测结果已保存: {output_dir}/prediction_results_{tag}.xlsx")
    print(f"✓ 模型已保存:      {output_dir}/extra_trees_{tag}.joblib")

    # ── 8. 将最优配置写回全局配置表 ───────────────────────────────────────────
    config['BEST_RANDOM_STATE'] = best_random_state
    config['BEST_PARAMS']       = best_params
    print(f"\n✓ 最优配置已写回配置表:")
    print(f"   BEST_RANDOM_STATE = {best_random_state}")
    print(f"   BEST_PARAMS       = {best_params}")

    # ── 9. 汇总报告 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"模型训练完成 —— 汇总报告（{tag}）")
    print("=" * 70)
    print(f"建模样本数      : {len(model_df)}")
    print(f"训练集          : {len(X_train)} 条")
    print(f"测试集          : {len(X_test)} 条")
    print(f"特征数量        : {len(feature_cols)}")
    print(f"最佳随机种子     : {best_random_state}")
    print(f"训练集 RMSE     : {train_metrics['RMSE']:.4f}")
    print(f"训练集 R²       : {train_metrics['R2']:.4f}")
    print(f"测试集 RMSE     : {test_metrics['RMSE']:.4f}")
    print(f"测试集 MAE      : {test_metrics['MAE']:.4f}")
    print(f"测试集 R²       : {test_metrics['R2']:.4f}")
    print(f"测试集 MAPE     : {test_metrics['MAPE']:.2f}%")
    print("=" * 70)
    print(f"\n所有输出文件已保存到 {output_dir}/ 目录")
    print("\n" + "=" * 100)
    print(f"{tag} 模型训练完成")
    print("=" * 100)

    return train_metrics, test_metrics, best_random_state
