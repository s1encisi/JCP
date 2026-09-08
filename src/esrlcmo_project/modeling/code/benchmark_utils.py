"""
benchmark_utils.py ── 14种算法基准评估共享工具模块
=====================================================
功能：
  - 数据加载与划分（直接复用 data_by_operation_mode_with_features.xlsx）
  - 14种机器学习算法统一定义
  - 训练 / 评估 / TOPSIS 综合排序
  - 结果汇总写入 Excel
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ── 算法导入 ─────────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE   = 'data_by_operation_mode_with_features.xlsx'
OUTPUT_DIR  = 'benchmarkoutputs'
RANDOM_SEED = 42
TEST_SIZE   = 0.1

# ─────────────────────────────────────────────────────────────────────────────
# 特征集定义（与上游脚本保持一致）
# ─────────────────────────────────────────────────────────────────────────────
FEATURES_WITH_VOLTAGE = {
    'three_stage': [
        '电积前液Cu（g/L）', '三段溶液温度℃', '三段电流强度A', '三段流量m3/h', '三段电压V',
        'year', 'month', 'day', 'hour', 'three_stage_power'
    ],
    'four_stage': [
        '电积前液Cu（g/L）', '四段溶液温度℃', '四段电流强度A', '四段流量m3/h', '四段电压V',
        'year', 'month', 'day', 'hour', 'four_stage_power'
    ],
    'serial': [
        '电积前液Cu（g/L）',
        '三段溶液温度℃', '三段电流强度A', '三段流量m3/h', '三段电压V',
        '四段溶液温度℃', '四段电流强度A', '四段流量m3/h', '四段电压V',
        'year', 'month', 'day', 'hour', 'total_power'
    ],
}

FEATURES_WITHOUT_VOLTAGE = {
    'three_stage': [
        '电积前液Cu（g/L）', '三段溶液温度℃', '三段电流强度A', '三段流量m3/h',
        'year', 'month', 'day', 'hour'
    ],
    'four_stage': [
        '电积前液Cu（g/L）', '四段溶液温度℃', '四段电流强度A', '四段流量m3/h',
        'year', 'month', 'day', 'hour'
    ],
    'serial': [
        '电积前液Cu（g/L）',
        '三段溶液温度℃', '三段电流强度A', '三段流量m3/h',
        '四段溶液温度℃', '四段电流强度A', '四段流量m3/h',
        'year', 'month', 'day', 'hour'
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# 目标变量与工况配置
# ─────────────────────────────────────────────────────────────────────────────
# 预测任务：Cu、As 使用含电压特征集；Voltage 使用不含电压特征集
TASK_CONFIGS = {
    'Cu': {
        'sheet_targets': {                       # sheet_name -> 目标列名
            'three_stage': '废铜液原液罐Cu（g/L）',
            'four_stage':  '废铜液原液罐Cu（g/L）',
            'serial':      '废铜液原液罐Cu（g/L）',
        },
        'features': FEATURES_WITH_VOLTAGE,
        'unit': 'g/L',
    },
    'As': {
        'sheet_targets': {
            'three_stage': '废铜液原液罐As（g/L）',
            'four_stage':  '废铜液原液罐As（g/L）',
            'serial':      '废铜液原液罐As（g/L）',
        },
        'features': FEATURES_WITH_VOLTAGE,
        'unit': 'g/L',
    },
    'Voltage': {
        'sheet_targets': {
            'three_stage': '三段电压V',
            'four_stage':  '四段电压V',
            'serial':      '三四段平均电压V',   # 串联时临时计算
        },
        'features': FEATURES_WITHOUT_VOLTAGE,
        'unit': 'V',
    },
}

OPERATION_MODES = ['three_stage', 'four_stage', 'serial']
OPERATION_LABELS = {
    'three_stage': '三段工况',
    'four_stage':  '四段工况',
    'serial':      '串联工况',
}

# ─────────────────────────────────────────────────────────────────────────────
# DNN 模型（PyTorch）
# ─────────────────────────────────────────────────────────────────────────────
class _CNNNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 1D CNN 模型，适用于特征序列数据
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        # 计算 CNN 输出维度
        cnn_output_dim = input_dim // 4 * 32
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x.squeeze(-1)


class _DNNNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_cnn(X_train, y_train, X_test, y_test,
               epochs=200, lr=1e-3, batch_size=64, random_state=RANDOM_SEED):
    """训练 CNN，返回 (pred_train, pred_test) numpy 数组"""
    torch.manual_seed(random_state)
    scaler = StandardScaler()
    Xtr = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    Xte = torch.tensor(scaler.transform(X_test),      dtype=torch.float32)
    ytr = torch.tensor(y_train.astype(float),          dtype=torch.float32)

    model     = _CNNNet(Xtr.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()
    loader    = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_train = model(Xtr).numpy()
        pred_test  = model(Xte).numpy()
    return pred_train, pred_test


def _train_dnn(X_train, y_train, X_test, y_test,
               epochs=200, lr=1e-3, batch_size=64, random_state=RANDOM_SEED):
    """训练 DNN，返回 (pred_train, pred_test) numpy 数组"""
    torch.manual_seed(random_state)
    scaler = StandardScaler()
    Xtr = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    Xte = torch.tensor(scaler.transform(X_test),      dtype=torch.float32)
    ytr = torch.tensor(y_train.astype(float),          dtype=torch.float32)

    model     = _DNNNet(Xtr.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()
    loader    = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_train = model(Xtr).numpy()
        pred_test  = model(Xte).numpy()
    return pred_train, pred_test


# ─────────────────────────────────────────────────────────────────────────────
# 14 种算法统一注册表
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAMES_ORDER = [
    'LinearRegression',
    'Lasso',
    'DecisionTree',
    'RandomForest',
    'GradientBoosting',
    'XGBoost',
    'LightGBM',
    'CatBoost',
    'SVR',
    'KNN',
    'BayesianRidge',
    'CNN',
    'DNN',
    'ExtraTrees',
]

# 中文全称映射（用于报告展示）
MODEL_FULLNAMES = {
    'LinearRegression': '线性回归（Linear Regression）',
    'Lasso':            '套索回归（Lasso Regression）',
    'DecisionTree':     '决策树回归器（Decision Tree Regressor）',
    'RandomForest':     '随机森林回归器（Random Forest Regressor）',
    'GradientBoosting': '梯度提升回归器（GBR）',
    'XGBoost':          '极端梯度提升回归器（XGBoost Regressor）',
    'LightGBM':         '轻量梯度提升回归器（LightGBM Regressor）',
    'CatBoost':         '类别梯度提升回归器（CatBoost Regressor）',
    'SVR':              '支持向量回归器（SVR）',
    'KNN':              'K近邻回归器（KNN Regressor）',
    'BayesianRidge':    '贝叶斯岭回归器（Bayesian Ridge Regressor）',
    'CNN':              '卷积神经网络回归器（CNN Regressor）',
    'DNN':              '深度神经网络回归器（DNN Regressor）',
    'ExtraTrees':       '极端随机树回归器（Extra Trees Regressor）',
}


def _build_sklearn_models(random_state=RANDOM_SEED):
    """构建 13 个 sklearn 兼容模型（不含 DNN）"""
    return {
        'LinearRegression': Pipeline([
            ('scaler', StandardScaler()), ('model', LinearRegression())
        ]),
        'Lasso': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(max_iter=10000, random_state=random_state))
        ]),
        'DecisionTree':     DecisionTreeRegressor(random_state=random_state),
        'RandomForest':     RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(random_state=random_state),
        'XGBoost':          xgb.XGBRegressor(random_state=random_state, verbosity=0, n_jobs=-1),
        'LightGBM':         lgb.LGBMRegressor(random_state=random_state, verbosity=-1, n_jobs=-1),
        'CatBoost':         CatBoostRegressor(random_seed=random_state, verbose=0),
        'SVR': Pipeline([
            ('scaler', StandardScaler()), ('model', SVR())
        ]),
        'KNN':              KNeighborsRegressor(n_jobs=-1),
        'BayesianRidge':    Pipeline([
            ('scaler', StandardScaler()),
            ('model', BayesianRidge())
        ]),
        'ShallowNN':        MLPRegressor(
            hidden_layer_sizes=(32,),  # 更简单的浅层网络
            max_iter=1000,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            alpha=1e-3,
            learning_rate_init=1e-3,
            solver='adam'
        ),
        'ExtraTrees':       ExtraTreesRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────────────────────────────────────
def calc_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return {'R2': r2, 'RMSE': rmse, 'MAPE': mape}


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────
def load_data(sheet_name, feature_cols, target_col, random_state=RANDOM_SEED):
    """
    读取指定 sheet，清理缺失值，返回 X_train, X_test, y_train, y_test（numpy 数组）。
    串联电压目标列 '三四段平均电压V' 自动计算。
    """
    df = pd.read_excel(DATA_FILE, sheet_name=sheet_name)

    if target_col == '三四段平均电压V' and '三四段平均电压V' not in df.columns:
        df['三四段平均电压V'] = (df['三段电压V'] + df['四段电压V']) / 2

    df = df.dropna(subset=feature_cols + [target_col])
    X  = df[feature_cols].values.astype(float)
    y  = df[target_col].values.astype(float)

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=random_state)


# ─────────────────────────────────────────────────────────────────────────────
# 核心：单工况 × 单目标 → 14 模型训练 + 评估
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark(sheet_name, feature_cols, target_col,
                  random_state=RANDOM_SEED, verbose=True):
    """
    训练 14 种模型并收集指标。
    返回 DataFrame，列：
      Model, Train_R2, Train_RMSE, Train_MAPE, Test_R2, Test_RMSE, Test_MAPE
    以及预测值字典 {model_name: {'train': y_pred_train, 'test': y_pred_test, 'true_train': y_train, 'true_test': y_test}}
    """
    X_train, X_test, y_train, y_test = load_data(
        sheet_name, feature_cols, target_col, random_state
    )
    if verbose:
        print(f"    数据: {len(X_train)+len(X_test)} 条  "
              f"(训练 {len(X_train)} / 测试 {len(X_test)})")

    sklearn_models = _build_sklearn_models(random_state)
    records = []
    predictions = {}

    for name in MODEL_NAMES_ORDER:
        pad = f"{name:<20}"
        if name == 'CNN':
            if verbose:
                print(f"    [{pad}] 训练中...", end='', flush=True)
            try:
                ptr, pte = _train_cnn(X_train, y_train, X_test, y_test,
                                      random_state=random_state)
                tr = calc_metrics(y_train, ptr)
                te = calc_metrics(y_test,  pte)
                status = f"Test R²={te['R2']:.4f}  RMSE={te['RMSE']:.4f}  MAPE={te['MAPE']:.2f}%"
                predictions[name] = {'train': ptr, 'test': pte, 'true_train': y_train, 'true_test': y_test}
            except Exception as e:
                tr = te = {'R2': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
                status = f"失败: {e}"
                predictions[name] = {'train': None, 'test': None, 'true_train': y_train, 'true_test': y_test}
        elif name == 'DNN':
            if verbose:
                print(f"    [{pad}] 训练中...", end='', flush=True)
            try:
                ptr, pte = _train_dnn(X_train, y_train, X_test, y_test,
                                      random_state=random_state)
                tr = calc_metrics(y_train, ptr)
                te = calc_metrics(y_test,  pte)
                status = f"Test R²={te['R2']:.4f}  RMSE={te['RMSE']:.4f}  MAPE={te['MAPE']:.2f}%"
                predictions[name] = {'train': ptr, 'test': pte, 'true_train': y_train, 'true_test': y_test}
            except Exception as e:
                tr = te = {'R2': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
                status = f"失败: {e}"
                predictions[name] = {'train': None, 'test': None, 'true_train': y_train, 'true_test': y_test}
        else:
            if verbose:
                print(f"    [{pad}] 训练中...", end='', flush=True)
            try:
                sklearn_models[name].fit(X_train, y_train)
                ptr = sklearn_models[name].predict(X_train)
                pte = sklearn_models[name].predict(X_test)
                tr = calc_metrics(y_train, ptr)
                te = calc_metrics(y_test,  pte)
                status = f"Test R²={te['R2']:.4f}  RMSE={te['RMSE']:.4f}  MAPE={te['MAPE']:.2f}%"
                predictions[name] = {'train': ptr, 'test': pte, 'true_train': y_train, 'true_test': y_test}
            except Exception as e:
                tr = te = {'R2': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
                status = f"失败: {e}"
                predictions[name] = {'train': None, 'test': None, 'true_train': y_train, 'true_test': y_test}

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
# TOPSIS 综合排序
# ─────────────────────────────────────────────────────────────────────────────
def topsis(df_metrics,
           benefit_cols=('Test_R2',),
           cost_cols=('Test_RMSE', 'Test_MAPE'),
           weights=None):
    """
    对 df_metrics 中的模型按 TOPSIS 进行综合排序。
    benefit_cols: 越大越好（R²）
    cost_cols:    越小越好（RMSE, MAPE）
    weights:      权重列表，顺序与 benefit_cols + cost_cols 一致，默认等权重
    返回带 TOPSIS_Score 和 TOPSIS_Rank 列的 DataFrame（已排序）
    """
    all_cols = list(benefit_cols) + list(cost_cols)
    df = df_metrics.copy().dropna(subset=all_cols).reset_index(drop=True)
    X  = df[all_cols].values.astype(float)

    # 向量归一化
    norms = np.sqrt((X ** 2).sum(axis=0))
    norms[norms == 0] = 1e-10
    X_norm = X / norms

    # 加权
    n = len(all_cols)
    w = np.ones(n) / n if weights is None else np.array(weights, float) / np.sum(weights)
    X_w = X_norm * w

    # 正负理想解
    ideal_pos = np.array([
        X_w[:, i].max() if c in benefit_cols else X_w[:, i].min()
        for i, c in enumerate(all_cols)
    ])
    ideal_neg = np.array([
        X_w[:, i].min() if c in benefit_cols else X_w[:, i].max()
        for i, c in enumerate(all_cols)
    ])

    d_pos = np.sqrt(((X_w - ideal_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((X_w - ideal_neg) ** 2).sum(axis=1))
    score = d_neg / (d_pos + d_neg + 1e-10)

    df['TOPSIS_Score'] = score
    df['TOPSIS_Rank']  = df['TOPSIS_Score'].rank(ascending=False).astype(int)
    return df.sort_values('TOPSIS_Rank').reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Excel 输出
# ─────────────────────────────────────────────────────────────────────────────
def save_to_excel(all_results: dict, output_path: str):
    """
    all_results: { sheet_name: DataFrame }
    将所有结果写入同一个 Excel 文件，每个 sheet_name 对应一个工作表。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in all_results.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    print(f"\n✓ 所有结果已保存至: {output_path}")
