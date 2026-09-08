# -*- coding: utf-8 -*-
"""
============================================================================
NSGA-II 多目标优化程序
============================================================================
本程序实现基于代理模型辅助的 NSGA-II 多目标优化算法，用于解决铜电积过程的多目标优化问题。

优化目标：
    1. 最小化 Cu_out（铜浓度）
    2. 最小化 As_out（砷浓度）
    3. 最小化 E_total（总能耗）
    4. 最大化 Net_profit（净利润）

约束条件：
    - Cu_out <= Cu_limit（铜浓度上限）
    - As_out >= As_min（砷浓度下限）
    - J <= J_max（电流密度上限）
    - V_cell <= V_cell_max（单槽电压上限）
    - Cu_As <= Cu_As_max（铜砷比上限）

支持三种工况优化：
    1. 三段工况（Three-Stage Operation）
    2. 四段工况（Four-Stage Operation）
    3. 串联工况（Series-Connected Operation）

使用方法：
    # 快速模式（种群30，迭代50）
    python nsga2_optimization.py --fast

    # 正常模式（种群300，迭代300）
    python nsga2_optimization.py --normal

    # 保存所有可行解（而非仅Pareto前沿）
    python nsga2_optimization.py --normal --all-feasible

============================================================================
"""

# ============================================================================
# 导入必要的库
# ============================================================================
import os
import sys
import warnings
import time
from datetime import datetime

import numpy as np
import joblib
import matplotlib

# ============================================================================
# 自动选择 matplotlib 后端
# ============================================================================
if sys.platform == 'win32':
    # Windows 环境使用交互式后端
    matplotlib.use('TkAgg')
else:
    # Linux/macOS 环境使用非交互式后端
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import pandas as pd

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# 忽略警告信息
warnings.filterwarnings('ignore')

# ============================================================================
# 全局超参数配置
# ============================================================================
PARAMS = {
    'Cu_limit': 8.0,       # 铜浓度上限 (g/L)
    'As_min': 4.0,         # 砷浓度下限 (g/L)
    'J_max': 340.0,        # 最大电流密度 (A/m²)
    'V_cell_max': 2.5,     # 单槽最大电压 (V)
    'Cu_As_max': 0.80,     # 铜砷比上限
    'n_cathode': 35,       # 阴极板数量
    'A_cathode': 1.100 * 1.029 * 2,  # 阴极板面积 (m²)
    'N_cell': 16,           # 电积槽数量
}

# NSGA-II 算法参数
NSGA2_PARAMS = {
    'pop_size': 600,        # 种群大小
    'n_gen': 300,           # 迭代代数
    'seed': 42,             # 随机种子
    'n_fronts': 3,          # 保留的非支配前沿数量
    'n_solutions': 50,     # 最终保留的解数量
    'save_all_feasible': False,  # 是否保存所有可行解
}

# 全局多目标评估基准点（用于 Hypervolume 计算）
# 目标顺序: [Cu_out, -As_out, E_total, -R_profit] (全部为最小化方向)
# 参考点必须在所有算法可能探索到的最差物理边界之外
GLOBAL_HV_REF_POINT = np.array([
    12.0,      # Cu_out: 物理上限约8，给足余量
    -0.0,      # -As_out: 物理上As不可能为负，取 -0.0 确保涵盖所有负值
    1.0e6,     # E_total: 设定一个极大的耗电量
    1.0e6      # -R_profit: 设定一个极大的亏损值
])

# 获取当前文件所在目录的绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'newnsga2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局代理模型实例
surrogate = None

# ============================================================================
# SCI 期刊绘图风格配置
# ============================================================================
def _setup_sci_style():
    """配置符合 SCI 期刊要求的 matplotlib 样式"""
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False,
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 12,
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 4.0,
        'ytick.major.size': 4.0,
        'xtick.top': True,
        'axes.grid': False,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.7',
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'lines.linewidth': 1.5,
        'patch.linewidth': 0.8,
    })

_setup_sci_style()

# 绘图配色方案
_CMAP_PROFIT = 'RdYlBu_r'

# 中英文工况名称映射
_MODE_EN = {
    '三段工况': 'Three-Stage Operation',
    '四段工况': 'Four-Stage Operation',
    '串联工况': 'Series-Connected Operation',
}

def _en(label):
    """获取工况的英文名称"""
    return _MODE_EN.get(label, label)

# 目标函数标签
_OBJ_LABELS = [
    r'$\mathrm{Cu_{out}}$ (g/L)',
    r'$\mathrm{As_{out}}$ (g/L)',
    r'$E_\mathrm{total}$ (kWh)',
    r'Net Profit (CNY)',
]

# ============================================================================
# 代理模型相关配置
# ============================================================================

# 模型文件路径配置（使用绝对路径）
MODEL_PATHS = {
    'three_stage': {
        'cu':      os.path.join(CURRENT_DIR, 'joblib', 'cu_three_stage', 'extra_trees_three_stage.joblib'),
        'as':      os.path.join(CURRENT_DIR, 'joblib', 'as_three_stage', 'extra_trees_three_stage.joblib'),
        'voltage': os.path.join(CURRENT_DIR, 'joblib', 'voltage_three_stage', 'extra_trees_three_stage.joblib'),
    },
    'four_stage': {
        'cu':      os.path.join(CURRENT_DIR, 'joblib', 'cu_four_stage', 'extra_trees_four_stage.joblib'),
        'as':      os.path.join(CURRENT_DIR, 'joblib', 'as_four_stage', 'extra_trees_four_stage.joblib'),
        'voltage': os.path.join(CURRENT_DIR, 'joblib', 'voltage_four_stage', 'extra_trees_four_stage.joblib'),
    },
    'serial': {
        'cu':      os.path.join(CURRENT_DIR, 'joblib', 'cu_serial', 'extra_trees_serial.joblib'),
        'as':      os.path.join(CURRENT_DIR, 'joblib', 'as_serial', 'extra_trees_serial.joblib'),
        'voltage': os.path.join(CURRENT_DIR, 'joblib', 'voltage_serial', 'extra_trees_serial.joblib'),
    },
}

# 特征列顺序配置（必须与训练时完全一致）
FEATURE_COLS = {
    'three_stage_cu_as': [
        '电积前液Cu（g/L）', '三段溶液温度℃', '三段电流强度A',
        '三段流量m3/h', '三段电压V', 'year', 'month', 'day', 'hour', 'three_stage_power'
    ],
    'three_stage_voltage': [
        '电积前液Cu（g/L）', '三段溶液温度℃', '三段电流强度A',
        '三段流量m3/h', 'year', 'month', 'day', 'hour'
    ],
    'four_stage_cu_as': [
        '电积前液Cu（g/L）', '四段溶液温度℃', '四段电流强度A',
        '四段流量m3/h', '四段电压V', 'year', 'month', 'day', 'hour', 'four_stage_power'
    ],
    'four_stage_voltage': [
        '电积前液Cu（g/L）', '四段溶液温度℃', '四段电流强度A',
        '四段流量m3/h', 'year', 'month', 'day', 'hour'
    ],
    'serial_cu_as': [
        '电积前液Cu（g/L）',
        '三段溶液温度℃', '三段电流强度A', '三段流量m3/h', '三段电压V',
        '四段溶液温度℃', '四段电流强度A', '四段流量m3/h', '四段电压V',
        'year', 'month', 'day', 'hour', 'total_power'
    ],
    'serial_voltage': [
        '电积前液Cu（g/L）',
        '三段溶液温度℃', '三段电流强度A', '三段流量m3/h',
        '四段溶液温度℃', '四段电流强度A', '四段流量m3/h',
        'year', 'month', 'day', 'hour'
    ],
}

# ============================================================================
# 代理模型类
# ============================================================================

class SurrogateModel:
    """
    统一的代理模型调用接口

    该类封装了所有预训练的 ExtraTrees 回归模型，提供高层次的预测接口。
    支持三段工况、四段工况和串联工况的铜浓度、砷浓度和电压预测。

    Attributes:
        _models (dict): 存储加载的模型，结构为 {工况模式: {目标: 模型}}
    """

    def __init__(self):
        """初始化代理模型，加载所有预训练模型"""
        self._models = {}
        self._load_all()

    def _load_all(self):
        """
        加载所有预训练的代理模型

        根据操作系统自动调整并行度：
        - Windows 环境：限制最多使用 4 个 CPU 核心
        - HPC/Linux 环境：使用全部 CPU 核心
        """
        # 环境检测：根据不同环境调整并行度
        if sys.platform == 'win32':
            # Windows 本地环境，限制并行度以避免系统过载
            n_jobs = min(os.cpu_count(), 4)
            print(f'Windows 环境，设置并行度为: {n_jobs}')
        else:
            # HPC 环境，使用全部核心
            n_jobs = -1
            print('HPC 环境，设置并行度为: 全部核心')

        # 加载每个工况的每个目标模型
        for mode, targets in MODEL_PATHS.items():
            self._models[mode] = {}
            for target, path in targets.items():
                print(f'正在加载模型: {path}')
                if os.path.exists(path):
                    self._models[mode][target] = joblib.load(path)
                    # 设置并行度
                    self._models[mode][target].n_jobs = n_jobs
                else:
                    raise FileNotFoundError(f'模型文件不存在: {path}')
        print('所有代理模型加载成功')

    def predict_three_stage(self, Cu_in, T3, I3, Q3, year, month, day, hour):
        """
        三段单独运行：预测 Cu_out, As_out, V3

        Args:
            Cu_in: 进液铜浓度 (g/L)
            T3: 三段溶液温度 (℃)
            I3: 三段电流强度 (A)
            Q3: 三段流量 (m³/h)
            year, month, day, hour: 时间特征

        Returns:
            tuple: (Cu_out, As_out, V3)
        """
        # 检查是否为批量输入
        is_batch = isinstance(Cu_in, (list, np.ndarray))

        if is_batch:
            n = len(Cu_in)
            # Step1: 先预测电压（不含电压特征）
            X_v = np.array([[Cu_in[i], T3[i], I3[i], Q3[i], year, month, day, hour] for i in range(n)])
            V3 = self._models['three_stage']['voltage'].predict(X_v)

            # Step2: 用预测电压预测Cu/As（含电压特征）
            X_ca = []
            for i in range(n):
                power3 = V3[i] * I3[i]
                X_ca.append([Cu_in[i], T3[i], I3[i], Q3[i], V3[i], year, month, day, hour, power3])
            X_ca = np.array(X_ca)
            Cu_out = self._models['three_stage']['cu'].predict(X_ca)
            As_out = self._models['three_stage']['as'].predict(X_ca)
            return Cu_out, As_out, V3
        else:
            # 单个输入处理
            X_v = np.array([[Cu_in, T3, I3, Q3, year, month, day, hour]])
            V3 = float(self._models['three_stage']['voltage'].predict(X_v)[0])
            power3 = V3 * I3
            X_ca = np.array([[Cu_in, T3, I3, Q3, V3, year, month, day, hour, power3]])
            Cu_out = float(self._models['three_stage']['cu'].predict(X_ca)[0])
            As_out = float(self._models['three_stage']['as'].predict(X_ca)[0])
            return Cu_out, As_out, V3

    def predict_four_stage(self, Cu_in, T4, I4, Q4, year, month, day, hour):
        """
        四段单独运行：预测 Cu_out, As_out, V4

        Args:
            Cu_in: 进液铜浓度 (g/L)
            T4: 四段溶液温度 (℃)
            I4: 四段电流强度 (A)
            Q4: 四段流量 (m³/h)
            year, month, day, hour: 时间特征

        Returns:
            tuple: (Cu_out, As_out, V4)
        """
        is_batch = isinstance(Cu_in, (list, np.ndarray))

        if is_batch:
            n = len(Cu_in)
            X_v = np.array([[Cu_in[i], T4[i], I4[i], Q4[i], year, month, day, hour] for i in range(n)])
            V4 = self._models['four_stage']['voltage'].predict(X_v)

            X_ca = []
            for i in range(n):
                power4 = V4[i] * I4[i]
                X_ca.append([Cu_in[i], T4[i], I4[i], Q4[i], V4[i], year, month, day, hour, power4])
            X_ca = np.array(X_ca)
            Cu_out = self._models['four_stage']['cu'].predict(X_ca)
            As_out = self._models['four_stage']['as'].predict(X_ca)
            return Cu_out, As_out, V4
        else:
            X_v = np.array([[Cu_in, T4, I4, Q4, year, month, day, hour]])
            V4 = float(self._models['four_stage']['voltage'].predict(X_v)[0])
            power4 = V4 * I4
            X_ca = np.array([[Cu_in, T4, I4, Q4, V4, year, month, day, hour, power4]])
            Cu_out = float(self._models['four_stage']['cu'].predict(X_ca)[0])
            As_out = float(self._models['four_stage']['as'].predict(X_ca)[0])
            return Cu_out, As_out, V4

    def predict_serial(self, Cu_in, T3, I3, Q3, T4, I4, Q4, year, month, day, hour):
        """
        三四段串联：端到端预测最终 Cu_out, As_out, V3, V4, V_avg

        Args:
            Cu_in: 进液铜浓度 (g/L)
            T3, I3, Q3: 三段温度、电流、流量
            T4, I4, Q4: 四段温度、电流、流量
            year, month, day, hour: 时间特征

        Returns:
            tuple: (Cu_out, As_out, V3, V4, V_avg)
        """
        is_batch = isinstance(Cu_in, (list, np.ndarray))

        if is_batch:
            n = len(Cu_in)
            # Step1: 使用 serial_voltage 模型直接预测三四段平均电压
            X_v = np.array([[Cu_in[i], T3[i], I3[i], Q3[i], T4[i], I4[i], Q4[i], year, month, day, hour] for i in range(n)])
            V_avg = self._models['serial']['voltage'].predict(X_v)

            # Step2: 串联Cu/As模型需要V3、V4两列，用V_avg近似代替（V3=V4=V_avg）
            X_ca = []
            for i in range(n):
                total_power = V_avg[i] * (I3[i] + I4[i])
                X_ca.append([
                    Cu_in[i],
                    T3[i], I3[i], Q3[i], V_avg[i],
                    T4[i], I4[i], Q4[i], V_avg[i],
                    year, month, day, hour, total_power
                ])
            X_ca = np.array(X_ca)
            Cu_out = self._models['serial']['cu'].predict(X_ca)
            As_out = self._models['serial']['as'].predict(X_ca)

            # V3 = V4 = V_avg（近似，串联时两段电压相近）
            return Cu_out, As_out, V_avg, V_avg, V_avg
        else:
            # Step1: 使用 serial_voltage 模型直接预测三四段平均电压
            X_v = np.array([[Cu_in, T3, I3, Q3, T4, I4, Q4, year, month, day, hour]])
            V_avg = float(self._models['serial']['voltage'].predict(X_v)[0])

            # Step2: 串联Cu/As模型需要V3、V4两列，用V_avg近似代替（V3=V4=V_avg）
            total_power = V_avg * (I3 + I4)
            X_ca = np.array([[
                Cu_in,
                T3, I3, Q3, V_avg,
                T4, I4, Q4, V_avg,
                year, month, day, hour, total_power
            ]])
            Cu_out = float(self._models['serial']['cu'].predict(X_ca)[0])
            As_out = float(self._models['serial']['as'].predict(X_ca)[0])

            # V3 = V4 = V_avg（近似，串联时两段电压相近）
            return Cu_out, As_out, V_avg, V_avg, V_avg

    def calc_energy(self, V3, I3, V4, I4, t, mode):
        """
        计算总电耗 (kWh)

        Args:
            V3, I3: 三段电压(V)和电流(A)
            V4, I4: 四段电压(V)和电流(A)
            t: 电积持续时间 (h)
            mode: 运行模式 ('three_stage', 'four_stage', 'serial')

        Returns:
            float: 总电耗 (kWh)
        """
        if mode == 'three_stage':
            return V3 * I3 * t / 1000
        elif mode == 'four_stage':
            return V4 * I4 * t / 1000
        else:  # serial
            return (V3 * I3 + V4 * I4) * t / 1000

    def calc_profit(self, Cu_in, Cu_out, As_out, Q, t, E_total, mode, econ_params=None):
        """
        计算单批次综合净利润 (元)

        公式：
            R_profit = R_Cu - R_save - C_e - C_op

            R_Cu   = (p_Cu - c_reprocess - c_copper_concentrate) × m_Cu
                   铜回收净收益，m_Cu = (Cu_in - Cu_out) * Q * t  [kg]

            R_save = c_As_treat × (As_out + Cu_out) × Q × t
                   危废处置节约收益（出液浓度越低，危废越少，节约越多）

            C_e    = p_e × E_total          电耗成本
            C_op   = c_op × t               设备运行成本

        Args:
            Cu_in: 进液铜浓度 (g/L)
            Cu_out: 出液铜浓度 (g/L)
            As_out: 出液砷浓度 (g/L)
            Q: 电解液流量 (m³/h)
            t: 电积持续时间 (h)
            E_total: 总电耗 (kWh)
            mode: 运行模式
            econ_params: 经济参数字典（可选）

        Returns:
            tuple: (R_profit, detail) 净利润和各项明细
        """
        # 默认经济参数（与图片中Table S5一致）
        p = econ_params or {
            'p_Cu':               98.44,    # 铜折算价格 (元/kg)
            'c_reprocess':         0.14,     # 黑铜板返炼成本 (元/kg)
            'c_copper_concentrate': 82.64,   # 铜精矿成本价格 (元/kg)
            'c_As_treat':          2.5,      # 砷危废处置单位成本 (元/kg)
            'p_e':                 0.6,      # 工业电价 (元/kWh)
            'c_op':              200.0,      # 设备运行成本 (元/h)
        }

        # 铜回收质量 (kg)
        m_Cu = np.maximum(0.0, (Cu_in - Cu_out) * Q * t)

        # 铜回收净收益
        R_Cu = (p['p_Cu'] - p['c_reprocess'] - p['c_copper_concentrate']) * m_Cu

        # 危废处置节约收益（As_out 取固定最大值 11.64，以最大化砷危废对经济效益的负面影响）
        As_out_fixed = 11.64
        R_save = p['c_As_treat'] * (As_out_fixed + Cu_out) * Q * t

        # 电耗成本
        C_e = p['p_e'] * E_total

        # 设备运行成本
        C_op = p['c_op'] * t

        # 综合净利润
        R_profit = R_Cu - R_save - C_e - C_op

        # 各项明细
        detail = {
            'R_Cu':      R_Cu,
            'R_save':    R_save,
            'C_e':       C_e,
            'C_op':      C_op,
            'R_profit':  R_profit,
            'm_Cu_kg':   m_Cu,
        }
        return R_profit, detail

# ============================================================================
# NSGA-II 优化问题定义
# ============================================================================

class ThreeStageOptProblem(Problem):
    """
    三段工况优化问题定义

    决策变量（5维）：
        0: Cu_in - 进液铜浓度 (29~55 g/L)
        1: T3 - 三段溶液温度 (40~65 ℃)
        2: I3 - 三段电流强度 (8000~27000 A)
        3: Q3 - 三段流量 (111~123 m³/h)
        4: t - 电积时间 (2~8 h)

    目标函数（最小化）：
        0: Cu_out - 出液铜浓度
        1: -As_out - 出液砷浓度（取负因为As要最大化）
        2: E_total - 总能耗
        3: -R_profit - 净利润（取负因为profit要最大化）

    约束条件（<=0 表示满足）：
        0: Cu_out - Cu_limit <= 0
        1: As_min - As_out <= 0
        2: J - J_max <= 0
        3: V_cell - V_cell_max <= 0
        4: Cu_As - Cu_As_max <= 0
    """

    def __init__(self, runtime, econ_params=None):
        """
        初始化三段工况优化问题

        Args:
            runtime: 运行时参数（包含year, month, day, hour）
            econ_params: 经济参数（可选）
        """
        self.runtime = runtime
        self.econ_params = econ_params

        # 决策变量边界 [Cu_in, T3, I3, Q3, t]
        xl = np.array([29.0, 40.0, 8000.0, 111.0, 2.0])
        xu = np.array([55.0, 65.0, 27000.0, 123.0, 8.0])

        # 调用父类初始化
        super().__init__(n_var=5, n_obj=4, n_constr=5, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        """
        向量化评估函数

        Args:
            X: 决策变量矩阵，形状 (N, 5)，N为个体数量
            out: 输出字典，用于存储目标函数值和约束函数值
        """
        p = PARAMS
        rt = self.runtime

        # 按列解析决策变量
        Cu_in, T3, I3, Q3, t = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

        # 批量预测（内部自动走 joblib 的多线程）
        Cu_out, As_out, V3 = surrogate.predict_three_stage(
            Cu_in, T3, I3, Q3, rt['year'], rt['month'], rt['day'], rt['hour'])

        # 批量计算能耗与利润
        E = surrogate.calc_energy(V3, I3, np.zeros_like(V3), np.zeros_like(I3), t, 'three_stage')
        R_profit, _ = surrogate.calc_profit(
            Cu_in, Cu_out, As_out, Q3, t, E, 'three_stage', self.econ_params)

        # 目标函数矩阵 (N, 4)
        out['F'] = np.column_stack([Cu_out, -As_out, E, -R_profit])

        # 约束函数矩阵 (N, 5)
        J = I3 / (p['n_cathode'] * p['A_cathode'])
        V_cell = V3 / p['N_cell']
        Cu_As = Cu_out / (As_out + 1e-8)

        out['G'] = np.column_stack([
            Cu_out - p['Cu_limit'],
            p['As_min'] - As_out,
            J - p['J_max'],
            V_cell - p['V_cell_max'],
            Cu_As - p['Cu_As_max']
        ])

        # 添加兜底：防止代理模型偶发的 NaN/Inf 导致算法中断
        out['F'] = np.nan_to_num(out['F'], nan=1e6, posinf=1e6, neginf=-1e6)
        out['G'] = np.nan_to_num(out['G'], nan=1e6, posinf=1e6, neginf=-1e6)


class FourStageOptProblem(Problem):
    """
    四段工况优化问题定义

    决策变量（5维）：
        0: Cu_in - 进液铜浓度 (29~55 g/L)
        1: T4 - 四段溶液温度 (40~65 ℃)
        2: I4 - 四段电流强度 (8000~29000 A)
        3: Q4 - 四段流量 (113~122 m³/h)
        4: t - 电积时间 (2~8 h)

    目标函数、约束条件同三段工况
    """

    def __init__(self, runtime, econ_params=None):
        self.runtime = runtime
        self.econ_params = econ_params

        xl = np.array([29.0, 40.0, 8000.0, 113.0, 2.0])
        xu = np.array([55.0, 65.0, 29000.0, 122.0, 8.0])
        super().__init__(n_var=5, n_obj=4, n_constr=5, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        p = PARAMS
        rt = self.runtime
        Cu_in, T4, I4, Q4, t = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

        Cu_out, As_out, V4 = surrogate.predict_four_stage(
            Cu_in, T4, I4, Q4, rt['year'], rt['month'], rt['day'], rt['hour'])

        E = surrogate.calc_energy(np.zeros_like(V4), np.zeros_like(I4), V4, I4, t, 'four_stage')
        R_profit, _ = surrogate.calc_profit(
            Cu_in, Cu_out, As_out, Q4, t, E, 'four_stage', self.econ_params)

        out['F'] = np.column_stack([Cu_out, -As_out, E, -R_profit])

        J = I4 / (p['n_cathode'] * p['A_cathode'])
        V_cell = V4 / p['N_cell']
        Cu_As = Cu_out / (As_out + 1e-8)

        out['G'] = np.column_stack([
            Cu_out - p['Cu_limit'],
            p['As_min'] - As_out,
            J - p['J_max'],
            V_cell - p['V_cell_max'],
            Cu_As - p['Cu_As_max']
        ])

        # 添加兜底：防止代理模型偶发的 NaN/Inf 导致算法中断
        out['F'] = np.nan_to_num(out['F'], nan=1e6, posinf=1e6, neginf=-1e6)
        out['G'] = np.nan_to_num(out['G'], nan=1e6, posinf=1e6, neginf=-1e6)


class SerialOptProblem(Problem):
    """
    串联工况优化问题定义

    决策变量（8维）：
        0: Cu_in - 进液铜浓度 (29~55 g/L)
        1: T3 - 三段溶液温度 (40~65 ℃)
        2: I3 - 三段电流强度 (10000~22000 A)
        3: Q3 - 三段流量 (113~122 m³/h)
        4: T4 - 四段溶液温度 (40~65 ℃)
        5: I4 - 四段电流强度 (10000~24000 A)
        6: Q4 - 四段流量 (113~122 m³/h)
        7: t - 电积时间 (2~8 h)

    目标函数、约束条件同三段工况
    """

    def __init__(self, runtime, econ_params=None):
        self.runtime = runtime
        self.econ_params = econ_params

        xl = np.array([29.0, 40.0, 10000.0, 113.0, 40.0, 10000.0, 114.0, 2.0])
        xu = np.array([55.0, 65.0, 22000.0, 122.0, 65.0, 24000.0, 122.0, 8.0])
        super().__init__(n_var=8, n_obj=4, n_constr=5, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        p = PARAMS
        rt = self.runtime
        Cu_in, T3, I3, Q3, T4, I4, Q4, t = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[:, 7]

        Cu_out, As_out, V3, V4, V_avg = surrogate.predict_serial(
            Cu_in, T3, I3, Q3, T4, I4, Q4, rt['year'], rt['month'], rt['day'], rt['hour'])

        E = surrogate.calc_energy(V3, I3, V4, I4, t, 'serial')
        Q_avg = (Q3 + Q4) / 2.0
        R_profit, _ = surrogate.calc_profit(
            Cu_in, Cu_out, As_out, Q_avg, t, E, 'serial', self.econ_params)

        out['F'] = np.column_stack([Cu_out, -As_out, E, -R_profit])

        J3 = I3 / (p['n_cathode'] * p['A_cathode'])
        J4 = I4 / (p['n_cathode'] * p['A_cathode'])
        J_max = np.maximum(J3, J4)
        V_cell = np.maximum(V3, V4) / p['N_cell']
        Cu_As = Cu_out / (As_out + 1e-8)

        out['G'] = np.column_stack([
            Cu_out - p['Cu_limit'],
            p['As_min'] - As_out,
            J_max - p['J_max'],
            V_cell - p['V_cell_max'],
            Cu_As - p['Cu_As_max']
        ])

        # 添加兜底：防止代理模型偶发的 NaN/Inf 导致算法中断
        out['F'] = np.nan_to_num(out['F'], nan=1e6, posinf=1e6, neginf=-1e6)
        out['G'] = np.nan_to_num(out['G'], nan=1e6, posinf=1e6, neginf=-1e6)

# ============================================================================
# NSGA-II 优化运行函数
# ============================================================================

def run_nsga2(problem, mode_label, pop_size=300, n_gen=300, seed=42,
              verbose=True, save_history=True, save_all_feasible=False, n_fronts=3,
              n_solutions=None):
    """
    运行 NSGA-II 多目标优化算法

    Args:
        problem: pymoo 优化问题实例
        mode_label: 工况标签（用于日志输出）
        pop_size: 种群大小（默认300）
        n_gen: 迭代代数（默认300）
        seed: 随机种子（默认42）
        verbose: 是否输出详细日志
        save_history: 是否保存收敛历史
        save_all_feasible: 是否保存所有可行解（而非仅Pareto前沿）

    Returns:
        tuple: (result, F_feasible, X_feasible, G_feasible, metrics)
            - result: pymoo 优化结果
            - F_feasible: 可行解的目标函数值矩阵 (N, 4)
            - X_feasible: 可行解的决策变量矩阵 (N, n_var)
            - G_feasible: 可行解的约束函数值矩阵 (N, n_constr)
            - metrics: 质量指标字典
    """
    print(f"\n{'='*60}")
    print(f" NSGA-II —— {mode_label} pop={pop_size} gen={n_gen} (Vectorized Mode)")
    print(f"{'='*60}")
    print(f" [1/6] 正在初始化优化问题...", flush=True)
    t_start = time.time()

    print(f" [2/6] 开始运行NSGA-II优化 (单进程+底层多线程并行)...", flush=True)
    result = minimize(
        problem,
        NSGA2(pop_size=pop_size),
        get_termination('n_gen', n_gen),
        seed=seed,
        verbose=verbose,
        save_history=save_history
    )

    print(f" [3/6] 优化完成，正在筛选可行解...", flush=True)

    # 处理优化结果
    if result.F is not None and result.X is not None:
        print(f" 原始 X 形状: {result.X.shape}")

        # 筛选满足约束条件的可行解
        fmask = (result.G <= 0).all(axis=1) if result.G is not None else np.ones(len(result.F), dtype=bool)
        F_all_feasible = result.F[fmask]
        X_all_feasible = result.X[fmask]
        G_all_feasible = result.G[fmask] if result.G is not None else None
        print(f" 所有可行解数量: {len(F_all_feasible)}")

        # 根据模式选择保存所有可行解或仅Pareto前沿
        if save_all_feasible:
            F_feasible = F_all_feasible
            X_feasible = X_all_feasible
            G_feasible = G_all_feasible
            print(f" 模式: 保存所有可行解")
        else:
            # 使用 pymoo 自带的 NonDominatedSorting 进行非支配排序
            # 保留前n个非支配前沿以获得更多解
            if len(F_all_feasible) > 0:
                nds = NonDominatedSorting()
                fronts = nds.do(F_all_feasible)
                # 只保留前n_fronts个前沿
                if len(fronts) > n_fronts:
                    fronts = fronts[:n_fronts]
                
                # 合并前3个前沿的索引
                front_indices = []
                for front in fronts:
                    front_indices.extend(front.tolist() if hasattr(front, 'tolist') else list(front))
                front_indices = list(set(front_indices))
                
                F_feasible = F_all_feasible[front_indices]
                X_feasible = X_all_feasible[front_indices]
                G_feasible = G_all_feasible[front_indices] if G_all_feasible is not None else None

                # 确保数组维度正确（处理只有一个解的情况）
                if len(X_feasible.shape) == 1:
                    X_feasible = X_feasible.reshape(1, -1)
                if len(F_feasible.shape) == 1:
                    F_feasible = F_feasible.reshape(1, -1)
                if G_feasible is not None and len(G_feasible.shape) == 1:
                    G_feasible = G_feasible.reshape(1, -1)
            print(f" 模式: 仅保存 Pareto 前沿解")
        
        # 计算当前可行解数量
        n_feasible = len(F_feasible)
        
        # 如果指定了解数量，且当前解数量多于目标，进行均匀采样
        if n_solutions is not None and n_feasible > n_solutions:
            indices = np.linspace(0, n_feasible - 1, n_solutions, dtype=int)
            F_feasible = F_feasible[indices]
            X_feasible = X_feasible[indices]
            G_feasible = G_feasible[indices] if G_feasible is not None else None
            print(f" 均匀采样: {n_feasible} -> {n_solutions}")
            n_feasible = n_solutions
    else:
        F_feasible = np.array([]).reshape(0, 4)
        X_feasible = np.array([]).reshape(0, problem.n_var)
        G_feasible = None
        n_feasible = 0
        F_all_feasible = np.array([]).reshape(0, 4)
        X_all_feasible = np.array([]).reshape(0, problem.n_var)
        G_all_feasible = None

    elapsed = time.time() - t_start
    print(f"\n 可行解: {n_feasible} / {len(result.F) if result.F is not None else 0}")

    # ---------------- 处理收敛历史数据 ----------------
    print(f" [4/6] 正在处理收敛历史数据...", flush=True)
    convergence_rows = []
    if hasattr(result, 'history') and result.history:
        for gen_idx, gen in enumerate(result.history):
            F_gen = gen.opt.get('F')
            if F_gen is None or len(F_gen) == 0:
                continue
            G_gen = gen.opt.get('G')
            if G_gen is not None:
                fm = (G_gen <= 0).all(axis=1)
                n_fg = int(fm.sum())
                F_fg = F_gen[fm]
            else:
                n_fg = 0
                F_fg = np.array([]).reshape(0, 4) # 改为空数组，跳过该代 HV 计算
            hv_g = sp_g = None
            if len(F_fg) >= 2:
                try:
                    hv_g = float(HV(ref_point=GLOBAL_HV_REF_POINT)(F_fg))
                    sp_g = float(SpacingIndicator()(F_fg))
                except Exception:
                    pass
            convergence_rows.append({
                'Generation': gen_idx + 1,
                'Best_Cu_out': float(F_gen[:, 0].min()),
                'Best_As_out': float(-F_gen[:, 1].min()),
                'Best_E_total': float(F_gen[:, 2].min()),
                'Best_Net_profit': float(-F_gen[:, 3].max()),
                'HV': hv_g,
                'Spacing': sp_g,
                'N_feasible': n_fg,
            })
    if convergence_rows:
        path = f'{OUTPUT_DIR}/convergence_{mode_label}.csv'
        pd.DataFrame(convergence_rows).to_csv(path, index=False)
        print(f" 收敛数据: {path}")

    # ---------------- 生成 Pareto 前沿数据 ----------------
    print(f" [5/6] 正在生成Pareto前沿数据...", flush=True)
    _VAR_NAMES = {
        '三段工况': ['Cu_in','TA','IA','QA','t'],
        '四段工况': ['Cu_in','TB','IB','QB','t'],
        '串联工况': ['Cu_in','TA','IA','QA','TB','IB','QB','t'],
    }
    var_names = _VAR_NAMES.get(mode_label, [f'x{i}' for i in range(problem.n_var)])
    pareto_rows = []
    for i in range(n_feasible):
        row = {'Method': 'NSGA-II', 'Condition': mode_label, 'Solution_ID': i+1}
        for j, vn in enumerate(var_names):
            row[vn] = float(X_feasible[i, j])
        row['Cu_out'] = float(F_feasible[i, 0])
        row['As_out'] = float(-F_feasible[i, 1])
        row['E_total'] = float(F_feasible[i, 2])
        row['Net_profit'] = float(-F_feasible[i, 3])
        if G_feasible is not None:
            for k, gn in enumerate(['G1_Cu','G2_As','G3_J','G4_V','G5_ratio']):
                row[gn] = float(G_feasible[i, k])
        pareto_rows.append(row)
    if pareto_rows:
        path = f'{OUTPUT_DIR}/pareto_nsga2_{mode_label.replace(" ", "_")}.csv'
        pd.DataFrame(pareto_rows).to_csv(path, index=False)
        print(f" Pareto CSV: {path}")

    # ---------------- 计算质量指标 ----------------
    print(f" [6/6] 正在计算质量指标...", flush=True)
    metrics = {'HV': None, 'Spacing': None, 'n_feasible': n_feasible}
    if n_feasible >= 2:
        hv_val = float(HV(ref_point=GLOBAL_HV_REF_POINT)(F_feasible))
        sp_val = float(SpacingIndicator()(F_feasible))
        As_real = -F_feasible[:, 1]
        metrics.update({
            'HV': hv_val,
            'Spacing': sp_val,
            'Mean_Cu_out': float(F_feasible[:, 0].mean()),
            'Std_Cu_out': float(F_feasible[:, 0].std()),
            'Range_Cu_out': float(np.ptp(F_feasible[:, 0])),
            'Mean_As_out': float(As_real.mean()),
            'Std_As_out': float(As_real.std()),
            'Range_As_out': float(np.ptp(As_real)),
            'Mean_E_total': float(F_feasible[:, 2].mean()),
            'Std_E_total': float(F_feasible[:, 2].std()),
            'Range_E_total': float(np.ptp(F_feasible[:, 2])),
            'Mean_Net_profit': float(-F_feasible[:, 3].mean()),
            'Std_Net_profit': float(F_feasible[:, 3].std()),
            'Range_Net_profit': float(np.ptp(F_feasible[:, 3])),
            'Total_time': elapsed,
            'N_evaluations': n_gen * pop_size,
        })
        print(f" HV={hv_val:.4f} Spacing={sp_val:.4f}")

    # 保存质量指标到CSV
    qm_path = f'{OUTPUT_DIR}/quality_metrics_nsga2.csv'
    qm_row = {'Method': 'NSGA-II', 'Condition': mode_label}
    qm_row.update(metrics)
    qm_new = pd.DataFrame([qm_row])
    if os.path.exists(qm_path):
        qm_old = pd.read_csv(qm_path)
        qm_old = qm_old[qm_old['Condition'] != mode_label]
        qm_new = pd.concat([qm_old, qm_new], ignore_index=True)
    qm_new.to_csv(qm_path, index=False)
    print(f" {mode_label} 优化完成！耗时: {elapsed:.2f}秒", flush=True)

    return result, F_feasible, X_feasible, G_feasible, metrics

# ============================================================================
# 可视化函数
# ============================================================================

def plot_pareto_3d(F_feasible, mode_label, save_path):
    """
    绘制 3D Pareto 前沿图

    Args:
        F_feasible: 可行解的目标函数值矩阵 (N, 4)
        mode_label: 工况标签
        save_path: 保存路径
    """
    if len(F_feasible) == 0:
        return
    profit = -F_feasible[:, 3]
    As_real = -F_feasible[:, 1]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    norm = mpl.colors.Normalize(vmin=profit.min(), vmax=profit.max())
    sc = ax.scatter(
        F_feasible[:, 0], As_real, F_feasible[:, 2],
        c=profit, cmap=_CMAP_PROFIT, norm=norm,
        s=28, alpha=0.80, edgecolors='none', depthshade=True, rasterized=True
    )

    ax.set_xlabel(_OBJ_LABELS[0], fontsize=11, labelpad=10)
    ax.set_ylabel(_OBJ_LABELS[1], fontsize=11, labelpad=10)
    ax.set_zlabel(_OBJ_LABELS[2], fontsize=11, labelpad=10)
    ax.tick_params(labelsize=9, direction='in')

    cbar = fig.colorbar(sc, ax=ax, shrink=0.50, pad=0.07, aspect=20)
    cbar.set_label(_OBJ_LABELS[3], fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.6)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#CCCCCC')
        pane.set_linewidth(0.5)

    ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.6)
    plt.tight_layout(pad=0.8)
    fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f" 3-D Pareto saved: {save_path}")


def plot_pareto_2d_projections(F_feasible, mode_label, save_path):
    """
    绘制 2D Pareto 前沿投影图（6个子图）

    Args:
        F_feasible: 可行解的目标函数值矩阵 (N, 4)
        mode_label: 工况标签
        save_path: 保存路径
    """
    if len(F_feasible) == 0:
        return

    F_plot = F_feasible.copy()
    F_plot[:, 1] = -F_plot[:, 1]
    F_plot[:, 3] = -F_plot[:, 3]

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    panel_tags = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    profit_vals = F_plot[:, 3]
    norm = mpl.colors.Normalize(vmin=profit_vals.min(), vmax=profit_vals.max())

    fig, axes = plt.subplots(2, 3, figsize=(16, 12), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.07, top=0.91, wspace=0.175, hspace=0.127)

    for ax, (xi, yi), tag in zip(axes.flat, pairs, panel_tags):
        ax.scatter(
            F_plot[:, xi], F_plot[:, yi],
            c=profit_vals, cmap=_CMAP_PROFIT, norm=norm,
            s=18, alpha=0.72, edgecolors='none', rasterized=True
        )
        ax.set_xlabel(_OBJ_LABELS[xi], fontsize=10)
        ax.set_ylabel(_OBJ_LABELS[yi], fontsize=10)
        ax.tick_params(labelsize=9, which='both', direction='in', top=True, right=True)
        ax.tick_params(axis='x', which='both', top=False)
        ax.tick_params(axis='y', which='both', right=False)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.grid(True, which='major', linestyle=':', linewidth=0.45, alpha=0.55, color='#AAAAAA')
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
        ax.text(0.04, 0.95, tag, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

    cax = fig.add_axes([0.905, 0.12, 0.018, 0.76])
    sm = mpl.cm.ScalarMappable(cmap=_CMAP_PROFIT, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(_OBJ_LABELS[3], fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.6)

    fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f" 2-D projections saved: {save_path}")


def plot_convergence(result, mode_label, save_path):
    """
    绘制收敛曲线图

    Args:
        result: pymoo 优化结果（包含 history）
        mode_label: 工况标签
        save_path: 保存路径
    """
    if not hasattr(result, 'history') or result.history is None:
        return

    best_cu = []
    for gen in result.history:
        F_gen = gen.opt.get('F')
        if F_gen is not None and len(F_gen) > 0:
            best_cu.append(float(F_gen[:, 0].min()))

    if not best_cu:
        return

    gens = np.arange(1, len(best_cu) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(gens, best_cu, alpha=0.10, color='#1f77b4', zorder=1)
    ax.plot(gens, best_cu, color='#1f77b4', linewidth=1.6, zorder=3, label=_en(mode_label))
    ax.annotate(
        f'Final: {best_cu[-1]:.4f} g/L',
        xy=(gens[-1], best_cu[-1]),
        xytext=(-80, 20),
        textcoords='offset points',
        fontsize=9.5,
        arrowprops=dict(arrowstyle='->', color='#444444', lw=0.8)
    )
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel(r'Best $\mathrm{Cu_{out}}$ (g/L)', fontsize=12)
    ax.set_ylim(min(best_cu) * 0.95, max(best_cu) * 1.05)
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=10)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(True, which='major', linestyle=':', linewidth=0.45, alpha=0.55, color='#AAAAAA')
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='0.7')
    plt.tight_layout(pad=0.8)
    fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f" Convergence saved: {save_path}")


# ============================================================================
# 妥协解选择与打印函数
# ============================================================================

def select_compromise_solutions(F_feasible, X_feasible, mode_label):
    """
    从 Pareto 前沿中选择典型妥协解

    选择策略：
    - quality: 质量优先（Cu_out 和 As_out 加权最小）
    - energy: 节能优先（E_total 最小）
    - profit: 经济优先（R_profit 最大）
    - balance: 均衡优化（所有目标加权最小）

    Args:
        F_feasible: 可行解的目标函数值矩阵 (N, 4)
        X_feasible: 可行解的决策变量矩阵 (N, n_var)
        mode_label: 工况标签

    Returns:
        dict: 包含四种妥协解的字典
    """
    if len(F_feasible) == 0:
        print(f" {mode_label} 无可行解")
        return {}

    # 归一化目标函数值
    F_min = F_feasible.min(axis=0)
    F_max = F_feasible.max(axis=0)
    denom = F_max - F_min
    denom[denom == 0] = 1e-10
    F_norm = (F_feasible - F_min) / denom

    # 选择四种典型妥协解
    results = {}
    for key, idx_fn in [
        ('quality', lambda: np.argmin(F_norm[:, 0] + F_norm[:, 1])),
        ('energy', lambda: np.argmin(F_norm[:, 2])),
        ('profit', lambda: np.argmin(F_norm[:, 3])),
        ('balance', lambda: np.argmin(F_norm.sum(axis=1))),
    ]:
        idx = idx_fn()
        results[key] = {'idx': idx, 'F': F_feasible[idx], 'X': X_feasible[idx]}
    return results


def print_compromise_solutions(solutions, mode_label, var_names):
    """
    打印妥协解的详细信息

    Args:
        solutions: 妥协解字典
        mode_label: 工况标签
        var_names: 决策变量名称列表
    """
    scene_names = {'quality': 'Quality', 'energy': 'Energy', 'profit': 'Profit', 'balance': 'Balance'}
    print(f"\n --- {mode_label} Solutions ---")
    for key, sol in solutions.items():
        print(f"\n [{scene_names[key]}]")
        vars_str = " ".join([f"x{i}={v:.2f}" for i, v in enumerate(sol['X'])])
        print(f" Variables: {vars_str}")
        print(f" Cu_out = {sol['F'][0]:.4f} g/L")
        print(f" As_out = {-sol['F'][1]:.4f} g/L")
        print(f" E_total = {sol['F'][2]:.4f} kWh")
        print(f" R_profit = {-sol['F'][3]:.2f}")


# ============================================================================
# Excel 导出函数
# ============================================================================

def _make_styles():
    """创建 Excel 单元格样式"""
    thin = Side(style='thin')
    return {
        'header_font': Font(name='Arial', bold=True, size=10),
        'data_font': Font(name='Arial', size=10),
        'center': Alignment(horizontal='center', vertical='center'),
        'left': Alignment(horizontal='left', vertical='center'),
        'border': Border(left=thin, right=thin, top=thin, bottom=thin),
        'header_fill': PatternFill('solid', start_color='D9E1F2'),
    }


def _apply_header(ws, row_idx, n_cols, styles):
    """应用表头样式"""
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=row_idx, column=c)
        cell.font = styles['header_font']
        cell.alignment = styles['center']
        cell.fill = styles['header_fill']
        cell.border = styles['border']


def _apply_data(cell, val, styles, is_str=False):
    """应用数据单元格样式"""
    cell.font = styles['data_font']
    cell.alignment = styles['left'] if is_str else styles['center']
    cell.border = styles['border']
    if isinstance(val, float):
        cell.value = round(val, 6)
    elif val is not None:
        cell.value = val


def save_nsga_excel(all_solutions, all_compromise, all_metrics, runtime,
                    template_path='NSGA.xlsx', output_path=None):
    """
    保存 NSGA-II 优化结果到 Excel 文件

    Args:
        all_solutions: 所有工况的解集字典
        all_compromise: 所有工况的妥协解字典
        all_metrics: 所有工况的质量指标字典
        runtime: 运行时参数
        template_path: Excel 模板路径
        output_path: 输出路径
    """
    if output_path is None:
        output_path = f'{OUTPUT_DIR}/NSGA_results.xlsx'

    if not os.path.exists(template_path):
        print(f" 模板文件 {template_path} 不存在，正在创建新的工作簿...")
        wb = Workbook()
        sheet_names = [
            'Condition 1 Pareto Front',
            'Condition 2 Pareto Front',
            'Condition 3 Pareto Front',
            'Compromise Solutions',
            'nsga_metrics',
            'quality_metrics_nsga'
        ]
        if 'Sheet' in wb.sheetnames:
            del wb['Sheet']
        for sheet_name in sheet_names:
            wb.create_sheet(title=sheet_name)

        st = _make_styles()

        # Condition 1 工作表
        ws1 = wb['Condition 1 Pareto Front']
        headers1 = ['Cu_in', 'TA', 'IA', 'QA', 't', 'Cu_out', 'As_out', 'E_total',
                    'Net profit\n(Ten thousand CNY)', 'VA', 'costs']
        for c, header in enumerate(headers1, 1):
            ws1.cell(row=1, column=c, value=header)
        _apply_header(ws1, 1, len(headers1), st)

        # Condition 2 工作表
        ws2 = wb['Condition 2 Pareto Front']
        headers2 = ['Cu_in', 'TB', 'IB', 'QB', 't', 'Cu_out', 'As_out', 'E_total',
                    'Net profit\n(Ten thousand CNY)', 'VB', 'costs']
        for c, header in enumerate(headers2, 1):
            ws2.cell(row=1, column=c, value=header)
        _apply_header(ws2, 1, len(headers2), st)

        # Condition 3 工作表
        ws3 = wb['Condition 3 Pareto Front']
        headers3 = ['Cu_in', 'TA', 'IA', 'QA', 'TB', 'IB', 'QB', 't', 'Cu_out', 'As_out',
                     'E_total', 'Net profit\n(Ten thousand CNY)', 'Vaverage', 'costs']
        for c, header in enumerate(headers3, 1):
            ws3.cell(row=1, column=c, value=header)
        _apply_header(ws3, 1, len(headers3), st)

        # Compromise Solutions 工作表
        ws_c = wb['Compromise Solutions']
        headers_c = ['Condition', 'Scenario', 'Cu_in', 'TA', 'IA', 'QA', 'TB', 'IB', 'QB',
                      't', 'Cu_out', 'As_out', 'E_total', 'Net profit']
        for c, header in enumerate(headers_c, 1):
            ws_c.cell(row=1, column=c, value=header)
        _apply_header(ws_c, 1, len(headers_c), st)

        # nsga_metrics 工作表
        ws_m = wb['nsga_metrics']
        headers_m = ['Condition', 'n_feasible', 'HV', 'Spacing']
        for c, header in enumerate(headers_m, 1):
            ws_m.cell(row=1, column=c, value=header)
        _apply_header(ws_m, 1, len(headers_m), st)

        # quality_metrics_nsga 工作表
        ws_q = wb['quality_metrics_nsga']
        headers_q = ['Condition', 'n_feasible', 'Mean_Cu_out', 'Std_Cu_out', 'Range_Cu_out',
                      'Mean_As_out', 'Std_As_out', 'Range_As_out', 'Mean_E_total', 'Std_E_total',
                      'Range_E_total', 'Mean_Net_profit', 'Std_Net_profit', 'Range_Net_profit']
        for c, header in enumerate(headers_q, 1):
            ws_q.cell(row=1, column=c, value=header)
        _apply_header(ws_q, 1, len(headers_q), st)
    else:
        wb = load_workbook(template_path)
        st = _make_styles()

    # 工况标签映射
    cond_label_map = {'三段工况': 'Condition 1', '四段工况': 'Condition 2', '串联工况': 'Condition 3'}
    sheet_defs = {
        '三段工况': ('Condition 1 Pareto Front', 'three_stage',
                     ['Cu_in', 'TA', 'IA', 'QA', 't'],
                     ['Cu_out', 'As_out', 'E_total', 'Net profit\n(Ten thousand CNY)', 'VA', 'costs']),
        '四段工况': ('Condition 2 Pareto Front', 'four_stage',
                     ['Cu_in', 'TB', 'IB', 'QB', 't'],
                     ['Cu_out', 'As_out', 'E_total', 'Net profit\n(Ten thousand CNY)', 'VB', 'costs']),
        '串联工况': ('Condition 3 Pareto Front', 'serial',
                     ['Cu_in', 'TA', 'IA', 'QA', 'TB', 'IB', 'QB', 't'],
                     ['Cu_out', 'As_out', 'E_total', 'Net profit\n(Ten thousand CNY)', 'Vaverage', 'costs']),
    }

    # 写入 Pareto 前沿数据
    for mode_label, (sheet_name, mode_key, dec_vars, obj_cols) in sheet_defs.items():
        if mode_label not in all_solutions:
            continue
        data = all_solutions[mode_label]
        F = data['F']
        X = data['X']
        ws = wb[sheet_name]

        # 清空现有数据
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row:
                cell.value = None

        if len(F) == 0:
            continue

        all_cols = dec_vars + obj_cols
        _apply_header(ws, 1, len(all_cols), st)

        for i in range(len(F)):
            Xi = X[i]
            Cu_out = float(F[i, 0])
            As_out = float(-F[i, 1])
            E_tot = float(F[i, 2])
            profit = float(-F[i, 3]) / 1e4

            # 根据工况类型重新计算电压值
            if mode_key == 'three_stage':
                Cu_in, T3, I3, Q3, t = Xi
                _, _, V = surrogate.predict_three_stage(
                    Cu_in, T3, I3, Q3, runtime['year'], runtime['month'], runtime['day'], runtime['hour'])
                V3 = float(V)
                V4 = 0.0
                Vavg = float(V)
                G = np.array([
                    Cu_out - PARAMS['Cu_limit'],
                    PARAMS['As_min'] - As_out,
                    I3 / (PARAMS['n_cathode'] * PARAMS['A_cathode']) - PARAMS['J_max'],
                    V3 / PARAMS['N_cell'] - PARAMS['V_cell_max'],
                    Cu_out / (As_out + 1e-8) - PARAMS['Cu_As_max']
                ])
            elif mode_key == 'four_stage':
                Cu_in, T4, I4, Q4, t = Xi
                _, _, V = surrogate.predict_four_stage(
                    Cu_in, T4, I4, Q4, runtime['year'], runtime['month'], runtime['day'], runtime['hour'])
                V3 = 0.0
                V4 = float(V)
                Vavg = float(V)
                G = np.array([
                    Cu_out - PARAMS['Cu_limit'],
                    PARAMS['As_min'] - As_out,
                    I4 / (PARAMS['n_cathode'] * PARAMS['A_cathode']) - PARAMS['J_max'],
                    V4 / PARAMS['N_cell'] - PARAMS['V_cell_max'],
                    Cu_out / (As_out + 1e-8) - PARAMS['Cu_As_max']
                ])
            else:  # serial
                Cu_in, T3, I3, Q3, T4, I4, Q4, t = Xi
                _, _, VA, VB, Va = surrogate.predict_serial(
                    Cu_in, T3, I3, Q3, T4, I4, Q4, runtime['year'], runtime['month'], runtime['day'], runtime['hour'])
                V3 = float(VA)
                V4 = float(VB)
                Vavg = float(Va)
                G = np.array([
                    Cu_out - PARAMS['Cu_limit'],
                    PARAMS['As_min'] - As_out,
                    max(I3, I4) / (PARAMS['n_cathode'] * PARAMS['A_cathode']) - PARAMS['J_max'],
                    max(V3, V4) / PARAMS['N_cell'] - PARAMS['V_cell_max'],
                    Cu_out / (As_out + 1e-8) - PARAMS['Cu_As_max']
                ])

            cost_val = float(-G.min())
            col_val = {vn: float(Xi[j]) for j, vn in enumerate(dec_vars)}

            # 更新目标值和辅助列
            if mode_key == 'three_stage':
                col_val.update({
                    'Cu_out': Cu_out,
                    'As_out': As_out,
                    'E_total': E_tot,
                    'Net profit\n(Ten thousand CNY)': profit,
                    'VA': V3,
                    'costs': cost_val
                })
            elif mode_key == 'four_stage':
                col_val.update({
                    'Cu_out': Cu_out,
                    'As_out': As_out,
                    'E_total': E_tot,
                    'Net profit\n(Ten thousand CNY)': profit,
                    'VB': V4,
                    'costs': cost_val
                })
            else:  # serial
                col_val.update({
                    'Cu_out': Cu_out,
                    'As_out': As_out,
                    'E_total': E_tot,
                    'Net profit\n(Ten thousand CNY)': profit,
                    'Vaverage': Vavg,
                    'costs': cost_val
                })

            for c_idx, col_name in enumerate(all_cols, start=1):
                _apply_data(ws.cell(row=i + 2, column=c_idx), col_val.get(col_name, ''), st)

        # 调整列宽
        for col in ws.columns:
            mx = max((len(str(cell.value or '')) for cell in col), default=8)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(mx + 4, 25)

    # 写入妥协解数据
    ws_c = wb['Compromise Solutions']
    for row in ws_c.iter_rows(min_row=2, max_row=ws_c.max_row):
        for cell in row:
            cell.value = None

    scene_en = {'quality': 'Quality priority', 'energy': 'Energy priority',
                'profit': 'Economic priority', 'balance': 'Balanced'}
    write_row = 2
    for mode_label, solutions in all_compromise.items():
        cond_str = cond_label_map.get(mode_label, mode_label)
        first = True
        for key, sol in solutions.items():
            Xi = sol['X']
            Cu_out = float(sol['F'][0])
            As_out = float(-sol['F'][1])
            E_tot = float(sol['F'][2])
            profit = float(-sol['F'][3]) / 1e4

            if len(Xi) == 8:
                Cu_in, T3, I3, Q3, T4, I4, Q4, t = Xi
                TB_v = float(T4)
                IB_v = float(I4)
                QB_v = float(Q4)
                TA_v = float(T3)
                IA_v = float(I3)
                QA_v = float(Q3)
            else:
                Cu_in, Tx, Ix, Qx, t = Xi
                if mode_label == '四段工况':
                    TB_v = float(Tx)
                    IB_v = float(Ix)
                    QB_v = float(Qx)
                    TA_v = None
                    IA_v = None
                    QA_v = None
                else:
                    TA_v = float(Tx)
                    IA_v = float(Ix)
                    QA_v = float(Qx)
                    TB_v = None
                    IB_v = None
                    QB_v = None

            row_vals = [cond_str if first else None, scene_en.get(key, key), float(Cu_in),
                        TA_v, IA_v, QA_v, TB_v, IB_v, QB_v, float(t),
                        Cu_out, As_out, E_tot, profit]
            for c_idx, val in enumerate(row_vals, start=1):
                _apply_data(ws_c.cell(row=write_row, column=c_idx), val, st, is_str=(c_idx <= 2))
            first = False
            write_row += 1

    # 写入 nsga_metrics 数据
    ws_m = wb['nsga_metrics']
    for row in ws_m.iter_rows(min_row=2, max_row=ws_m.max_row):
        for cell in row:
            cell.value = None
    for r, (ml, m) in enumerate(all_metrics.items(), start=2):
        for c, val in enumerate([cond_label_map.get(ml, ml), m.get('n_feasible', 0),
                                   m.get('HV'), m.get('Spacing')], start=1):
            _apply_data(ws_m.cell(row=r, column=c), val, st, is_str=(c == 1))

    # 写入 quality_metrics_nsga 数据
    ws_q = wb['quality_metrics_nsga']
    for row in ws_q.iter_rows(min_row=2, max_row=ws_q.max_row):
        for cell in row:
            cell.value = None
    for r, (ml, m) in enumerate(all_metrics.items(), start=2):
        vals = [cond_label_map.get(ml, ml), m.get('n_feasible', 0),
                m.get('Mean_Cu_out'), m.get('Std_Cu_out'), m.get('Range_Cu_out'),
                m.get('Mean_As_out'), m.get('Std_As_out'), m.get('Range_As_out'),
                m.get('Mean_E_total'), m.get('Std_E_total'), m.get('Range_E_total'),
                m.get('Mean_Net_profit'), m.get('Std_Net_profit'), m.get('Range_Net_profit')]
        for c, val in enumerate(vals, start=1):
            _apply_data(ws_q.cell(row=r, column=c), val, st, is_str=(c == 1))

    wb.save(output_path)
    print(f"\n NSGA_results.xlsx 已保存: {output_path}")


def save_solutions_to_excel(F_feasible, X_feasible, solutions, mode_label, var_names, output_dir):
    """保存解集到 Excel 文件"""
    os.makedirs(output_dir, exist_ok=True)
    file_path = f"{output_dir}/solutions_{mode_label}.xlsx"
    has_data = False

    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        if len(F_feasible) > 0:
            df = pd.DataFrame()
            for i, vn in enumerate(var_names):
                df[vn] = X_feasible[:, i]
            df['Cu_out (g/L)'] = F_feasible[:, 0]
            df['As_out (g/L)'] = -F_feasible[:, 1]
            df['E_total (kWh)'] = F_feasible[:, 2]
            df['R_profit (万元)'] = -F_feasible[:, 3] / 1e4
            df.to_excel(writer, sheet_name='可行解集', index=False)
            has_data = True

        if solutions:
            rows = []
            for key, sol in solutions.items():
                row = {'场景': {'quality': '质量优先', 'energy': '节能优先',
                                'profit': '经济优先', 'balance': '均衡优化'}[key]}
                for i, vn in enumerate(var_names):
                    row[vn] = sol['X'][i]
                row.update({'Cu_out(g/L)': sol['F'][0], 'As_out(g/L)': -sol['F'][1],
                            'E_total(kWh)': sol['F'][2], 'R_profit(万元)': -sol['F'][3] / 1e4})
                rows.append(row)
            pd.DataFrame(rows).to_excel(writer, sheet_name='最优妥协解', index=False)
            has_data = True

        if not has_data:
            pd.DataFrame({'提示': ['无可行解和最优解']}).to_excel(writer, sheet_name='空数据集', index=False)

    print(f" 解集保存: {file_path}")


def save_metrics_to_excel(all_metrics, output_dir):
    """保存评估指标到 Excel 文件"""
    os.makedirs(output_dir, exist_ok=True)
    rows = [{'工况': ml, '可行解数': m['n_feasible'], 'HV': m['HV'], 'Spacing': m['Spacing']}
            for ml, m in all_metrics.items()]
    pd.DataFrame(rows).to_excel(f"{output_dir}/metrics.xlsx", index=False)
    print(f" 评价指标保存")


def save_si_statistics(all_solutions, output_dir):
    """保存统计信息到 CSV 文件"""
    os.makedirs(output_dir, exist_ok=True)
    s1_rows, s3_rows = [], []
    obj_names = ['Cu_out', 'As_out', 'E_total', 'Net_profit']
    constr_names = ['G1_Cu_limit', 'G2_As_min', 'G3_J_max', 'G4_V_cell', 'G5_Cu_As_ratio']

    for cond, data in all_solutions.items():
        F = data.get('F')
        X = data.get('X')
        G = data.get('G')
        if F is None or len(F) == 0:
            continue
        vn = data.get('var_names', [f'x{i}' for i in range(X.shape[1])])

        # 目标函数统计
        F_d = F.copy()
        F_d[:, 1] = -F_d[:, 1]
        F_d[:, 3] = -F_d[:, 3] / 1e4

        for j, v in enumerate(vn):
            col = X[:, j]
            s1_rows.append({
                'Method': 'NSGA-II', 'Condition': cond, 'Variable': v,
                'Mean': float(col.mean()), 'Std': float(col.std()),
                'Min': float(col.min()), 'Q25': float(np.percentile(col, 25)),
                'Median': float(np.median(col)), 'Q75': float(np.percentile(col, 75)),
                'Max': float(col.max())
            })
        for k, on in enumerate(obj_names):
            col = F_d[:, k]
            s1_rows.append({
                'Method': 'NSGA-II', 'Condition': cond, 'Variable': on,
                'Mean': float(col.mean()), 'Std': float(col.std()),
                'Min': float(col.min()), 'Q25': float(np.percentile(col, 25)),
                'Median': float(np.median(col)), 'Q75': float(np.percentile(col, 75)),
                'Max': float(col.max())
            })

        # 约束条件统计
        if G is not None:
            for k, cn in enumerate(constr_names):
                col = G[:, k]
                viol = col[col > 0]
                s3_rows.append({
                    'Method': 'NSGA-II', 'Condition': cond, 'Constraint': cn,
                    'Violation_rate': float(len(viol) / max(len(col), 1)),
                    'Mean_violation': float(viol.mean()) if len(viol) > 0 else 0.0,
                    'Max_violation': float(viol.max()) if len(viol) > 0 else 0.0,
                    'Mean_slack': float((-col).mean())
                })

    if s1_rows:
        pd.DataFrame(s1_rows).to_csv(f'{output_dir}/table_s1_decision_variables.csv', index=False)
        print(f" Table S1 saved")
    if s3_rows:
        pd.DataFrame(s3_rows).to_csv(f'{output_dir}/table_s3_constraints.csv', index=False)
        print(f" Table S3 saved")


def save_compromise_csv(all_compromise, output_dir):
    """保存妥协解到 CSV 文件"""
    os.makedirs(output_dir, exist_ok=True)
    scene_en = {'quality': 'Quality Priority', 'energy': 'Energy Priority',
                'profit': 'Profit Priority', 'balance': 'Balanced'}
    rows = []

    for cond, solutions in all_compromise.items():
        for key, sol in solutions.items():
            row = {
                'Method': 'NSGA-II', 'Condition': cond, 'Scenario': scene_en.get(key, key),
                'Cu_out': float(sol['F'][0]), 'As_out': float(-sol['F'][1]),
                'E_total': float(sol['F'][2]), 'Net_profit': float(-sol['F'][3]) / 1e4
            }
            for j, vn in enumerate(sol.get('var_names', [])):
                row[vn] = float(sol['X'][j])
            rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(f'{output_dir}/table_s4_compromise_solutions.csv', index=False)
        print(f" Table S4 saved")


# ============================================================================
# 主入口函数
# ============================================================================

def main():
    """
    主函数：执行 NSGA-II 多目标优化

    支持的命令行参数：
        --fast: 使用快速模式（种群30，迭代50）
        --normal: 使用正常模式（种群300，迭代300）
        --all-feasible: 保存所有可行解（而非仅Pareto前沿）
    """
    print("=" * 60, flush=True)
    print(" NSGA-II 多目标优化程序启动 (向量化高性能版)", flush=True)
    print("=" * 60, flush=True)

    # 加载代理模型
    print("\n[阶段 1/5] 正在加载代理模型...", flush=True)
    global surrogate
    surrogate = SurrogateModel()
    print("[阶段 1/5] 代理模型加载完成！", flush=True)

    # 不绑定环境默认模式，由用户通过命令行参数确定
    FAST_MODE = False

    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='NSGA-II 多目标优化程序')
    parser.add_argument('--fast', action='store_true',
                        help='使用快速模式（小种群和迭代次数）')
    parser.add_argument('--normal', action='store_true',
                        help='使用正常模式（大种群和迭代次数）')
    parser.add_argument('--all-feasible', action='store_true',
                        help='保存所有可行解（而非仅Pareto前沿）')
    parser.add_argument('--n-fronts', type=int, default=3,
                        help='保留前n个非支配前沿（默认3，仅在非--all-feasible模式下生效）')
    parser.add_argument('--n-solutions', type=int, default=None,
                        help='指定最终保留的解数量（优先于--n-fronts和--all-feasible）')
    parser.add_argument('--pop-size', type=int, default=None,
                        help='种群大小（默认: fast=30, normal=900）')
    parser.add_argument('--n-gen', type=int, default=None,
                        help='迭代代数（默认: fast=50, normal=600）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认42）')
    args = parser.parse_args()

    # 命令行参数确定运行模式
    if args.fast:
        FAST_MODE = True
    elif args.normal:
        FAST_MODE = False

    # 设置算法参数（从全局配置读取）
    pop_size = NSGA2_PARAMS['pop_size']
    n_gen = NSGA2_PARAMS['n_gen']
    seed = NSGA2_PARAMS['seed']
    n_solutions = NSGA2_PARAMS['n_solutions']
    n_fronts = NSGA2_PARAMS['n_fronts']
    SAVE_ALL_FEASIBLE = NSGA2_PARAMS['save_all_feasible']

    # 设置随机种子
    np.random.seed(seed)

    # 获取当前时间作为运行时参数
    _now = datetime.now()
    runtime = {
        'year': _now.year,
        'month': _now.month,
        'day': _now.day,
        'hour': _now.hour
    }

    # 打印运行配置信息
    print(f"\n{'=' * 60}")
    print(" 运行配置信息")
    print(f"{'=' * 60}")
    print(f" 时间戳: {_now.strftime('%Y-%m-%d %H:%M')}")
    print(f" 运行环境: {'本地 Windows' if sys.platform == 'win32' else 'HPC'}")
    print(f" 运行模式: {'快速模式' if FAST_MODE else '正常模式'}")
    print(f" 保存模式: {'所有可行解' if SAVE_ALL_FEASIBLE else '仅Pareto前沿'}")
    print(f" 种群大小: {pop_size}")
    print(f" 迭代代数: {n_gen}")
    print(f" 随机种子: {seed}")
    if n_solutions:
        print(f" 目标解数量: {n_solutions}")
    else:
        print(f" 保留前沿数: {n_fronts}")
    print(f" Table S7 约束: Cu_limit={PARAMS['Cu_limit']}, As_min={PARAMS['As_min']}, "
          f"J_max={PARAMS['J_max']}, V_cell_max={PARAMS['V_cell_max']}, Cu_As_max={PARAMS['Cu_As_max']}")
    print(f"{'=' * 60}\n")

    # 定义三种工况的配置
    all_metrics = {}
    all_solutions = {}
    all_compromise = {}
    configs = [
        (ThreeStageOptProblem(runtime), '三段工况',
         ['Cu_in(g/L)', 'T3(℃)', 'I3(A)', 'Q3(m³/h)', 't(h)'],
         ['Cu_in', 'TA', 'IA', 'QA', 't']),
        (FourStageOptProblem(runtime), '四段工况',
         ['Cu_in(g/L)', 'T4(℃)', 'I4(A)', 'Q4(m³/h)', 't(h)'],
         ['Cu_in', 'TB', 'IB', 'QB', 't']),
        (SerialOptProblem(runtime), '串联工况',
         ['Cu_in(g/L)', 'T3(℃)', 'I3(A)', 'Q3(m³/h)', 'T4(℃)', 'I4(A)', 'Q4(m³/h)', 't(h)'],
         ['Cu_in', 'TA', 'IA', 'QA', 'TB', 'IB', 'QB', 't']),
    ]

    print(f"[阶段 2/5] 开始三种工况的优化计算...\n", flush=True)

    # 依次优化三种工况
    for idx, (problem, mode_label, print_vars, var_names) in enumerate(configs, 1):
        print(f"\n>>> 正在处理第 {idx}/3 个工况: {mode_label} <<<")
        result, F, X, G_feas, metrics = run_nsga2(
            problem, mode_label, pop_size, n_gen, seed,
            verbose=False, save_history=True, save_all_feasible=SAVE_ALL_FEASIBLE,
            n_fronts=n_fronts, n_solutions=n_solutions
        )

        # 保存结果
        all_metrics[mode_label] = metrics
        all_solutions[mode_label] = {'F': F, 'X': X, 'G': G_feas, 'var_names': var_names}

        # 生成可视化
        tag = mode_label.replace('工况', '')
        plot_pareto_3d(F, mode_label, f'{OUTPUT_DIR}/pareto_3d_{tag}.png')
        plot_pareto_2d_projections(F, mode_label, f'{OUTPUT_DIR}/pareto_2d_{tag}.png')
        plot_convergence(result, mode_label, f'{OUTPUT_DIR}/convergence_{tag}.png')

        # 选择妥协解
        sol = select_compromise_solutions(F, X, mode_label)
        print_compromise_solutions(sol, mode_label, print_vars)
        for s in sol.values():
            s['var_names'] = var_names
        all_compromise[mode_label] = sol

        # 保存解集
        save_solutions_to_excel(F, X, sol, mode_label, print_vars, OUTPUT_DIR)
        print(f" {mode_label} 处理完成 ({idx}/3)\n")

    # 打印汇总信息
    print(f"[阶段 3/5] 所有优化完成，正在生成汇总结果...")
    print(f"\n{'=' * 60}")
    print(f" {'工况':<10} {'可行解':>8} {'HV':>12} {'Spacing':>14}")
    print(f" {'-' * 48}")
    for ml, m in all_metrics.items():
        hv = f"{m['HV']:.4f}" if m['HV'] is not None else 'N/A'
        sp = f"{m['Spacing']:.4f}" if m['Spacing'] is not None else 'N/A'
        print(f" {ml:<10} {m['n_feasible']:>8} {hv:>12} {sp:>14}")

    # 保存评估指标和统计数据
    print(f"[阶段 4/5] 正在保存评估指标和统计数据...")
    save_metrics_to_excel(all_metrics, OUTPUT_DIR)
    save_si_statistics(all_solutions, OUTPUT_DIR)
    save_compromise_csv(all_compromise, OUTPUT_DIR)

    # 生成最终 Excel 文件
    print(f"[阶段 5/5] 正在生成 NSGA_results.xlsx ...")
    save_nsga_excel(
        all_solutions, all_compromise, all_metrics, runtime,
        template_path='NSGA.xlsx', output_path=f'{OUTPUT_DIR}/NSGA_results.xlsx'
    )

    print(f"\n{'=' * 60}")
    print(f" 所有任务已完成！输出目录: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
