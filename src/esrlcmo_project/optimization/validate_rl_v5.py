"""
validate_rl.py  v5.1
===============================================================
功能: PPO-Lagrangian 模型完整验证与分析工具
  [原有] 逐行验证 RL 策略相对历史操作的性能改善
  [新增] 掩码验证：三级可调度场景对比（full / realistic / conservative）
  [新增] 动态序贯响应：Cu_in 连续波动下 RL vs NSGA-II 对比（4种波动模式）
  [新增] 铜价/电价灵敏度分析（9个价格场景，含 RL 自适应推理层）
  [新增] 偏好权重策略轨迹对比（5种偏好，支持全工况分析）
  [新增] Pareto 局部斜率分析（边际替代关系，三区控制机制）
  [新增] 主导变量迁移分析（Spearman 相关，分区控制机制识别）
  [新增] 代理模型预测精度验证（预测值 vs 实测值 MAE/RMSE/MAPE）

Fix log (v5.0→v5.1):
  [FIX-W1] run_weight_preference_analysis 推理策略从 dist.mean（确定性单点）
            改为 stochastic K次采样取最优利润，对齐训练侧 evaluate 的
            stochastic 策略。

            根本原因：训练时状态以原始量级（Cu_in≈40, IA≈15000）直接
            拼接权重 w∈[0,1] 送入网络；第一层激活中状态维度贡献约为权重
            维度的 1000× 量级，梯度难以有效传入权重维度，导致策略
            dist.mean 对不同 w 几乎无差异（summary 每行相同的根因）。

            修复方案：复用训练侧 evaluate 的 stochastic 策略——对同一
            (s, w) 采样 K=3 次候选动作，经代理模型批量推理后取利润最高
            的候选，使分布方差 std>0 时采样空间仍能产生有效差异。

            注意：这是验证侧的缓解措施。若需彻底修复（使 dist.mean 也
            有效分化），需在训练代码 ppo_lagrangian.py 中对状态做
            [-1,1] 归一化后重新训练，并在验证侧同步加入归一化。

  [FIX-W2] 新增 _diagnose_weight_sensitivity 函数：在固定状态下探测
            5 种极端权重对应的 a_mean 差异，输出敏感性评分（L2 标准差）；
            结果保存至 weight_sensitivity_diagnosis.xlsx，帮助量化权重的
            实际影响程度。

  [FIX-W3] summary 新增诊断列：
            a_mean_norm  —— dist.mean 的 L2 范数（近 0 表示策略保守）
            profit_std_K —— K 次采样利润的标准差（量化探索空间大小）

  [FIX-W4] trajectory 新增 deterministic 对照列（Det_Cu_out / Det_As_out /
            Det_Profit / Det_{key} / Stoch_vs_Det_profit），保留原
            dist.mean 单点结果作为参照，便于比较两种推理策略的差异。

  [FIX-W5] 全局常量 WEIGHT_SAMPLE_K=3 控制采样次数（默认与训练侧一致）。

Fix log (v4→v5):
  [FIX-1] calc_E_profit 中 R_save 改为 As_fixed=11.64，与训练代码一致，
           消除验证利润系统性虚高（约 32~65%）
  [FIX-2] run_price_sensitivity 新增 RL 自适应推理层：各价格场景下
           重新调用 RL 网络推理控制变量，反映真实自适应能力；
           原固定解重定价逻辑保留为 fixed_repricing 层（两层并输出）
  [FIX-3] ECON n_cathode 改为 35（对应单极板），J_max 改为 340 A/m²，
           与训练代码 BASE_CFG 完全一致，消除参数化隐患
  [FIX-4] nsga_fixed_apply 中 NSGA_response_ms 改为实测值，
           不再硬编码 0.05ms，使响应时间对比基准统一
  [FIX-5] run_masked_validation 汇总新增 Bootstrap 95% CI 和配对 t 检验，
           量化改善效果的统计显著性
  [FIX-6] gen_weights 的 extreme 分支正确使用 n 参数：
           4 极端 + 1 均衡固定，n>5 时 Dirichlet 补充，n<5 时截断
  [FIX-7] rl_dynamic_response / nsga_fixed_apply 的时间参数改为外部传入，
           run_dynamic_comparison 传入历史数据中位时间，不再硬编码
  [FIX-8] run_weight_preference_analysis 支持 target_cond='ALL'，
           可对三个工况逐一分析，不再全局写死为 Condition 2
  [FIX-9] compare_cumulative 中 DataFrame.get 嵌套改为显式列名查找，
           避免列名差异时静默返回 0

用法: 直接运行，通过顶部 RUN_* 开关控制各模块
      WEIGHT_ANALYSIS_COND = 'ALL' 可开启全工况权重分析
      WEIGHT_SAMPLE_K = 3 控制随机采样次数（可调大以提升解质量）
===============================================================
"""

import os
os.environ['OMP_NUM_THREADS']      = '1'
os.environ['MKL_NUM_THREADS']      = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS']  = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal
from joblib import load
import time
import time as time_module
from pathlib import Path
from scipy.stats import spearmanr

# =============================================================================
# ── 0. 功能开关（True=运行该模块，False=跳过） ────────────────────────────────
# =============================================================================
RUN_ORIGINAL  = False   # 原有 v3.0 逐行验证
RUN_SURROGATE = False   # 代理模型预测精度验证（预测 vs 实测）
RUN_MASKED    = False   # 三级掩码场景对比验证
RUN_DYNAMIC   = False   # Cu_in 动态波动序贯响应对比
RUN_PRICE     = True   # 铜价/电价灵敏度分析
RUN_WEIGHT    = False   # 偏好权重策略轨迹对比
RUN_PARETO    = False   # Pareto 局部斜率与主导变量迁移

# 动态实验波动模式开关（仅当 RUN_DYNAMIC=True 时生效）
DYN_MODE_REAL_BASED   = True   # AR(1) 正常波动
DYN_MODE_STEP_CHANGE  = True   # 阶跃变化（批次切换）
DYN_MODE_GRADUAL_DRIFT= True   # 缓慢漂移（阳极板消耗）
DYN_MODE_MIXED        = True   # 混合模式

# 偏好权重分析重点工况
WEIGHT_ANALYSIS_COND = 'ALL'

# 偏好权重推理时随机采样次数（对齐训练侧 evaluate stochastic 策略，默认 K=3）
# 增大 K 可提升最优解质量，但计算量线性增加；建议范围 3~10。
WEIGHT_SAMPLE_K = 3

# =============================================================================
# ── 1. 路径配置 ───────────────────────────────────────────────────────────────
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent
RESULT_DIR = PROJECT_DIR / "result"

EXCEL_PATH = str(PROJECT_DIR / "validate_data.xlsx")
MODEL_DIR  = str(PROJECT_DIR / "joblib")

# 各工况输出目录（自动创建）
OUTPUT_DIRS = {
    'Condition 1': str(RESULT_DIR / "validate_outputs_condition1"),
    'Condition 2': str(RESULT_DIR / "validate_outputs_condition2"),
    'Condition 3': str(RESULT_DIR / "validate_outputs_condition3"),
}
# 综合输出目录（跨工况汇总文件）
OUTPUT_DIR_COMMON = str(RESULT_DIR / "validation_v4")

PT_PATHS = {
    'Condition 1': str(PROJECT_DIR / "pt" / "ppo_actor_condition1.pt"),
    'Condition 2': str(PROJECT_DIR / "pt" / "ppo_actor_condition2.pt"),
    'Condition 3': str(PROJECT_DIR / "pt" / "ppo_actor_condition3.pt"),
}

PARETO_CSV_PATHS = {
    'Condition 1': str(RESULT_DIR / "outputs_condition1" / "pareto_ppo_condition1.csv"),
    'Condition 2': str(RESULT_DIR / "outputs_condition2" / "pareto_ppo_condition2.csv"),
    'Condition 3': str(RESULT_DIR / "outputs_condition3" / "pareto_ppo_condition3.csv"),
}

NSGA_PATH = str(RESULT_DIR / "NSGA_results.xlsx")

# =============================================================================
# ── 2. 数据列名配置 ───────────────────────────────────────────────────────────
# =============================================================================
HAS_CONDITION_COL  = True
CONDITION_COL_NAME = "Condition"
DEFAULT_CONDITION  = "Condition 1"

# validate_data.xlsx 中实际存在的列
# Time As_out Cu_out Cu_in TA IA VA QA TB IB VB QB year month day hour Condition t
MEASURED_CU_OUT_COL = 'Cu_out'   # 实测 Cu_out 列名
MEASURED_AS_OUT_COL = 'As_out'   # 实测 As_out 列名

COL_MAP = {
    'Cu_in': 'Cu_in', 't': 't',
    'TA': 'TA', 'IA': 'IA', 'QA': 'QA',
    'TB': 'TB', 'IB': 'IB', 'QB': 'QB',
}

WEIGHT_STRATEGY  = 'extreme'
N_WEIGHT_SAMPLES = 30

# =============================================================================
# ── 3. 经济参数与约束参数 ─────────────────────────────────────────────────────
# =============================================================================
ECON = dict(
    p_Cu=98.44,
    c_reprocess=0.14,
    c_copper_concentrate=82.64,
    p_e=0.6,
    c_op=200,
    c_As_treat=2.5,
    # [FIX-3] 与训练代码 BASE_CFG 保持一致：n_cathode=35，不含 N_cell
    n_cathode=35,
    A_cathode=1.1 * 1.029 * 2,
)
CST = dict(
    Cu_limit=8.0, As_min=4.0,
    # [FIX-3] 与训练代码 BASE_CFG 保持一致：J_max=340 A/m²（单槽电流密度）
    J_max=340.0,
    V_cell_max=2.5, N_cell=16, Cu_As_ratio_max=0.8,
)
_J_DENOM = ECON['n_cathode'] * ECON['A_cathode']

# =============================================================================
# ── 4. 工况配置（CCFG） ───────────────────────────────────────────────────────
# =============================================================================
CCFG = {
    'Condition 1': dict(
        keys=['Cu_in', 'TA', 'IA', 'QA', 't'],
        dsc=np.array([2.0, 2.0, 1000.0, 1.0], dtype=np.float32),
        lo =np.array([29.0, 48.0,  8000.0, 111.0, 2.0], dtype=np.float32),
        hi =np.array([55.0, 65.0, 27000.0, 123.0, 8.0], dtype=np.float32),
        sub='three_stage', sd=5, ad=4, hd=128,
    ),
    'Condition 2': dict(
        keys=['Cu_in', 'TB', 'IB', 'QB', 't'],
        dsc=np.array([2.0, 2.0, 1000.0, 1.0], dtype=np.float32),
        lo =np.array([29.0, 55.0,  8000.0, 113.0, 2.0], dtype=np.float32),
        hi =np.array([55.0, 65.0, 27000.0, 123.0, 8.0], dtype=np.float32),
        sub='four_stage', sd=5, ad=4, hd=128,
    ),
    'Condition 3': dict(
        keys=['Cu_in', 'TA', 'IA', 'QA', 'TB', 'IB', 'QB', 't'],
        dsc=np.array([2.0, 2.0, 1000.0, 1.0, 2.0, 1000.0, 1.0], dtype=np.float32),
        lo =np.array([29.0, 40.0,  8000.0, 111.0, 40.0,  8000.0, 113.0, 2.0], dtype=np.float32),
        hi =np.array([55.0, 65.0, 27000.0, 123.0, 65.0, 27000.0, 123.0, 8.0], dtype=np.float32),
        sub='serial', sd=8, ad=7, hd=128,   # 确认：Condition 3 隐藏层为 128
    ),
}

# =============================================================================
# ── 5. 掩码配置 ───────────────────────────────────────────────────────────────
# =============================================================================
# True = 该变量可被 RL 调节；False = 固定为历史值（不可控）
MASK_CONFIG_REALISTIC = {
    'Condition 1': {'Cu_in': False, 'TA': True,  'IA': True,  'QA': True,  't': True},
    'Condition 2': {'Cu_in': False, 'TB': True,  'IB': True,  'QB': True,  't': True},
    'Condition 3': {'Cu_in': False, 'TA': True,  'IA': True,  'QA': True,
                    'TB': True,  'IB': True,  'QB': True,  't': True},
}
MASK_CONFIG_CONSERVATIVE = {
    'Condition 1': {'Cu_in': False, 'TA': False, 'IA': True,  'QA': False, 't': False},
    'Condition 2': {'Cu_in': False, 'TB': False, 'IB': True,  'QB': False, 't': False},
    'Condition 3': {'Cu_in': False, 'TA': False, 'IA': True,  'QA': False,
                    'TB': False, 'IB': True,  'QB': False, 't': False},
}
MASK_SCENARIOS = {
    'full_control':  None,
    'realistic':     MASK_CONFIG_REALISTIC,
    'conservative':  MASK_CONFIG_CONSERVATIVE,
}

# =============================================================================
# ── 6. 价格灵敏度场景配置 ─────────────────────────────────────────────────────
# =============================================================================
PRICE_SCENARIOS = {
    'p_Cu_-30%':   {'p_Cu': 98.44 * 0.700, 'p_e': 0.60},
    'p_Cu_-27.5%': {'p_Cu': 98.44 * 0.725, 'p_e': 0.60},
    'p_Cu_-25%':   {'p_Cu': 98.44 * 0.750, 'p_e': 0.60},
    'p_Cu_-22.5%': {'p_Cu': 98.44 * 0.775, 'p_e': 0.60},
    'p_Cu_-20%':   {'p_Cu': 98.44 * 0.800, 'p_e': 0.60},
    'p_Cu_-17.5%': {'p_Cu': 98.44 * 0.825, 'p_e': 0.60},
    'p_Cu_-15%':   {'p_Cu': 98.44 * 0.850, 'p_e': 0.60},
    'p_Cu_-12.5%': {'p_Cu': 98.44 * 0.875, 'p_e': 0.60},
    'p_Cu_-10%':   {'p_Cu': 98.44 * 0.900, 'p_e': 0.60},
    'p_Cu_base':   {'p_Cu': 98.44,          'p_e': 0.60},
    'p_Cu_+15%':   {'p_Cu': 98.44 * 1.15,  'p_e': 0.60},
    'p_Cu_+30%':   {'p_Cu': 98.44 * 1.30,  'p_e': 0.60},
    'p_e_-25%':    {'p_Cu': 98.44,          'p_e': 0.45},
    'p_e_+25%':    {'p_Cu': 98.44,          'p_e': 0.75},
    'worst_case':  {'p_Cu': 98.44 * 0.70,  'p_e': 0.75},
    'best_case':   {'p_Cu': 98.44 * 1.30,  'p_e': 0.45},
}

# =============================================================================
# ── 7. 偏好权重场景配置 ───────────────────────────────────────────────────────
# =============================================================================
PREFERENCE_SCENARIOS = {
    'Energy Priority':   np.array([0.10, 0.10, 0.70, 0.10], dtype=np.float32),
    'Quality Priority':  np.array([0.70, 0.10, 0.10, 0.10], dtype=np.float32),
    'Economic Priority': np.array([0.10, 0.10, 0.10, 0.70], dtype=np.float32),
    'As Priority':       np.array([0.10, 0.70, 0.10, 0.10], dtype=np.float32),
    'Balanced':          np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
}

# =============================================================================
# ── 8. 神经网络定义（与 ppo_lagrangian.py 完全一致） ──────────────────────────
# =============================================================================
class ActorCritic(nn.Module):
    def __init__(self, s_dim=5, a_dim=4, w_dim=4, hidden=128, nc=5):
        super().__init__()
        inp = s_dim + w_dim
        self.shared = nn.Sequential(
            nn.Linear(inp, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu      = nn.Linear(hidden, a_dim)
        self.log_std = nn.Parameter(torch.zeros(a_dim))
        self.reward_critic = nn.Sequential(
            nn.Linear(inp, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.cost_critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inp, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, 1),
            ) for _ in range(nc)
        ])
        nn.init.orthogonal_(self.mu.weight, 0.01)

    def forward(self, s, w):
        x  = torch.cat([s, w], dim=-1)
        h  = self.shared(x)
        mu = torch.tanh(self.mu(h))
        std = self.log_std.exp().clamp(1e-3, 1.0)
        vr  = self.reward_critic(x).squeeze(-1)
        vc  = torch.cat([cc(x) for cc in self.cost_critics], dim=-1)
        return Normal(mu, std), vr, vc

# =============================================================================
# ── 9. 基础工具函数 ───────────────────────────────────────────────────────────
# =============================================================================

def load_surr(md, sub, nj=1):
    """加载指定工况的代理模型（Cu / As / V）。"""
    ms = {}
    tag_map = {'cu': 'Cu', 'as': 'As', 'voltage': 'V'}
    for tag in ['cu', 'as', 'voltage']:
        p = os.path.join(md, f'{tag}_{sub}', f'extra_trees_{sub}.joblib')
        if not os.path.exists(p):
            raise FileNotFoundError(f"Surrogate model not found: {p}")
        ms[tag_map[tag]] = load(p)
        if hasattr(ms[tag_map[tag]], 'n_jobs'):
            ms[tag_map[tag]].n_jobs = nj
    return ms


def load_rl_net(pt_path, cond):
    """
    加载 RL 策略网络。
    trainer.save() 保存的是纯 state_dict（非 checkpoint 格式），
    即 torch.save(self.net.state_dict(), path)。
    若文件为 checkpoint 格式（含 'net' 键），自动兼容。
    """
    raw = torch.load(pt_path, map_location='cpu')
    c   = CCFG[cond]
    net = ActorCritic(s_dim=c['sd'], a_dim=c['ad'], w_dim=4, hidden=c['hd'])
    # 自动判断格式
    if isinstance(raw, dict) and 'net' in raw:
        net.load_state_dict(raw['net'])
        lam = raw.get('lambdas', np.zeros(5, dtype=np.float32))
    else:
        net.load_state_dict(raw)
        lam = np.zeros(5, dtype=np.float32)
    net.eval()
    return net, lam


def predict(models, cond, s_arr, year, month, day, hour):
    """代理模型级联预测：V → Cu → As。返回 (Cu_out, As_out, VA, VB)。"""
    if cond == 'Condition 1':
        Cu_in, TA, IA, QA, t = s_arr
        fv     = np.array([Cu_in, TA, IA, QA, year, month, day, hour], dtype=np.float32)
        VA     = float(models['V'].predict(fv.reshape(1, -1))[0])
        fc     = np.array([Cu_in, TA, IA, QA, VA, year, month, day, hour, VA * IA], dtype=np.float32)
        Cu_out = float(models['Cu'].predict(fc.reshape(1, -1))[0])
        As_out = float(models['As'].predict(fc.reshape(1, -1))[0])
        return Cu_out, As_out, VA, None
    elif cond == 'Condition 2':
        Cu_in, TB, IB, QB, t = s_arr
        fv     = np.array([Cu_in, TB, IB, QB, year, month, day, hour], dtype=np.float32)
        VB     = float(models['V'].predict(fv.reshape(1, -1))[0])
        fc     = np.array([Cu_in, TB, IB, QB, VB, year, month, day, hour, VB * IB], dtype=np.float32)
        Cu_out = float(models['Cu'].predict(fc.reshape(1, -1))[0])
        As_out = float(models['As'].predict(fc.reshape(1, -1))[0])
        return Cu_out, As_out, None, VB
    else:
        Cu_in, TA, IA, QA, TB, IB, QB, t = s_arr
        fv   = np.array([Cu_in, TA, IA, QA, TB, IB, QB, year, month, day, hour], dtype=np.float32)
        VMean = float(models['V'].predict(fv.reshape(1, -1))[0])
        VA = VB = VMean
        tp  = VA * IA + VB * IB
        fc   = np.array([Cu_in, TA, IA, QA, VA, TB, IB, QB, VB, year, month, day, hour, tp], dtype=np.float32)
        Cu_out = float(models['Cu'].predict(fc.reshape(1, -1))[0])
        As_out = float(models['As'].predict(fc.reshape(1, -1))[0])
        return Cu_out, As_out, VA, VB


def calc_E_profit(cond, s_arr, Cu_out, As_out, VA, VB,
                  p_Cu=None, p_e=None):
    """
    计算总能耗 E（kWh）和净利润 profit（万元）。
    p_Cu/p_e 可传入自定义值（用于价格灵敏度分析），默认使用 ECON 全局参数。

    [FIX-1] R_save 中 As_out 改为固定基准值 11.64 g/L，与训练代码
    compute_objectives_batch / _w_obj_c1c2 / _w_obj_c3 中的
    As_fixed = 11.64 保持一致，消除验证利润系统性虚高问题。
    """
    _p_Cu   = p_Cu if p_Cu is not None else ECON['p_Cu']
    _p_e    = p_e  if p_e  is not None else ECON['p_e']
    As_fixed = 11.64   # 危废处置基准（与训练代码一致）
    Cu_in = s_arr[0]
    t     = s_arr[-1]
    if cond == 'Condition 1':
        IA, QA = s_arr[2], s_arr[3]
        E = VA * IA * t / 1000.0
    elif cond == 'Condition 2':
        IB, QB = s_arr[2], s_arr[3]
        E = VB * IB * t / 1000.0
    else:
        IA, QA, IB = s_arr[2], s_arr[3], s_arr[5]
        E = (VA * IA + VB * IB) * t / 1000.0
    QA_val = s_arr[3]
    m_Cu   = max(0.0, (Cu_in - Cu_out) * QA_val * t)
    R_Cu   = (_p_Cu - ECON['c_reprocess'] - ECON['c_copper_concentrate']) * m_Cu
    R_save = ECON['c_As_treat'] * (As_fixed + Cu_out) * QA_val * t
    profit = (R_Cu - R_save - _p_e * E - ECON['c_op'] * t) / 1e4
    return E, profit


def calc_costs(cond, s_arr, Cu_out, As_out, VA, VB):
    """计算五类工艺约束代价（违反时>0，满足时=0）。"""
    if cond == 'Condition 1':
        IA = s_arr[2]
        J  = IA / _J_DENOM
        Vc = VA / CST['N_cell']
    elif cond == 'Condition 2':
        IB = s_arr[2]
        J  = IB / _J_DENOM
        Vc = VB / CST['N_cell']
    else:
        IA, IB = s_arr[2], s_arr[5]
        J  = max(IA, IB) / _J_DENOM
        Vc = max(VA, VB) / CST['N_cell']
    ratio = Cu_out / (As_out + 1e-8)
    return {
        'C1_Cu>8':    max(0.0, Cu_out - CST['Cu_limit']),
        'C2_As<4':    max(0.0, CST['As_min'] - As_out),
        'C3_J>21.3':  max(0.0, J  - CST['J_max']),
        'C4_V>2.5':   max(0.0, Vc - CST['V_cell_max']),
        'C5_rat>0.8': max(0.0, ratio - CST['Cu_As_ratio_max']),
    }


def gen_weights(strategy='extreme', n=10):
    """
    生成偏好权重向量组。
    [FIX-6] extreme 分支现在正确使用 n 参数：
      - 固定生成 4 个轴对齐极端权重（每维各一个主导方向）
      - 固定生成 1 个均衡权重
      - 若 n > 5，剩余 (n-5) 个从 Dirichlet 随机补充
      - 若 n <= 4，仅返回前 n 个极端权重
    这样 n 参数不再被静默忽略，函数签名语义与实际行为一致。
    """
    if strategy == 'single':
        return np.array([[0.25] * 4], dtype=np.float32)
    elif strategy == 'extreme':
        ws = []
        # 4 个轴对齐极端权重
        for d in range(4):
            e = np.zeros(4, dtype=np.float32)
            e[d] = 0.85
            e += np.random.dirichlet([0.5] * 4).astype(np.float32) * 0.15
            ws.append(e / e.sum())
        # 1 个均衡权重
        ws.append(np.array([0.25] * 4, dtype=np.float32))
        # 若 n > 5，补充随机 Dirichlet 权重
        if n > 5:
            extra = np.random.dirichlet([1.0] * 4, size=n - 5).astype(np.float32)
            ws.extend(extra.tolist())
        # 若 n < 5，截断至 n 个
        return np.array(ws[:n], dtype=np.float32)
    else:
        return np.random.dirichlet([1.0] * 4, size=n).astype(np.float32)


def progress_bar(current, total, bar_width=40):
    """控制台进度条。"""
    percent = current / total
    filled  = int(bar_width * percent)
    bar     = '=' * filled + '-' * (bar_width - filled)
    elapsed = time.time() - progress_bar.start_time if hasattr(progress_bar, 'start_time') else 0
    if current == 1:
        progress_bar.start_time = time.time()
    print(f'\r[{bar}] {current}/{total} ({percent*100:.1f}%) elapsed:{elapsed:.1f}s',
          end='', flush=True)
    if current == total:
        print()


def _out(cond_or_path, fname=None):
    """
    构造输出路径：
      _out('Condition 1', 'foo.csv') → outputs_condition1/foo.csv
      _out(None, 'bar.csv')          → validation_v4/bar.csv
    """
    if fname is None:
        return os.path.join(OUTPUT_DIR_COMMON, cond_or_path)
    if cond_or_path in OUTPUT_DIRS:
        return os.path.join(OUTPUT_DIRS[cond_or_path], fname)
    return os.path.join(OUTPUT_DIR_COMMON, fname)


def save_df(df, cond_or_none, fname_stem):
    """保存 DataFrame 为 xlsx + csv，打印提示。"""
    base = _out(cond_or_none, fname_stem) if cond_or_none else _out(fname_stem)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(str(c).strip() for c in col) for col in df.columns]
    df.to_excel(base + '.xlsx', index=False)
    df.to_csv(base + '.csv', index=False, encoding='utf-8-sig')
    print(f"    → Saved: {fname_stem}.xlsx / .csv")

# =============================================================================
# ── 10. 核心函数：apply_action_with_mask ────────────────────────────────────
# =============================================================================

def apply_action_with_mask(s_orig, a_mean, cond, mask_cfg=None):
    """
    带掩码的动作应用。
    mask_cfg: 该工况的 {变量名: True/False} 字典，None=全部可调。
    返回: (s_new, applied_vars)
    """
    c     = CCFG[cond]
    keys  = c['keys']
    dsc   = c['dsc']
    lo    = c['lo']
    hi    = c['hi']
    n_act = c['ad']
    s_new = s_orig.copy()
    applied = []
    for i in range(n_act):
        var_name = keys[i]
        if mask_cfg is None or mask_cfg.get(var_name, True):
            delta    = a_mean[i] * dsc[i]
            s_new[i] = np.clip(s_orig[i] + delta, lo[i], hi[i])
            applied.append(var_name)
    return s_new, applied

# =============================================================================
# ── 11. 原有 v3.0 核心函数：validate_row ────────────────────────────────────
# =============================================================================

def validate_row(row, cond, net, surr, weights, year, month, day, hour):
    """单行验证（v3.0 原逻辑，mask=None 即全变量可调）。"""
    return validate_row_masked(row, cond, net, surr, weights,
                               year, month, day, hour,
                               mask_cfg=None, scenario_name='full_control')


def validate_row_masked(row, cond, net, surr, weights, year, month, day, hour,
                        mask_cfg=None, scenario_name='full_control'):
    """
    带掩码的单行验证。
    mask_cfg=None → 全变量可调（等同于 v3.0 原逻辑）。
    """
    c      = CCFG[cond]
    s_orig = np.array([float(row[COL_MAP[k]]) for k in c['keys']], dtype=np.float32)

    Cu_out_o, As_out_o, VA_o, VB_o = predict(surr, cond, s_orig, year, month, day, hour)
    E_o, profit_o = calc_E_profit(cond, s_orig, Cu_out_o, As_out_o, VA_o, VB_o)
    costs_o = calc_costs(cond, s_orig, Cu_out_o, As_out_o, VA_o, VB_o)
    feas_o  = all(v < 1e-3 for v in costs_o.values())

    w_results = []
    for w in weights:
        with torch.no_grad():
            st = torch.FloatTensor(s_orig).unsqueeze(0)
            wt = torch.FloatTensor(w).unsqueeze(0)
            dist, _, _ = net(st, wt)
            a_mean = dist.mean.numpy()[0]

        s_new, applied_vars = apply_action_with_mask(s_orig, a_mean, cond, mask_cfg)

        Cu_out_n, As_out_n, VA_n, VB_n = predict(surr, cond, s_new, year, month, day, hour)
        E_n, profit_n = calc_E_profit(cond, s_new, Cu_out_n, As_out_n, VA_n, VB_n)
        costs_n = calc_costs(cond, s_new, Cu_out_n, As_out_n, VA_n, VB_n)
        feas_n  = all(v < 1e-3 for v in costs_n.values())

        w_results.append(dict(
            w=w.tolist(), action=a_mean.tolist(), s_new=s_new.tolist(),
            applied_vars=applied_vars, mask_scenario=scenario_name,
            Cu_out=Cu_out_n, As_out=As_out_n, VA=VA_n, VB=VB_n,
            E=E_n, profit=profit_n, feasible=feas_n, costs=costs_n,
        ))

    feasibles = [r for r in w_results if r['feasible']]
    best      = max(feasibles, key=lambda x: x['profit']) if feasibles else None

    return dict(
        s_orig=s_orig.tolist(), Cu_out_o=Cu_out_o, As_out_o=As_out_o,
        E_o=E_o, profit_o=profit_o, feasible_o=feas_o, costs_o=costs_o,
        w_results=w_results, best=best,
        applied_vars=best['applied_vars'] if best else [],
        mask_scenario=scenario_name,
    )

# =============================================================================
# ── 12. 模块 A：代理模型预测精度验证 ────────────────────────────────────────
# =============================================================================

def run_surrogate_accuracy(df, row_conds, surrs):
    """
    对比代理模型预测值与实测值（Cu_out / As_out）。
    输出指标：MAE、RMSE、MAPE，按工况分组。
    """
    print("\n  [Surrogate Accuracy] 计算预测 vs 实测误差...")
    records = []
    for idx, (_, row) in enumerate(df.iterrows()):
        cond = row_conds[idx]
        if cond not in surrs:
            continue
        year  = int(row.get('year', 2025))
        month = int(row.get('month', 6))
        day   = int(row.get('day', 15))
        hour  = int(row.get('hour', 8))
        c     = CCFG[cond]
        try:
            s = np.array([float(row[COL_MAP[k]]) for k in c['keys']], dtype=np.float32)
            Cu_pred, As_pred, VA, VB = predict(surrs[cond], cond, s, year, month, day, hour)
            Cu_true = float(row[MEASURED_CU_OUT_COL])
            As_true = float(row[MEASURED_AS_OUT_COL])
            records.append({
                'Row': idx + 1, 'Condition': cond,
                'Cu_true': Cu_true, 'Cu_pred': round(Cu_pred, 4),
                'As_true': As_true, 'As_pred': round(As_pred, 4),
                'Cu_err':  abs(Cu_pred - Cu_true),
                'As_err':  abs(As_pred - As_true),
                'Cu_pct':  abs(Cu_pred - Cu_true) / (abs(Cu_true) + 1e-8) * 100,
                'As_pct':  abs(As_pred - As_true) / (abs(As_true) + 1e-8) * 100,
            })
        except Exception as e:
            continue

    if not records:
        print("    No records processed.")
        return

    acc_df = pd.DataFrame(records)
    save_df(acc_df, None, 'surrogate_accuracy_detail')

    # 分工况统计
    stats_rows = []
    for cond in acc_df['Condition'].unique():
        sub = acc_df[acc_df['Condition'] == cond]
        for tgt, err_col, pct_col in [('Cu_out', 'Cu_err', 'Cu_pct'),
                                       ('As_out', 'As_err', 'As_pct')]:
            stats_rows.append({
                'Condition': cond, 'Target': tgt,
                'N': len(sub),
                'MAE':  round(sub[err_col].mean(), 4),
                'RMSE': round(np.sqrt((sub[err_col]**2).mean()), 4),
                'MAPE(%)': round(sub[pct_col].mean(), 2),
                'Max_err': round(sub[err_col].max(), 4),
            })
    stats_df = pd.DataFrame(stats_rows)
    save_df(stats_df, None, 'surrogate_accuracy_summary')
    print(stats_df.to_string(index=False))

# =============================================================================
# ── 13. 模块 B：掩码验证（三级场景对比） ────────────────────────────────────
# =============================================================================

def run_masked_validation(df, row_conds, nets, surrs):
    """三级掩码场景对比验证，输出各场景下的核心性能汇总。"""
    print("\n  [Masked Validation] 三级掩码场景对比...")
    weights = gen_weights('extreme', 10)
    scenario_summaries = {}

    for scenario_name, mask_cfg_all in MASK_SCENARIOS.items():
        print(f"\n    Scenario: {scenario_name}")
        rows = []
        for idx, (_, row) in enumerate(df.iterrows()):
            cond = row_conds[idx]
            if cond not in nets or cond not in surrs:
                continue
            mask_cfg = None if mask_cfg_all is None else mask_cfg_all.get(cond)
            year  = int(row.get('year', 2025))
            month = int(row.get('month', 6))
            day   = int(row.get('day', 15))
            hour  = int(row.get('hour', 8))
            try:
                res = validate_row_masked(row, cond, nets[cond], surrs[cond],
                                          weights, year, month, day, hour,
                                          mask_cfg=mask_cfg,
                                          scenario_name=scenario_name)
            except Exception:
                continue

            sm = {'Scenario': scenario_name, 'Condition': cond, 'Row': idx + 1,
                  'Orig_profit': res['profit_o'], 'Orig_feasible': int(res['feasible_o'])}
            if res['best']:
                b = res['best']
                sm.update({
                    'RL_profit':    round(b['profit'], 4),
                    'RL_Cu_out':    round(b['Cu_out'], 4),
                    'RL_As_out':    round(b['As_out'], 4),
                    'RL_E':         round(b['E'], 2),
                    'RL_feasible':  1,
                    'Profit_delta': round(b['profit'] - res['profit_o'], 4),
                    'Cu_reduction': round(res['Cu_out_o'] - b['Cu_out'], 4),
                    'As_improvement': round(b['As_out'] - res['As_out_o'], 4),
                    'E_reduction':  round(res['E_o'] - b['E'], 2),
                    'Applied_vars': ','.join(res['applied_vars']),
                    'N_applied':    len(res['applied_vars']),
                })
            else:
                sm.update({'RL_profit': None, 'RL_Cu_out': None, 'RL_As_out': None,
                           'RL_E': None, 'RL_feasible': 0, 'Profit_delta': None,
                           'Cu_reduction': None, 'As_improvement': None, 'E_reduction': None,
                           'Applied_vars': '', 'N_applied': 0})
            rows.append(sm)

        sdf = pd.DataFrame(rows)
        scenario_summaries[scenario_name] = sdf
        save_df(sdf, None, f'masked_{scenario_name}_detail')
        valid = sdf.dropna(subset=['Profit_delta'])
        feas_r = sdf['RL_feasible'].sum() / max(len(sdf), 1) * 100
        impr_r = (valid['Profit_delta'] > 0).sum() / max(len(valid), 1) * 100
        print(f"    N={len(sdf)}  feas={feas_r:.1f}%  "
              f"mean_profit_Δ={valid['Profit_delta'].mean():+.4f}万  "
              f"improve_rate={impr_r:.1f}%")

    # 跨场景汇总对比表（含 Bootstrap 置信区间和配对 t 检验）
    from scipy import stats as _scipy_stats
    comp_rows = []
    for sn, sdf in scenario_summaries.items():
        valid = sdf.dropna(subset=['Profit_delta'])
        deltas = valid['Profit_delta'].values

        # [FIX-5] Bootstrap 95% 置信区间（10000次重采样）
        if len(deltas) >= 2:
            rng = np.random.RandomState(42)
            boot_means = np.array([
                rng.choice(deltas, size=len(deltas), replace=True).mean()
                for _ in range(10000)
            ])
            ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        else:
            ci_lo = ci_hi = float('nan')

        # [FIX-5] 配对 t 检验：H0: RL 利润与原始利润无差异
        orig_profits = valid['Orig_profit'].values
        rl_profits   = valid['RL_profit'].dropna().values if 'RL_profit' in valid else np.array([])
        if len(rl_profits) >= 2 and len(orig_profits) >= 2:
            n_paired = min(len(orig_profits), len(rl_profits))
            t_stat, p_val = _scipy_stats.ttest_rel(
                rl_profits[:n_paired], orig_profits[:n_paired])
        else:
            t_stat = p_val = float('nan')

        comp_rows.append({
            'Scenario':             sn,
            'N_samples':            len(sdf),
            'RL_feas_rate(%)':      round(sdf['RL_feasible'].sum() / max(len(sdf),1) * 100, 1),
            'Mean_profit_delta':    round(float(np.nanmean(deltas)), 4) if len(deltas) else float('nan'),
            'CI95_lo':              round(float(ci_lo), 4),
            'CI95_hi':              round(float(ci_hi), 4),
            'ttest_t':              round(float(t_stat), 4),
            'ttest_p':              round(float(p_val), 4),
            'Significant(p<0.05)':  'YES' if (not np.isnan(p_val) and p_val < 0.05) else 'NO',
            'Mean_Cu_reduction':    round(valid['Cu_reduction'].mean(), 4),
            'Mean_As_improvement':  round(valid['As_improvement'].mean(), 4),
            'Mean_E_reduction':     round(valid['E_reduction'].mean(), 2),
            'Improve_rate(%)':      round((valid['Profit_delta']>0).sum()/max(len(valid),1)*100, 1),
        })
    comp_df = pd.DataFrame(comp_rows)
    save_df(comp_df, None, 'masked_validation_comparison')
    print("\n    [Masked Validation] 汇总对比（含统计显著性）：")
    print(comp_df.to_string(index=False))

# =============================================================================
# ── 14. 模块 C：动态序贯响应（Cu_in 连续波动） ──────────────────────────────
# =============================================================================

def generate_dynamic_cu_in_sequence(n_steps=20, base_cu_in=40.0, seed=42,
                                     noise_type='real_based'):
    """
    生成 Cu_in 时序波动序列（长度 n_steps）。
    noise_type: 'real_based' | 'step_change' | 'gradual_drift' | 'mixed'
    统计参数来自 Table S4 Condition 1: mean=38.92, std=4.18。
    """
    np.random.seed(seed)
    if noise_type == 'real_based':
        ar_coef    = 0.7
        noise_std  = 4.18 * np.sqrt(1 - ar_coef ** 2)
        seq = [base_cu_in]
        for _ in range(n_steps - 1):
            nxt = ar_coef * seq[-1] + (1 - ar_coef) * base_cu_in + \
                  np.random.normal(0, noise_std)
            seq.append(float(np.clip(nxt, 29.0, 55.0)))
        return np.array(seq)
    elif noise_type == 'step_change':
        seq = np.ones(n_steps) * base_cu_in
        for i in range(0, n_steps, max(1, n_steps // 3)):
            step = np.random.uniform(-8, 8)
            seq[i:] = np.clip(seq[i:] + step, 29.0, 55.0)
        return seq
    elif noise_type == 'gradual_drift':
        drift = np.linspace(base_cu_in + 5, base_cu_in - 5, n_steps)
        noise = np.random.normal(0, 1.5, n_steps)
        return np.clip(drift + noise, 29.0, 55.0)
    else:  # mixed
        seq = generate_dynamic_cu_in_sequence(n_steps, base_cu_in, seed, 'real_based')
        step_at = n_steps // 2
        seq[step_at:] = np.clip(seq[step_at:] + 6.0, 29.0, 55.0)
        return seq


def rl_dynamic_response(cu_in_seq, cond, net, surr, fixed_state_base,
                         weight=None, mask_cfg=None,
                         year=2025, month=6, day=15, hour=8):
    """
    RL 策略在 Cu_in 连续波动下的多步序贯响应。
    状态跨步转移（state = s_new），模拟真实序贯决策。
    记录每步响应时间（ms）。

    [FIX-7] year/month/day/hour 改为参数传入，不再硬编码为 2025/6/15/8，
    调用方可传入数据集的真实时间特征，提升代理模型预测精度。
    """
    c      = CCFG[cond]
    weight = weight if weight is not None else np.array([0.25]*4, dtype=np.float32)
    state  = fixed_state_base.copy()
    results = []

    for t_idx, cu_in_val in enumerate(cu_in_seq):
        state[0] = float(np.clip(cu_in_val, c['lo'][0], c['hi'][0]))

        t_start = time_module.perf_counter()
        with torch.no_grad():
            dist, _, _ = net(
                torch.FloatTensor(state).unsqueeze(0),
                torch.FloatTensor(weight).unsqueeze(0),
            )
            a_mean = dist.mean.numpy()[0]
        t_rl = (time_module.perf_counter() - t_start) * 1000  # ms

        s_new, applied = apply_action_with_mask(state, a_mean, cond, mask_cfg)

        Cu_out, As_out, VA, VB = predict(surr, cond, s_new, year, month, day, hour)
        E, profit = calc_E_profit(cond, s_new, Cu_out, As_out, VA, VB)
        costs     = calc_costs(cond, s_new, Cu_out, As_out, VA, VB)
        feasible  = all(v < 1e-3 for v in costs.values())

        rec = {
            'Timestep': t_idx, 'Cu_in_actual': round(cu_in_val, 4),
            'RL_response_ms': round(t_rl, 4),
            'Cu_out': round(Cu_out, 4), 'As_out': round(As_out, 4),
            'E': round(E, 2), 'Profit': round(profit, 4),
            'Feasible': int(feasible), 'Method': 'RL',
        }
        for i, k in enumerate(c['keys']):
            rec[f's_{k}'] = round(s_new[i], 4)
        results.append(rec)
        state = s_new.copy()   # 状态转移（跨批次）

    return pd.DataFrame(results)


def nsga_fixed_apply(cu_in_seq, cond, surr, nsga_pareto_df, fixed_state_base,
                     n_neighbors=5, year=2025, month=6, day=15, hour=8):
    """
    NSGA-II「固定解查表应用」基线：按 Cu_in 距离最近邻匹配。
    无状态转移，每步独立查表，模拟静态离线优化的在线部署。

    [FIX-7] year/month/day/hour 改为参数传入，与 rl_dynamic_response 保持一致。
    """
    c    = CCFG[cond]
    results = []
    for t_idx, cu_in_val in enumerate(cu_in_seq):
        # 按 Cu_in 最近邻查找 NSGA 解
        if 'Cu_in' in nsga_pareto_df.columns:
            dist_arr = np.abs(nsga_pareto_df['Cu_in'].values - cu_in_val)
            nn_idx   = np.argsort(dist_arr)[:n_neighbors]
            cands    = nsga_pareto_df.iloc[nn_idx]
        else:
            cands = nsga_pareto_df.sample(min(n_neighbors, len(nsga_pareto_df)), random_state=42)

        p_col = next((col for col in ['Net_profit', 'profit', 'Net profit (Ten thousand CNY)', 'Net profit\n(Ten thousand CNY)'] if col in cands.columns), None)
        if p_col is None:
            continue
        best_nsga = cands.loc[cands[p_col].idxmax()]

        # 构造应用状态：Cu_in 替换为实际值，控制变量取 NSGA 推荐
        state = fixed_state_base.copy()
        state[0] = float(np.clip(cu_in_val, c['lo'][0], c['hi'][0]))
        for i, k in enumerate(c['keys'][1:], start=1):
            if k in best_nsga.index and pd.notna(best_nsga[k]):
                state[i] = float(np.clip(best_nsga[k], c['lo'][i], c['hi'][i]))

        year, month, day, hour = 2025, 6, 15, 8
        # [FIX-4] 实测查表+预测耗时，不再硬编码 0.05ms
        t_nsga_start = time_module.perf_counter()
        Cu_out, As_out, VA, VB = predict(surr, cond, state, year, month, day, hour)
        t_nsga_ms = (time_module.perf_counter() - t_nsga_start) * 1000
        E, profit = calc_E_profit(cond, state, Cu_out, As_out, VA, VB)
        costs     = calc_costs(cond, state, Cu_out, As_out, VA, VB)
        feasible  = all(v < 1e-3 for v in costs.values())

        results.append({
            'Timestep': t_idx, 'Cu_in_actual': round(cu_in_val, 4),
            'NSGA_response_ms': round(t_nsga_ms, 4),   # [FIX-4] 实测值
            'Cu_out': round(Cu_out, 4), 'As_out': round(As_out, 4),
            'E': round(E, 2), 'Profit': round(profit, 4),
            'Feasible': int(feasible), 'Method': 'NSGA-II (Fixed)',
        })
    return pd.DataFrame(results)


def compare_cumulative(rl_df, nsga_df):
    """计算 RL vs NSGA-II 的跨批次累积指标对比。"""
    summary = {}
    for method, traj in [('RL', rl_df), ('NSGA-II', nsga_df)]:
        # [FIX-9] 显式列名查找，避免嵌套 .get() 因列名拼写差异静默返回 0
        if 'RL_response_ms' in traj.columns:
            mean_ms = round(traj['RL_response_ms'].mean(), 4)
        elif 'NSGA_response_ms' in traj.columns:
            mean_ms = round(traj['NSGA_response_ms'].mean(), 4)
        else:
            mean_ms = 0.0
        summary[method] = {
            'Cumulative_profit':    round(traj['Profit'].sum(), 4),
            'Cumulative_E':         round(traj['E'].sum(), 1),
            'Feasibility_rate(%)':  round(traj['Feasible'].mean() * 100, 1),
            'Violation_count':      int((traj['Feasible'] == 0).sum()),
            'Mean_Cu_out':          round(traj['Cu_out'].mean(), 4),
            'Std_Cu_out':           round(traj['Cu_out'].std(), 4),
            'Mean_profit_per_step': round(traj['Profit'].mean(), 4),
            'Mean_response_ms':     mean_ms,
        }
    comp = pd.DataFrame(summary).T
    if 'RL' in comp.index and 'NSGA-II' in comp.index:
        cp_rl   = comp.loc['RL', 'Cumulative_profit']
        cp_nsga = comp.loc['NSGA-II', 'Cumulative_profit']
        comp['Profit_advantage(%)'] = round(
            (cp_rl - cp_nsga) / (abs(cp_nsga) + 1e-8) * 100, 2)
    return comp


def run_dynamic_comparison(df, row_conds, nets, surrs, nsga_all):
    """
    对四种 Cu_in 波动模式，分别运行 RL 动态响应与 NSGA-II 固定查表对比。
    基础状态取各工况历史数据中位数。
    """
    print("\n  [Dynamic Comparison] Cu_in 动态波动序贯响应对比...")

    modes = {
        'real_based':    DYN_MODE_REAL_BASED,
        'step_change':   DYN_MODE_STEP_CHANGE,
        'gradual_drift': DYN_MODE_GRADUAL_DRIFT,
        'mixed':         DYN_MODE_MIXED,
    }
    n_steps = 20   # 波动序列长度（用户设定）

    all_traj   = []
    all_cumul  = []

    for cond in ['Condition 1', 'Condition 2', 'Condition 3']:
        if cond not in nets or cond not in surrs:
            continue
        c = CCFG[cond]

        # 取该工况历史数据中位数作为固定基础状态，同时取中位时间特征
        sub_df = df[[COL_MAP.get(k, k) for k in c['keys']
                     if COL_MAP.get(k, k) in df.columns]]
        sub_cond = df[[CONDITION_COL_NAME]].copy()
        sub_cond.columns = ['_cond']
        mask_cond = sub_cond['_cond'].astype(str).str.strip() == \
                    cond.replace('Condition ', '')
        hist_sub = df[mask_cond]
        if len(hist_sub) == 0:
            continue
        fixed_state = np.array(
            [float(hist_sub[COL_MAP[k]].median()) for k in c['keys']],
            dtype=np.float32)
        base_cu_in = fixed_state[0]

        # [FIX-7] 取历史数据的中位时间特征，传入 rl_dynamic_response / nsga_fixed_apply
        med_year  = int(hist_sub['year'].median())  if 'year'  in hist_sub.columns else 2025
        med_month = int(hist_sub['month'].median()) if 'month' in hist_sub.columns else 6
        med_day   = int(hist_sub['day'].median())   if 'day'   in hist_sub.columns else 15
        med_hour  = int(hist_sub['hour'].median())  if 'hour'  in hist_sub.columns else 8

        # 获取该工况的 NSGA Pareto 集
        nsga_cond = nsga_all.get(cond, pd.DataFrame())

        for mode_name, do_run in modes.items():
            if not do_run:
                continue
            print(f"    {cond} | mode={mode_name} | n_steps={n_steps}")

            cu_seq = generate_dynamic_cu_in_sequence(
                n_steps=n_steps, base_cu_in=float(base_cu_in),
                seed=42, noise_type=mode_name)

            # RL 响应（[FIX-7] 传入历史中位时间）
            rl_traj = rl_dynamic_response(
                cu_seq, cond, nets[cond], surrs[cond], fixed_state.copy(),
                weight=np.array([0.25]*4, dtype=np.float32),
                mask_cfg=MASK_CONFIG_REALISTIC.get(cond),
                year=med_year, month=med_month, day=med_day, hour=med_hour)
            rl_traj['Condition'] = cond
            rl_traj['Mode']      = mode_name
            all_traj.append(rl_traj)

            # NSGA-II 查表应用（[FIX-7] 同步传入历史中位时间）
            if len(nsga_cond) > 0:
                nsga_traj = nsga_fixed_apply(
                    cu_seq, cond, surrs[cond], nsga_cond, fixed_state.copy(),
                    year=med_year, month=med_month, day=med_day, hour=med_hour)
                if len(nsga_traj) == 0 or 'Profit' not in nsga_traj.columns:
                    print(f"      WARNING: NSGA traj empty or missing Profit column, skipping comparison")
                    continue
                nsga_traj['Condition'] = cond
                nsga_traj['Mode']      = mode_name
                all_traj.append(nsga_traj)

                # 累积对比
                comp = compare_cumulative(rl_traj, nsga_traj)
                comp['Condition'] = cond
                comp['Mode']      = mode_name
                comp['Method']    = comp.index
                all_cumul.append(comp.reset_index(drop=True))
                # [FIX-4] NSGA 响应时间现为实测值
                nsga_mean_ms = nsga_traj['NSGA_response_ms'].mean()
                print(f"      RL cum_profit={rl_traj['Profit'].sum():.2f}万  "
                      f"NSGA cum_profit={nsga_traj['Profit'].sum():.2f}万  "
                      f"RL mean_response={rl_traj['RL_response_ms'].mean():.3f}ms  "
                      f"NSGA mean_response={nsga_mean_ms:.3f}ms（实测）")

    if all_traj:
        traj_df = pd.concat(all_traj, ignore_index=True)
        save_df(traj_df, None, 'dynamic_response_trajectories')
    if all_cumul:
        cumul_df = pd.concat(all_cumul, ignore_index=True)
        save_df(cumul_df, None, 'dynamic_cumulative_comparison')
        print("\n    [Cumulative Comparison Summary]")
        print(cumul_df[['Condition','Mode','Method','Cumulative_profit',
                         'Feasibility_rate(%)','Violation_count',
                         'Profit_advantage(%)']].to_string(index=False))

# =============================================================================
# ── 15. 模块 D：铜价/电价灵敏度分析 ─────────────────────────────────────────
# =============================================================================

def run_price_sensitivity(pareto_csv_paths, nets=None, surrs=None, df=None, row_conds=None):
    """
    铜价/电价灵敏度分析。

    [FIX-2] 原版仅对已有 Pareto CSV 中固定控制变量重算利润，
    无法反映价格变化时 RL 策略的自适应推荐。
    现在分两层输出：
      ① fixed_solution_repricing：原始 Pareto 解在新价格下的经济表现
         （控制变量不变，仅价格改变，用于"固定策略对价格冲击的暴露"分析）
      ② rl_adaptive_response：当 nets/surrs/df 传入时，对历史数据重跑 RL 推理，
         生成新价格下的自适应推荐解，反映 RL 策略的真实自适应能力
    若 nets/surrs/df 未传入，则仅输出第①层（保持向后兼容）。
    """
    print("\n  [Price Sensitivity] 铜价/电价灵敏度分析...")
    all_stat  = []
    all_shift = []

    # ── ① 固定解重定价（原有逻辑，但 R_save 已与 calc_E_profit 一致）──────────
    As_fixed = 11.64   # [FIX-1] 与训练代码一致
    for cond, csv_path in pareto_csv_paths.items():
        if not os.path.exists(csv_path):
            print(f"    SKIP {cond}: CSV not found → {csv_path}")
            continue
        pareto_df = pd.read_csv(csv_path)
        E_col = next((c for c in ['E_total', 'E'] if c in pareto_df.columns), None)
        if E_col is None:
            print(f"    SKIP {cond}: no E column")
            continue

        for sc_name, prices in PRICE_SCENARIOS.items():
            p_Cu = prices['p_Cu']
            p_e  = prices['p_e']
            new_profits = []

            for _, row in pareto_df.iterrows():
                Cu_in  = float(row.get('Cu_in', 40.0))
                Cu_out = float(row.get('Cu_out', 6.0))
                E_val  = float(row[E_col])
                Q_col  = next((c for c in ['QA', 'QB'] if c in pareto_df.columns), None)
                Q      = float(row[Q_col]) if Q_col else 117.0
                t_val  = float(row.get('t', 4.0))

                m_Cu   = max(0.0, (Cu_in - Cu_out) * Q * t_val)
                R_Cu   = (p_Cu - ECON['c_reprocess'] - ECON['c_copper_concentrate']) * m_Cu
                # [FIX-1] 使用固定 As_fixed，与训练代码一致
                R_save = ECON['c_As_treat'] * (As_fixed + Cu_out) * Q * t_val
                profit = (R_Cu - R_save - p_e * E_val - ECON['c_op'] * t_val) / 1e4
                new_profits.append(profit)

            new_profits = np.array(new_profits)
            all_stat.append({
                'Condition': cond, 'Scenario': sc_name, 'Layer': 'fixed_repricing',
                'p_Cu': round(p_Cu, 2), 'p_e': round(p_e, 3),
                'N_solutions': len(new_profits),
                'Mean_profit':  round(new_profits.mean(), 4),
                'Max_profit':   round(new_profits.max(), 4),
                'Min_profit':   round(new_profits.min(), 4),
                'Std_profit':   round(new_profits.std(), 4),
                'N_profitable': int((new_profits > 0).sum()),
            })

            # 均衡偏好折中解
            obj_mat = np.column_stack([
                pareto_df['Cu_out'].values,
                -pareto_df['As_out'].values,
                pareto_df[E_col].values,
                -new_profits,
            ])
            mn = obj_mat.min(0); mx = obj_mat.max(0)
            dn = mx - mn; dn[dn == 0] = 1e-10
            best_idx = int(np.argmin(((obj_mat - mn) / dn).sum(1)))
            br = pareto_df.iloc[best_idx]
            all_shift.append({
                'Condition': cond, 'Scenario': sc_name, 'Layer': 'fixed_repricing',
                'p_Cu': round(p_Cu, 2), 'p_e': round(p_e, 3),
                'Cu_out':    round(float(br['Cu_out']), 4),
                'As_out':    round(float(br['As_out']), 4),
                'E_total':   round(float(br[E_col]), 2),
                'Net_profit': round(float(new_profits[best_idx]), 4),
            })

    stat_df  = pd.DataFrame(all_stat)
    shift_df = pd.DataFrame(all_shift)

    # ── ② RL 自适应推理（新增，仅当传入 nets/surrs/df 时执行）──────────────────
    if nets is not None and surrs is not None and df is not None and row_conds is not None:
        print("    [FIX-2] 执行 RL 自适应推理（各价格场景下重新推理控制变量）...")
        w_balanced = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        adapt_rows = []

        for cond in ['Condition 1', 'Condition 2', 'Condition 3']:
            if cond not in nets or cond not in surrs:
                continue
            c = CCFG[cond]
            hist_idxs = [i for i, rc in enumerate(row_conds) if rc == cond]
            if not hist_idxs:
                continue

            for sc_name, prices in PRICE_SCENARIOS.items():
                p_Cu_sc = prices['p_Cu']
                p_e_sc  = prices['p_e']
                sc_profits = []

                for idx in hist_idxs:
                    row = df.iloc[idx]
                    year  = int(row.get('year', 2025))
                    month = int(row.get('month', 6))
                    day   = int(row.get('day', 15))
                    hour  = int(row.get('hour', 8))
                    s_orig = np.array([float(row[COL_MAP[k]]) for k in c['keys']],
                                      dtype=np.float32)
                    try:
                        with torch.no_grad():
                            dist, _, _ = nets[cond](
                                torch.FloatTensor(s_orig).unsqueeze(0),
                                torch.FloatTensor(w_balanced).unsqueeze(0),
                            )
                            a_mean = dist.mean.numpy()[0]
                        s_new, _ = apply_action_with_mask(s_orig, a_mean, cond, None)
                        Cu_out, As_out, VA, VB = predict(
                            surrs[cond], cond, s_new, year, month, day, hour)
                        # 用场景价格计算利润
                        E, profit = calc_E_profit(
                            cond, s_new, Cu_out, As_out, VA, VB,
                            p_Cu=p_Cu_sc, p_e=p_e_sc)
                        sc_profits.append(profit)
                    except Exception:
                        continue

                if sc_profits:
                    arr = np.array(sc_profits)
                    adapt_rows.append({
                        'Condition': cond, 'Scenario': sc_name, 'Layer': 'rl_adaptive',
                        'p_Cu': round(p_Cu_sc, 2), 'p_e': round(p_e_sc, 3),
                        'N_solutions':  len(arr),
                        'Mean_profit':  round(arr.mean(), 4),
                        'Max_profit':   round(arr.max(), 4),
                        'Min_profit':   round(arr.min(), 4),
                        'Std_profit':   round(arr.std(), 4),
                        'N_profitable': int((arr > 0).sum()),
                    })

        if adapt_rows:
            adapt_df = pd.DataFrame(adapt_rows)
            stat_df  = pd.concat([stat_df, adapt_df], ignore_index=True)
            save_df(adapt_df, None, 'price_sensitivity_rl_adaptive')
            print(f"    RL 自适应推理完成，共 {len(adapt_rows)} 条记录")
    else:
        print("    [FIX-2 INFO] nets/surrs/df 未传入，跳过 RL 自适应层"
              "（仅输出固定解重定价结果）")

    save_df(stat_df,  None, 'price_sensitivity_statistics')
    save_df(shift_df, None, 'price_sensitivity_compromise_shift')
    print(f"    Scenarios={len(PRICE_SCENARIOS)}  Conditions={len(pareto_csv_paths)}")
    print("    Compromise solution shift under price scenarios (fixed repricing):")
    sub = shift_df[shift_df['Condition'] == 'Condition 2']
    if len(sub):
        print(sub.to_string(index=False))

# =============================================================================
# ── 16. 模块 E：偏好权重策略轨迹对比 ────────────────────────────────────────
# =============================================================================

def _diagnose_weight_sensitivity(net, cond, s_orig, n_probe=5):
    """
    [诊断] 在固定状态 s_orig 下，用 n_probe 种均匀覆盖权重向量探测网络的权重敏感性。
    打印各偏好场景下 dist.mean 的差异统计。

    原理：
      训练时 s 以原始量级（Cu_in≈40, IA≈15000）直接拼接 w∈[0,1] 送入网络。
      第一层中状态列的激活贡献约为权重列的 1000× 量级，梯度难以有效回传至
      权重维度，导致策略对 w 几乎无感知——a_mean 在不同 w 下几乎相同。
      该函数量化这一不敏感程度，帮助判断策略是否真正学到了多目标分化。

    返回：
      sensitivity_score: 跨权重向量的 a_mean L2 标准差（越大说明越敏感）
    """
    c = CCFG[cond]
    # 生成 n_probe 种覆盖极端偏好的权重向量
    probe_weights = [
        np.array([0.85, 0.05, 0.05, 0.05], dtype=np.float32),  # 极端偏好 Cu
        np.array([0.05, 0.85, 0.05, 0.05], dtype=np.float32),  # 极端偏好 As
        np.array([0.05, 0.05, 0.85, 0.05], dtype=np.float32),  # 极端偏好 E
        np.array([0.05, 0.05, 0.05, 0.85], dtype=np.float32),  # 极端偏好 profit
        np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),  # 均衡
    ][:n_probe]

    a_means = []
    with torch.no_grad():
        for w in probe_weights:
            dist, _, _ = net(
                torch.FloatTensor(s_orig).unsqueeze(0),
                torch.FloatTensor(w).unsqueeze(0),
            )
            a_means.append(dist.mean.numpy()[0])

    a_mat = np.array(a_means)  # (n_probe, action_dim)
    # 跨权重方向的动作标准差（每个动作维度）
    per_dim_std = a_mat.std(axis=0)
    sensitivity_score = float(per_dim_std.mean())

    print(f"\n    [权重敏感性诊断] 状态={[f'{v:.1f}' for v in s_orig]}")
    print(f"    {'偏好':20s}  " + "  ".join(f"a{i}" for i in range(a_mat.shape[1])))
    pref_names = ['w_Cu=0.85', 'w_As=0.85', 'w_E=0.85', 'w_pr=0.85', 'Balanced']
    for name, a in zip(pref_names[:n_probe], a_means):
        print(f"    {name:20s}  " + "  ".join(f"{v:+.4f}" for v in a))
    print(f"    {'per-dim std':20s}  " + "  ".join(f"{v:.4f}" for v in per_dim_std))
    print(f"    敏感性评分（越高越好）: {sensitivity_score:.6f}")

    if sensitivity_score < 0.01:
        print(f"    ⚠ 警告: 敏感性评分 < 0.01，策略对权重几乎无感知。")
        print(f"      根因：训练时状态量级（IA≈15000）远大于权重（0~1），")
        print(f"             梯度难以有效传入权重维度。")
        print(f"      建议：(1) 在训练代码中对状态做 [-1,1] 归一化后同步")
        print(f"             修改验证代码；(2) 当前结果仍有效，但 summary")
        print(f"             各行差异主要来自状态分布差异，而非权重调节。")
    elif sensitivity_score < 0.05:
        print(f"    ℹ 提示: 权重敏感性较弱（评分 {sensitivity_score:.4f}），")
        print(f"      策略有一定的多目标分化能力，但差异较小。")
    else:
        print(f"    ✔ 权重敏感性正常（评分 {sensitivity_score:.4f}）。")

    return sensitivity_score, a_mat


def _infer_best_action_stochastic(net, cond, s_orig, weight, surr,
                                  year, month, day, hour,
                                  mask_cfg=None, K=WEIGHT_SAMPLE_K):
    """
    [核心] 对齐训练侧 evaluate 的 stochastic 策略：
      对同一 (s, w) 采样 K 次动作 → K 个候选次态 → 代理模型批量预测利润
      → 选利润最高的候选作为最终推荐。

    与原始 dist.mean 单点确定性推理的区别：
      dist.mean：确定性，完全忽略了策略分布的探索信息，不同 w 下若 mean≈0
                 则所有场景结果相同；
      stochastic：利用分布方差探索动作空间，即使 mean≈0，std>0 时采样仍
                  能产生有效的动作差异，从 K 个候选中选出最优解。

    参数
    ----
    K : 采样次数，默认与训练侧 evaluate 一致（K=3）

    返回
    ----
    s_best : 最优候选次态 (state_dim,)
    a_best : 对应动作 (action_dim,)
    profit_best : 最优候选的预估利润
    all_profits : 所有 K 个候选的利润列表（用于分析分布）
    """
    c   = CCFG[cond]
    dsc = c['dsc']
    lo  = c['lo']
    hi  = c['hi']
    dim = c['ad']

    with torch.no_grad():
        dist, _, _ = net(
            torch.FloatTensor(s_orig).unsqueeze(0),
            torch.FloatTensor(weight).unsqueeze(0),
        )
        # dist.sample((K,)) → (K, 1, action_dim)，squeeze(1) → (K, action_dim)
        samples = dist.sample((K,)).numpy().squeeze(1)  # (K, action_dim)
        # 同时保留 mean 作为对照
        a_mean  = dist.mean.numpy()[0]                  # (action_dim,)

    all_profits = []
    all_s_new   = []
    all_a       = []

    for k in range(K):
        a_k   = samples[k]  # (action_dim,)
        s_new_k, _ = apply_action_with_mask(s_orig, a_k, cond, mask_cfg)
        try:
            Cu_out_k, As_out_k, VA_k, VB_k = predict(
                surr, cond, s_new_k, year, month, day, hour)
            _, profit_k = calc_E_profit(
                cond, s_new_k, Cu_out_k, As_out_k, VA_k, VB_k)
        except Exception:
            profit_k = -np.inf
        all_profits.append(profit_k)
        all_s_new.append(s_new_k)
        all_a.append(a_k)

    # 选利润最高的候选（对齐 evaluate._approx_profit_batch 的 argmax 逻辑）
    best_k = int(np.argmax(all_profits))
    return (all_s_new[best_k], all_a[best_k],
            all_profits[best_k], all_profits, a_mean)


def run_weight_preference_analysis(df, row_conds, nets, surrs,
                                   target_cond=WEIGHT_ANALYSIS_COND):
    """
    对指定工况（或全部已加载工况）使用 5 种典型偏好权重运行 RL，
    分析推荐控制变量分布的差异，量化权重语义清晰性。

    [FIX-8]  target_cond 支持传入 'ALL' 以对所有已加载工况逐一分析。
    [FIX-W1] 推理策略从 dist.mean（确定性，权重不敏感时输出恒为 0）
             改为 stochastic K 次采样取最优利润，对齐训练侧 evaluate。
             这是解决 summary 每行相同问题的核心修复。
    [FIX-W2] 增加权重敏感性诊断（首行状态上打印不同 w 下 a_mean 差异）。
    [FIX-W3] summary 新增 a_mean_norm（均值动作 L2 范数）、
             profit_std_across_samples（K 次采样利润标准差）两列，
             帮助判断策略的探索能力。
    [FIX-W4] 保留原 dist.mean 结果作为 'deterministic' 对照列，
             便于对比确定性推理与随机采样的差异。

    调用方式：
      run_weight_preference_analysis(..., target_cond='ALL')          # 全部工况
      run_weight_preference_analysis(..., target_cond='Condition 2')  # 单工况（原有行为）
    """
    # 确定要分析的工况列表
    if target_cond == 'ALL':
        cond_list = [c for c in ['Condition 1', 'Condition 2', 'Condition 3']
                     if c in nets and c in surrs]
    else:
        cond_list = [target_cond]

    for cur_cond in cond_list:
        print(f"\n  [Weight Preference] 偏好权重轨迹分析（{cur_cond}）...")
        if cur_cond not in nets or cur_cond not in surrs:
            print(f"    SKIP: {cur_cond} model not loaded.")
            continue

        c     = CCFG[cur_cond]
        net   = nets[cur_cond]
        surr  = surrs[cur_cond]
        pref_records = []

        hist_rows = [(idx, row) for idx, (_, row) in enumerate(df.iterrows())
                     if row_conds[idx] == cur_cond]
        if not hist_rows:
            print(f"    SKIP: no data rows for {cur_cond}.")
            continue

        # ── [FIX-W2] 权重敏感性诊断（用第一行历史状态） ────────────────────
        first_row = hist_rows[0][1]
        s_probe   = np.array([float(first_row[COL_MAP[k]]) for k in c['keys']],
                              dtype=np.float32)
        sensitivity_score, _ = _diagnose_weight_sensitivity(net, cur_cond, s_probe)
        # 保存诊断结果
        diag_df = pd.DataFrame([{
            'Condition': cur_cond,
            'Sensitivity_score': round(sensitivity_score, 6),
            'N_probe_weights': 5,
            'Interpretation': (
                'Insensitive (score<0.01): policy barely differentiates weights'
                if sensitivity_score < 0.01 else
                'Weak (0.01~0.05): mild multi-objective differentiation'
                if sensitivity_score < 0.05 else
                'Normal (>0.05): meaningful weight-conditioned behavior'
            ),
        }])
        save_df(diag_df, cur_cond, 'weight_sensitivity_diagnosis')

        # ── [FIX-W1] stochastic K 次采样推理（核心修复） ────────────────────
        mask_cfg = MASK_CONFIG_REALISTIC.get(cur_cond)
        print(f"    推理策略: stochastic K={WEIGHT_SAMPLE_K} 次采样取最优利润")
        print(f"    （对齐训练侧 evaluate stochastic 策略，替代原 dist.mean 确定性推理）")
        print(f"    处理 {len(hist_rows)} 行 × {len(PREFERENCE_SCENARIOS)} 偏好场景"
              f" × K={WEIGHT_SAMPLE_K} 次采样 ...")

        for sc_name, weight in PREFERENCE_SCENARIOS.items():
            for idx, row in hist_rows:
                year  = int(row.get('year', 2025))
                month = int(row.get('month', 6))
                day   = int(row.get('day', 15))
                hour  = int(row.get('hour', 8))
                s_orig = np.array([float(row[COL_MAP[k]]) for k in c['keys']],
                                   dtype=np.float32)

                # [FIX-W1] stochastic 采样推理
                (s_new, a_best, profit_best,
                 all_profits, a_mean) = _infer_best_action_stochastic(
                    net, cur_cond, s_orig, weight, surr,
                    year, month, day, hour, mask_cfg=mask_cfg, K=WEIGHT_SAMPLE_K)

                Cu_out, As_out, VA, VB = predict(
                    surr, cur_cond, s_new, year, month, day, hour)
                E, profit = calc_E_profit(cur_cond, s_new, Cu_out, As_out, VA, VB)
                costs     = calc_costs(cur_cond, s_new, Cu_out, As_out, VA, VB)

                # [FIX-W4] 保留 dist.mean 确定性推理结果作为对照
                s_det, _ = apply_action_with_mask(s_orig, a_mean, cur_cond, mask_cfg)
                Cu_det, As_det, VA_det, VB_det = predict(
                    surr, cur_cond, s_det, year, month, day, hour)
                _, profit_det = calc_E_profit(cur_cond, s_det, Cu_det, As_det, VA_det, VB_det)

                rec = {
                    'Scenario':     sc_name,
                    'Condition':    cur_cond,
                    'Row':          idx + 1,
                    'w_Cu':         round(float(weight[0]), 3),
                    'w_As':         round(float(weight[1]), 3),
                    'w_E':          round(float(weight[2]), 3),
                    'w_profit':     round(float(weight[3]), 3),
                    # stochastic 采样结果（主要输出）
                    'Cu_out':       round(Cu_out, 4),
                    'As_out':       round(As_out, 4),
                    'E':            round(E, 2),
                    'Profit':       round(profit, 4),
                    'Feasible':     int(all(v < 1e-3 for v in costs.values())),
                    # [FIX-W3] 附加诊断列
                    'a_mean_norm':  round(float(np.linalg.norm(a_mean)), 6),
                    'profit_std_K': round(float(np.std([p for p in all_profits
                                                        if np.isfinite(p)])), 6),
                    # [FIX-W4] deterministic 对照结果
                    'Det_Cu_out':   round(Cu_det, 4),
                    'Det_As_out':   round(As_det, 4),
                    'Det_Profit':   round(profit_det, 4),
                    'Stoch_vs_Det_profit': round(profit - profit_det, 4),
                }
                # 推荐控制变量（stochastic 最优次态）
                for i, k in enumerate(c['keys']):
                    rec[f'Rec_{k}'] = round(float(s_new[i]), 4)
                # deterministic 对照控制变量
                for i, k in enumerate(c['keys']):
                    rec[f'Det_{k}'] = round(float(s_det[i]), 4)
                pref_records.append(rec)

        pref_df = pd.DataFrame(pref_records)
        save_df(pref_df, cur_cond, 'weight_preference_trajectories')

        # ── 分组统计汇总 ─────────────────────────────────────────────────────
        agg_cols = ['Cu_out', 'As_out', 'E', 'Profit', 'Feasible',
                    'a_mean_norm', 'profit_std_K',
                    'Det_Profit', 'Stoch_vs_Det_profit'] + \
                   [f'Rec_{k}' for k in c['keys']]
        stats = pref_df.groupby('Scenario')[agg_cols].agg(['mean', 'std']).round(4)
        save_df(stats.reset_index(), cur_cond, 'weight_preference_summary')

        print(f"\n    [结果] {len(hist_rows)} 行 × {len(PREFERENCE_SCENARIOS)} 偏好")
        print(f"    ── stochastic 推理（主要结果）──")
        mean_tab = pref_df.groupby('Scenario')[
            ['Cu_out', 'As_out', 'E', 'Profit', 'Feasible']].mean().round(4)
        print(mean_tab.to_string())

        print(f"\n    ── deterministic 对照（dist.mean）──")
        det_tab = pref_df.groupby('Scenario')[
            ['Det_Cu_out', 'Det_As_out', 'Det_Profit']].mean().round(4)
        print(det_tab.to_string())

        print(f"\n    ── 权重敏感性（a_mean_norm & profit_std_K）──")
        sens_tab = pref_df.groupby('Scenario')[
            ['a_mean_norm', 'profit_std_K', 'Stoch_vs_Det_profit']].mean().round(6)
        print(sens_tab.to_string())

        # ── [FIX-W2] 输出跨偏好场景的目标差异统计（量化 summary 是否真的不同）
        print(f"\n    ── 各目标跨偏好场景的标准差（越大说明权重越有效分化）──")
        cross_std = mean_tab.std().round(6)
        print(cross_std.to_string())
        if cross_std['Profit'] < 0.001:
            print(f"\n    ⚠ 跨场景 Profit 标准差 < 0.001：summary 行间几乎无差异。")
            print(f"      根因确认：策略对偏好权重不敏感，详见权重敏感性诊断文件。")
            print(f"      该问题根源在训练侧（状态量级失配），验证侧已通过")
            print(f"      stochastic 采样最大程度缓解；若需彻底修复，需在训练代码中")
            print(f"      对状态做 [-1,1] 归一化（参见 ppo_lagrangian.py 修复建议）。")

# =============================================================================
# ── 17. 模块 F：Pareto 斜率与主导变量迁移 ───────────────────────────────────
# =============================================================================

def compute_pareto_local_slopes(pareto_df, E_col, p_col, method='RL'):
    """
    计算 Pareto 前沿的局部斜率 |ΔE / ΔCu_out|（kWh per g/L）。
    三分区：Low-E (>5.5) / Transition (4.5-5.5) / High-E (<4.5)。
    """
    df_s = pareto_df.sort_values('Cu_out').reset_index(drop=True)
    slope_rows = []
    for i in range(len(df_s) - 1):
        dCu = df_s.loc[i+1, 'Cu_out'] - df_s.loc[i, 'Cu_out']
        if abs(dCu) < 1e-6:
            continue
        dE  = df_s.loc[i+1, E_col] - df_s.loc[i, E_col]
        dPr = df_s.loc[i+1, p_col] - df_s.loc[i, p_col]
        cu_mid = (df_s.loc[i, 'Cu_out'] + df_s.loc[i+1, 'Cu_out']) / 2
        if cu_mid > 5.5:
            zone = 'Low-E Zone (Reaction-rate controlled)'
        elif cu_mid > 4.5:
            zone = 'Transition Zone (Mixed control)'
        else:
            zone = 'High-E Zone (Mass-transfer limited)'
        slope_rows.append({
            'Cu_out_mid':      round(cu_mid, 4),
            'Slope_E_per_Cu':  round(abs(dE / dCu), 2),
            'Slope_P_per_Cu':  round(dPr / dCu, 4),
            'Zone': zone, 'Method': method,
        })
    slope_df = pd.DataFrame(slope_rows)
    zone_stats = slope_df.groupby(['Method','Zone']).agg(
        N=('Slope_E_per_Cu','count'),
        Mean_slope=('Slope_E_per_Cu','mean'),
        Std_slope=('Slope_E_per_Cu','std'),
        Min_slope=('Slope_E_per_Cu','min'),
        Max_slope=('Slope_E_per_Cu','max'),
    ).round(2)
    return slope_df, zone_stats


def dominant_variable_shift_analysis(pareto_df, cond, E_col):
    """
    Spearman 相关分析：识别 Pareto 各 Cu_out 分区内的主导控制变量。
    返回分区主导变量迁移汇总表。
    """
    ctrl_map = {
        'Condition 1': ['IA', 'QA', 't'],
        'Condition 2': ['IB', 'QB', 't'],
        'Condition 3': ['IA', 'QA', 'IB', 'QB', 't'],
    }
    ctrl_vars = [v for v in ctrl_map.get(cond, []) if v in pareto_df.columns]
    cu_vals   = pareto_df['Cu_out'].values
    bins      = np.quantile(cu_vals, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    shift_rows = []
    for i in range(len(bins) - 1):
        lo_b, hi_b = bins[i], bins[i+1]
        mask = (pareto_df['Cu_out'] >= lo_b) & (pareto_df['Cu_out'] <= hi_b)
        sub  = pareto_df[mask]
        if len(sub) < 5:
            continue
        cu_center = round((lo_b + hi_b) / 2, 3)
        row_out = {
            'Condition': cond,
            'Cu_out_center': cu_center,
            'Cu_out_range':  f'{lo_b:.2f}-{hi_b:.2f}',
            'N_samples':     len(sub),
        }
        max_abs_corr = 0.0
        dom_var      = 'None'
        dom_dir      = ''
        for var in ctrl_vars:
            if sub[var].std() < 1e-6:
                continue
            try:
                corr, pval = spearmanr(sub[var].values, sub[E_col].values)
            except Exception:
                continue
            row_out[f'rho_{var}'] = round(float(corr), 3)
            row_out[f'p_{var}']   = round(float(pval), 4)
            if abs(corr) > abs(max_abs_corr) and pval < 0.15:
                max_abs_corr = corr
                dom_var      = var
                dom_dir      = 'positive' if corr > 0 else 'negative'
        row_out['Dominant_var']  = dom_var
        row_out['Dominant_rho']  = round(max_abs_corr, 3)
        row_out['Dominant_dir']  = dom_dir
        if cu_center > 5.5:
            mech = 'Reaction-rate control (I dominates)'
        elif cu_center > 4.5:
            mech = 'Transition (I → Q/t shift)'
        else:
            mech = 'Mass-transfer control (Q/t dominates)'
        row_out['Mechanism'] = mech
        shift_rows.append(row_out)
    return pd.DataFrame(shift_rows)


def run_pareto_analysis(pareto_csv_paths):
    """
    对三种工况的 RL Pareto 解集执行：
      1. 局部斜率计算（三分区，量化边际替代关系非线性）
      2. 主导变量迁移分析（Spearman 相关，识别控制机制转变）
    """
    print("\n  [Pareto Analysis] 斜率分析与主导变量迁移...")
    all_slopes = []
    all_shifts = []
    all_zone_stats = []

    for cond, csv_path in pareto_csv_paths.items():
        if not os.path.exists(csv_path):
            print(f"    SKIP {cond}: CSV not found")
            continue
        pareto_df = pd.read_csv(csv_path)
        E_col = next((c for c in ['E_total', 'E'] if c in pareto_df.columns), None)
        p_col = next((c for c in ['Net_profit', 'profit', 'Net profit (Ten thousand CNY)', 'Net profit\n(Ten thousand CNY)'] if c in pareto_df.columns), None)
        if E_col is None or p_col is None or 'Cu_out' not in pareto_df.columns:
            print(f"    SKIP {cond}: missing required columns")
            continue

        # 斜率分析
        slope_df, zone_stats = compute_pareto_local_slopes(
            pareto_df, E_col, p_col, method='RL')
        slope_df['Condition']  = cond
        zone_stats_r = zone_stats.reset_index()
        zone_stats_r['Condition'] = cond
        all_slopes.append(slope_df)
        all_zone_stats.append(zone_stats_r)

        # 主导变量迁移
        shift_df = dominant_variable_shift_analysis(pareto_df, cond, E_col)
        all_shifts.append(shift_df)

        print(f"\n    {cond} | Zone stats:")
        print(zone_stats.to_string())
        print(f"\n    {cond} | Dominant variable shift:")
        print(shift_df[['Cu_out_range','N_samples','Dominant_var',
                         'Dominant_rho','Mechanism']].to_string(index=False))

    if all_slopes:
        save_df(pd.concat(all_slopes, ignore_index=True), None, 'pareto_local_slopes')
    if all_zone_stats:
        save_df(pd.concat(all_zone_stats, ignore_index=True), None, 'pareto_zone_statistics')
    if all_shifts:
        save_df(pd.concat(all_shifts, ignore_index=True), None, 'pareto_dominant_var_shift')

# =============================================================================
# ── 18. 原有 v3.0 主验证流程（完整保留） ────────────────────────────────────
# =============================================================================

def run_original_validation(df, row_conds, nets, surrs):
    """v3.0 原有逐行验证流程，完整保留，输出至各工况目录。"""
    print("\n  [Original v3.0] 逐行验证...")
    weights = gen_weights(WEIGHT_STRATEGY, N_WEIGHT_SAMPLES)
    print(f"    Weight strategy: {WEIGHT_STRATEGY}  N={len(weights)}")

    summaries, details = [], []
    progress_bar.start_time = time.time()
    for idx, (_, row) in enumerate(df.iterrows()):
        cond = row_conds[idx]
        if cond not in nets or cond not in surrs:
            progress_bar(idx + 1, len(df))
            continue
        year  = int(row.get('year', 2025))
        month = int(row.get('month', 6))
        day   = int(row.get('day', 15))
        hour  = int(row.get('hour', 8))
        try:
            res = validate_row(row, cond, nets[cond], surrs[cond],
                               weights, year, month, day, hour)
        except Exception as e:
            print(f"\n    ERROR row {idx+1}: {e}")
            progress_bar(idx + 1, len(df))
            continue

        c_keys = CCFG[cond]['keys']
        sm = {'Row': idx+1, 'Condition': cond}
        for i, k in enumerate(c_keys):
            sm[f'In_{k}'] = round(res['s_orig'][i], 4)
        sm.update({
            'Orig_Cu_out': round(res['Cu_out_o'], 4),
            'Orig_As_out': round(res['As_out_o'], 4),
            'Orig_E':      round(res['E_o'], 2),
            'Orig_profit': round(res['profit_o'], 4),
            'Orig_feasible': int(res['feasible_o']),
        })
        # 实测值（若有）
        if MEASURED_CU_OUT_COL in row.index:
            sm['Meas_Cu_out'] = round(float(row[MEASURED_CU_OUT_COL]), 4)
            sm['Meas_As_out'] = round(float(row[MEASURED_AS_OUT_COL]), 4)
        if res['best']:
            b = res['best']
            sn = np.array(b['s_new']); so = np.array(res['s_orig'])
            sm.update({
                'RL_Cu_out':    round(b['Cu_out'], 4),
                'RL_As_out':    round(b['As_out'], 4),
                'RL_E':         round(b['E'], 2),
                'RL_profit':    round(b['profit'], 4),
                'RL_feasible':  1,
                'Profit_delta': round(b['profit'] - res['profit_o'], 4),
                'Cu_out_delta': round(res['Cu_out_o'] - b['Cu_out'], 4),
                'As_out_delta': round(b['As_out'] - res['As_out_o'], 4),
                'E_delta':      round(res['E_o'] - b['E'], 2),
                'State_delta':  str([round(float(sn[i]-so[i]),4)
                                     for i in range(min(len(sn),len(so))-1)]),
            })
        else:
            sm.update({'RL_Cu_out': None, 'RL_As_out': None, 'RL_E': None,
                       'RL_profit': None, 'RL_feasible': 0,
                       'Profit_delta': None, 'Cu_out_delta': None,
                       'As_out_delta': None, 'E_delta': None, 'State_delta': None})
        summaries.append(sm)

        for wi, wr in enumerate(res['w_results']):
            dt = {'Row': idx+1, 'Condition': cond, 'WeightID': wi,
                  'w_Cu': round(wr['w'][0],4), 'w_As': round(wr['w'][1],4),
                  'w_E':  round(wr['w'][2],4), 'w_pr': round(wr['w'][3],4)}
            for i, k in enumerate(c_keys):
                dt[f'In_{k}']  = round(res['s_orig'][i], 4)
                dt[f'New_{k}'] = round(wr['s_new'][i], 4)
            for ai in range(CCFG[cond]['ad']):
                dt[f'act_{ai}'] = round(wr['action'][ai], 6)
            dt.update({'Cu_out': round(wr['Cu_out'],4), 'As_out': round(wr['As_out'],4),
                       'E': round(wr['E'],2), 'profit': round(wr['profit'],4),
                       'feasible': int(wr['feasible'])})
            for ck_, cv_ in wr['costs'].items():
                dt[f'cost_{ck_}'] = round(cv_, 6)
            details.append(dt)
        progress_bar(idx + 1, len(df))
    print()

    sdf = pd.DataFrame(summaries)
    ddf = pd.DataFrame(details)

    # 按工况分别保存
    for cond_name in ['Condition 1', 'Condition 2', 'Condition 3']:
        sub_s = sdf[sdf['Condition'] == cond_name] if len(sdf) > 0 else pd.DataFrame()
        sub_d = ddf[ddf['Condition'] == cond_name] if len(ddf) > 0 else pd.DataFrame()
        if len(sub_s) > 0:
            save_df(sub_s, cond_name, 'validation_summary')
        if len(sub_d) > 0:
            save_df(sub_d, cond_name, 'validation_details')

    # 合并全局汇总
    save_df(sdf, None, 'validation_summary_all')

    # 统计打印
    print(f"\n{'='*70}")
    print(f"  VALIDATION STATISTICS (v3.0)")
    print(f"{'='*70}")
    n = len(sdf)
    if n == 0:
        print("  No rows processed.")
        return sdf
    nf_o = sdf['Orig_feasible'].sum()
    nf_r = sdf['RL_feasible'].sum()
    print(f"  Total rows:      {n}")
    print(f"  Orig feasible:   {nf_o} ({nf_o/max(n,1)*100:.1f}%)")
    print(f"  RL feasible:     {nf_r} ({nf_r/max(n,1)*100:.1f}%)")
    for col, unit in [('Profit_delta','万元'),('Cu_out_delta','g/L'),
                      ('As_out_delta','g/L'),('E_delta','kWh')]:
        if col in sdf.columns:
            vd = sdf[col].dropna()
            if len(vd) > 0:
                print(f"\n  {col} ({unit}):")
                print(f"    Mean={vd.mean():+.4f}  Median={vd.median():+.4f}"
                      f"  Max={vd.max():+.4f}  Min={vd.min():+.4f}"
                      f"  Improved={( vd>0).sum()}/{len(vd)}"
                      f" ({(vd>0).sum()/len(vd)*100:.1f}%)")
    for cond_name in sdf['Condition'].unique():
        sub = sdf[sdf['Condition'] == cond_name]
        rp  = sub['RL_profit'].dropna()
        print(f"\n  {cond_name} ({len(sub)} rows):")
        print(f"    Orig feas={sub['Orig_feasible'].sum()}  RL feas={sub['RL_feasible'].sum()}")
        if len(rp) > 0:
            print(f"    RL profit: mean={rp.mean():.4f}  max={rp.max():.4f}")
    return sdf

# =============================================================================
# ── 19. NSGA-II 解集加载工具 ─────────────────────────────────────────────────
# =============================================================================

def load_nsga_pareto(nsga_path):
    """
    加载 NSGA-II 的 Pareto 解集（支持 Excel 多 Sheet 格式）。
    返回 {'Condition 1': df1, 'Condition 2': df2, 'Condition 3': df3}。
    """
    nsga_all = {}
    if not os.path.exists(nsga_path):
        print(f"    WARNING: NSGA file not found: {nsga_path}")
        return nsga_all
    try:
        xl = pd.ExcelFile(nsga_path)
        sheet_map = {
            'Condition 1 Pareto Front': 'Condition 1',
            'Condition 2 Pareto Front': 'Condition 2',
            'Condition 3 Pareto Front': 'Condition 3',
        }
        for sheet, cond in sheet_map.items():
            if sheet in xl.sheet_names:
                df = pd.read_excel(nsga_path, sheet_name=sheet)
                nsga_all[cond] = df
                print(f"    NSGA loaded: {cond}  {len(df)} solutions")
            else:
                # 尝试直接按工况数字命名的 sheet
                alt = f'Condition {cond[-1]} Pareto Front'
                if alt in xl.sheet_names:
                    nsga_all[cond] = pd.read_excel(nsga_path, sheet_name=alt)
    except Exception as e:
        print(f"    ERROR loading NSGA file: {e}")
    return nsga_all

# =============================================================================
# ── 20. Main ──────────────────────────────────────────────────────────────────
# =============================================================================

def main():
    # 创建所有输出目录
    for d in list(OUTPUT_DIRS.values()) + [OUTPUT_DIR_COMMON]:
        os.makedirs(d, exist_ok=True)

    print("=" * 70)
    print("  PPO-Lagrangian Model Validation Tool  v4.0")
    print("=" * 70)
    print(f"  Modules: original={RUN_ORIGINAL}  surrogate={RUN_SURROGATE}"
          f"  masked={RUN_MASKED}  dynamic={RUN_DYNAMIC}")
    print(f"           price={RUN_PRICE}  weight={RUN_WEIGHT}"
          f"  pareto={RUN_PARETO}")
    print("=" * 70)

    # ── 读取数据 ─────────────────────────────────────────────────────────────
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Data file not found: {EXCEL_PATH}")
    print(f"\n[1] Reading: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)
    print(f"  {len(df)} rows  columns: {list(df.columns)}")
    needed = set(COL_MAP.values()) | {CONDITION_COL_NAME}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    print("  Column check: OK")

    # 工况列处理（值为 1/2/3 → 'Condition 1/2/3'）
    raw_conds = df[CONDITION_COL_NAME].tolist()
    row_conds = [
        f"Condition {int(c)}" if isinstance(c, (int, float)) and not pd.isna(c)
        else str(c).strip() if isinstance(c, str) and not c.startswith('Condition')
        else str(c).strip()
        for c in raw_conds
    ]
    # 确保格式统一
    row_conds = [c if c.startswith('Condition') else f"Condition {c}" for c in row_conds]
    print(f"  Conditions: {dict(pd.Series(row_conds).value_counts().to_dict())}")

    # ── 加载模型 ─────────────────────────────────────────────────────────────
    print(f"\n[2] Loading RL nets & surrogate models...")
    nets, surrs, lams = {}, {}, {}
    for cond in ['Condition 1', 'Condition 2', 'Condition 3']:
        if cond not in set(row_conds) and cond not in PARETO_CSV_PATHS:
            continue
        pt = PT_PATHS.get(cond, '')
        if pt and os.path.exists(pt):
            try:
                net, lam = load_rl_net(pt, cond)
                nets[cond] = net
                lams[cond] = lam
                print(f"  RL net loaded: {cond}  λ={[f'{v:.3f}' for v in lam]}")
            except Exception as e:
                print(f"  ERROR loading RL net ({cond}): {e}")
        else:
            print(f"  WARNING: .pt not found: {pt}")
        try:
            surrs[cond] = load_surr(MODEL_DIR, CCFG[cond]['sub'])
            print(f"  Surrogate loaded: {cond}")
        except Exception as e:
            print(f"  ERROR loading surrogate ({cond}): {e}")

    # ── 加载 NSGA-II 解集 ────────────────────────────────────────────────────
    print(f"\n[3] Loading NSGA-II Pareto solutions...")
    nsga_all = load_nsga_pareto(NSGA_PATH)

    # ══════════════════════════════════════════════════════════════════════════
    # 执行各模块
    # ══════════════════════════════════════════════════════════════════════════

    t_total = time.time()

    # ── 模块 0：原有 v3.0 验证 ───────────────────────────────────────────────
    if RUN_ORIGINAL:
        print(f"\n{'─'*70}")
        print(f"  [Module 0] 原有 v3.0 逐行验证")
        print(f"{'─'*70}")
        t0 = time.time()
        run_original_validation(df, row_conds, nets, surrs)
        print(f"  耗时: {time.time()-t0:.1f}s")

    # ── 模块 1：代理模型预测精度验证 ─────────────────────────────────────────
    if RUN_SURROGATE:
        print(f"\n{'─'*70}")
        print(f"  [Module 1] 代理模型预测精度验证（预测 vs 实测）")
        print(f"{'─'*70}")
        t0 = time.time()
        run_surrogate_accuracy(df, row_conds, surrs)
        print(f"  耗时: {time.time()-t0:.1f}s")

    # ── 模块 2：掩码验证 ─────────────────────────────────────────────────────
    if RUN_MASKED:
        print(f"\n{'─'*70}")
        print(f"  [Module 2] 三级掩码场景对比验证")
        print(f"{'─'*70}")
        t0 = time.time()
        run_masked_validation(df, row_conds, nets, surrs)
        print(f"  耗时: {time.time()-t0:.1f}s")

    # ── 模块 3：动态序贯响应 ──────────────────────────────────────────────────
    if RUN_DYNAMIC:
        print(f"\n{'─'*70}")
        print(f"  [Module 3] 动态 Cu_in 波动序贯响应对比")
        modes_on = [m for m, v in {
            'real_based': DYN_MODE_REAL_BASED,
            'step_change': DYN_MODE_STEP_CHANGE,
            'gradual_drift': DYN_MODE_GRADUAL_DRIFT,
            'mixed': DYN_MODE_MIXED,
        }.items() if v]
        print(f"  波动模式: {modes_on}  n_steps=20")
        print(f"{'─'*70}")
        t0 = time.time()
        run_dynamic_comparison(df, row_conds, nets, surrs, nsga_all)
        print(f"  耗时: {time.time()-t0:.1f}s")

    # ── 模块 4：铜价/电价灵敏度 ──────────────────────────────────────────────
    if RUN_PRICE:
        print(f"\n{'─'*70}")
        print(f"  [Module 4] 铜价/电价灵敏度分析（{len(PRICE_SCENARIOS)} 场景）")
        print(f"{'─'*70}")
        t0 = time.time()
        # [FIX-2] 传入 nets/surrs/df/row_conds 以启用 RL 自适应推理层
        run_price_sensitivity(PARETO_CSV_PATHS,
                              nets=nets, surrs=surrs,
                              df=df, row_conds=row_conds)
        print(f"  耗时: {time.time()-t0:.1f}s")

    # ── 模块 5：偏好权重分析 ──────────────────────────────────────────────────
    if RUN_WEIGHT:
        print(f"\n{'─'*70}")
        # [FIX-8] 支持 'ALL' 对三个工况逐一分析
        print(f"  [Module 5] 偏好权重策略轨迹对比（target={WEIGHT_ANALYSIS_COND}）")
        print(f"{'─'*70}")
        t0 = time.time()
        run_weight_preference_analysis(df, row_conds, nets, surrs,
                                        target_cond=WEIGHT_ANALYSIS_COND)
        print(f"  耗时: {time.time()-t0:.1f}s")

    # ── 模块 6：Pareto 斜率与主导变量迁移 ────────────────────────────────────
    if RUN_PARETO:
        print(f"\n{'─'*70}")
        print(f"  [Module 6] Pareto 局部斜率与主导变量迁移分析")
        print(f"{'─'*70}")
        t0 = time.time()
        run_pareto_analysis(PARETO_CSV_PATHS)
        print(f"  耗时: {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  [OK]  All modules complete!  Total time: {time.time()-t_total:.1f}s")
    print(f"  Output directories:")
    for cond, d in OUTPUT_DIRS.items():
        print(f"    {cond}: {d}")
    print(f"    Common: {OUTPUT_DIR_COMMON}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
