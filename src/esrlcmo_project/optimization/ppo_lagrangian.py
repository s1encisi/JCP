"""
PPO-Lagrangian Safe RL for Copper Electrowinning
=================================================
支持三种工况（condition）统一入口，通过命令行 --condition 1/2/3 切换：

  Condition 1 — Unit A  (three_stage)
    State/Action : [Cu_in, TA, IA, QA, t] / [ΔCu_in, ΔTA, ΔIA, ΔQA]
    Surrogate    : extra_trees_three_stage.joblib  (V:8维, Cu/As:10维)

  Condition 2 — Unit B  (four_stage)
    State/Action : [Cu_in, TB, IB, QB, t] / [ΔCu_in, ΔTB, ΔIB, ΔQB]
    Surrogate    : extra_trees_four_stage.joblib   (V:8维, Cu/As:10维)

  Condition 3 — Unit A+B Serial cascade
    State/Action : [Cu_in, TA, IA, QA, TB, IB, QB, t] / [ΔCu_in, ΔTA, ΔIA, ΔQA, ΔTB, ΔIB, ΔQB]
    Surrogate    : extra_trees_serial.joblib        (V:11维, Cu/As:14维)

架构说明：
  [REFACTOR] 两阶段 rollout 架构，彻底解决多进程模型加载死锁：
    阶段1: fork 出 n_workers 个子进程，只做策略推理+状态转移，不碰 ExtraTrees
    阶段2: 主进程拿回 raw 轨迹，统一做批量 ExtraTrees 预测

Fix log:
  [FIX-1]  Lagrangian λᵢ 独立更新（每个约束一个）
  [FIX-2]  episode 达到 max_ep_steps 后终止
  [FIX-3]  feasibility 阈值 1e-3
  [FIX-4]  rollout buffer 存完整代价向量 (N×5)
  [FIX-5]  profit 列名鲁棒查找
  [FIX-6]  build_features 特征维度与实际训练特征完全一致
  [FIX-7]  每次 rollout 末尾都记录 metrics
  [FIX-8]  BC 预训练噪声维度对齐
  [FIX-9]  CUDA 不可用时自动回退 CPU
  [FIX-10] Condition 2 dec_var_map 注释修正（原写错为 TA/IA/QA，实为 TB/IB/QB）
  [FIX-11] save_rl_excel 统一抽出，消除三份独立副本中的列名偏差
            (Etotal/E_total/VA/VB/Vaverage/costs/costss 全部在一处对齐)
  [FIX-12] Compromise Solutions 分支改为依据 dec_vars 长度自动填充，
            消除原 Condition 2/3 中 "if len==5: TA/IA/QA设None" 的 fragile 硬编码
  [FIX-13] Worker Pool 死锁问题彻底解决：
            新架构不再使用持久化 Pool，每次 rollout 临时 fork 进程，从根源消灭死锁。
  [FIX-14] _select_by_hv_contribution O(M³) 性能修复：
            新增第一级防护（MAX_BEFORE_HV=500 随机预采样）+ 第二级批量剪枝（BATCH模式），
            复杂度从 O(M³·logM) 降至 O(M²·logM/BATCH)，运行时间从数小时降至分钟级。
  [FIX-15] quality_metrics_ppo.csv 多轮覆盖问题：改为追加写入（mode='a'），
            三轮评估数据全部保留；新增 Strategy 列以区分各轮。
  [FIX-16] Feasibility_rate 语义修正：原错误地用宽松候选解数计算，
            现改为严格可行解计数 feas_count（costs_max<feasible_threshold）。
  [FIX-17] 删除废弃函数 _worker_rollout_persistent：调用不存在的
            _worker_predict_batch，若意外触发会抛 NameError，已彻底移除。
  [FIX-18] plot_pareto 中 _scatter ax 参数修复：原版内部硬编码 axes[0] 导致
            NSGA-II 点从未被画到左图（Cu-As），现将两个子图都作为参数显式传入。
  [FIX-19] evaluate() stochastic/nsga_perturb 策略单步推理瓶颈消除：
            原 _approx_profit 对 K=3 个候选动作逐一调用 predict_all_single，
            每步触发 3×3=9 次单点 ExtraTrees 推理；新 _approx_profit_batch 将
            K 个候选次态堆叠为 (K, state_dim) 矩阵，一次 predict_all_batch 完成，
            推理次数降为 3 次（批量），评估速度提升约 2–4×。

Optimization log:
  [OPT-1]  批量 ExtraTrees 推理：主进程统一批量预测，提高效率
  [OPT-2]  多进程并行 rollout：n_workers=4 CPU worker 并行采集，主进程专职 GPU update
  [OPT-3]  ExtraTrees n_jobs 控制：每模型单线程，避免 OpenMP 争抢
  [OPT-4]  每个约束独立 cost critic（5 头），数学上严格正确
  [OPT-5]  weights 在每个 episode 开始时重采样
  [OPT-6]  evaluate 记录 episode 内所有步的可行解
  [OPT-7]  BC 预训练后 PPO 阶段 lr warm-up（线性升至目标 lr）
  [OPT-8]  Pareto 解集去重 + hypervolume 指标输出
  [OPT-9]  训练曲线新增每约束 λ 变化子图
  [OPT-10] 极端权重采样：30% 概率某维度>0.8，提升 Pareto 前沿边界覆盖
  [OPT-11] 随机探索评估：stochastic 策略采样3次取最高 profit
  [OPT-12] 动态 cost_limit：训练后期按 0.95^(update//50) 衰减，0.05→0.02
  [OPT-13] 动态 entropy_coef：前半程 0.008，后半程 0.003
  [OPT-14] 分层评估策略：三轮（确定性/随机/NSGA扰动），合并 Pareto 过滤
  [OPT-CFG] total_steps=20M，充分探索训练空间

Improvement log (超越 NSGA-II 专项改进):
  [IMP-1] 解锚 entropy：BC 预训练后前 100 次 update 大幅提高 entropy_coef(0.05)，
           打破 NSGA 邻域锚定效应；后续恢复 0.008/0.003 动态策略
  [IMP-2] Dominance bonus：reward 中加入"支配 NSGA 解"的额外奖励(+0.5)，
           给 agent 明确的"超越 NSGA"激励信号；NSGA Pareto 点在主进程初始化时一次性载入
  [IMP-3] 宽松可行解收集：evaluate 阶段先用 costs_max<0.05 收集候选，
           最终输出仍严格过滤 costs_max<1e-3，大幅提升 Pareto 多样性
"""

# ── 必须在所有 import 之前设置，防止 worker 子进程 OpenMP 争抢 ────────────────
import os
os.environ['OMP_NUM_THREADS']      = '1'
os.environ['MKL_NUM_THREADS']      = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS']  = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import time
import warnings
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from joblib import load
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent

# =============================================================================
# ── 0. 进度追踪工具（Progress Tracker） ──────────────────────────────────────
# =============================================================================

class ProgressTracker:
    """
    统一进度显示工具，支持：
      - 阶段级进度（stage）：显示当前处于哪一大阶段
      - 步骤级进度（step）：阶段内子步骤进度条
      - 时间统计：已用时 / 预估剩余时间
      - 实时指标打印
    使用方法：
      tracker = ProgressTracker(total_stages=6)
      tracker.start_stage(1, "Loading surrogate models")
      tracker.update_step(current, total, extra_info="loss=0.01")
      tracker.end_stage("Done. 3 models loaded.")
    """

    _BAR_WIDTH = 30   # 进度条宽度（字符数）
    _SEP = "─" * 68   # 分隔线

    def __init__(self, total_stages: int = 6):
        self.total_stages   = total_stages
        self.current_stage  = 0
        self.stage_name     = ""
        self._t_global      = time.time()
        self._t_stage       = time.time()
        self._t_step        = time.time()
        self._last_pct      = -1   # 避免重复打印相同百分比

    # ── 内部工具 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """将秒格式化为可读字符串，如 '1h23m' 或 '45s'。"""
        seconds = max(0.0, seconds)
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}min"
        else:
            return f"{seconds/3600:.1f}h"

    @staticmethod
    def _bar(current: int, total: int, width: int = 30) -> str:
        """生成 ASCII 进度条，如 '[████████░░░░░░] 53%'。"""
        if total <= 0:
            return f"[{'?'*width}] ?%"
        pct  = min(1.0, current / total)
        done = int(pct * width)
        return (f"[{'█' * done}{'░' * (width - done)}]"
                f" {pct*100:5.1f}%  {current}/{total}")

    # ── 公开接口 ──────────────────────────────────────────────────────────────
    def start_stage(self, stage_num: int, description: str):
        """标记新阶段开始，打印标题横幅。"""
        self.current_stage = stage_num
        self.stage_name    = description
        self._t_stage      = time.time()
        self._last_pct     = -1
        global_elapsed = self._fmt_time(time.time() - self._t_global)
        print(f"\n{self._SEP}")
        print(f"  [阶段 {stage_num}/{self.total_stages}]  {description}")
        print(f"  全局已用时: {global_elapsed}")
        print(self._SEP)

    def end_stage(self, summary: str = ""):
        """标记当前阶段结束，打印耗时摘要。"""
        elapsed = self._fmt_time(time.time() - self._t_stage)
        tag = f"✔ 阶段 {self.current_stage} 完成"
        if summary:
            print(f"  {tag}  |  {summary}  |  耗时 {elapsed}")
        else:
            print(f"  {tag}  |  耗时 {elapsed}")

    def update_step(self, current: int, total: int, prefix: str = "",
                    extra: str = "", min_interval: float = 2.0):
        """
        打印子步骤进度条（节流：默认每 min_interval 秒最多打印一次）。
        current: 当前步（从 1 开始）
        total  : 总步数
        prefix : 行前缀，如 "  BC epoch"
        extra  : 附加信息，如 "loss=0.0123"
        """
        now = time.time()
        pct = int(current / max(total, 1) * 100)
        # 节流：间隔不足且百分比未变化则跳过
        if (now - self._t_step < min_interval) and (pct == self._last_pct):
            return
        self._t_step  = now
        self._last_pct = pct

        stage_elapsed = self._fmt_time(now - self._t_stage)
        # 估算剩余时间
        if current > 0:
            per_step = (now - self._t_stage) / current
            remain   = self._fmt_time(per_step * (total - current))
            eta_str  = f"剩余≈{remain}"
        else:
            eta_str = "计算中..."

        bar = self._bar(current, total, self._BAR_WIDTH)
        line = f"  {prefix}  {bar}"
        if extra:
            line += f"  | {extra}"
        line += f"  | 段耗时 {stage_elapsed}  {eta_str}"
        print(line, flush=True)

    def update_step_force(self, current: int, total: int, prefix: str = "",
                          extra: str = ""):
        """强制打印进度（无节流），用于关键节点。"""
        self._t_step = 0.0  # 清零节流计时器
        self.update_step(current, total, prefix=prefix, extra=extra, min_interval=0.0)

    def print_info(self, msg: str, indent: int = 2):
        """打印普通信息行（带缩进）。"""
        print(" " * indent + msg, flush=True)

    def print_metric(self, label: str, value, unit: str = ""):
        """打印单条指标，如 '    Cu_out  : 5.23 g/L'。"""
        val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"    {label:<22s}: {val_str} {unit}", flush=True)

    def print_section(self, title: str):
        """打印二级小节标题。"""
        print(f"\n  ┌─ {title} {'─'*(50-len(title))}┐", flush=True)

    def print_section_end(self):
        print(f"  └{'─'*53}┘", flush=True)


# 全局唯一 tracker 实例（在 main 中初始化）
_TRACKER: ProgressTracker = None


def _get_tracker() -> ProgressTracker:
    """获取全局 tracker，若未初始化则返回一个静默 tracker。"""
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = ProgressTracker(total_stages=6)
    return _TRACKER


# ── 全局 HV 参考点（内联定义，无需外部文件）───────────────────────────────────
# 目标顺序: [Cu_out, -As_out, E_total, -R_profit]（全部最小化方向）
# 参考点位于所有算法可能探索到的最差物理边界之外
GLOBAL_HV_REF_POINT = np.array([
    12.0,    # Cu_out：物理上限约 8，给足余量
    -0.0,    # -As_out：As 不可能为负，取 -0.0 确保涵盖所有负值
    1.0e6,   # E_total：极大耗电量上界
    1.0e6,   # -R_profit：极大亏损上界
])


# =============================================================================
# ── 1. 命令行解析 ─────────────────────────────────────────────────────────────
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='PPO-Lagrangian Safe RL — Copper Electrowinning')
    parser.add_argument('--condition', type=int, choices=[1, 2, 3], required=True,
                        help='工况编号：1=Unit A / 2=Unit B / 3=Serial A+B')
    parser.add_argument('--fast', action='store_true',
                        help='快速模式：减少计算量用于调试')
    return parser.parse_args()


# =============================================================================
# ── 2. 工况配置（仅差异部分，公共参数统一写在 BASE_CFG） ──────────────────────
# =============================================================================

# ── 所有条件共享的超参数 ──────────────────────────────────────────────────────
BASE_CFG = dict(
    # 路径配置
    model_dir        = str(PROJECT_DIR / "joblib"),        # 代理模型保存路径
    nsga_path        = str(PROJECT_DIR / "NSGA.xlsx"),     # NSGA-II 解的Excel文件路径
    rl_xlsx_template = str(PROJECT_DIR / "RL.xlsx"),       # RL结果Excel模板路径

    # 经济参数
    p_Cu=98.44,              # 铜价（元/吨）
    c_reprocess=0.14,        # 再处理成本
    c_copper_concentrate=82.64, # 铜精矿成本
    p_e=0.6,                 # 电价（元/kWh）
    c_op=200,                # 操作成本
    c_As_treat=2.5,          # 砷处理成本

    # 阴极规格
    n_cathode=35,            # 阴极数量
    A_cathode=1.1 * 1.029 * 2, # 单个阴极面积（m²）

    # 安全约束阈值（参见 Table S7）
    Cu_limit=8.0,            # 铜离子浓度上限（g/L）
    As_min=4.0,              # 砷离子浓度下限（g/L）
    J_max=340.0,             # 电流密度上限（A/m²）
    V_cell_max=2.5,          # 单槽电压上限（V）
    N_cell=16,               # 电解槽数量
    Cu_As_ratio_max=0.8,     # 铜砷浓度比上限

    # 公共维度
    weight_dim=4,             # 权重向量维度（对应4个目标）
    n_constraints=5,          # 约束数量

    # 奖励缩放：[Cu_out, -As_out, E_total, -profit]
    scale=[10.0, 10.0, 1000.0, 5000.0],

    # Episode / 训练参数
    max_ep_steps=30,          # 每个episode的最大步数
    device='cuda',            # 计算设备（'cuda'或'cpu'）
    n_workers=4,              # CPU worker数量（并行采集轨迹）
    per_worker_steps=4000,    # 每个worker每次rollout的步数
    batch_size=2048,          # PPO批量大小
    total_steps=20_000_000,   # 总训练步数

    # PPO 超参数
    lr_actor=1.5e-4,          # 策略网络学习率
    gamma=0.95,               # 折扣因子
    lam_gae=0.95,             # GAE lambda参数
    eps_clip=0.2,             # PPO裁剪参数
    ppo_epochs=6,             # 每个batch的PPO更新轮数

    # Lagrangian 约束参数
    entropy_coef=0.005,       # 熵正则化系数（基础值）
    lr_lambda=1e-3,           # Lagrangian乘子学习率
    cost_limit=0.05,          # 约束违反阈值（动态衰减）

    # BC 预训练
    bc_epochs=150,            # BC预训练轮数
    bc_lr=3e-4,               # BC预训练学习率

    # PPO lr warm-up
    warmup_updates=150,       # 线性warm-up更新次数

    # 评估参数
    eval_episodes=3000,        # 总评估轮数
    traj_limit=200,            # 轨迹记录限制（避免内存爆炸）
    feasible_threshold=1e-3,   # 严格可行解阈值
    candidate_threshold=0.05,   # 宽松候选解阈值
    pareto_target_size=300,    # Pareto解集目标大小

    seed=42,                   # 随机种子
)

# ── 快速调试模式（--fast 时覆盖上述参数）──────────────────────────────────────
FAST_CFG = dict(
    n_workers=4,          # 相同
    per_worker_steps=2000, # 普通=4000, 100
    batch_size=1024,       # 普通=2048, 512
    total_steps=10_000_000,    # 普通=20_000_000, 5_000
    bc_epochs=75,          # 普通=150, 2
    warmup_updates=75,     # 普通=150, 2
    eval_episodes=1500,     # 普通=3000, 10
    traj_limit=100,         # 普通=200, 5
    pareto_target_size=150, # 普通=300, 10
)

# ── 各 condition 特有字段 ─────────────────────────────────────────────────────
_CONDITION_OVERRIDES = {
    1: dict(
        output_dir = str(PROJECT_DIR / "outputs_condition1"),
        # State: [Cu_in, TA, IA, QA, t]  Action: [ΔCu_in, ΔTA, ΔIA, ΔQA]
        state_dim=5, action_dim=4,
        Cu_in_range=(29.0, 55.0),
        T_range=(48.0, 65.0),   # TA
        I_range=(8000.0, 27000.0),
        Q_range=(111.0, 123.0),
        t_range=(2.0, 8.0),
        dCu_in=2.0, dT=2.0, dI=1000.0, dQ=1.0,
        model_subdir='three_stage',
        model_suffix='three_stage',
    ),
    2: dict(
        output_dir = str(PROJECT_DIR / "outputs_condition2"),
        # State: [Cu_in, TB, IB, QB, t]  Action: [ΔCu_in, ΔTB, ΔIB, ΔQB]
        # [FIX-10] Condition 2 决策变量为 TB/IB/QB（非 TA/IA/QA）
        state_dim=5, action_dim=4,
        Cu_in_range=(29.0, 55.0),
        T_range=(55.0, 65.0),   # TB
        I_range=(8000.0, 27000.0),
        Q_range=(113.0, 123.0),
        t_range=(2.0, 8.0),
        dCu_in=2.0, dT=2.0, dI=1000.0, dQ=1.0,
        model_subdir='four_stage',
        model_suffix='four_stage',
    ),
    3: dict(
        output_dir = str(PROJECT_DIR / "outputs_condition3"),
        # State: [Cu_in, TA, IA, QA, TB, IB, QB, t]
        # Action: [ΔCu_in, ΔTA, ΔIA, ΔQA, ΔTB, ΔIB, ΔQB]
        state_dim=8, action_dim=7,
        Cu_in_range=(29.0, 55.0),
        TA_range=(40.0, 65.0), IA_range=(8000.0, 27000.0), QA_range=(111.0, 123.0),
        TB_range=(40.0, 65.0), IB_range=(8000.0, 27000.0), QB_range=(113.0, 123.0),
        t_range=(2.0, 8.0),
        dCu_in=2.0, dTA=2.0, dIA=1000.0, dQA=1.0,
        dTB=2.0,    dIB=1000.0, dQB=1.0,
        model_subdir='serial',
        model_suffix='serial',
    ),
}

# ── 全局 CFG / 派生常量（在 make_cfg 中初始化）──────────────────────────────
CFG      = None  # 全局配置字典
_J_DENOM = None  # 电流密度分母（n_cathode * A_cathode）
_SCALE_V = None  # 奖励缩放向量
_DELTA_SC = None # 动作缩放因子
_S_LO    = None  # 状态下界
_S_HI    = None  # 状态上界
DEVICE   = None  # 计算设备（cuda/cpu）
CONDITION = None # 工况编号（1/2/3）


def make_cfg(condition: int, fast: bool = False):
    """合并 BASE_CFG 与对应条件的 override，初始化所有全局常量。"""
    global CFG, _J_DENOM, _SCALE_V, _DELTA_SC, _S_LO, _S_HI, DEVICE, CONDITION

    CONDITION = condition
    CFG = {**BASE_CFG, **_CONDITION_OVERRIDES[condition]}
    if fast:
        CFG = {**CFG, **FAST_CFG}

    _J_DENOM  = CFG['n_cathode'] * CFG['A_cathode']
    _SCALE_V  = np.array(CFG['scale'], dtype=np.float32)

    if condition == 1:
        _DELTA_SC = np.array([CFG['dCu_in'], CFG['dT'], CFG['dI'], CFG['dQ']], dtype=np.float32)
        _S_LO = np.array([CFG['Cu_in_range'][0], CFG['T_range'][0],
                          CFG['I_range'][0], CFG['Q_range'][0],
                          CFG['t_range'][0]], dtype=np.float32)
        _S_HI = np.array([CFG['Cu_in_range'][1], CFG['T_range'][1],
                          CFG['I_range'][1], CFG['Q_range'][1],
                          CFG['t_range'][1]], dtype=np.float32)
    elif condition == 2:
        _DELTA_SC = np.array([CFG['dCu_in'], CFG['dT'], CFG['dI'], CFG['dQ']], dtype=np.float32)
        _S_LO = np.array([CFG['Cu_in_range'][0], CFG['T_range'][0],
                          CFG['I_range'][0], CFG['Q_range'][0],
                          CFG['t_range'][0]], dtype=np.float32)
        _S_HI = np.array([CFG['Cu_in_range'][1], CFG['T_range'][1],
                          CFG['I_range'][1], CFG['Q_range'][1],
                          CFG['t_range'][1]], dtype=np.float32)
    else:  # condition == 3
        _DELTA_SC = np.array([
            CFG['dCu_in'],
            CFG['dTA'], CFG['dIA'], CFG['dQA'],
            CFG['dTB'], CFG['dIB'], CFG['dQB'],
        ], dtype=np.float32)
        _S_LO = np.array([
            CFG['Cu_in_range'][0],
            CFG['TA_range'][0], CFG['IA_range'][0], CFG['QA_range'][0],
            CFG['TB_range'][0], CFG['IB_range'][0], CFG['QB_range'][0],
            CFG['t_range'][0],
        ], dtype=np.float32)
        _S_HI = np.array([
            CFG['Cu_in_range'][1],
            CFG['TA_range'][1], CFG['IA_range'][1], CFG['QA_range'][1],
            CFG['TB_range'][1], CFG['IB_range'][1], CFG['QB_range'][1],
            CFG['t_range'][1],
        ], dtype=np.float32)

    torch.manual_seed(CFG['seed'])
    np.random.seed(CFG['seed'])

    # [FIX-9] CUDA 不可用时自动回退 CPU
    _dev = CFG['device']
    if _dev == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA unavailable, falling back to CPU.")
        _dev = 'cpu'
    DEVICE = torch.device(_dev)

    os.makedirs(CFG['output_dir'], exist_ok=True)
    print(f"[CFG] Condition {condition}  |  device={DEVICE}"
          f"  |  state_dim={CFG['state_dim']}  action_dim={CFG['action_dim']}"
          f"  |  output → {CFG['output_dir']}")


def _out(fname: str) -> str:
    """构造输出文件路径（需在 make_cfg 之后调用）。"""
    return os.path.join(CFG['output_dir'], fname)


# =============================================================================
# ── 3. Surrogate 模型加载 ─────────────────────────────────────────────────────
# =============================================================================

def load_models(model_dir: str, n_jobs: int = 1) -> dict:
    """
    [OPT-3] n_jobs 控制每模型内部线程数。
    worker 进程用 n_jobs=1 避免 OpenMP 争抢；主进程 feature-check 用 n_jobs=-1。
    """
    suffix = CFG['model_suffix']
    models = {
        'Cu': load(f"{model_dir}/cu_{CFG['model_subdir']}/extra_trees_{suffix}.joblib"),
        'As': load(f"{model_dir}/as_{CFG['model_subdir']}/extra_trees_{suffix}.joblib"),
        'V':  load(f"{model_dir}/voltage_{CFG['model_subdir']}/extra_trees_{suffix}.joblib"),
    }
    for m in models.values():
        if hasattr(m, 'n_jobs'):
            m.n_jobs = n_jobs
    return models


# =============================================================================
# ── 4. 特征构建 & 批量预测（按 condition 分支） ───────────────────────────────
# =============================================================================

def _build_features_c1c2(Cu_in_arr, T_arr, I_arr, Q_arr, V_arr=None,
                          month=6, day=15, year=2025, hour=8):
    """
    Condition 1/2 公用特征构建。
    V_arr=None → 8维电压特征；V_arr!=None → 10维浓度特征（含 V 和 power）。
    """
    N = len(Cu_in_arr)
    if V_arr is None:
        feat = np.empty((N, 8), dtype=np.float32)
        feat[:, 0] = Cu_in_arr; feat[:, 1] = T_arr
        feat[:, 2] = I_arr;     feat[:, 3] = Q_arr
        feat[:, 4] = year;      feat[:, 5] = month
        feat[:, 6] = day;       feat[:, 7] = hour
    else:
        feat = np.empty((N, 10), dtype=np.float32)
        feat[:, 0] = Cu_in_arr; feat[:, 1] = T_arr
        feat[:, 2] = I_arr;     feat[:, 3] = Q_arr
        feat[:, 4] = V_arr;     feat[:, 5] = year
        feat[:, 6] = month;     feat[:, 7] = day
        feat[:, 8] = hour;      feat[:, 9] = V_arr * I_arr   # stage_power
    return feat


def _build_features_c3_voltage(Cu_in_arr, TA_arr, IA_arr, QA_arr,
                                TB_arr, IB_arr, QB_arr,
                                month=6, day=15, year=2025, hour=8):
    """Condition 3 电压模型特征，shape=(N, 11)。"""
    N = len(Cu_in_arr)
    feat = np.empty((N, 11), dtype=np.float32)
    feat[:, 0] = Cu_in_arr; feat[:, 1] = TA_arr; feat[:, 2] = IA_arr; feat[:, 3] = QA_arr
    feat[:, 4] = TB_arr;    feat[:, 5] = IB_arr; feat[:, 6] = QB_arr
    feat[:, 7] = year;      feat[:, 8] = month;  feat[:, 9] = day; feat[:, 10] = hour
    return feat


def _build_features_c3_conc(Cu_in_arr, TA_arr, IA_arr, QA_arr, VA_arr,
                              TB_arr, IB_arr, QB_arr, VB_arr, total_power_arr,
                              month=6, day=15, year=2025, hour=8):
    """Condition 3 浓度模型特征，shape=(N, 14)。"""
    N = len(Cu_in_arr)
    feat = np.empty((N, 14), dtype=np.float32)
    feat[:, 0] = Cu_in_arr; feat[:, 1] = TA_arr; feat[:, 2] = IA_arr; feat[:, 3] = QA_arr
    feat[:, 4] = VA_arr;    feat[:, 5] = TB_arr; feat[:, 6] = IB_arr; feat[:, 7] = QB_arr
    feat[:, 8] = VB_arr;    feat[:, 9] = year;   feat[:, 10] = month; feat[:, 11] = day
    feat[:, 12] = hour;     feat[:, 13] = total_power_arr  # VA*IA + VB*IB
    return feat


def predict_all_batch(models, *state_cols, month=6, day=15, year=2025, hour=8):
    """
    [OPT-1] 对整批状态做级联预测 V → Cu → As。

    Condition 1/2: state_cols = (Cu_in, T, I, Q, t)
                   返回 (Cu_out, As_out, V)
    Condition 3:   state_cols = (Cu_in, TA, IA, QA, TB, IB, QB, t)
                   返回 (Cu_out, As_out, VMean, VA, VB)
    """
    if CONDITION in (1, 2):
        Cu_in_arr, T_arr, I_arr, Q_arr, t_arr = state_cols
        fv = _build_features_c1c2(Cu_in_arr, T_arr, I_arr, Q_arr, None,
                                   month, day, year, hour)
        V_arr = models['V'].predict(fv).astype(np.float32)
        fc = _build_features_c1c2(Cu_in_arr, T_arr, I_arr, Q_arr, V_arr,
                                   month, day, year, hour)
        Cu_out = models['Cu'].predict(fc).astype(np.float32)
        As_out = models['As'].predict(fc).astype(np.float32)
        return Cu_out, As_out, V_arr

    else:  # Condition 3
        Cu_in_arr, TA_arr, IA_arr, QA_arr, TB_arr, IB_arr, QB_arr, t_arr = state_cols
        fv = _build_features_c3_voltage(Cu_in_arr, TA_arr, IA_arr, QA_arr,
                                         TB_arr, IB_arr, QB_arr, month, day, year, hour)
        VMean_arr = models['V'].predict(fv).astype(np.float32)
        # 串联：VA = VB = VMean（统一均值电压）
        VA_arr = VB_arr = VMean_arr
        tp_arr = VA_arr * IA_arr + VB_arr * IB_arr
        fc = _build_features_c3_conc(Cu_in_arr, TA_arr, IA_arr, QA_arr, VA_arr,
                                      TB_arr, IB_arr, QB_arr, VB_arr, tp_arr,
                                      month, day, year, hour)
        Cu_out = models['Cu'].predict(fc).astype(np.float32)
        As_out = models['As'].predict(fc).astype(np.float32)
        return Cu_out, As_out, VMean_arr, VA_arr, VB_arr


def predict_all_single(models, *state_scalars, month=6, day=15, year=2025, hour=8):
    """单步预测（供 feature-check 和 evaluate._approx_profit 调用）。"""
    if CONDITION in (1, 2):
        Cu_in, T, I, Q, t = state_scalars
        fv = np.array([Cu_in, T, I, Q, year, month, day, hour], dtype=np.float32)
        V = float(models['V'].predict(fv.reshape(1, -1))[0])
        fc = np.array([Cu_in, T, I, Q, V, year, month, day, hour, V * I], dtype=np.float32)
        Cu_out = float(models['Cu'].predict(fc.reshape(1, -1))[0])
        As_out = float(models['As'].predict(fc.reshape(1, -1))[0])
        return Cu_out, As_out, V
    else:
        Cu_in, TA, IA, QA, TB, IB, QB, t = state_scalars
        fv = np.array([Cu_in, TA, IA, QA, TB, IB, QB, year, month, day, hour], dtype=np.float32)
        VMean = float(models['V'].predict(fv.reshape(1, -1))[0])
        tp = VMean * IA + VMean * IB
        fc = np.array([Cu_in, TA, IA, QA, VMean, TB, IB, QB, VMean,
                       year, month, day, hour, tp], dtype=np.float32)
        Cu_out = float(models['Cu'].predict(fc.reshape(1, -1))[0])
        As_out = float(models['As'].predict(fc.reshape(1, -1))[0])
        return Cu_out, As_out, VMean


def check_feature_alignment(models: dict) -> bool:
    """快速检查模型期望特征数与实际构建特征数是否一致。"""
    try:
        if CONDITION == 1:
            Cu_out, As_out, V = predict_all_single(models, 40, 58, 15000, 117, 4)
            n_v, n_c = 8, 10
        elif CONDITION == 2:
            Cu_out, As_out, V = predict_all_single(models, 40, 61, 15000, 118, 4)
            n_v, n_c = 8, 10
        else:
            Cu_out, As_out, V = predict_all_single(models, 40, 58, 15000, 117, 55, 12000, 118, 4)
            n_v, n_c = 11, 14
        print(f"[Feature check OK]")
        print(f"  Voltage model: expects {models['V'].n_features_in_}  got {n_v}")
        print(f"  Cu/As  model:  expects {models['Cu'].n_features_in_}  got {n_c}")
        print(f"  Test predict: V={V:.2f}V  Cu_out={Cu_out:.2f}g/L  As_out={As_out:.2f}g/L")
        assert models['V'].n_features_in_ == n_v
        assert models['Cu'].n_features_in_ == n_c
        return True
    except Exception as e:
        print(f"[Feature check FAILED] {e}")
        for nm, m in models.items():
            if hasattr(m, 'n_features_in_'):
                print(f"  Model '{nm}': n_features_in_={m.n_features_in_}")
        return False


# =============================================================================
# ── 5. 目标函数 & 约束代价（按 condition 分支） ───────────────────────────────
# =============================================================================

def compute_objectives_batch(state_cols, Cu_out_arr, As_out_arr, V_out):
    """
    向量化经济目标计算，返回 (E_arr, profit_arr)。

    profit 公式（全条件统一）：
      m_Cu   = max(0, (Cu_in - Cu_out) × Q × t)
      R_Cu   = (p_Cu - c_reprocess - c_copper_concentrate) × m_Cu
      R_save = c_As_treat × (11.64 + Cu_out) × Q × t   [As_out 固定为行业最大值]
      profit = (R_Cu - R_save - p_e×E - c_op×t) / 1e4
    """
    As_fixed = 11.64  # 危废处置基准：固定 As_out，使成本偏保守
    if CONDITION in (1, 2):
        Cu_in_arr, T_arr, I_arr, Q_arr, t_arr = state_cols
        V_arr = V_out                          # shape (N,)
        E_arr      = V_arr * I_arr * t_arr / 1000.0
        m_Cu_arr   = np.maximum(0.0, (Cu_in_arr - Cu_out_arr) * Q_arr * t_arr)
        R_Cu_arr   = (CFG['p_Cu'] - CFG['c_reprocess'] - CFG['c_copper_concentrate']) * m_Cu_arr
        R_save_arr = CFG['c_As_treat'] * (As_fixed + Cu_out_arr) * Q_arr * t_arr
        profit_arr = (R_Cu_arr - R_save_arr - CFG['p_e'] * E_arr - CFG['c_op'] * t_arr) / 1e4
    else:
        Cu_in_arr, TA_arr, IA_arr, QA_arr, TB_arr, IB_arr, QB_arr, t_arr = state_cols
        VMean_arr, VA_arr, VB_arr = V_out      # tuple of three (N,) arrays
        # 串联能耗 = Unit A + Unit B
        E_arr      = (VA_arr * IA_arr + VB_arr * IB_arr) * t_arr / 1000.0
        m_Cu_arr   = np.maximum(0.0, (Cu_in_arr - Cu_out_arr) * QA_arr * t_arr)
        R_Cu_arr   = (CFG['p_Cu'] - CFG['c_reprocess'] - CFG['c_copper_concentrate']) * m_Cu_arr
        R_save_arr = CFG['c_As_treat'] * (As_fixed + Cu_out_arr) * QA_arr * t_arr
        profit_arr = (R_Cu_arr - R_save_arr - CFG['p_e'] * E_arr - CFG['c_op'] * t_arr) / 1e4
    return E_arr, profit_arr


def compute_costs_batch(Cu_out_arr, As_out_arr, I_arr_or_tuple, V_arr_or_tuple) -> np.ndarray:
    """
    向量化约束代价计算，输出 shape=(N, 5)。

    约束（Table S7）：
      C1: Cu_out <= Cu_limit          → max(0, Cu_out - Cu_limit)
      C2: As_out >= As_min            → max(0, As_min - As_out)   [下限约束]
      C3: J = I / (n_cathode×A) <= J_max
      C4: V_cell = V / N_cell <= V_cell_max
      C5: Cu_out/As_out <= Cu_As_ratio_max

    Condition 3: C3/C4 取两单元较大值（串联系统取严格侧）。
    """
    if CONDITION in (1, 2):
        I_arr = I_arr_or_tuple
        V_arr = V_arr_or_tuple
        J_arr     = I_arr / _J_DENOM
        Vcell_arr = V_arr / CFG['N_cell']
    else:
        IA_arr, IB_arr = I_arr_or_tuple
        VA_arr, VB_arr = V_arr_or_tuple
        J_arr     = np.maximum(IA_arr, IB_arr) / _J_DENOM
        Vcell_arr = np.maximum(VA_arr, VB_arr) / CFG['N_cell']

    ratio_arr = Cu_out_arr / (As_out_arr + 1e-8)
    c = np.stack([
        np.maximum(0.0, Cu_out_arr - CFG['Cu_limit']),     # C1
        np.maximum(0.0, CFG['As_min'] - As_out_arr),       # C2：下限约束
        np.maximum(0.0, J_arr      - CFG['J_max']),        # C3
        np.maximum(0.0, Vcell_arr  - CFG['V_cell_max']),   # C4
        np.maximum(0.0, ratio_arr  - CFG['Cu_As_ratio_max']),  # C5
    ], axis=1)
    return c.astype(np.float32)


# =============================================================================
# ── 6. Actor-Critic 网络 ──────────────────────────────────────────────────────
# =============================================================================

class ActorCritic(nn.Module):
    """
    Weight-conditioned PPO Actor-Critic，带 [OPT-4] 独立 cost critic 头。

    输入：[state(state_dim) ‖ weight(weight_dim)]
    输出：
      dist  — Normal(mu, std) 动作分布
      v_r   — 奖励值函数  shape=(B,)
      v_c   — 代价值函数  shape=(B, n_constraints)
    """
    def __init__(self, s_dim: int, a_dim: int, w_dim: int = 4,
                 hidden: int = 128, n_constraints: int = 5):
        super().__init__()
        inp = s_dim + w_dim
        self.n_constraints = n_constraints

        # 共享骨干（Actor + 所有 Critic 共用底层表示）
        self.shared = nn.Sequential(
            nn.Linear(inp, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # Actor head
        self.mu      = nn.Linear(hidden, a_dim)
        self.log_std = nn.Parameter(torch.zeros(a_dim))
        # 奖励 Critic
        self.reward_critic = nn.Sequential(
            nn.Linear(inp, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # [OPT-4] 每个约束独立一个 cost critic 头，数学上严格
        self.cost_critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inp, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, 1),
            ) for _ in range(n_constraints)
        ])
        nn.init.orthogonal_(self.mu.weight, 0.01)

    def forward(self, s: torch.Tensor, w: torch.Tensor):
        x = torch.cat([s, w], dim=-1)
        h = self.shared(x)
        mu  = torch.tanh(self.mu(h))
        std = self.log_std.exp().clamp(1e-3, 1.0)
        v_r = self.reward_critic(x).squeeze(-1)
        v_c = torch.cat([cc(x) for cc in self.cost_critics], dim=-1)
        return Normal(mu, std), v_r, v_c


# =============================================================================
# ── 7. GAE ────────────────────────────────────────────────────────────────────
# =============================================================================

def compute_gae(rewards: np.ndarray, values: np.ndarray,
                dones: np.ndarray, next_val: float) -> np.ndarray:
    """广义优势估计（GAE）；奖励和每条约束代价分别独立调用。"""
    adv = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        nv = next_val if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + CFG['gamma'] * nv * (1 - dones[t]) - values[t]
        gae   = delta + CFG['gamma'] * CFG['lam_gae'] * (1 - dones[t]) * gae
        adv[t] = gae
    return adv


# =============================================================================
# ── 8. Worker 进程（无模型版，只做轨迹采样） ──────────────────────────────────
# [REFACTOR] 彻底去掉持久化Pool和多进程模型加载，从根源消灭死锁。
# 新架构：worker只做策略推理+状态转移（纯numpy/torch），
#         主进程拿到raw轨迹后统一做ExtraTrees批量预测。
# =============================================================================

def _worker_sample_only(args):
    """
    新架构的 worker 函数：只负责轨迹采样，不加载任何 ExtraTrees 模型。
    主进程拿到 nxt_states 后统一批量预测，彻底消灭多进程模型加载问题。
    
    参数:
        args: 包含 worker_id, net_state_dict, cfg, delta_sc, s_lo, s_hi, nsga_states, steps, seed 的元组
    
    返回值:
        包含原始轨迹数据的字典，包括状态、动作、对数概率、下一状态、权重和终止标志
    """

    (worker_id, net_state_dict, cfg, delta_sc, s_lo, s_hi,
     nsga_states, steps, seed) = args

    np.random.seed(seed + worker_id * 1000)
    torch.manual_seed(seed + worker_id * 1000)

    net = ActorCritic(cfg['state_dim'], cfg['action_dim'],
                      cfg['weight_dim'], n_constraints=cfg['n_constraints'])
    net.load_state_dict(net_state_dict)
    net.eval()

    cond = cfg.get('_condition', 1)

    def _sample_state():
        if nsga_states is not None and np.random.rand() < 0.5:
            s = nsga_states[np.random.randint(len(nsga_states))].copy().astype(np.float32)
        else:
            if cond in (1, 2):
                s = np.array([
                    np.random.uniform(*cfg['Cu_in_range']),
                    np.random.uniform(*cfg['T_range']),
                    np.random.uniform(*cfg['I_range']),
                    np.random.uniform(*cfg['Q_range']),
                    np.random.uniform(*cfg['t_range']),
                ], dtype=np.float32)
            else:
                s = np.array([
                    np.random.uniform(*cfg['Cu_in_range']),
                    np.random.uniform(*cfg['TA_range']),
                    np.random.uniform(*cfg['IA_range']),
                    np.random.uniform(*cfg['QA_range']),
                    np.random.uniform(*cfg['TB_range']),
                    np.random.uniform(*cfg['IB_range']),
                    np.random.uniform(*cfg['QB_range']),
                    np.random.uniform(*cfg['t_range']),
                ], dtype=np.float32)
        return np.clip(s, s_lo, s_hi)

    def _sample_weights():
        if np.random.rand() < 0.3:
            e = np.zeros(4, dtype=np.float32)
            e[np.random.randint(4)] = 0.85
            e += np.random.dirichlet([0.5] * 4).astype(np.float32) * 0.15
            return e / e.sum()
        return np.random.dirichlet([1.0] * 4).astype(np.float32)

    raw_states, raw_actions, raw_logps = [], [], []
    raw_weights, raw_dones, raw_nxt_st = [], [], []
    state   = _sample_state()
    weights = _sample_weights()
    ep_step = 0

    with torch.no_grad():
        for _ in range(steps):
            s_t = torch.FloatTensor(state).unsqueeze(0)
            w_t = torch.FloatTensor(weights).unsqueeze(0)
            dist, _, _ = net(s_t, w_t)
            action = dist.sample()
            lp     = dist.log_prob(action).sum(-1).item()
            a_np   = action.numpy()[0]
            delta  = a_np * delta_sc
            new_s  = state.copy()
            dim    = len(delta)
            new_s[:dim] = np.clip(state[:dim] + delta, s_lo[:dim], s_hi[:dim])
            new_s = np.clip(new_s, s_lo, s_hi)

            raw_states.append(state.copy())
            raw_actions.append(a_np.copy())
            raw_logps.append(lp)
            raw_weights.append(weights.copy())
            raw_nxt_st.append(new_s.copy())
            ep_step += 1
            done = (ep_step >= cfg['max_ep_steps'])
            raw_dones.append(float(done))
            state = new_s
            if done:
                state   = _sample_state()
                weights = _sample_weights()
                ep_step = 0

    return dict(
        s   = np.array(raw_states,  dtype=np.float32),
        a   = np.array(raw_actions, dtype=np.float32),
        lp  = np.array(raw_logps,   dtype=np.float32),
        nxt = np.array(raw_nxt_st,  dtype=np.float32),
        w   = np.array(raw_weights, dtype=np.float32),
        d   = np.array(raw_dones,   dtype=np.float32),
    )


def _predict_batch_main(nxt: np.ndarray, models: dict):
    """
    主进程版批量预测函数。
    直接使用主进程已加载的models，对批量状态进行预测，接口与原worker版一致。
    
    参数:
        nxt: 批量的下一状态数组
        models: 包含Cu、As、V模型的字典
    
    返回值:
        预测结果，包括Cu_out, As_out, V相关值, E_arr, profit_arr, costs_b
    """

    c = CONDITION
    if c in (1, 2):
        fv     = _w_build_c1c2_volt(nxt[:, 0], nxt[:, 1], nxt[:, 2], nxt[:, 3])
        V_arr  = models['V'].predict(fv).astype(np.float32)
        fc     = _w_build_c1c2_conc(nxt[:, 0], nxt[:, 1], nxt[:, 2], nxt[:, 3],
                                     V_arr, nxt[:, 2])
        Cu_out = models['Cu'].predict(fc).astype(np.float32)
        As_out = models['As'].predict(fc).astype(np.float32)
        E_arr, profit_arr = _w_obj_c1c2(nxt[:, 0], nxt[:, 2], nxt[:, 3], nxt[:, 4],
                                         Cu_out, As_out, V_arr)
        costs_b = _w_cost_c1c2(Cu_out, As_out, nxt[:, 2], V_arr)
        return Cu_out, As_out, V_arr, None, None, E_arr, profit_arr, costs_b
    else:
        fv    = _w_build_c3_volt(nxt[:, 0], nxt[:, 1], nxt[:, 2], nxt[:, 3],
                                  nxt[:, 4], nxt[:, 5], nxt[:, 6])
        VMean = models['V'].predict(fv).astype(np.float32)
        VA_arr = VB_arr = VMean
        tp    = VA_arr * nxt[:, 2] + VB_arr * nxt[:, 5]
        fc    = _w_build_c3_conc(nxt[:, 0], nxt[:, 1], nxt[:, 2], nxt[:, 3], VA_arr,
                                  nxt[:, 4], nxt[:, 5], nxt[:, 6], VB_arr, tp)
        Cu_out = models['Cu'].predict(fc).astype(np.float32)
        As_out = models['As'].predict(fc).astype(np.float32)
        E_arr, profit_arr = _w_obj_c3(nxt[:, 0], nxt[:, 1], nxt[:, 2], nxt[:, 3],
                                       nxt[:, 4], nxt[:, 5], nxt[:, 6], nxt[:, 7],
                                       Cu_out, As_out, VMean, VA_arr, VB_arr)
        costs_b = _w_cost_c3(Cu_out, As_out, nxt[:, 2], nxt[:, 5], VA_arr, VB_arr)
        return Cu_out, As_out, VMean, VA_arr, VB_arr, E_arr, profit_arr, costs_b


# ── 以下旧版worker函数保留签名供参考，实际不再调用 ────────────────────────────
def _init_worker(model_dir, condition, cfg, delta_sc, s_lo, s_hi, scale_v, j_denom,
                 nsga_pts=None):
    """
    旧版 worker 初始化函数（已废弃）。
    原功能：每个 worker 进程只执行一次：加载模型 + 恢复全局常量。
    
    注意：此函数已废弃，新架构不再使用持久化 Pool，每次 rollout 临时 fork 进程。
    """
    pass  # 已废弃


# ── worker 内部轻量特征/目标函数（仅供 _predict_batch_main 调用）───────────────────

def _w_build_c1c2_volt(Cu_in, T, I, Q, month=6, day=15, year=2025, hour=8):
    N = len(Cu_in)
    f = np.empty((N, 8), dtype=np.float32)
    f[:, 0]=Cu_in; f[:, 1]=T; f[:, 2]=I; f[:, 3]=Q
    f[:, 4]=year;  f[:, 5]=month; f[:, 6]=day; f[:, 7]=hour
    return f

def _w_build_c1c2_conc(Cu_in, T, I, Q, V, I_orig, month=6, day=15, year=2025, hour=8):
    N = len(Cu_in)
    f = np.empty((N, 10), dtype=np.float32)
    f[:, 0]=Cu_in; f[:, 1]=T; f[:, 2]=I; f[:, 3]=Q; f[:, 4]=V
    f[:, 5]=year;  f[:, 6]=month; f[:, 7]=day; f[:, 8]=hour; f[:, 9]=V*I_orig
    return f

def _w_build_c3_volt(Cu_in, TA, IA, QA, TB, IB, QB, month=6, day=15, year=2025, hour=8):
    N = len(Cu_in)
    f = np.empty((N, 11), dtype=np.float32)
    f[:, 0]=Cu_in; f[:, 1]=TA; f[:, 2]=IA; f[:, 3]=QA
    f[:, 4]=TB;    f[:, 5]=IB; f[:, 6]=QB
    f[:, 7]=year;  f[:, 8]=month; f[:, 9]=day; f[:, 10]=hour
    return f

def _w_build_c3_conc(Cu_in, TA, IA, QA, VA, TB, IB, QB, VB, tp,
                      month=6, day=15, year=2025, hour=8):
    N = len(Cu_in)
    f = np.empty((N, 14), dtype=np.float32)
    f[:, 0]=Cu_in; f[:, 1]=TA; f[:, 2]=IA; f[:, 3]=QA; f[:, 4]=VA
    f[:, 5]=TB;    f[:, 6]=IB; f[:, 7]=QB; f[:, 8]=VB
    f[:, 9]=year;  f[:, 10]=month; f[:, 11]=day; f[:, 12]=hour; f[:, 13]=tp
    return f

def _w_obj_c1c2(Cu_in, I, Q, t, Cu_out, As_out, V):
    E = V * I * t / 1000.0
    m  = np.maximum(0.0, (Cu_in - Cu_out) * Q * t)
    R  = (CFG['p_Cu'] - CFG['c_reprocess'] - CFG['c_copper_concentrate']) * m
    Rs = CFG['c_As_treat'] * (11.64 + Cu_out) * Q * t
    pr = (R - Rs - CFG['p_e'] * E - CFG['c_op'] * t) / 1e4
    return E, pr

def _w_obj_c3(Cu_in, TA, IA, QA, TB, IB, QB, t, Cu_out, As_out, VMean, VA, VB):
    E = (VA * IA + VB * IB) * t / 1000.0
    m  = np.maximum(0.0, (Cu_in - Cu_out) * QA * t)
    R  = (CFG['p_Cu'] - CFG['c_reprocess'] - CFG['c_copper_concentrate']) * m
    Rs = CFG['c_As_treat'] * (11.64 + Cu_out) * QA * t
    pr = (R - Rs - CFG['p_e'] * E - CFG['c_op'] * t) / 1e4
    return E, pr

def _w_cost_c1c2(Cu_out, As_out, I, V):
    J = I / _J_DENOM
    Vc = V / CFG['N_cell']
    r  = Cu_out / (As_out + 1e-8)
    c  = np.stack([
        np.maximum(0.0, Cu_out - CFG['Cu_limit']),
        np.maximum(0.0, CFG['As_min'] - As_out),
        np.maximum(0.0, J  - CFG['J_max']),
        np.maximum(0.0, Vc - CFG['V_cell_max']),
        np.maximum(0.0, r  - CFG['Cu_As_ratio_max']),
    ], axis=1)
    return c.astype(np.float32)

def _w_cost_c3(Cu_out, As_out, IA, IB, VA, VB):
    J  = np.maximum(IA, IB) / _J_DENOM
    Vc = np.maximum(VA, VB) / CFG['N_cell']
    r  = Cu_out / (As_out + 1e-8)
    c  = np.stack([
        np.maximum(0.0, Cu_out - CFG['Cu_limit']),
        np.maximum(0.0, CFG['As_min'] - As_out),
        np.maximum(0.0, J  - CFG['J_max']),
        np.maximum(0.0, Vc - CFG['V_cell_max']),
        np.maximum(0.0, r  - CFG['Cu_As_ratio_max']),
    ], axis=1)
    return c.astype(np.float32)


def _sample_state_worker():
    """Worker 内随机采样初始状态（与 NSGA 状态各 50%）。"""
    cfg = CFG
    cond = CONDITION
    if cond in (1, 2):
        return np.array([
            np.random.uniform(*cfg['Cu_in_range']),
            np.random.uniform(*cfg['T_range']),
            np.random.uniform(*cfg['I_range']),
            np.random.uniform(*cfg['Q_range']),
            np.random.uniform(*cfg['t_range']),
        ], dtype=np.float32)
    else:
        return np.array([
            np.random.uniform(*cfg['Cu_in_range']),
            np.random.uniform(*cfg['TA_range']), np.random.uniform(*cfg['IA_range']),
            np.random.uniform(*cfg['QA_range']),
            np.random.uniform(*cfg['TB_range']), np.random.uniform(*cfg['IB_range']),
            np.random.uniform(*cfg['QB_range']),
            np.random.uniform(*cfg['t_range']),
        ], dtype=np.float32)


def _sample_weights_worker():
    """[OPT-10] Worker 内极端权重采样。"""
    if np.random.rand() < 0.3:
        e = np.zeros(4, dtype=np.float32)
        e[np.random.randint(4)] = 0.85
        e += np.random.dirichlet([0.5] * 4).astype(np.float32) * 0.15
        return e / e.sum()
    return np.random.dirichlet([1.0] * 4).astype(np.float32)


def _dominance_bonus_batch(Cu_out_b, As_out_b, E_b, profit_b,
                            nsga_pts, bonus: float = 0.5) -> np.ndarray:
    """
    [IMP-2] 逐样本判断是否支配了 NSGA Pareto 集合中的至少一个点。
    若支配则加 bonus，否则 0。

    nsga_pts : shape (M, 4)，列顺序 = [Cu_out, -As_out, E, -profit]（全部最小化方向）
    返回      : shape (N,) float32
    """
    if nsga_pts is None or len(nsga_pts) == 0:
        return np.zeros(len(Cu_out_b), dtype=np.float32)

    my_obj = np.stack([
        Cu_out_b,
        -As_out_b,
        E_b,
        -profit_b,
    ], axis=1).astype(np.float64)   # (N, 4)

    nsga = nsga_pts.astype(np.float64)   # (M, 4)

    # 广播支配判断：my_obj[:,None,:] <= nsga[None,:,:]  → (N, M, 4)
    le = my_obj[:, None, :] <= nsga[None, :, :]   # (N, M, 4)
    lt = my_obj[:, None, :] <  nsga[None, :, :]   # (N, M, 4)
    dominates_any = (le.all(axis=2) & lt.any(axis=2)).any(axis=1)  # (N,)
    return (dominates_any.astype(np.float32) * bonus)


# =============================================================================
# ── 9. Behavioural Cloning 预训练 ─────────────────────────────────────────────
# =============================================================================

def pretrain_bc(net: ActorCritic, nsga_df: pd.DataFrame, opt):
    """[OPT-7] 行为克隆预训练，让策略网络从 NSGA-II 解中学习初始方向。"""
    if CONDITION in (1, 2):
        state_cols = ['Cu_in', 'TA' if CONDITION == 1 else 'TB',
                      'IA' if CONDITION == 1 else 'IB',
                      'QA' if CONDITION == 1 else 'QB', 't']
        noise_s = np.array([2.0, 1.0, 500.0, 0.5, 0.5], dtype=np.float32)
    else:
        state_cols = ['Cu_in', 'TA', 'IA', 'QA', 'TB', 'IB', 'QB', 't']
        noise_s = np.array([2.0, 1.0, 500.0, 0.5, 1.0, 500.0, 0.5, 0.5], dtype=np.float32)

    states = nsga_df[state_cols].values.astype(np.float32)
    tracker = _get_tracker()
    tracker.print_section("Behavioural Cloning Pretraining")
    tracker.print_info(f"epochs={CFG['bc_epochs']}  lr={CFG['bc_lr']}  "
                       f"samples={len(states)}  batch_size={CFG['batch_size']}")
    n_batches = max(1, (len(states) + CFG['batch_size'] - 1) // CFG['batch_size'])
    tracker.print_info(f"每 epoch 约 {n_batches} 个 mini-batch")
    bc_opt = optim.Adam(net.parameters(), lr=CFG['bc_lr'])
    t0_bc = time.time()
    best_loss = float('inf')
    for ep in range(CFG['bc_epochs']):
        idx = np.random.permutation(len(states))
        ep_loss = 0.0
        for i in range(0, len(idx), CFG['batch_size']):
            bi    = idx[i:i+CFG['batch_size']]
            s_tgt = states[bi]
            s_src = np.clip(
                s_tgt - np.random.randn(*s_tgt.shape).astype(np.float32) * noise_s,
                _S_LO, _S_HI)
            # 目标动作：归一化到 [-1, 1] 的增量
            a_tgt = np.clip((s_tgt[:, :CFG['action_dim']] - s_src[:, :CFG['action_dim']])
                            / _DELTA_SC, -1.0, 1.0)
            w = np.random.dirichlet([1] * 4, size=len(bi)).astype(np.float32)
            dist, _, _ = net(torch.FloatTensor(s_src).to(DEVICE),
                             torch.FloatTensor(w).to(DEVICE))
            loss = -dist.log_prob(torch.FloatTensor(a_tgt).to(DEVICE)).mean()
            bc_opt.zero_grad(); loss.backward(); bc_opt.step()
            ep_loss += loss.item()
        best_loss = min(best_loss, ep_loss)
        # 每 10 epoch 打印一次进度（比原来更频繁，信息更丰富）
        if (ep + 1) % 10 == 0 or (ep + 1) == CFG['bc_epochs']:
            tracker.update_step_force(
                ep + 1, CFG['bc_epochs'],
                prefix="  BC",
                extra=f"loss={ep_loss:.4f}  best={best_loss:.4f}  "
                      f"lr={bc_opt.param_groups[0]['lr']:.2e}",
            )
    elapsed_bc = time.time() - t0_bc
    tracker.print_info(f"BC 预训练完成  最终 loss={ep_loss:.4f}  "
                       f"最优 loss={best_loss:.4f}  总耗时={elapsed_bc:.0f}s")
    tracker.print_section_end()


# =============================================================================
# ── 10. Pareto 过滤 & 超体积指标 ─────────────────────────────────────────────
# =============================================================================

def _pareto_filter(df: pd.DataFrame) -> pd.DataFrame:
    """保留 Pareto 非支配解（最小化 Cu_out，最大化 As_out/profit，最小化 E）。"""
    if len(df) == 0:
        return df
    E_col = 'E' if 'E' in df.columns else 'E_total'
    p_col = None
    for col in df.columns:
        if 'profit' in col.lower():
            p_col = col
            break
    if p_col is None:
        p_col = 'Net_profit' if 'Net_profit' in df.columns else None
    if p_col is None:
        return df
    F = np.column_stack([
        df['Cu_out'].values,
        -df['As_out'].values,
        df[E_col].values,
        -df[p_col].values,
    ])
    n  = len(F)
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        dominated[i] = False
        nd[dominated] = False
    return df[nd].reset_index(drop=True)


def _hv_presample(df: pd.DataFrame, budget: int, E_col: str, p_col: str) -> pd.DataFrame:
    """
    极端点保留的分层预采样，替代纯随机预采样。

    纯随机预采样的风险：Pareto 前沿极端点（某维度最优解）数量少，
    被随机丢弃概率高，导致最终输出前沿在边界区域出现缺口。

    本函数采用"先锁定极端点，再随机补充"策略：
      1. 对每个目标维度各锁定最优的 n_extreme 个点（去重后合并）
      2. 剩余槽位从非极端点随机填充
    这样无论 M 多大，前沿的四个角落都会被保留，
    HV 贡献计算阶段的参考空间不会因为边界点缺失而严重低估。
    """
    n = len(df)
    if n <= budget:
        return df

    F = np.column_stack([
        df['Cu_out'].values,
        -df['As_out'].values,
        df[E_col].values,
        -df[p_col].values,
    ])

    # 每个维度保留最优的 n_extreme 个点（约占 budget 的 5%，至少 5 个）
    n_extreme = max(5, budget // 20)
    extreme_idx = set()
    for dim in range(F.shape[1]):
        best_k = np.argsort(F[:, dim])[:n_extreme]
        extreme_idx.update(best_k.tolist())

    extreme_idx = list(extreme_idx)
    n_random = budget - len(extreme_idx)
    non_extreme = [i for i in range(n) if i not in set(extreme_idx)]
    rng = np.random.RandomState(42)
    if n_random > 0 and len(non_extreme) > 0:
        fill = rng.choice(non_extreme,
                          size=min(n_random, len(non_extreme)),
                          replace=False).tolist()
    else:
        fill = []

    return df.iloc[extreme_idx + fill].reset_index(drop=True)


def _select_by_hv_contribution(df: pd.DataFrame, target_size: int = 300) -> pd.DataFrame:
    """
    基于超体积贡献剔除冗余解，保持 Pareto 前沿多样性。

    [PERF FIX] 原实现为 O(M³·logM) 的逐点暴力剔除，M=3000 时需数十小时。
    新实现采用两级加速策略：
      1. 若候选解超过 MAX_BEFORE_HV，先用 _hv_presample 极端点保留预采样。
         _hv_presample 锁定每目标维度最优的若干点再随机补充，
         避免纯随机采样丢失前沿边界点导致多样性损失。
      2. 批量剪枝（BATCH 模式）：每轮一次性计算所有点的 HV 贡献，批量移除
         贡献最小的 BATCH 个点，将外层迭代次数从 O(M) 降至 O(M/BATCH)。
    复杂度：O((M/BATCH) × M × HV_cost)  ≈ O(M²·logM / BATCH)，
    BATCH=10 时比原版快约 10 倍，实际运行时间从数小时降至分钟级。
    """
    if len(df) <= target_size:
        return df

    E_col = 'E' if 'E' in df.columns else 'E_total'
    p_col = None
    for col in df.columns:
        if 'profit' in col.lower():
            p_col = col
            break
    if p_col is None:
        p_col = 'Net_profit' if 'Net_profit' in df.columns else None
    if p_col is None:
        return df

    # ── 第一级防护：极端点保留预采样，防止 HV 迭代规模失控 ────────────────────
    MAX_BEFORE_HV = 500
    if len(df) > MAX_BEFORE_HV:
        before = len(df)
        df = _hv_presample(df, MAX_BEFORE_HV, E_col, p_col)
        print(f"    [HV prune] 预采样 {before} → {len(df)} 个点"
              f"（含各维度极端点保护）", flush=True)

    try:
        from pymoo.indicators.hv import HV
        # E_col 和 p_col 已在函数开头提取，此处直接使用

        F = np.column_stack([
            df['Cu_out'].values, -df['As_out'].values,
            df[E_col].values,   -df[p_col].values,
        ])
        ref = F.max(axis=0) * 1.1
        indices = list(range(len(F)))

        # ── 第二级加速：批量剪枝，每轮移除贡献最小的 BATCH 个点 ──────────────
        # BATCH 越大速度越快，但每轮贡献估算的精度略降（因为移除多个点后
        # 各点贡献值会相互影响）；实践中 BATCH=10 在精度和速度间取得良好平衡。
        BATCH = max(1, (len(indices) - target_size) // 20)
        hv_calc = HV(ref_point=ref)

        while len(indices) > target_size:
            F_cur = F[indices]
            hv_full = float(hv_calc(F_cur))

            # 计算每个点的 HV 贡献 = HV_full - HV_without_i
            contributions = np.empty(len(indices), dtype=np.float64)
            for k in range(len(indices)):
                mask = np.ones(len(indices), dtype=bool)
                mask[k] = False
                contributions[k] = hv_full - float(hv_calc(F_cur[mask]))

            # 批量移除贡献最小的若干个点
            n_remove = min(BATCH, len(indices) - target_size)
            remove_local = np.argsort(contributions)[:n_remove]
            # 从大到小移除，避免索引偏移
            for ri in sorted(remove_local.tolist(), reverse=True):
                indices.pop(ri)
            print(f"    [HV prune] 剩余 {len(indices)} 个解，"
                  f"本批移除 {n_remove} 个贡献最低点", flush=True)

        return df.iloc[indices].reset_index(drop=True)

    except ImportError:
        print("    [HV prune] pymoo 未安装，回退随机采样。")
        return df.sample(n=target_size, random_state=42).reset_index(drop=True)


def _compute_hypervolume_2d(df: pd.DataFrame) -> float:
    """2D Pareto 超体积（Cu_out vs As_out），作为快速诊断指标。"""
    if len(df) == 0:
        return 0.0
    pts = df[['Cu_out', 'As_out']].values.astype(float)
    ref_min = np.array([pts[:, 0].max() * 1.1, pts[:, 1].min() * 0.9])
    pts_min = np.column_stack([pts[:, 0], -pts[:, 1]])
    ref_min2 = np.array([ref_min[0], -ref_min[1]])
    mask = np.all(pts_min < ref_min2, axis=1)
    pts_min = pts_min[mask]
    if len(pts_min) == 0:
        return 0.0
    pts_min = pts_min[pts_min[:, 0].argsort()]
    hv = 0.0; prev_x = ref_min[0]
    for x, y in pts_min[::-1]:
        hv += (prev_x - x) * (ref_min2[1] - y)
        prev_x = x
    return float(hv)


def _compute_hypervolume_4d(df: pd.DataFrame) -> float:
    """4D 超体积，使用 GLOBAL_HV_REF_POINT 保证跨算法可比性。"""
    if len(df) == 0:
        return 0.0
    try:
        from pymoo.indicators.hv import HV
        E_col = 'E' if 'E' in df.columns else 'E_total'
        p_col = None
        for col in df.columns:
            if 'profit' in col.lower():
                p_col = col
                break
        if p_col is None:
            p_col = 'Net_profit' if 'Net_profit' in df.columns else None
        if p_col is None:
            return 0.0
        F = np.column_stack([
            df['Cu_out'].values, -df['As_out'].values,
            df[E_col].values,   -df[p_col].values,
        ])
        return float(HV(ref_point=GLOBAL_HV_REF_POINT)(F))
    except ImportError:
        return 0.0


# =============================================================================
# ── 11. PPO-Lagrangian 训练器 ─────────────────────────────────────────────────
# =============================================================================

class PPOLagrangian:
    """
    PPO-Lagrangian 安全强化学习训练器。
    核心特性详见文件头 Optimization log / Fix log。
    
    新架构：两阶段 rollout
    - 阶段1: 并行轨迹采样（worker无模型）
    - 阶段2: 主进程统一批量ExtraTrees预测
    """

    def __init__(self, models: dict, nsga_df: pd.DataFrame):
        self.nsga_df = nsga_df
        self.models  = models
        cond = CONDITION    

        # NSGA 状态列（按条件）
        if cond == 1:
            state_cols = ['Cu_in', 'TA', 'IA', 'QA', 't']
        elif cond == 2:
            state_cols = ['Cu_in', 'TB', 'IB', 'QB', 't']
        else:
            state_cols = ['Cu_in', 'TA', 'IA', 'QA', 'TB', 'IB', 'QB', 't']
        self.nsga_states = self.nsga_df[state_cols].values.astype(np.float32)

        # [FIX-13] 先初始化不涉及 GPU 的状态，再创建 Pool
        # 避免 fork 时 CUDA 锁以"已锁定"状态被复制到子进程导致死锁
        self.lambdas = np.zeros(CFG['n_constraints'], dtype=np.float32)
        self.metrics = dict(
            ep_rewards=[], ep_costs=[], ep_profits=[], ep_Cu_out=[], ep_As_out=[],
            lambdas_hist=[]
        )
        self.update_count = 0
        self.worker_pool  = None
        self.nsga_pts     = self._build_nsga_pts()  # [IMP-2] dominance bonus用

        self._init_worker_pool()  # 新版：no-op

        self.net = ActorCritic(
            CFG['state_dim'], CFG['action_dim'], CFG['weight_dim'],
            n_constraints=CFG['n_constraints']  
        ).to(DEVICE)

        self.opt = optim.Adam(self.net.parameters(), lr=CFG['lr_actor'])
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.opt, start_factor=0.1, end_factor=1.0, total_iters=CFG['warmup_updates'])

    def _init_worker_pool(self):
        """初始化worker池（新架构下无需初始化）。
        
        新架构不再使用持久化Pool，每次rollout_parallel临时fork进程，
        从根源消灭多进程模型加载死锁问题。
        """
        print("[POOL] 无模型worker模式，跳过持久化Pool初始化。", flush=True)
        self.worker_pool = None

    def _build_nsga_pts(self) -> np.ndarray:
        """
        从 nsga_df 构建 (M, 4) Pareto 矩阵，列=[Cu_out,-As_out,E,-profit]。
        兼容不同列名（E/E_total, profit/Net_profit）。
        
        [IMP-2] 用于计算 dominance bonus，给agent明确的"超越NSGA"激励信号。
        """

        df = self.nsga_df
        E_col = 'E' if 'E' in df.columns else ('E_total' if 'E_total' in df.columns else None)
        p_col = None
        for col in df.columns:
            if 'profit' in col.lower():
                p_col = col
                break
        if p_col is None:
            p_col = 'Net_profit' if 'Net_profit' in df.columns else None
        if E_col is None or p_col is None or 'Cu_out' not in df.columns:
            print("[IMP-2 WARNING] NSGA DataFrame 缺少目标列，dominance bonus 已禁用。")
            return None
        pts = np.column_stack([
            df['Cu_out'].values.astype(np.float64),
            -df['As_out'].values.astype(np.float64),
            df[E_col].values.astype(np.float64),
            -df[p_col].values.astype(np.float64),
        ])
        print(f"[IMP-2] NSGA Pareto pts loaded: {len(pts)} solutions → dominance bonus active.")
        return pts

    def pretrain(self):
        """[OPT-7] BC 预训练，完成后重置 LR scheduler。"""
        pretrain_bc(self.net, self.nsga_df, self.opt)
        # BC 后重置 scheduler，确保 PPO 阶段从 warm-up 起点开始
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.opt, start_factor=0.1, end_factor=1.0, total_iters=CFG['warmup_updates'])

    def sample_weights(self, n: int = 1) -> np.ndarray:
        """[OPT-10] 极端权重采样：30% 某维>0.8，提升 Pareto 前沿边界覆盖。"""
        results = []
        for _ in range(n):
            if np.random.rand() < 0.3:
                e = np.zeros(4, dtype=np.float32)
                e[np.random.randint(4)] = 0.85
                e += np.random.dirichlet([0.5] * 4).astype(np.float32) * 0.15
                results.append(e / e.sum())
            else:
                results.append(np.random.dirichlet([1.0] * 4).astype(np.float32))
        return np.array(results, dtype=np.float32)

    def rollout_parallel(self) -> dict:
        """
        两阶段rollout，彻底解决多进程模型加载死锁：
        - 阶段1: fork出n_workers个子进程，只做策略推理+状态转移，不碰ExtraTrees
        - 阶段2: 主进程拿回raw轨迹，统一做批量ExtraTrees预测
        
        返回值:
            包含轨迹数据的字典，用于后续的PPO更新
        """
        self.net.eval()
        net_state = {k: v.cpu() for k, v in self.net.state_dict().items()}

        cfg_copy = dict(CFG)
        cfg_copy['_condition'] = CONDITION

        args_list = [
            (wid, net_state, cfg_copy, _DELTA_SC, _S_LO, _S_HI,
             self.nsga_states, CFG['per_worker_steps'],
             CFG['seed'] + self.update_count * 100)
            for wid in range(CFG['n_workers'])
        ]

        # ── 阶段1：并行轨迹采样（worker无模型）──────────────────────────────
        # 使用fork模式创建临时进程池，worker只做策略推理和状态转移
        ctx = mp.get_context('fork')
        with ctx.Pool(processes=CFG['n_workers']) as pool:
            raw_bufs = pool.map(_worker_sample_only, args_list)

        # ── 阶段2：主进程统一批量ExtraTrees预测 ─────────────────────────────
        # 收集所有worker的下一状态，统一进行批量预测，提高效率并避免多进程模型加载问题
        all_nxt = np.concatenate([b['nxt'] for b in raw_bufs], axis=0)
        (Cu_out_all, As_out_all, _, _, _, 
         E_all, profit_all, costs_all) = _predict_batch_main(all_nxt, self.models)

        split_pts     = np.cumsum([len(b['nxt']) for b in raw_bufs[:-1]])
        Cu_splits     = np.split(Cu_out_all, split_pts)
        As_splits     = np.split(As_out_all, split_pts)
        E_splits      = np.split(E_all,      split_pts)
        profit_splits = np.split(profit_all, split_pts)
        costs_splits  = np.split(costs_all,  split_pts)

        # ── critic values在CPU上计算，避免与GPU训练冲突 ──────────────────────
        net_cpu = self.net.cpu()
        net_cpu.eval()
        CHUNK = 512
        bufs = []
        for i, b in enumerate(raw_bufs):
            Cu_b = Cu_splits[i]; As_b = As_splits[i]
            E_b  = E_splits[i];  pr_b = profit_splits[i]
            costs_b = costs_splits[i]; w_mat = b['w']

            obj_b = np.stack([Cu_b, -As_b, E_b, -pr_b], axis=1) / _SCALE_V[None]
            r_b   = -np.einsum('ni,ni->n', w_mat, obj_b)
            dom_bonus = _dominance_bonus_batch(Cu_b, As_b, E_b, pr_b,
                                               self.nsga_pts, bonus=0.5)
            r_b = r_b + dom_bonus

            s_arr = b['s']; w_arr = b['w']
            v_r_list, v_c_list = [], []
            with torch.no_grad():
                for j in range(0, len(s_arr), CHUNK):
                    _, v_r, v_c = net_cpu(torch.FloatTensor(s_arr[j:j+CHUNK]),
                                          torch.FloatTensor(w_arr[j:j+CHUNK]))
                    v_r_list.append(v_r.numpy())
                    v_c_list.append(v_c.numpy())
            v_r_arr = np.concatenate(v_r_list)
            v_c_arr = np.concatenate(v_c_list)
            with torch.no_grad():
                _, nv_r, nv_c = net_cpu(torch.FloatTensor(b['nxt'][-1:]),
                                        torch.FloatTensor(w_arr[-1:]))

            bufs.append(dict(
                s=s_arr, a=b['a'], lp=b['lp'], r=r_b, c=costs_b,
                v_r=v_r_arr, v_c=v_c_arr, d=b['d'], w=w_arr,
                next_v_r=nv_r.item(), next_v_c=nv_c.numpy()[0],
                ep_metrics={
                    'Cu_out_last': float(Cu_b[-1]),
                    'As_out_last': float(As_b[-1]),
                    'profit_last': float(pr_b[-1]),
                    'r_sum':       float(r_b.sum()),
                    'cost_sum':    float(costs_b.sum()),
                }
            ))

        self.net.to(DEVICE)
        self.net.train()

        merged = {k: np.concatenate([b[k] for b in bufs], axis=0)
                  for k in ('s', 'a', 'lp', 'r', 'c', 'v_r', 'v_c', 'd', 'w')}
        all_adv_r, all_ret_r, all_adv_c, all_ret_c = [], [], [], []
        for b in bufs:
            adv_r = compute_gae(b['r'], b['v_r'], b['d'], b['next_v_r'])
            ret_r = adv_r + b['v_r']
            adv_c_per = np.zeros_like(b['c'])
            ret_c_per = np.zeros_like(b['c'])
            for k in range(CFG['n_constraints']):
                adv_c_per[:, k] = compute_gae(b['c'][:, k], b['v_c'][:, k],
                                               b['d'], b['next_v_c'][k])
                ret_c_per[:, k] = adv_c_per[:, k] + b['v_c'][:, k]
            all_adv_r.append(adv_r); all_ret_r.append(ret_r)
            all_adv_c.append(adv_c_per); all_ret_c.append(ret_c_per)
        merged['adv_r'] = np.concatenate(all_adv_r)
        merged['ret_r'] = np.concatenate(all_ret_r)
        merged['adv_c'] = np.concatenate(all_adv_c)
        merged['ret_c'] = np.concatenate(all_ret_c)

        for b in bufs:
            m = b['ep_metrics']
            self.metrics['ep_rewards'].append(m['r_sum'])
            self.metrics['ep_costs'].append(m['cost_sum'])
            self.metrics['ep_profits'].append(m['profit_last'])
            self.metrics['ep_Cu_out'].append(m['Cu_out_last'])
            self.metrics['ep_As_out'].append(m['As_out_last'])
        return merged

    def update(self, buf: dict):
        adv_r   = buf['adv_r'].astype(np.float32)
        adv_c   = buf['adv_c'].astype(np.float32)
        ret_r   = buf['ret_r'].astype(np.float32)
        ret_c   = buf['ret_c'].astype(np.float32)
        lam_vec = self.lambdas
        adv_lag  = adv_r - (adv_c * lam_vec[None, :]).sum(1)
        adv_norm = ((adv_lag - adv_lag.mean()) / (adv_lag.std() + 1e-8)).astype(np.float32)

        s_t       = torch.FloatTensor(buf['s']).to(DEVICE)
        a_t       = torch.FloatTensor(buf['a']).to(DEVICE)
        lp_old_t  = torch.FloatTensor(buf['lp']).to(DEVICE)
        w_t       = torch.FloatTensor(buf['w']).to(DEVICE)
        adv_t     = torch.FloatTensor(adv_norm).to(DEVICE)
        ret_r_t   = torch.FloatTensor(ret_r).to(DEVICE)
        ret_c_t   = torch.FloatTensor(ret_c).to(DEVICE)
        n = len(s_t)

        # [IMP-1] 动态 entropy_coef（三段式）：
        #   前 100 次 update：0.05（解锚阶段，打破 BC 预训练的 NSGA 邻域锚定）
        #   前半程（100~total/2）：0.008（正常探索）
        #   后半程：0.003（收敛精化）
        total_updates = CFG['total_steps'] // (CFG['n_workers'] * CFG['per_worker_steps'])
        if self.update_count < 100:
            ent_coef = 0.05
        elif self.update_count < total_updates // 2:
            ent_coef = 0.008
        else:
            ent_coef = 0.003

        for _ in range(CFG['ppo_epochs']):
            perm = torch.randperm(n)
            for i in range(0, n, CFG['batch_size']):
                bi = perm[i:i+CFG['batch_size']]
                dist, v_r_pred, v_c_pred = self.net(s_t[bi], w_t[bi])
                ratio    = (dist.log_prob(a_t[bi]).sum(-1) - lp_old_t[bi]).exp()
                clip_r   = ratio.clamp(1 - CFG['eps_clip'], 1 + CFG['eps_clip'])
                actor_loss       = -torch.min(ratio * adv_t[bi], clip_r * adv_t[bi]).mean()
                critic_loss      = 0.5 * (v_r_pred - ret_r_t[bi]).pow(2).mean()
                cost_critic_loss = 0.5 * (v_c_pred - ret_c_t[bi]).pow(2).mean()
                entropy_loss     = -ent_coef * dist.entropy().mean()
                loss = actor_loss + critic_loss + cost_critic_loss + entropy_loss
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

        # [OPT-7] warm-up scheduler
        self.scheduler.step()

        # [OPT-12] 动态 cost_limit：0.05 → 0.02
        dyn_limit = max(0.02, CFG['cost_limit'] * (0.95 ** (self.update_count // 50)))
        violation_mask = buf['c'] > 0
        mean_costs = np.zeros(CFG['n_constraints'], dtype=np.float32)
        for k in range(CFG['n_constraints']):
            violated = buf['c'][violation_mask[:, k], k]
            if len(violated) > 0:
                mean_costs[k] = violated.mean()
        for k in range(CFG['n_constraints']):
            self.lambdas[k] = max(
                0.0, self.lambdas[k] + CFG['lr_lambda'] * (mean_costs[k] - dyn_limit))
        self.update_count += 1
        self.metrics['lambdas_hist'].append(self.lambdas.copy())

    def train(self):
        cond = CONDITION
        rollout_steps = CFG['n_workers'] * CFG['per_worker_steps']
        total_updates = CFG['total_steps'] // rollout_steps
        step = 0
        t0 = time.time()

        tracker = _get_tracker()
        tracker.print_section("PPO-Lagrangian 强化学习训练")
        tracker.print_info(f"Workers={CFG['n_workers']}  per_worker={CFG['per_worker_steps']}"
                           f"  rollout/update={rollout_steps:,}  total={CFG['total_steps']:,}")
        tracker.print_info(f"总 updates={total_updates}  batch_size={CFG['batch_size']}"
                           f"  ppo_epochs={CFG['ppo_epochs']}  n_constraints={CFG['n_constraints']}")
        tracker.print_info(f"lr_actor={CFG['lr_actor']:.2e}  lr_lambda={CFG['lr_lambda']:.2e}"
                           f"  cost_limit={CFG['cost_limit']}  warmup={CFG['warmup_updates']} updates")
        training_history = []

        # 断点续训
        ckpt_path = _out(f'checkpoint_condition{cond}.pt')
        if os.path.exists(ckpt_path):
            print(f"[INFO] Loading checkpoint from {ckpt_path}")
            ck = torch.load(ckpt_path, map_location=DEVICE)
            self.net.load_state_dict(ck['net'])
            self.opt.load_state_dict(ck['optimizer'])
            self.scheduler.load_state_dict(ck['scheduler'])
            self.lambdas      = ck['lambdas']
            self.metrics      = ck['metrics']
            self.update_count = ck['update_count']
            step              = ck['step']
            training_history  = ck['training_history']
            print(f"[INFO] Resumed from step {step:,} / {CFG['total_steps']:,}"
                  f"  (update {self.update_count}/{total_updates})")

        while step < CFG['total_steps']:
            t_rollout = time.time()
            buf = self.rollout_parallel()
            t_update = time.time()
            self.update(buf)
            step += rollout_steps

            training_history.append({
                'Update_step': self.update_count, 'Step': step,
                'Reward_sum':  float(np.mean(self.metrics['ep_rewards'][-20:] or [0])),
                'Cost_sum':    float(np.mean(self.metrics['ep_costs'][-20:] or [0])),
                'Cost_Cu':     float(buf['c'][:, 0].mean()),
                'Cost_As':     float(buf['c'][:, 1].mean()),
                'Cost_J':      float(buf['c'][:, 2].mean()),
                'Cost_V':      float(buf['c'][:, 3].mean()),
                'Cost_ratio':  float(buf['c'][:, 4].mean()),
                'Lambda_Cu':   float(self.lambdas[0]),
                'Lambda_As':   float(self.lambdas[1]),
                'Lambda_J':    float(self.lambdas[2]),
                'Lambda_V':    float(self.lambdas[3]),
                'Lambda_ratio':float(self.lambdas[4]),
                'Profit':      float(np.mean(self.metrics['ep_profits'][-20:] or [0])),
                'Cu_out':      float(np.mean(self.metrics['ep_Cu_out'][-20:] or [0])),
                'As_out':      float(np.mean(self.metrics['ep_As_out'][-20:] or [0])),
                'Learning_rate': float(self.opt.param_groups[0]['lr']),
                'Time_elapsed':  time.time() - t0,
                'Rollout_time':  t_update - t_rollout,
                'Update_time':   time.time() - t_update,
            })

            # 每 10 次 update 打印一行详细进度
            if self.update_count % 10 == 0:
                elapsed  = time.time() - t0
                remain   = elapsed / max(step, 1) * (CFG['total_steps'] - step)
                m = self.metrics
                lstr = ' '.join(f'{v:.3f}' for v in self.lambdas)
                # 动态成本限制
                dyn_limit = max(0.02, CFG['cost_limit'] * (0.95 ** (self.update_count // 50)))
                # 判断当前训练阶段
                total_upd = CFG['total_steps'] // (CFG['n_workers'] * CFG['per_worker_steps'])
                if self.update_count < 100:
                    phase = "解锚(ent=0.05)"
                elif self.update_count < total_upd // 2:
                    phase = "探索(ent=0.008)"
                else:
                    phase = "收敛(ent=0.003)"
                rollout_t = training_history[-1]['Rollout_time']
                update_t  = training_history[-1]['Update_time']
                tracker.update_step_force(
                    step, CFG['total_steps'],
                    prefix=f"  PPO [{phase}]",
                    extra=(
                        f"upd={self.update_count}/{total_updates}"
                        f"  reward={np.mean(m['ep_rewards'][-20:] or [0]):7.3f}"
                        f"  cost={np.mean(m['ep_costs'][-20:] or [0]):.4f}"
                        f"  cost_lim={dyn_limit:.3f}"
                        f"  λ=[{lstr}]"
                        f"  profit={np.mean(m['ep_profits'][-20:] or [0]):.2f}万"
                        f"  Cu={np.mean(m['ep_Cu_out'][-20:] or [0]):.3f}"
                        f"  As={np.mean(m['ep_As_out'][-20:] or [0]):.3f}"
                        f"  lr={self.opt.param_groups[0]['lr']:.2e}"
                        f"  rollout={rollout_t:.1f}s  update={update_t:.1f}s"
                        f"  剩余≈{tracker._fmt_time(remain)}"
                    ),
                )
                pd.DataFrame(training_history).to_csv(
                    _out(f'ppo_training_history_condition{cond}.csv'), index=False)

            # 每 100 次 update 保存 checkpoint
            if self.update_count % 100 == 0:
                ck = dict(net=self.net.state_dict(), optimizer=self.opt.state_dict(),
                          scheduler=self.scheduler.state_dict(), lambdas=self.lambdas,
                          metrics=self.metrics, update_count=self.update_count,
                          step=step, training_history=training_history)
                torch.save(ck, ckpt_path)
                tracker.print_info(
                    f"[CKPT] 第 {self.update_count} 次 update 保存检查点 → {ckpt_path}"
                    f"  (step={step:,}/{CFG['total_steps']:,})")

        # 训练完成，保存最终 history
        hist_csv = _out(f'ppo_training_history_condition{cond}.csv')
        pd.DataFrame(training_history).to_csv(hist_csv, index=False)
        lam_rows = [{
            'Update_step': e['Update_step'],
            'Lambda_Cu': e['Lambda_Cu'], 'Lambda_As': e['Lambda_As'],
            'Lambda_J': e['Lambda_J'],  'Lambda_V': e['Lambda_V'],
            'Lambda_ratio': e['Lambda_ratio'],
            'Mean_cost_Cu': e['Cost_Cu'], 'Mean_cost_As': e['Cost_As'],
            'Mean_cost_J': e['Cost_J'],   'Mean_cost_V': e['Cost_V'],
            'Mean_cost_ratio': e['Cost_ratio'],
        } for e in training_history]
        pd.DataFrame(lam_rows).to_csv(
            _out(f'lambda_evolution_condition{cond}.csv'), index=False)
        total_time = time.time() - t0
        tracker.print_info(
            f"训练完成  Condition {cond}  total_time={total_time/3600:.2f}h"
            f"  updates={self.update_count}")
        tracker.print_section_end()

    def save(self):
        path = _out(f'ppo_actor_condition{CONDITION}.pt')
        torch.save(self.net.state_dict(), path)
        print(f"[Saved] Model → {path}")

    def _approx_profit_batch(self, state: np.ndarray,
                              samples: np.ndarray) -> np.ndarray:
        """
        [FIX-19] 批量利润近似，替代原逐个调用 _approx_profit 的单点推理瓶颈。

        原实现对 K 个候选动作逐一调用 predict_all_single，每次独立触发 3 次
        ExtraTrees 推理（V→Cu→As），K=3 时共 9 次单点推理。
        新实现将 K 个候选次态堆叠成 shape=(K, state_dim) 的批量输入，
        调用一次 predict_all_batch，仅触发 3 次批量推理（各 K 行），
        推理次数不变但批量化使 ExtraTrees 的 predict 开销摊薄，
        实测约快 2–4×（取决于 ExtraTrees 树的数量和 K 的大小）。

        参数
        ----
        state   : 当前状态 (state_dim,)
        samples : 候选动作矩阵 (K, action_dim)，来自 dist.sample((K,))

        返回
        ----
        profits : shape (K,) float32，每个候选动作对应的预估利润
        """
        K   = len(samples)
        dim = len(_DELTA_SC)

        # 防御性 squeeze：兼容 dist.sample((K,)) 返回 (K,1,action_dim) 的情况
        if samples.ndim == 3:
            samples = samples.squeeze(1)   # (K, 1, action_dim) → (K, action_dim)

        # 计算所有候选次态：(K, state_dim)
        next_states = np.tile(state, (K, 1)).astype(np.float32)
        deltas = samples * _DELTA_SC[None, :]           # (K, action_dim)
        next_states[:, :dim] = np.clip(
            state[:dim] + deltas, _S_LO[:dim], _S_HI[:dim])
        next_states = np.clip(next_states, _S_LO, _S_HI)

        # 一次批量 ExtraTrees 推理，涵盖全部 K 个候选次态
        if CONDITION in (1, 2):
            Cu_out, As_out, V = predict_all_batch(
                self.models,
                next_states[:, 0], next_states[:, 1],
                next_states[:, 2], next_states[:, 3],
                next_states[:, 4])
            m_Cu = np.maximum(0.0,
                (next_states[:, 0] - Cu_out) * next_states[:, 3] * next_states[:, 4])
            E    = V * next_states[:, 2] * next_states[:, 4] / 1000.0
            Q    = next_states[:, 3]
            t    = next_states[:, 4]
        else:
            Cu_out, As_out, VMean, _VA, _VB = predict_all_batch(
                self.models,
                next_states[:, 0], next_states[:, 1], next_states[:, 2],
                next_states[:, 3], next_states[:, 4], next_states[:, 5],
                next_states[:, 6], next_states[:, 7])
            m_Cu = np.maximum(0.0,
                (next_states[:, 0] - Cu_out) * next_states[:, 3] * next_states[:, 7])
            E    = VMean * (next_states[:, 2] + next_states[:, 5]) * next_states[:, 7] / 1000.0
            Q    = next_states[:, 3]
            t    = next_states[:, 7]

        R  = (CFG['p_Cu'] - CFG['c_reprocess'] - CFG['c_copper_concentrate']) * m_Cu
        Rs = CFG['c_As_treat'] * (11.64 + Cu_out) * Q * t
        return ((R - Rs - CFG['p_e'] * E - CFG['c_op'] * t) / 1e4).astype(np.float32)

    def evaluate(self, n_episodes: int = None, strategy: str = 'mixed') -> pd.DataFrame:
        """
        [OPT-14] 分层评估策略。
        strategy: 'deterministic' | 'stochastic' | 'nsga_perturb' | 'mixed'
        """
        if n_episodes is None:
            n_episodes = CFG['eval_episodes']
        cond = CONDITION
        results = []; trajectories = []
        self.net.eval()
        lo, hi = _S_LO, _S_HI
        stochastic_episodes = int(n_episodes * 2 / 3)
        t0_eval = time.time()
        TRAJ_LIMIT = CFG['traj_limit']   # 只记录前 N 个 episode 的轨迹，避免内存爆炸
        feas_count = 0     # 增量计数严格可行解，避免 O(N²) 扫描
        total_steps_count = 0  # 记录总步数，用于准确计算 Feasibility_rate

        tracker = _get_tracker()
        tracker.print_info(f"[Eval:{strategy}] 开始评估  n_episodes={n_episodes}"
                           f"  strategy={strategy}  max_ep_steps={CFG['max_ep_steps']}"
                           f"  feasible_threshold={CFG['feasible_threshold']}"
                           f"  candidate_threshold={CFG['candidate_threshold']}")

        # 创建 GPU 张量缓冲区，避免每步重建导致内存泄漏
        s_buf = torch.zeros(1, CFG['state_dim'], dtype=torch.float32, device=DEVICE)
        w_buf = torch.zeros(1, CFG['weight_dim'], dtype=torch.float32, device=DEVICE)

        for ep_idx in range(n_episodes):
            w = self.sample_weights()[0]

            if strategy == 'nsga_perturb':
                if cond in (1, 2):
                    noise = np.array([2.0, 1.0, 500.0, 0.5, 0.5], dtype=np.float32)
                else:
                    noise = np.array([2.0, 1.0, 500.0, 0.5, 1.0, 500.0, 0.5, 0.5], dtype=np.float32)
                state = self.nsga_states[np.random.randint(len(self.nsga_states))].copy()
                state = np.clip(state + np.random.randn(len(state)).astype(np.float32) * noise, lo, hi)
            elif strategy == 'mixed':
                # mixed 策略：50% NSGA 初始化，50% 随机采样
                if np.random.rand() < 0.5:
                    state = self.nsga_states[np.random.randint(len(self.nsga_states))].copy()
                else:
                    state = np.clip(_sample_state_worker(), lo, hi)
            else:
                # deterministic / stochastic：全随机采样，保证评估覆盖全状态空间
                state = np.clip(_sample_state_worker(), lo, hi)
            state = np.clip(state, lo, hi)

            use_stochastic = (
                strategy in ('stochastic', 'nsga_perturb') or
                (strategy == 'mixed' and ep_idx < stochastic_episodes)
            )

            ep_states = []
            for _step in range(CFG['max_ep_steps']):
                total_steps_count += 1
                
                # 使用缓冲区，避免每步创建新张量
                s_buf.copy_(torch.FloatTensor(state).unsqueeze(0))
                w_buf.copy_(torch.FloatTensor(w).unsqueeze(0))
                
                with torch.no_grad():
                    dist, _, _ = self.net(s_buf, w_buf)
                    if strategy == 'stochastic':
                        nd = torch.distributions.Normal(
                            dist.mean, (dist.stddev * 2.0).clamp(1e-3, 2.0))
                        a_np = nd.sample().cpu().numpy()[0]
                    elif use_stochastic:
                        # [FIX-19] 批量采样 K 个候选动作，用一次 predict_all_batch
                        # 替代原来 K 次 predict_all_single，推理次数从 K×3 降为 3。
                        # dist 的 batch_shape=(1,)，sample((3,)) 返回 (3,1,action_dim)，
                        # 需 squeeze(1) 压掉 batch 维，得到 (3, action_dim)。
                        samples = dist.sample((3,)).cpu().numpy().squeeze(1)  # (3, action_dim)
                        profits = self._approx_profit_batch(state, samples)
                        a_np = samples[int(np.argmax(profits))]
                    else:
                        a_np = dist.mean.cpu().numpy()[0]
                delta = a_np * _DELTA_SC
                new_s = state.copy()
                dim   = len(delta)
                new_s[:dim] = np.clip(state[:dim] + delta, lo[:dim], hi[:dim])
                state = np.clip(new_s, lo, hi)
                ep_states.append(state.copy())

            ep_states = np.array(ep_states, dtype=np.float32)
            # 批量推理
            if cond in (1, 2):
                Cu_out_b, As_out_b, V_b = predict_all_batch(
                    self.models, ep_states[:, 0], ep_states[:, 1],
                    ep_states[:, 2], ep_states[:, 3], ep_states[:, 4])
                E_b, profit_b = compute_objectives_batch(
                    (ep_states[:, 0], ep_states[:, 1], ep_states[:, 2],
                     ep_states[:, 3], ep_states[:, 4]),
                    Cu_out_b, As_out_b, V_b)
                costs_b = compute_costs_batch(Cu_out_b, As_out_b, ep_states[:, 2], V_b)
                V_col_vals = V_b
            else:
                Cu_out_b, As_out_b, VMean_b, VA_b, VB_b = predict_all_batch(
                    self.models,
                    ep_states[:, 0], ep_states[:, 1], ep_states[:, 2], ep_states[:, 3],
                    ep_states[:, 4], ep_states[:, 5], ep_states[:, 6], ep_states[:, 7])
                E_b, profit_b = compute_objectives_batch(
                    (ep_states[:, 0], ep_states[:, 1], ep_states[:, 2], ep_states[:, 3],
                     ep_states[:, 4], ep_states[:, 5], ep_states[:, 6], ep_states[:, 7]),
                    Cu_out_b, As_out_b, (VMean_b, VA_b, VB_b))
                costs_b = compute_costs_batch(Cu_out_b, As_out_b,
                                              (ep_states[:, 2], ep_states[:, 5]),
                                              (VA_b, VB_b))
                V_col_vals = VMean_b

            for _step in range(CFG['max_ep_steps']):
                costs = costs_b[_step]
                # [IMP-3] 宽松门槛收集候选解（后续严格过滤再输出），提升 Pareto 多样性
                if costs.max() < CFG['candidate_threshold']:
                    row = {
                        'Cu_in': ep_states[_step, 0],
                        'Cu_out': float(Cu_out_b[_step]),
                        'As_out': float(As_out_b[_step]),
                        'E': float(E_b[_step]),
                        'profit': float(profit_b[_step]),
                        'V': float(V_col_vals[_step]),
                        'costs_max': float(costs.max()),
                    }
                    if cond == 1:
                        row.update({'TA': ep_states[_step, 1], 'IA': ep_states[_step, 2],
                                    'QA': ep_states[_step, 3], 't':  ep_states[_step, 4]})
                    elif cond == 2:
                        row.update({'TB': ep_states[_step, 1], 'IB': ep_states[_step, 2],
                                    'QB': ep_states[_step, 3], 't':  ep_states[_step, 4]})
                    else:
                        row.update({'TA': ep_states[_step, 1], 'IA': ep_states[_step, 2],
                                    'QA': ep_states[_step, 3], 'TB': ep_states[_step, 4],
                                    'IB': ep_states[_step, 5], 'QB': ep_states[_step, 6],
                                    't':  ep_states[_step, 7]})
                    results.append(row)
                    # 增量计数严格可行解（O(1)，避免 O(N²) 扫描）
                    if costs.max() < CFG['feasible_threshold']:
                        feas_count += 1

                # 只保存前 TRAJ_LIMIT 个 episode 的轨迹，避免内存爆炸
                if ep_idx < TRAJ_LIMIT:
                    trajectories.append({
                        'Episode_ID': ep_idx, 'Step': _step,
                        'Cu_out': float(Cu_out_b[_step]), 'As_out': float(As_out_b[_step]),
                        'E_total': float(E_b[_step]), 'Net_profit': float(profit_b[_step]),
                        'Feasible': int(costs.max() < CFG['feasible_threshold']),
                        'Weight_Cu': float(w[0]), 'Weight_As': float(w[1]),
                        'Weight_E': float(w[2]),  'Weight_profit': float(w[3]),
                    })

            # 每 200 个 episode 打印一次评估进度（比原来 500 更频繁）
            if (ep_idx + 1) % 200 == 0 or (ep_idx + 1) == n_episodes:
                elapsed_eval = time.time() - t0_eval
                cand_count   = len(results)
                feas_rate    = feas_count / max(total_steps_count, 1) * 100
                tracker.update_step(
                    ep_idx + 1, n_episodes,
                    prefix=f"  Eval[{strategy}]",
                    extra=(
                        f"候选解={cand_count}"
                        f"  严格可行={feas_count}"
                        f"  可行率={feas_rate:.2f}%"
                        f"  总步={total_steps_count:,}"
                    ),
                    min_interval=0.0,
                )

        self.net.train()
        if trajectories:
            traj_df = pd.DataFrame(trajectories)
            traj_df[traj_df['Episode_ID'] < 200].to_csv(
                _out(f'episode_trajectories_condition{cond}.csv'), index=False)

        df = pd.DataFrame(results)
        if len(df) == 0:
            return df
        # [IMP-3] 严格可行解过滤（宽松收集后严格输出）
        df_strict = df[df['costs_max'] < CFG['feasible_threshold']].copy()
        if len(df_strict) == 0:
            print(f"    [IMP-3] 无严格可行解（costs_max<{CFG['feasible_threshold']}），回退使用宽松解集（<{CFG['candidate_threshold']}）")
            df_strict = df.copy()
        df = _pareto_filter(df_strict)
        TARGET_SIZE = CFG['pareto_target_size']
        print(f"    [Evaluate] Pareto size before HV selection: {len(df)}")
        df = _select_by_hv_contribution(df, target_size=TARGET_SIZE)
        print(f"    [Evaluate] Pareto size after HV selection: {len(df)}")

        # 保存标准 Pareto CSV
        pareto_std = df.copy()
        pareto_std['Method']      = 'PPO-Lagrangian'
        pareto_std['Condition']   = f'Condition {cond}'
        pareto_std['Solution_ID'] = range(1, len(pareto_std) + 1)
        pareto_std = pareto_std.rename(columns={'E': 'E_total', 'profit': 'Net_profit'})
        pareto_std.to_csv(_out(f'pareto_ppo_condition{cond}.csv'), index=False)

        # 质量指标 CSV（追加模式，保留所有三轮评估数据）
        qm = {
            'Method': 'PPO-Lagrangian', 'Condition': f'Condition {cond}',
            'Strategy': strategy,
            'N_solutions':   len(df),
            'Feasibility_rate': feas_count / max(total_steps_count, 1),
            'Mean_Cu_out':  float(df['Cu_out'].mean()),
            'Std_Cu_out':   float(df['Cu_out'].std()),
            'Range_Cu_out': float(df['Cu_out'].max() - df['Cu_out'].min()),
            'Mean_As_out':  float(df['As_out'].mean()),
            'Std_As_out':   float(df['As_out'].std()),
            'Range_As_out': float(df['As_out'].max() - df['As_out'].min()),
        }
        qm_path = _out(f'quality_metrics_ppo.csv')
        qm_df   = pd.DataFrame([qm])
        # 首轮写入时创建文件（带表头），后续轮次追加（无表头）
        write_header = not os.path.exists(qm_path)
        qm_df.to_csv(qm_path, index=False, mode='a', header=write_header)
        return df

    def generate_weight_policy_mapping(self, n_samples: int = 1000):
        """记录权重→策略映射，用于分析 Pareto 前沿结构。"""
        print(f"  [Weight-policy mapping] sampling {n_samples} points ...")
        mapping_data = []
        self.net.eval()
        for _ in range(n_samples):
            w = self.sample_weights()[0]
            s = np.clip(_sample_state_worker(), _S_LO, _S_HI)
            with torch.no_grad():
                dist, _, _ = self.net(torch.FloatTensor(s).unsqueeze(0).to(DEVICE),
                                      torch.FloatTensor(w).unsqueeze(0).to(DEVICE))
                a = dist.mean.cpu().numpy()[0]
            row = {f'w{i}': float(w[i]) for i in range(4)}
            row.update({f'a{i}': float(a[i]) for i in range(len(a))})
            mapping_data.append(row)
        self.net.train()
        path = _out(f'weight_policy_map_condition{CONDITION}.csv')
        pd.DataFrame(mapping_data).to_csv(path, index=False)
        print(f"  [Saved] weight-policy map → {path}")

    # ── 可视化 ──────────────────────────────────────────────────────────────

    def plot_training(self):
        """绘制训练曲线（含每约束 λ 子图）。"""
        cond = CONDITION
        hist_csv = _out(f'ppo_training_history_condition{cond}.csv')
        if not os.path.exists(hist_csv):
            return
        th = pd.read_csv(hist_csv)
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'PPO-Lagrangian Training — Condition {cond}', fontsize=13)

        def _plot(ax, col, title, ylabel):
            if col in th.columns:
                ax.plot(th['Update_step'], th[col]); ax.set_title(title)
                ax.set_xlabel('Update step'); ax.set_ylabel(ylabel); ax.grid(True)

        _plot(axes[0, 0], 'Reward_sum',  'Reward',         'Reward sum')
        _plot(axes[0, 1], 'Cost_sum',    'Total cost',     'Cost sum')
        _plot(axes[1, 0], 'Profit',      'Profit (万CNY)', 'Profit')
        _plot(axes[1, 1], 'Cu_out',      'Cu_out',         'g/L')

        ax_l = axes[2, 0]
        for k, name in enumerate(['Cu', 'As', 'J', 'V', 'ratio']):
            col = f'Lambda_{name}'
            if col in th.columns:
                ax_l.plot(th['Update_step'], th[col], label=f'λ_{name}')
        ax_l.set_title('Lagrange multipliers'); ax_l.legend(fontsize=8); ax_l.grid(True)

        _plot(axes[2, 1], 'Learning_rate', 'Learning rate', 'lr')
        plt.tight_layout()
        path = _out(f'training_curve_condition{cond}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[Saved] training curve → {path}")

    def plot_pareto(self, rl_results: pd.DataFrame, nsga_df: pd.DataFrame):
        """绘制 RL vs NSGA Pareto 对比图（Cu_out-As_out 2D + E-profit 2D）。"""
        cond = CONDITION
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Pareto Front Comparison — Condition {cond}', fontsize=13)

        # [FIX-11] 修复原版 _scatter 内部硬编码 axes[0] 导致 ax 参数对左图无效的问题。
        # 现在两个子图都通过参数传入，_scatter 本身不再引用外部 axes 变量。
        def _scatter(ax_cu_as, ax_e_profit, df, label, marker, alpha=0.7):
            if len(df) == 0:
                return
            E_col = 'E' if 'E' in df.columns else 'E_total'
            p_col = 'profit' if 'profit' in df.columns else 'Net_profit'
            if 'Cu_out' in df.columns and 'As_out' in df.columns:
                ax_cu_as.scatter(df['Cu_out'], df['As_out'],
                                 label=label, marker=marker, alpha=alpha, s=20)
            if E_col in df.columns and p_col in df.columns:
                ax_e_profit.scatter(df[E_col], df[p_col],
                                    label=label, marker=marker, alpha=alpha, s=20)

        axes[0].set_xlabel('Cu_out (g/L)'); axes[0].set_ylabel('As_out (g/L)')
        axes[0].set_title('Cu_out vs As_out')
        axes[1].set_xlabel('E_total (kWh)'); axes[1].set_ylabel('Profit (万CNY)')
        axes[1].set_title('Energy vs Profit')

        _scatter(axes[0], axes[1], nsga_df, 'NSGA-II', 's', alpha=0.5)
        _scatter(axes[0], axes[1], rl_results, f'PPO-Lag (C{cond})', 'o')

        for ax in axes:
            ax.legend(fontsize=8); ax.grid(True)
        plt.tight_layout()
        path = _out(f'pareto_comparison_condition{cond}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[Saved] Pareto plot → {path}")


# =============================================================================
# ── 12. save_rl_excel（统一，消除三份副本的列名不一致） ───────────────────────
# =============================================================================

def _make_xl_styles() -> dict:
    thin = Side(style='thin')
    return {
        'hf':    Font(name='Arial', bold=True, size=10),
        'df':    Font(name='Arial', size=10),
        'ca':    Alignment(horizontal='center', vertical='center'),
        'la':    Alignment(horizontal='left',   vertical='center'),
        'bd':    Border(left=thin, right=thin, top=thin, bottom=thin),
        'hfill': PatternFill('solid', start_color='D9E1F2'),
    }

def _xl_header(ws, row: int, ncols: int, st: dict):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = st['hf']; cell.alignment = st['ca']
        cell.fill = st['hfill']; cell.border = st['bd']

def _xl_data(cell, val, st: dict, is_str: bool = False):
    cell.font = st['df']
    cell.alignment = st['la'] if is_str else st['ca']
    cell.border = st['bd']
    if val is None:
        return
    if isinstance(val, float):
        if not np.isnan(val):
            cell.value = round(val, 6)
    else:
        cell.value = val


# 模板各 sheet 的标准列头定义（统一在此处维护，消除三份副本的偏差）
# [FIX-11] 统一使用 E_total（原 C1 中误写为 Etotal）；
#          VA/VB/Vaverage 各在自己的 sheet；cost 列统一叫 costs（原 C2 误用 costss）
_SHEET_HEADERS = {
    'Condition 1 Pareto Front':
        ['Cu_in', 'TA', 'IA', 'QA', 't',
         'Cu_out', 'As_out', 'E_total',
         'Net profit\n(Ten thousand  CNY)', 'VA', 'costs'],
    'Condition 2 Pareto Front':
        ['Cu_in', 'TB', 'IB', 'QB', 't',
         'Cu_out', 'As_out', 'E_total',
         'Net profit\n(Ten thousand  CNY)', 'VB', 'costs'],
    'Condition 3 Pareto Front':
        ['Cu_in', 'TA', 'IA', 'QA', 'TB', 'IB', 'QB', 't',
         'Cu_out', 'As_out', 'E_total',
         'Net profit\n(Ten thousand  CNY)', 'Vaverage', 'costs'],
    'Compromise Solutions':
        ['Condition', 'Scenario', 'Cu_in',
         'TA', 'IA', 'QA', 'TB', 'IB', 'QB', 't',
         'Cu_out', 'As_out', 'E_total', 'Net profit'],
    'rl_metrics':
        ['Condition', 'N_solutions', 'Hypervolume', 'Spacing'],
    'reward':
        ['Update_step', 'Condition 1', 'Condition 2', 'Condition 3'],
    'quality_metrics_ppo':
        ['Condition', 'N_solutions',
         'Mean_Cu_out', 'Std_Cu_out', 'Range_Cu_out',
         'Mean_As_out', 'Std_As_out', 'Range_As_out',
         'Mean_E_total', 'Std_E_total', 'Range_E_total',
         'Mean_Net_profit', 'Std_Net_profit', 'Range_Net_profit'],
}

def _ensure_workbook(template_path: str) -> Workbook:
    """读取模板或新建工作簿（含标准 sheet 及列头）。"""
    if os.path.exists(template_path):
        return load_workbook(template_path)
    print(f"[INFO] Template not found, creating new workbook: {template_path}")
    wb = Workbook()
    if 'Sheet' in wb.sheetnames:
        del wb['Sheet']
    st = _make_xl_styles()
    for sheet_name, headers in _SHEET_HEADERS.items():
        ws = wb.create_sheet(title=sheet_name)
        _xl_header(ws, 1, len(headers), st)
        for c_idx, h in enumerate(headers, start=1):
            ws.cell(row=1, column=c_idx).value = h
    return wb


def save_rl_excel(rl_results: pd.DataFrame, training_history_path: str,
                  condition_label: str, template_path: str, output_path: str):
    """
    将 PPO-Lagrangian 结果写入 Excel，与 RL.xlsx 模板格式完全一致。

    参数
    ----
    rl_results            : Pareto 过滤后的 DataFrame
    training_history_path : ppo_training_history_conditionX.csv 路径
    condition_label       : 'Condition 1' / 'Condition 2' / 'Condition 3'
    template_path         : RL.xlsx 模板路径（不存在时自动新建）
    output_path           : 输出 Excel 路径
    """
    wb = _ensure_workbook(template_path)
    st = _make_xl_styles()

    # 决策变量定义（按条件）
    # [FIX-10] Condition 2 决策变量是 TB/IB/QB（非 TA/IA/QA）
    dec_var_map = {
        'Condition 1': ['Cu_in', 'TA', 'IA', 'QA', 't'],
        'Condition 2': ['Cu_in', 'TB', 'IB', 'QB', 't'],
        'Condition 3': ['Cu_in', 'TA', 'IA', 'QA', 'TB', 'IB', 'QB', 't'],
    }
    voltage_col_map = {
        'Condition 1': 'VA',
        'Condition 2': 'VB',
        'Condition 3': 'Vaverage',
    }
    pareto_sheet_map = {
        'Condition 1': 'Condition 1 Pareto Front',
        'Condition 2': 'Condition 2 Pareto Front',
        'Condition 3': 'Condition 3 Pareto Front',
    }

    dec_vars   = dec_var_map[condition_label]
    V_col      = voltage_col_map[condition_label]
    sheet_name = pareto_sheet_map[condition_label]

    # ── Pareto Front sheet ───────────────────────────────────────────────────
    ws = wb[sheet_name]
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.value = None

    if len(rl_results) > 0:
        obj_cols = _SHEET_HEADERS[sheet_name][len(dec_vars):]
        all_cols = dec_vars + obj_cols
        _xl_header(ws, 1, len(all_cols), st)
        for c_idx, h in enumerate(all_cols, start=1):
            ws.cell(row=1, column=c_idx).value = h

        for i, row_data in rl_results.iterrows():
            rr      = i - rl_results.index[0] + 2
            E_val   = float(row_data.get('E', row_data.get('E_total', 0)))
            profit  = float(row_data.get('profit', row_data.get('Net_profit', 0)))
            V_val   = float(row_data.get('V', row_data.get('VA', row_data.get('VB',
                            row_data.get('VMean', row_data.get('Vaverage', 0))))))
            cost_v  = float(row_data.get('costs_max', 0))
            col_val = {dv: float(row_data.get(dv, 0)) for dv in dec_vars}
            col_val.update({
                'Cu_out': float(row_data['Cu_out']),
                'As_out': float(row_data['As_out']),
                'E_total': E_val,
                'Net profit\n(Ten thousand  CNY)': profit,
                V_col: V_val,
                'costs': cost_v,
            })
            for c_idx, col_name in enumerate(all_cols, start=1):
                _xl_data(ws.cell(row=rr, column=c_idx),
                         col_val.get(col_name, ''), st)

    # ── Compromise Solutions sheet ───────────────────────────────────────────
    ws_c = wb['Compromise Solutions']
    for row in ws_c.iter_rows(min_row=2, max_row=ws_c.max_row):
        for cell in row:
            cell.value = None

    if len(rl_results) > 0:
        E_col = 'E' if 'E' in rl_results.columns else 'E_total'
        p_col = 'profit' if 'profit' in rl_results.columns else 'Net_profit'
        obj_mat = np.column_stack([
            rl_results['Cu_out'].values,
            -rl_results['As_out'].values,
            rl_results[E_col].values,
            -rl_results[p_col].values,
        ])
        mn = obj_mat.min(axis=0); mx = obj_mat.max(axis=0)
        dn = mx - mn; dn[dn == 0] = 1e-10
        obj_norm = (obj_mat - mn) / dn
        scene_map = [
            ('Quality priority',  int(np.argmin(obj_norm[:, 0] + obj_norm[:, 1]))),
            ('Energy priority',   int(np.argmin(obj_norm[:, 2]))),
            ('Economic priority', int(np.argmin(obj_norm[:, 3]))),
            ('Balanced',          int(np.argmin(obj_norm.sum(axis=1)))),
        ]
        write_row = 2
        for s_idx, (scenario, row_idx) in enumerate(scene_map):
            rr = rl_results.iloc[row_idx]
            cond_write = condition_label if s_idx == 0 else None
            E_v  = float(rr.get(E_col, 0))
            pr_v = float(rr.get(p_col, 0))

            # [FIX-12] 统一按 dec_vars 提取，无需 fragile 的 len==5 硬编码
            def _g(col_name):
                val = rr.get(col_name, None)
                return float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else None

            row_vals = [
                cond_write, scenario,
                _g('Cu_in'),
                _g('TA'), _g('IA'), _g('QA'),
                _g('TB'), _g('IB'), _g('QB'),
                _g('t'),
                float(rr['Cu_out']), float(rr['As_out']),
                E_v, pr_v,
            ]
            for c_idx, val in enumerate(row_vals, start=1):
                _xl_data(ws_c.cell(row=write_row, column=c_idx),
                         val, st, is_str=(c_idx <= 2))
            write_row += 1

    # ── rl_metrics sheet ─────────────────────────────────────────────────────
    ws_m = wb['rl_metrics']
    for row in ws_m.iter_rows(min_row=2, max_row=ws_m.max_row):
        for cell in row:
            cell.value = None

    cond_row_map = {'Condition 1': 2, 'Condition 2': 3, 'Condition 3': 4}
    r = cond_row_map.get(condition_label, 2)
    n_sol = len(rl_results)

    hv_val = sp_val = None
    if n_sol >= 2:
        try:
            from pymoo.indicators.hv import HV
            from pymoo.indicators.spacing import SpacingIndicator
            E_c = 'E' if 'E' in rl_results.columns else 'E_total'
            p_c = 'profit' if 'profit' in rl_results.columns else 'Net_profit'
            F_rl = np.column_stack([
                rl_results['Cu_out'].values, -rl_results['As_out'].values,
                rl_results[E_c].values,      -rl_results[p_c].values,
            ])
            ref_pt = F_rl.max(axis=0) * 1.1
            hv_val = float(HV(ref_point=ref_pt)(F_rl))
            sp_val = float(SpacingIndicator()(F_rl))
        except (ImportError, Exception):
            pass

    for c_idx, val in enumerate([condition_label, n_sol, hv_val, sp_val], start=1):
        _xl_data(ws_m.cell(row=r, column=c_idx), val, st, is_str=(c_idx == 1))

    # ── reward (training history) sheet ──────────────────────────────────────
    ws_r = wb['reward']
    for row in ws_r.iter_rows(min_row=2, max_row=ws_r.max_row):
        for cell in row:
            cell.value = None

    reward_col_map = {'Condition 1': 2, 'Condition 2': 3, 'Condition 3': 4}
    r_col = reward_col_map.get(condition_label, 2)
    if os.path.exists(training_history_path):
        th = pd.read_csv(training_history_path)
        for i, row_data in th.iterrows():
            rr_idx = i + 2
            if ws_r.cell(row=rr_idx, column=1).value is None:
                ws_r.cell(row=rr_idx, column=1).value = int(row_data['Update_step'])
            ws_r.cell(row=rr_idx, column=r_col).value = float(row_data.get('Reward_sum', 0))

    # ── quality_metrics_ppo sheet ─────────────────────────────────────────────
    ws_q = wb['quality_metrics_ppo']
    for row in ws_q.iter_rows(min_row=2, max_row=ws_q.max_row):
        for cell in row:
            cell.value = None

    if n_sol > 0:
        E_c = 'E' if 'E' in rl_results.columns else 'E_total'
        p_c = 'profit' if 'profit' in rl_results.columns else 'Net_profit'
        q_vals = [
            condition_label, n_sol,
            float(rl_results['Cu_out'].mean()),  float(rl_results['Cu_out'].std()),
            float(rl_results['Cu_out'].max() - rl_results['Cu_out'].min()),
            float(rl_results['As_out'].mean()),  float(rl_results['As_out'].std()),
            float(rl_results['As_out'].max() - rl_results['As_out'].min()),
            float(rl_results[E_c].mean()),       float(rl_results[E_c].std()),
            float(rl_results[E_c].max() - rl_results[E_c].min()),
            float(rl_results[p_c].mean()),       float(rl_results[p_c].std()),
            float(rl_results[p_c].max() - rl_results[p_c].min()),
        ]
        for c_idx, val in enumerate(q_vals, start=1):
            _xl_data(ws_q.cell(row=r, column=c_idx), val, st, is_str=(c_idx == 1))

    wb.save(output_path)
    print(f"  [Saved] RL Excel: {output_path}")


# =============================================================================
# ── 13. 结果合并（三个 condition 训练完成后调用） ────────────────────────────
# =============================================================================

def merge_rl_results(base_dir: str, final_output: str):
    """
    将 Condition 1/2/3 的 pareto_ppo_condition 文件合并为完整的 RL_final.xlsx。
    前提：pareto_ppo_condition{1,2,3}.csv 已存在。
    """
    from openpyxl import Workbook as _WB

    # 决策变量映射
    dec_var_map = {
        'Condition 1': ['Cu_in', 'TA', 'IA', 'QA', 't'],
        'Condition 2': ['Cu_in', 'TB', 'IB', 'QB', 't'],
        'Condition 3': ['Cu_in', 'TA', 'IA', 'QA', 'TB', 'IB', 'QB', 't'],
    }

    voltage_col_map = {
        'Condition 1': 'VA',
        'Condition 2': 'VB',
        'Condition 3': 'Vaverage',
    }

    files = {
        'Condition 1': os.path.join(base_dir, 'pareto_ppo_condition1.csv'),
        'Condition 2': os.path.join(base_dir, 'pareto_ppo_condition2.csv'),
        'Condition 3': os.path.join(base_dir, 'pareto_ppo_condition3.csv'),
    }
    for cond, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"[merge] 找不到 {cond} 文件: {path}")

    print("=== 开始合并 RL 结果 ===")
    sheet_names = list(_SHEET_HEADERS.keys())
    
    # 读取所有文件数据
    dfs = {}
    for cond, path in files.items():
        print(f"  读取 {cond} ...")
        df = pd.read_csv(path)
        dfs[cond] = df

    merged = {}
    
    # ── A. 处理各工况的 Pareto Front ─────────────────────────────────────────
    for cond in files:
        df = dfs[cond].copy()
        # 选择需要的列
        dec_vars = dec_var_map[cond]
        obj_cols = ['Cu_out', 'As_out', 'E_total', 'Net_profit', 'V', 'costs_max']
        selected_cols = dec_vars + obj_cols
        pareto_df = df[selected_cols]
        # 重命名列以匹配模板
        rename_map = {'V': voltage_col_map[cond], 'costs_max': 'costs', 'Net_profit': 'Net profit\n(Ten thousand  CNY)'}
        pareto_df = pareto_df.rename(columns=rename_map)
        merged[f'{cond} Pareto Front'] = pareto_df

    # ── B. 生成 Compromise Solutions ───────────────────────────────────────
    comp_dfs = []
    for cond, df in dfs.items():
        if len(df) == 0:
            continue
        
        # 计算归一化目标矩阵
        obj_mat = np.column_stack([
            df['Cu_out'].values,
            -df['As_out'].values,
            df['E_total'].values,
            -df['Net_profit'].values,
        ])
        mn = obj_mat.min(axis=0); mx = obj_mat.max(axis=0)
        dn = mx - mn; dn[dn == 0] = 1e-10
        obj_norm = (obj_mat - mn) / dn
        
        # 选择不同场景的解
        scene_map = [
            ('Quality priority',  int(np.argmin(obj_norm[:, 0] + obj_norm[:, 1]))),
            ('Energy priority',   int(np.argmin(obj_norm[:, 2]))),
            ('Economic priority', int(np.argmin(obj_norm[:, 3]))),
            ('Balanced',          int(np.argmin(obj_norm.sum(axis=1)))),
        ]
        
        for scenario, row_idx in scene_map:
            rr = df.iloc[row_idx]
            row_data = {
                'Condition': cond,
                'Scenario': scenario,
                'Cu_in': float(rr['Cu_in']),
                'TA': float(rr.get('TA', 0)) if 'TA' in rr else None,
                'IA': float(rr.get('IA', 0)) if 'IA' in rr else None,
                'QA': float(rr.get('QA', 0)) if 'QA' in rr else None,
                'TB': float(rr.get('TB', 0)) if 'TB' in rr else None,
                'IB': float(rr.get('IB', 0)) if 'IB' in rr else None,
                'QB': float(rr.get('QB', 0)) if 'QB' in rr else None,
                't': float(rr['t']),
                'Cu_out': float(rr['Cu_out']),
                'As_out': float(rr['As_out']),
                'E_total': float(rr['E_total']),
                'Net profit': float(rr['Net_profit']),
            }
            comp_dfs.append(row_data)

    merged['Compromise Solutions'] = pd.DataFrame(comp_dfs) if comp_dfs else pd.DataFrame()

    # ── C. 生成 rl_metrics ─────────────────────────────────────────────────
    rl_metrics_data = []
    for cond, df in dfs.items():
        n_sol = len(df)
        # 计算超体积和间距
        hv_val = None
        sp_val = None
        
        try:
            from pymoo.indicators.hv import HV
            from pymoo.indicators.spacing import SpacingIndicator
            F_rl = np.column_stack([
                df['Cu_out'].values, -df['As_out'].values,
                df['E_total'].values, -df['Net_profit'].values,
            ])
            # 使用全局参考点
            hv_val = float(HV(ref_point=GLOBAL_HV_REF_POINT)(F_rl))
            sp_val = float(SpacingIndicator()(F_rl))
        except (ImportError, Exception) as e:
            print(f"  Warning: HV calculation failed for {cond}: {e}")
            pass
        
        rl_metrics_data.append({
            'Condition': cond,
            'N_solutions': n_sol,
            'Hypervolume': hv_val,
            'Spacing': sp_val,
        })

    merged['rl_metrics'] = pd.DataFrame(rl_metrics_data)

    # ── D. reward：从原来的Excel文件中读取 ───────────────────────────
    reward_files = {
        'Condition 1': os.path.join(base_dir, 'outputs_condition1', 'RL_results_condition1.xlsx'),
        'Condition 2': os.path.join(base_dir, 'outputs_condition2', 'RL_results_condition2.xlsx'),
        'Condition 3': os.path.join(base_dir, 'outputs_condition3', 'RL_results_condition3.xlsx'),
    }
    reward_dfs = []
    for cond, path in reward_files.items():
        if os.path.exists(path):
            try:
                xl = pd.ExcelFile(path)
                if 'reward' in xl.sheet_names:
                    df = pd.read_excel(path, sheet_name='reward')
                    if not df.empty:
                        # 重命名列以匹配模板
                        if 'reward' in df.columns:
                            df = df.rename(columns={'reward': cond})
                        reward_dfs.append(df)
            except Exception as e:
                print(f"  Warning: Failed to read reward data from {cond}: {e}")
                pass
    
    if reward_dfs:
        # 以 Update_step 为键做外连接
        df_merged = reward_dfs[0]
        for df_next in reward_dfs[1:]:
            df_merged = pd.merge(df_merged, df_next, on='Update_step', how='outer')
        merged['reward'] = df_merged.sort_values('Update_step').reset_index(drop=True)
    else:
        merged['reward'] = pd.DataFrame()

    # ── E. 生成 quality_metrics_ppo ───────────────────────────────────────
    quality_data = []
    for cond, df in dfs.items():
        if len(df) == 0:
            continue
        
        quality_data.append({
            'Condition': cond,
            'N_solutions': len(df),
            'Mean_Cu_out': float(df['Cu_out'].mean()),
            'Std_Cu_out': float(df['Cu_out'].std()),
            'Range_Cu_out': float(df['Cu_out'].max() - df['Cu_out'].min()),
            'Mean_As_out': float(df['As_out'].mean()),
            'Std_As_out': float(df['As_out'].std()),
            'Range_As_out': float(df['As_out'].max() - df['As_out'].min()),
            'Mean_E_total': float(df['E_total'].mean()),
            'Std_E_total': float(df['E_total'].std()),
            'Range_E_total': float(df['E_total'].max() - df['E_total'].min()),
            'Mean_Net_profit': float(df['Net_profit'].mean()),
            'Std_Net_profit': float(df['Net_profit'].std()),
            'Range_Net_profit': float(df['Net_profit'].max() - df['Net_profit'].min()),
        })

    merged['quality_metrics_ppo'] = pd.DataFrame(quality_data) if quality_data else pd.DataFrame()

    print("\n  写入合并文件 ...")
    st  = _make_xl_styles()
    wb  = _WB()
    del wb['Sheet']

    for sn in sheet_names:
        ws = wb.create_sheet(title=sn)
        df = merged[sn]
        if df.empty:
            print(f"  ⚠ 跳过 {sn}（无数据）")
            continue
        _xl_header(ws, 1, len(df.columns), st)
        for c_idx, col_name in enumerate(df.columns, start=1):
            ws.cell(row=1, column=c_idx).value = col_name
        for r_idx, row_data in df.iterrows():
            excel_row = r_idx - df.index[0] + 2
            for c_idx, val in enumerate(row_data, start=1):
                is_str = isinstance(val, str)
                _xl_data(ws.cell(row=excel_row, column=c_idx), val, st, is_str=is_str)
        print(f"  ✔ {sn} ({len(df)} 行)")

    wb.save(final_output)
    print(f"\n=== 合并完成 → {final_output} ===")


# =============================================================================
# ── 14. Main ──────────────────────────────────────────────────────────────────
# =============================================================================

if __name__ == '__main__':
    # [FIX-13] 不再使用全局 set_start_method；spawn context 在 _init_worker_pool
    # 中通过 mp.get_context('spawn') 按需创建，避免重复调用副作用。

    args = parse_args()
    cond = args.condition
    make_cfg(cond, fast=args.fast)

    # ── 初始化全局进度追踪器 ──────────────────────────────────────────────────
    _TRACKER = ProgressTracker(total_stages=6)
    tracker  = _TRACKER

    t_start = time.time()
    print("=" * 68)
    print(f"  PPO-Lagrangian Safe RL — Condition {cond}"
          + ("  [FAST MODE]" if args.fast else ""))
    print(f"  Constraints: Cu_limit={CFG['Cu_limit']}  As_min={CFG['As_min']}"
          f"  Cu_As_ratio_max={CFG['Cu_As_ratio_max']}")
    print(f"  Device: {DEVICE}  |  Workers: {CFG['n_workers']}"
          f"  |  Seed: {CFG['seed']}")
    print(f"  Rollout/update: {CFG['n_workers']*CFG['per_worker_steps']:,} steps"
          f"  |  Total: {CFG['total_steps']:,} steps"
          f"  ≈ {CFG['total_steps']//(CFG['n_workers']*CFG['per_worker_steps'])} updates")
    print("=" * 68)

    # ── 阶段 1 / 6：加载代理模型 ─────────────────────────────────────────────
    tracker.start_stage(1, "Loading surrogate models")
    t1 = time.time()
    tracker.print_info(f"目录: {CFG['model_dir']}")
    tracker.print_info("正在加载 Cu / As / Voltage ExtraTrees 模型 ...")
    models = load_models(CFG['model_dir'], n_jobs=-1)
    tracker.print_info(f"已加载模型: {list(models.keys())}")
    tracker.end_stage(f"3 个代理模型加载完成  耗时 {time.time()-t1:.1f}s")

    # ── 阶段 2 / 6：特征对齐检查 ─────────────────────────────────────────────
    tracker.start_stage(2, "Feature alignment check")
    tracker.print_info("检查代理模型输入特征维度是否与训练时一致 ...")
    if not check_feature_alignment(models):
        raise RuntimeError("Feature mismatch — abort.")
    tracker.end_stage("特征对齐通过 ✔")

    # ── 阶段 3 / 6：加载 NSGA-II 解 + 初始化 Trainer ────────────────────────
    cond_sheet = f'Condition {cond} Pareto Front'
    tracker.start_stage(3, f"Loading NSGA-II Pareto solutions + Trainer init")
    tracker.print_info(f"NSGA 文件: {CFG['nsga_path']}")
    tracker.print_info(f"读取 Sheet: '{cond_sheet}' ...")
    nsga_df = pd.read_excel(CFG['nsga_path'], sheet_name=cond_sheet)
    tracker.print_info(f"NSGA 解数量: {len(nsga_df)}  列: {list(nsga_df.columns)}")

    tracker.print_info("初始化 PPOLagrangian 训练器 (含 Worker Pool init) ...")
    t3 = time.time()
    trainer = PPOLagrangian(models, nsga_df)
    tracker.end_stage(f"Trainer 就绪  耗时 {time.time()-t3:.1f}s")

    # ── 阶段 4 / 6：Behavioural Cloning 预训练 ───────────────────────────────
    tracker.start_stage(4, "Behavioural Cloning pretraining")
    t4 = time.time()
    trainer.pretrain()
    tracker.end_stage(f"BC 预训练完成  耗时 {time.time()-t4:.1f}s")

    # ── 阶段 5 / 6：PPO-Lagrangian 训练 ──────────────────────────────────────
    tracker.start_stage(5, "PPO-Lagrangian training")
    t5 = time.time()
    trainer.train()
    trainer.save()
    tracker.end_stage(f"训练完成  耗时 {(time.time()-t5)/3600:.2f}h")

    # ── 阶段 6 / 6：分层评估 ─────────────────────────────────────────────────
    tracker.start_stage(6, "Tiered Evaluation (3-round)")
    tier_ep = max(10, CFG['eval_episodes'] // 3)
    tracker.print_info(f"每轮 episodes={tier_ep}  策略: deterministic / stochastic / nsga_perturb")

    # 清除上一次运行留下的旧质量指标文件，确保三轮数据从头累积
    _qm_path = _out(f'quality_metrics_ppo.csv')
    if os.path.exists(_qm_path):
        os.remove(_qm_path)

    tracker.print_info("Round 1/3: Deterministic ...")
    rl_r1 = trainer.evaluate(n_episodes=tier_ep, strategy='deterministic')
    tracker.print_info(f"  → {len(rl_r1)} 可行解")

    tracker.print_info("Round 2/3: Stochastic std×2 ...")
    rl_r2 = trainer.evaluate(n_episodes=tier_ep, strategy='stochastic')
    tracker.print_info(f"  → {len(rl_r2)} 可行解")

    tracker.print_info("Round 3/3: NSGA-II init + perturbation ...")
    rl_r3 = trainer.evaluate(n_episodes=tier_ep, strategy='nsga_perturb')
    tracker.print_info(f"  → {len(rl_r3)} 可行解")

    all_results = [r for r in [rl_r1, rl_r2, rl_r3] if len(r) > 0]
    if all_results:
        rl_results = pd.concat(all_results, ignore_index=True)
        rl_results = _pareto_filter(rl_results)
    else:
        rl_results = pd.DataFrame()
    tracker.print_info(f"合并 Pareto 前沿: {len(rl_results)} 个非支配解")

    tracker.print_info("Weight-policy mapping (2000 samples) ...")
    trainer.generate_weight_policy_mapping(n_samples=2000)

    if len(rl_results):
        hv2 = _compute_hypervolume_2d(rl_results)
        hv4 = _compute_hypervolume_4d(rl_results)
        tracker.print_metric("Hypervolume 2D (Cu-As)", hv2)
        tracker.print_metric("Hypervolume 4D (global ref)", hv4)
        rl_results.to_csv(_out(f'rl_pareto_condition{cond}.csv'), index=False)
        tracker.print_info(f"保存 CSV: rl_pareto_condition{cond}.csv")

        tracker.print_info(f"生成 RL_results_condition{cond}.xlsx ...")
        save_rl_excel(
            rl_results,
            training_history_path=_out(f'ppo_training_history_condition{cond}.csv'),
            condition_label=f'Condition {cond}',
            template_path=CFG['rl_xlsx_template'],
            output_path=_out(f'RL_results_condition{cond}.xlsx'),
        )
    else:
        tracker.print_info("[WARNING] 未找到可行解！")
        tracker.print_info("建议：1) 放松 cost_limit；2) 增大 eval_episodes；"
                           "3) 查看 lambdas_hist 是否收敛")

    trainer.plot_training()
    trainer.plot_pareto(rl_results, nsga_df)

    total_elapsed = time.time() - t_start
    tracker.end_stage(f"评估完成")

    print(f"\n{'='*68}")
    print(f"  ✔  全流程完成  Condition {cond}"
          f"  总耗时 {total_elapsed/3600:.2f}h  ({tracker._fmt_time(total_elapsed)})")
    print(f"  输出目录: {CFG['output_dir']}")
    print(f"{'='*68}")
