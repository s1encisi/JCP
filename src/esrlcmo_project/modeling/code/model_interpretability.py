"""
模型可解释性分析脚本 v2 —— SCI 顶刊绘图风格
修复内容：
  - PDP 使用原始列名传给 partial_dependence，展示时才用英文标签
  - 字体统一为 Times New Roman（英文）
  - 所有图宽高比 4:3，DPI=600
  - feature_importance_all_models：删除大标题、删除左侧行标签、子图标题单行（无单位）、间距减半
  - shap_all_models_summary：删除大标题、间距减半
  - 单模型 shap 图：子图间距减半、标题下移、colorbar 标签改为 Feature Value
  - 新增 shap_beeswarm_all_models.png：9 个蜂群图合并大图
  - PDP 全面修正：传原始列名、4:3 比例、Times New Roman、无中文

新增功能（不改变任何绘图逻辑）：
  - 每张图片生成对应的 Excel 文件，保存到 EXCEL_ROOT 目录，方便在 Origin 中重新绘制
"""

import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import shap
import joblib
from sklearn.inspection import partial_dependence

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 0.  全局样式
# ══════════════════════════════════════════════════════════════════════════════
def setup_style():
    mpl.rcParams.update({
        'font.family':        'serif',
        'font.serif':         ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset':   'stix',
        'axes.unicode_minus': False,
        'figure.dpi':         150,
        'savefig.dpi':        600,
        'axes.linewidth':     0.8,
        'axes.spines.top':    True,
        'axes.spines.right':  True,
        'xtick.major.width':  0.8,
        'ytick.major.width':  0.8,
        'xtick.direction':    'in',
        'ytick.direction':    'in',
        'font.size':          9,
        'axes.titlesize':     10,
        'axes.labelsize':     9,
        'legend.fontsize':    8,
        'legend.framealpha':  0.85,
        'legend.edgecolor':   '#cccccc',
    })

setup_style()

# ══════════════════════════════════════════════════════════════════════════════
# 1.  配色
# ══════════════════════════════════════════════════════════════════════════════
PALETTE_TARGET = {
    'voltage': '#2166AC',
    'cu':      '#D6604D',
    'as':      '#4DAC26',
}
SHAP_CMAP = LinearSegmentedColormap.from_list(
    'shap_bwr', ['#3288BD', '#FFFFBF', '#D53E4F'], N=256
)

# ══════════════════════════════════════════════════════════════════════════════
# 2.  模型配置（features 保持原始列名，用于训练/PDP）
# ══════════════════════════════════════════════════════════════════════════════
MODEL_CONFIGS = {
    'voltage_three_stage': {
        'model_path':   'outputs/voltage_three_stage/extra_trees_three_stage.joblib',
        'sheet':        'three_stage',
        'features':     ['电积前液Cu（g/L）','三段溶液温度℃','三段电流强度A','三段流量m3/h',
                         'year','month','day','hour'],
        'target':       '三段电压V',
        'unit':         'V',
        'label':        'Condition1 | Voltage',
        'target_group': 'voltage',
        'stage_group':  'three_stage',
    },
    'voltage_four_stage': {
        'model_path':   'outputs/voltage_four_stage/extra_trees_four_stage.joblib',
        'sheet':        'four_stage',
        'features':     ['电积前液Cu（g/L）','四段溶液温度℃','四段电流强度A','四段流量m3/h',
                         'year','month','day','hour'],
        'target':       '四段电压V',
        'unit':         'V',
        'label':        'Condition2 | Voltage',
        'target_group': 'voltage',
        'stage_group':  'four_stage',
    },
    'voltage_serial': {
        'model_path':   'outputs/voltage_serial/extra_trees_serial.joblib',
        'sheet':        'serial',
        'features':     ['电积前液Cu（g/L）','三段溶液温度℃','三段电流强度A','三段流量m3/h',
                         '四段溶液温度℃','四段电流强度A','四段流量m3/h','year','month','day','hour'],
        'target':       '三四段平均电压V',
        'unit':         'V',
        'label':        'Condition3 | Voltage',
        'target_group': 'voltage',
        'stage_group':  'serial',
        'special':      'serial_voltage',
    },
    'cu_three_stage': {
        'model_path':   'outputs/cu_three_stage/extra_trees_three_stage.joblib',
        'sheet':        'three_stage',
        'features':     ['电积前液Cu（g/L）','三段溶液温度℃','三段电流强度A','三段流量m3/h',
                         '三段电压V','year','month','day','hour','three_stage_power'],
        'target':       '废铜液原液罐Cu（g/L）',
        'unit':         'g/L',
        'label':        'Condition1 | Cu',
        'target_group': 'cu',
        'stage_group':  'three_stage',
    },
    'cu_four_stage': {
        'model_path':   'outputs/cu_four_stage/extra_trees_four_stage.joblib',
        'sheet':        'four_stage',
        'features':     ['电积前液Cu（g/L）','四段溶液温度℃','四段电流强度A','四段流量m3/h',
                         '四段电压V','year','month','day','hour','four_stage_power'],
        'target':       '废铜液原液罐Cu（g/L）',
        'unit':         'g/L',
        'label':        'Condition2 | Cu',
        'target_group': 'cu',
        'stage_group':  'four_stage',
    },
    'cu_serial': {
        'model_path':   'outputs/cu_serial/extra_trees_serial.joblib',
        'sheet':        'serial',
        'features':     ['电积前液Cu（g/L）','三段溶液温度℃','三段电流强度A','三段流量m3/h','三段电压V',
                         '四段溶液温度℃','四段电流强度A','四段流量m3/h','四段电压V',
                         'year','month','day','hour','total_power'],
        'target':       '废铜液原液罐Cu（g/L）',
        'unit':         'g/L',
        'label':        'Condition3 | Cu',
        'target_group': 'cu',
        'stage_group':  'serial',
    },
    'as_three_stage': {
        'model_path':   'outputs/as_three_stage/extra_trees_three_stage.joblib',
        'sheet':        'three_stage',
        'features':     ['电积前液Cu（g/L）','三段溶液温度℃','三段电流强度A','三段流量m3/h',
                         '三段电压V','year','month','day','hour','three_stage_power'],
        'target':       '废铜液原液罐As（g/L）',
        'unit':         'g/L',
        'label':        'Condition1 | As',
        'target_group': 'as',
        'stage_group':  'three_stage',
    },
    'as_four_stage': {
        'model_path':   'outputs/as_four_stage/extra_trees_four_stage.joblib',
        'sheet':        'four_stage',
        'features':     ['电积前液Cu（g/L）','四段溶液温度℃','四段电流强度A','四段流量m3/h',
                         '四段电压V','year','month','day','hour','four_stage_power'],
        'target':       '废铜液原液罐As（g/L）',
        'unit':         'g/L',
        'label':        'Condition2 | As',
        'target_group': 'as',
        'stage_group':  'four_stage',
    },
    'as_serial': {
        'model_path':   'outputs/as_serial/extra_trees_serial.joblib',
        'sheet':        'serial',
        'features':     ['电积前液Cu（g/L）','三段溶液温度℃','三段电流强度A','三段流量m3/h','三段电压V',
                         '四段溶液温度℃','四段电流强度A','四段流量m3/h','四段电压V',
                         'year','month','day','hour','total_power'],
        'target':       '废铜液原液罐As（g/L）',
        'unit':         'g/L',
        'label':        'Condition3 | As',
        'target_group': 'as',
        'stage_group':  'serial',
    },
}

# 特征名 → 显示标签（纯英文，用于图中轴标签）
FEAT_DISPLAY = {
    '电积前液Cu（g/L）':  'Cu_in',
    '三段溶液温度℃':      'T$_A$',
    '三段电流强度A':       'I$_A$',
    '三段流量m3/h':        'Q$_A$',
    '三段电压V':           'V$_A$',
    'three_stage_power':   'P$_A$',
    '四段溶液温度℃':      'T$_B$',
    '四段电流强度A':       'I$_B$',
    '四段流量m3/h':        'Q$_B$',
    '四段电压V':           'V$_B$',
    'four_stage_power':    'P$_B$',
    'total_power':         'P$_{total}$',
    'year':                'Year',
    'month':               'Month',
    'day':                 'Day',
    'hour':                'Hour',
}

# Origin 中不支持 LaTeX，提供纯文本版显示标签
FEAT_DISPLAY_PLAIN = {
    '电积前液Cu（g/L）':  'Cu_in',
    '三段溶液温度℃':      'TA',
    '三段电流强度A':       'IA',
    '三段流量m3/h':        'QA',
    '三段电压V':           'VA',
    'three_stage_power':   'PA',
    '四段溶液温度℃':      'TB',
    '四段电流强度A':       'IB',
    '四段流量m3/h':        'QB',
    '四段电压V':           'VB',
    'four_stage_power':    'PB',
    'total_power':         'Ptotal',
    'year':                'Year',
    'month':               'Month',
    'day':                 'Day',
    'hour':                'Hour',
}

def disp(col):
    """原始列名 → 显示标签（含 LaTeX，用于绘图）"""
    return FEAT_DISPLAY.get(col, col)

def disp_plain(col):
    """原始列名 → 纯文本显示标签（用于 Excel，Origin 友好）"""
    return FEAT_DISPLAY_PLAIN.get(col, col)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE  = str(PROJECT_ROOT / 'data' / 'data_by_operation_mode_with_features.xlsx')
OUT_ROOT   = str(PROJECT_ROOT / 'modeling' / 'interpretability_outputs')
EXCEL_ROOT = OUT_ROOT   # Excel 输出根目录

# ══════════════════════════════════════════════════════════════════════════════
# 3.  辅助
# ══════════════════════════════════════════════════════════════════════════════
def ensure_dirs():
    for sub in ['feature_importance', 'shap', 'pdp']:
        os.makedirs(os.path.join(OUT_ROOT, sub), exist_ok=True)
    # Excel 子目录
    for sub in ['feature_importance', 'shap', 'pdp']:
        os.makedirs(os.path.join(EXCEL_ROOT, sub), exist_ok=True)


def save_excel(df_or_dict, path, description=''):
    """
    保存 DataFrame 或 {sheet_name: DataFrame} 字典到 Excel。
    路径不存在时自动创建父目录。
    """
    import time
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        try:
            if isinstance(df_or_dict, pd.DataFrame):
                df_or_dict.to_excel(path, index=False, engine='openpyxl')
            else:
                with pd.ExcelWriter(path, engine='openpyxl') as writer:
                    for sheet, df in df_or_dict.items():
                        df.to_excel(writer, sheet_name=sheet[:31], index=False)
            print(f'      [Excel] {description}: {path}')
            return
        except PermissionError:
            attempt += 1
            if attempt < max_attempts:
                print(f'      [Excel] Permission error, retrying ({attempt}/{max_attempts})...')
                time.sleep(1)
            else:
                print(f'      [Excel] Failed to save after {max_attempts} attempts: {path}')
                raise


def load_data_and_model(key, cfg):
    """
    返回 (model, X_original, y)
    X_original 的列名保持原始中文/英文，供 partial_dependence 使用
    """
    model = joblib.load(cfg['model_path'])
    df = pd.read_excel(DATA_FILE, sheet_name=cfg['sheet'])
    if cfg.get('special') == 'serial_voltage':
        df['三四段平均电压V'] = (df['三段电压V'] + df['四段电压V']) / 2
    feat = cfg['features']
    tgt  = cfg['target']
    mdf  = df.dropna(subset=[tgt] + feat).copy()
    X    = mdf[feat].copy()
    y    = mdf[tgt].values
    return model, X, y


def make_bar_colors(imp_sorted, base_color):
    """生成从浅→深的渐变颜色列表"""
    norm = (imp_sorted - imp_sorted.min()) / (imp_sorted.max() - imp_sorted.min() + 1e-10)
    base = mcolors.to_rgb(base_color)
    colors = [
        tuple(base[c] * 0.35 + 0.65 * (1 - n * 0.65) for c in range(3))
        for n in norm
    ]
    colors[-1] = base
    return colors


# ══════════════════════════════════════════════════════════════════════════════
# 4.  特征重要性综合图  +  Excel
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_importance_all(results_dict):
    target_order = ['voltage', 'cu', 'as']
    stage_order  = ['three_stage', 'four_stage', 'serial']
    subplot_titles = {
        ('voltage', 'three_stage'): 'Condition 1 | Voltage',
        ('voltage', 'four_stage'):  'Condition 2 | Voltage',
        ('voltage', 'serial'):      'Condition 3 | Voltage',
        ('cu',      'three_stage'): 'Condition 1 | Cu',
        ('cu',      'four_stage'):  'Condition 2 | Cu',
        ('cu',      'serial'):      'Condition 3 | Cu',
        ('as',      'three_stage'): 'Condition 1 | As',
        ('as',      'four_stage'):  'Condition 2 | As',
        ('as',      'serial'):      'Condition 3 | As',
    }

    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5))
    fig.subplots_adjust(hspace=0.30, wspace=0.20)

    # ── Excel 数据收集（综合图，每个子图对应一个 sheet）──
    excel_sheets = {}

    for ri, tgt in enumerate(target_order):
        for ci, stg in enumerate(stage_order):
            ax = axes[ri, ci]
            found_key = next(
                (k for k, v in results_dict.items()
                 if v['target_group'] == tgt and v['stage_group'] == stg), None
            )
            if found_key is None:
                ax.axis('off'); continue

            info  = results_dict[found_key]
            imp   = np.array(info['importances'])
            feat  = info['original_features']
            color = PALETTE_TARGET[tgt]

            idx = np.argsort(imp)
            n   = min(len(idx), 10)
            idx = idx[-n:]
            imp_s  = imp[idx]
            feat_s = [disp(feat[i]) for i in idx]

            bar_colors = make_bar_colors(imp_s, color)
            bars = ax.barh(range(n), imp_s, color=bar_colors,
                           edgecolor='white', linewidth=0.5, height=0.65)
            for bar, val in zip(bars, imp_s):
                ax.text(bar.get_width() + imp_s.max() * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}', va='center', ha='left',
                        fontsize=10, fontweight='bold', color='#333333')

            ax.set_yticks(range(n))
            ax.set_yticklabels(feat_s, fontsize=12, fontweight='bold')
            ax.set_xlabel('Feature Importance', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', labelsize=12, length=3)
            ax.set_xlim(0, imp_s.max() * 1.25)
            # 删除子图标题
            # ax.set_title(subplot_titles[(tgt, stg)], fontsize=11,
            #              fontweight='bold', pad=5, color=color)
            ax.grid(axis='x', linestyle='--', linewidth=0.4, alpha=0.5, color='#bbbbbb')
            # 加粗子图边框
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
            # 设置 X 轴刻度标签加粗
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')

            # 收集该子图数据（bottom→top 排列，Origin 横条图习惯）
            sheet_name = f'{tgt}_{stg}'
            excel_sheets[sheet_name] = pd.DataFrame({
                'Feature':    [disp_plain(feat[i]) for i in idx],   # 无 LaTeX
                'Feature_raw': [feat[i] for i in idx],
                'Importance': imp_s,
            })

    out = os.path.join(OUT_ROOT, 'feature_importance', 'feature_importance_all_models.png')
    fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  ✓ Feature importance combined: {out}')

    # 保存综合图对应 Excel（每个子图一个 sheet）
    save_excel(
        excel_sheets,
        os.path.join(EXCEL_ROOT, 'feature_importance', 'feature_importance_all_models.xlsx'),
        'FI combined (9 subplots)'
    )

    # 单模型图
    for key, info in results_dict.items():
        _plot_single_fi(key, info)


def _plot_single_fi(key, info):
    imp   = np.array(info['importances'])
    feat  = info['original_features']
    color = PALETTE_TARGET[info['target_group']]

    idx    = np.argsort(imp)
    imp_s  = imp[idx]
    feat_s = [disp(feat[i]) for i in idx]

    h = max(3.0, len(feat_s) * 0.38)
    fig, ax = plt.subplots(figsize=(h * 4/3, h))

    bar_colors = make_bar_colors(imp_s, color)
    bars = ax.barh(range(len(feat_s)), imp_s, color=bar_colors,
                   edgecolor='white', linewidth=0.5, height=0.65)
    for bar, val in zip(bars, imp_s):
        ax.text(bar.get_width() + imp_s.max() * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', ha='left', fontsize=7.5, color='#333333')

    ax.set_yticks(range(len(feat_s)))
    ax.set_yticklabels(feat_s, fontsize=8.5)
    ax.set_xlabel('Feature Importance', fontsize=9.5)
    ax.set_xlim(0, imp_s.max() * 1.30)
    ax.set_title(f'Feature Importance — {info["label"]}', fontsize=10, fontweight='bold', pad=7)
    ax.grid(axis='x', linestyle='--', linewidth=0.45, alpha=0.55, color='#bbbbbb')
    fig.tight_layout()

    out = os.path.join(OUT_ROOT, 'feature_importance', f'fi_{key}.png')
    fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'    ✓ {key}: {out}')

    # ── Excel（与图完全一致，bottom→top 顺序）──
    df_excel = pd.DataFrame({
        'Feature':     [disp_plain(feat[i]) for i in idx],
        'Feature_raw': [feat[i] for i in idx],
        'Importance':  imp_s,
    })
    save_excel(
        df_excel,
        os.path.join(EXCEL_ROOT, 'feature_importance', f'fi_{key}.xlsx'),
        f'FI single {key}'
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5.  特征重要性汇总 Excel（原有，保留不变）
# ══════════════════════════════════════════════════════════════════════════════
def export_importance_excel(results_dict):
    rows = []
    for key, info in results_dict.items():
        for feat, imp in zip(info['original_features'], info['importances']):
            rows.append({
                'Model Key':         key,
                'Label':             info['label'],
                'Target Group':      info['target_group'],
                'Stage Group':       info['stage_group'],
                'Feature (raw)':     feat,
                'Feature (display)': disp(feat),
                'Importance':        imp,
            })
    df_all = pd.DataFrame(rows)
    df_all['Rank'] = df_all.groupby('Model Key')['Importance'].rank(
        ascending=False, method='min').astype(int)

    path = os.path.join(OUT_ROOT, 'feature_importance', 'feature_importance_summary.xlsx')
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        df_all.to_excel(writer, sheet_name='All Models', index=False)
        for key, grp in df_all.groupby('Model Key'):
            grp.sort_values('Rank').to_excel(
                writer, sheet_name=key[:31], index=False)
        pivot = df_all.pivot_table(
            index='Feature (display)', columns='Model Key',
            values='Importance', aggfunc='mean')
        pivot.to_excel(writer, sheet_name='Pivot Table')
    print(f'  ✓ Excel: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SHAP
# ══════════════════════════════════════════════════════════════════════════════
def compute_shap(model, X_orig, max_samples=300):
    """X_orig 保持原始列名"""
    if len(X_orig) > max_samples:
        idx = np.random.choice(len(X_orig), max_samples, replace=False)
        X_s = X_orig.iloc[idx].reset_index(drop=True)
    else:
        X_s = X_orig.reset_index(drop=True)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_s)
    return sv, X_s


def _draw_beeswarm(ax, sv, X_s, sorted_idx, cmap):
    """在给定 ax 上画蜂群图，使用原始列名的显示标签"""
    n_feat = len(X_s.columns)
    for row_i, fi in enumerate(sorted_idx):
        sv_col = sv[:, fi]
        x_col  = X_s.iloc[:, fi].values
        x_norm = (x_col - x_col.min()) / (x_col.max() - x_col.min() + 1e-10)
        colors = cmap(x_norm)
        jitter = np.random.uniform(-0.3, 0.3, size=len(sv_col))
        ax.scatter(sv_col, row_i + jitter, c=colors, s=6,
                   alpha=0.6, linewidths=0, zorder=2)
    ax.axvline(0, color='#555555', linewidth=0.8, linestyle='--', zorder=1)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([disp(X_s.columns[i]) for i in sorted_idx], fontsize=8)
    ax.set_xlabel('SHAP Value', fontsize=9)
    ax.grid(axis='x', linestyle='--', linewidth=0.4, alpha=0.45, color='#cccccc')


def plot_shap_single(key, cfg, sv, X_s):
    """单模型 SHAP：左条形 + 右蜂群，4:3，间距减半，colorbar 标签改 Feature Value"""
    color  = PALETTE_TARGET[cfg['target_group']]
    n_feat = len(X_s.columns)
    h = max(4.5, n_feat * 0.45 + 1.5)
    w = h * 4 / 3 * 2

    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    fig.subplots_adjust(wspace=0.22, top=0.88)

    mean_abs   = np.abs(sv).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)        # ascending
    feat_labels = [disp(X_s.columns[i]) for i in sorted_idx]
    imp_s = mean_abs[sorted_idx]

    # ── 左：条形 ──
    ax0 = axes[0]
    bar_colors = make_bar_colors(imp_s, color)
    bars = ax0.barh(range(n_feat), imp_s, color=bar_colors,
                    edgecolor='white', linewidth=0.4, height=0.65)
    for bar, val in zip(bars, imp_s):
        ax0.text(bar.get_width() + imp_s.max() * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f'{val:.4f}', va='center', ha='left', fontsize=7, color='#333333')
    ax0.set_yticks(range(n_feat))
    ax0.set_yticklabels(feat_labels, fontsize=8)
    ax0.set_xlabel('Mean |SHAP Value|', fontsize=9)
    ax0.set_xlim(0, imp_s.max() * 1.30)
    ax0.set_title('SHAP Feature Importance', fontsize=9.5, fontweight='bold')
    ax0.grid(axis='x', linestyle='--', linewidth=0.4, alpha=0.5, color='#cccccc')

    # ── 右：蜂群 ──
    ax1 = axes[1]
    _draw_beeswarm(ax1, sv, X_s, sorted_idx, SHAP_CMAP)
    ax1.set_title('SHAP Beeswarm Plot', fontsize=9.5, fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap=SHAP_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, shrink=0.55, pad=0.02, aspect=22)
    cbar.set_label('Feature Value', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(f'SHAP Analysis — {cfg["label"]}',
                 fontsize=11, fontweight='bold', y=0.96)

    out = os.path.join(OUT_ROOT, 'shap', f'shap_{key}.png')
    fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'    ✓ SHAP {key}: {out}')

    # ── Excel ──
    # Sheet 1：条形图数据（mean |SHAP|，bottom→top 顺序）
    df_bar = pd.DataFrame({
        'Feature':        [disp_plain(X_s.columns[i]) for i in sorted_idx],
        'Feature_raw':    [X_s.columns[i] for i in sorted_idx],
        'Mean_abs_SHAP':  imp_s,
    })
    # Sheet 2：蜂群图原始数据（每列一个特征，行为每个样本的 SHAP 值）
    shap_df = pd.DataFrame(
        sv[:, sorted_idx],
        columns=[disp_plain(X_s.columns[i]) for i in sorted_idx]
    )
    # Sheet 3：对应特征原始值（归一化前，用于颜色映射参考）
    feat_df = pd.DataFrame(
        X_s.iloc[:, sorted_idx].values,
        columns=[disp_plain(X_s.columns[i]) for i in sorted_idx]
    )
    save_excel(
        {'Bar_MeanAbsSHAP': df_bar, 'Beeswarm_SHAP_Values': shap_df, 'Beeswarm_Feature_Values': feat_df},
        os.path.join(EXCEL_ROOT, 'shap', f'shap_{key}.xlsx'),
        f'SHAP single {key}'
    )


def plot_shap_all_bar(all_shap_data):
    """3×3 综合 SHAP 条形图，无大标题，间距减半"""
    target_order = ['voltage', 'cu', 'as']
    stage_order  = ['three_stage', 'four_stage', 'serial']
    subplot_titles = {
        ('voltage', 'three_stage'): 'Condition 1 | Voltage',
        ('voltage', 'four_stage'):  'Condition 2 | Voltage',
        ('voltage', 'serial'):      'Condition 3 | Voltage',
        ('cu',      'three_stage'): 'Condition 1 | Cu',
        ('cu',      'four_stage'):  'Condition 2 | Cu',
        ('cu',      'serial'):      'Condition 3 | Cu',
        ('as',      'three_stage'): 'Condition 1 | As',
        ('as',      'four_stage'):  'Condition 2 | As',
        ('as',      'serial'):      'Condition 3 | As',
    }

    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5))
    fig.subplots_adjust(hspace=0.30, wspace=0.26)

    excel_sheets = {}

    for ri, tgt in enumerate(target_order):
        for ci, stg in enumerate(stage_order):
            ax = axes[ri, ci]
            found_key = next(
                (k for k, (sv, X_s, cfg) in all_shap_data.items()
                 if cfg['target_group'] == tgt and cfg['stage_group'] == stg), None
            )
            if found_key is None:
                ax.axis('off'); continue

            sv, X_s, cfg = all_shap_data[found_key]
            color    = PALETTE_TARGET[tgt]
            mean_abs = np.abs(sv).mean(axis=0)
            idx      = np.argsort(mean_abs)
            n        = min(len(idx), 10)
            idx      = idx[-n:]
            feat_s   = [disp(X_s.columns[i]) for i in idx]
            imp_s    = mean_abs[idx]

            bar_colors = make_bar_colors(imp_s, color)
            bars = ax.barh(range(n), imp_s, color=bar_colors,
                           edgecolor='white', linewidth=0.4, height=0.65)
            for bar, val in zip(bars, imp_s):
                ax.text(bar.get_width() + imp_s.max() * 0.03,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}', va='center', ha='left', fontsize=6, color='#333333')

            ax.set_yticks(range(n))
            ax.set_yticklabels(feat_s, fontsize=7)
            ax.set_xlabel('Mean |SHAP|', fontsize=7.5)
            ax.set_xlim(0, imp_s.max() * 1.38)
            ax.set_title(subplot_titles[(tgt, stg)], fontsize=8.5,
                         fontweight='bold', pad=5, color=color)
            ax.grid(axis='x', linestyle='--', linewidth=0.35, alpha=0.5, color='#cccccc')
            ax.tick_params(axis='x', labelsize=6.5)

            sheet_name = f'{tgt}_{stg}'
            excel_sheets[sheet_name] = pd.DataFrame({
                'Feature':       [disp_plain(X_s.columns[i]) for i in idx],
                'Feature_raw':   [X_s.columns[i] for i in idx],
                'Mean_abs_SHAP': imp_s,
            })

    out = os.path.join(OUT_ROOT, 'shap', 'shap_all_models_summary.png')
    fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  ✓ SHAP combined bar: {out}')

    save_excel(
        excel_sheets,
        os.path.join(EXCEL_ROOT, 'shap', 'shap_all_models_summary.xlsx'),
        'SHAP combined bar (9 subplots)'
    )


def plot_shap_all_beeswarm(all_shap_data):
    """3×3 综合蜂群图，无大标题，间距减半"""
    target_order = ['voltage', 'cu', 'as']
    stage_order  = ['three_stage', 'four_stage', 'serial']
    subplot_titles = {
        ('voltage', 'three_stage'): 'Condition 1 | Voltage',
        ('voltage', 'four_stage'):  'Condition 2 | Voltage',
        ('voltage', 'serial'):      'Condition 3 | Voltage',
        ('cu',      'three_stage'): 'Condition 1 | Cu',
        ('cu',      'four_stage'):  'Condition 2 | Cu',
        ('cu',      'serial'):      'Condition 3 | Cu',
        ('as',      'three_stage'): 'Condition 1 | As',
        ('as',      'four_stage'):  'Condition 2 | As',
        ('as',      'serial'):      'Condition 3 | As',
    }

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.22, wspace=0.19)

    # 用于 Excel：每个子图一个 sheet，保存 SHAP 值 + 特征值
    shap_sheets = {}
    feat_sheets = {}

    for ri, tgt in enumerate(target_order):
        for ci, stg in enumerate(stage_order):
            ax = axes[ri, ci]
            found_key = next(
                (k for k, (sv, X_s, cfg) in all_shap_data.items()
                 if cfg['target_group'] == tgt and cfg['stage_group'] == stg), None
            )
            if found_key is None:
                ax.axis('off'); continue

            sv, X_s, cfg = all_shap_data[found_key]
            color    = PALETTE_TARGET[tgt]
            mean_abs = np.abs(sv).mean(axis=0)
            sorted_idx = np.argsort(mean_abs)

            n = min(len(sorted_idx), 10)
            sorted_idx = sorted_idx[-n:] if n < len(sorted_idx) else sorted_idx

            n_feat = len(sorted_idx)
            for row_i, fi in enumerate(sorted_idx):
                sv_col = sv[:, fi]
                x_col  = X_s.iloc[:, fi].values
                x_norm = (x_col - x_col.min()) / (x_col.max() - x_col.min() + 1e-10)
                colors = SHAP_CMAP(x_norm)
                jitter = np.random.uniform(-0.28, 0.28, size=len(sv_col))
                ax.scatter(sv_col, row_i + jitter, c=colors, s=5,
                           alpha=0.55, linewidths=0, zorder=2)

            ax.axvline(0, color='#555555', linewidth=0.7, linestyle='--', zorder=1)
            ax.set_yticks(range(n_feat))
            ax.set_yticklabels([disp(X_s.columns[i]) for i in sorted_idx], fontsize=12, fontweight='bold')
            ax.set_xlabel('SHAP Value', fontsize=14, fontweight='bold')
            # 删除子图标题
            # ax.set_title(subplot_titles[(tgt, stg)], fontsize=11,
            #              fontweight='bold', pad=5, color=color)
            ax.grid(axis='x', linestyle='--', linewidth=0.35, alpha=0.45, color='#cccccc')
            ax.tick_params(axis='x', labelsize=12, length=3)
            # 加粗子图边框
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
            # 设置 X 轴刻度标签加粗
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')

            if ci == 2:
                sm = plt.cm.ScalarMappable(cmap=SHAP_CMAP, norm=plt.Normalize(0, 1))
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.03, aspect=25)
                cbar.set_label('Feature Value', fontsize=10, fontweight='bold')
                cbar.ax.tick_params(labelsize=10)
                # 设置颜色条刻度标签加粗
                for label in cbar.ax.get_yticklabels():
                    label.set_fontweight('bold')

            # 收集 Excel 数据
            col_names = [disp_plain(X_s.columns[i]) for i in sorted_idx]
            sheet_name = f'{tgt}_{stg}'
            shap_sheets[sheet_name] = pd.DataFrame(
                sv[:, sorted_idx], columns=col_names)
            feat_sheets[sheet_name] = pd.DataFrame(
                X_s.iloc[:, sorted_idx].values, columns=col_names)

    out = os.path.join(OUT_ROOT, 'shap', 'shap_beeswarm_all_models.png')
    fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  ✓ SHAP beeswarm combined: {out}')

    # Excel：两个文件，SHAP 值 和 特征原始值，各 9 个 sheet
    save_excel(
        shap_sheets,
        os.path.join(EXCEL_ROOT, 'shap', 'shap_beeswarm_all_models_SHAP_values.xlsx'),
        'Beeswarm combined SHAP values'
    )
    save_excel(
        feat_sheets,
        os.path.join(EXCEL_ROOT, 'shap', 'shap_beeswarm_all_models_feature_values.xlsx'),
        'Beeswarm combined feature values'
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PDP
# ══════════════════════════════════════════════════════════════════════════════
def _draw_pdp_ax(ax, model, X_orig, orig_feat, color, unit,
                 ice_max_samples=100, show_ylabel=True,
                 fontsize_label=10.5, fontsize_tick=9.5, fontsize_title=11,
                 linewidth_avg=2.2, panel_label=None):
    """
    核心绘图函数：在单个 Axes 上绘制 ICE + PDP average 曲线（SCI 顶刊风格）

    Parameters
    ----------
    ax              : matplotlib Axes
    model           : fitted estimator
    X_orig          : pd.DataFrame, original feature matrix
    orig_feat       : str, raw column name
    color           : str, hex color for the average line & fill
    unit            : str, target unit string
    ice_max_samples : int, max ICE curves to draw (subsampled for speed)
    show_ylabel     : bool, whether to draw y-axis label
    fontsize_*      : font sizes
    linewidth_avg   : float, linewidth of average PDP line
    panel_label     : str or None, panel letter label (e.g. '(a)')

    Returns
    -------
    rec : dict with PDP data (for Excel export), or None on error
    """
    feat_idx = list(X_orig.columns).index(orig_feat)

    try:
        # Compute both ICE curves and the average
        pdp_res = partial_dependence(
            model, X_orig, [feat_idx],
            kind='both', grid_resolution=60
        )
        grid_vals = pdp_res['grid_values'][0]
        avg_pred  = pdp_res['average'][0]
        ice_preds = pdp_res['individual'][0]   # shape (n_samples, n_grid)
    except Exception as e:
        ax.text(0.5, 0.5, f'PDP error:\n{str(e)[:80]}',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=6, wrap=True)
        # 取消错误情况下的标题显示
        # ax.set_title(disp(orig_feat), fontsize=fontsize_title, fontweight='bold',
        #              color=color, pad=5)
        return None

    # ── ICE lines (subsampled) ────────────────────────────────────────────────
    n_ice = min(ice_max_samples, ice_preds.shape[0])
    ice_idx = np.random.choice(ice_preds.shape[0], n_ice, replace=False)

    # Build a slightly lighter / transparent tint from the main color
    base_rgb = mcolors.to_rgb(color)
    ice_color = tuple(min(1.0, c * 0.55 + 0.45) for c in base_rgb)

    for i in ice_idx:
        ax.plot(grid_vals, ice_preds[i], color=ice_color,
                linewidth=0.55, alpha=0.35, zorder=1)

    # ── Average PDP line ──────────────────────────────────────────────────────
    ax.plot(grid_vals, avg_pred, color=color,
            linewidth=linewidth_avg, zorder=4, label='average',
            solid_capstyle='round')

    # ── Subtle fill under the average curve ──────────────────────────────────
    ax.fill_between(grid_vals, avg_pred.min(), avg_pred,
                    alpha=0.10, color=color, zorder=0)

    # ── Rug plot (feature distribution ticks) ────────────────────────────────
    feat_vals = X_orig.iloc[:, feat_idx].values
    y_range   = avg_pred.max() - avg_pred.min()
    rug_y     = avg_pred.min() - y_range * 0.10
    ax.plot(feat_vals, np.full(len(feat_vals), rug_y),
            '|', color='#999999', alpha=0.35, markersize=3.5,
            markeredgewidth=0.55, zorder=2)

    # ── Mean reference line ───────────────────────────────────────────────────
    ax.axvline(feat_vals.mean(), color='#777777', linewidth=0.75,
               linestyle=':', alpha=0.65, zorder=3)

    # ── Axis formatting ───────────────────────────────────────────────────────
    display_name = disp(orig_feat)
    # 为x轴标题添加单位
    unit_map = {
        'Cu_in': 'g/L',
        'T$_A$': '°C',
        'I$_A$': 'A',
        'Q$_A$': 'm³/h',
        'V$_A$': 'V',
        'P$_A$': 'kWh',
        'T$_B$': '°C',
        'I$_B$': 'A',
        'Q$_B$': 'm³/h',
        'V$_B$': 'V',
        'P$_B$': 'kWh',
        'P$_{total}$': 'kWh',
        'Year': '',
        'Month': '',
        'Day': '',
        'Hour': ''
    }
    x_unit = unit_map.get(display_name, '')
    if x_unit:
        ax.set_xlabel(f'{display_name} ({x_unit})', fontsize=fontsize_label, fontweight='bold')
    else:
        ax.set_xlabel(display_name, fontsize=fontsize_label, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel(f'Partial Dep. ({unit})',
                      fontsize=fontsize_label - 0.5, fontweight='bold')

    # 取消子图标题
    # ax.set_title(display_name, fontsize=fontsize_title,
    #              fontweight='bold', color=color, pad=5)

    ax.tick_params(labelsize=fontsize_tick, direction='in',
                   which='both', length=3.5, width=0.8)
    ax.tick_params(axis='x', labelsize=fontsize_tick)
    ax.tick_params(axis='y', labelsize=fontsize_tick)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')

    ax.grid(linestyle='--', linewidth=0.40, alpha=0.45, color='#cccccc', zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

    # ── Panel label (a), (b) … ────────────────────────────────────────────────
    if panel_label is not None:
        ax.text(0.03, 0.97, panel_label, transform=ax.transAxes,
                fontsize=fontsize_title, fontweight='bold',
                va='top', ha='left', color='#222222',
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.75, pad=1.5))

    # ── Legend (average line only) ────────────────────────────────────────────
    leg = ax.legend(fontsize=fontsize_tick, loc='upper right',
                    framealpha=0.88, edgecolor='#cccccc',
                    handlelength=1.6, handletextpad=0.4,
                    borderpad=0.5, labelspacing=0.3)
    for ltext in leg.get_texts():
        ltext.set_fontweight('bold')

    return {
        'feature':     disp_plain(orig_feat),
        'feature_raw': orig_feat,
        'grid_values': grid_vals,
        'avg_pred':    avg_pred,
        'feat_mean':   feat_vals.mean(),
        'rug_values':  feat_vals,
    }


def plot_pdp_single(key, cfg, model, X_orig, top_n=6):
    """
    PDP 图（双输出模式）
    ①  6-panel 合并图  →  pdp_{key}.png          （原有逻辑，保留）
    ②  每个特征单独子图  →  pdp_individual/{key}/  （新增，ICE + average 风格）

    配色、字体、样式全面升级至 SCI 顶刊标准：
      - ICE 细线（浅色半透明）+ 粗 average 线
      - Times New Roman、4:3 比例
      - 带 panel 编号 (a)(b)… 的单独子图
      - rug 图、均值参考虚线
    """
    color  = PALETTE_TARGET[cfg['target_group']]
    n_feat = min(top_n, len(X_orig.columns))
    unit   = cfg['unit']

    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:n_feat]
    top_feats   = [X_orig.columns[i] for i in top_idx]

    # ── ① 6-panel 合并图 ────────────────────────────────────────────────────
    ncols = 3
    nrows = int(np.ceil(n_feat / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.0, nrows * 3.0),
                             constrained_layout=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    panel_letters = list('abcdefghijklmnopqrstuvwxyz')
    pdp_records = []

    for plot_i, orig_feat in enumerate(top_feats):
        ri, ci = divmod(plot_i, ncols)
        ax = axes[ri, ci]
        rec = _draw_pdp_ax(
            ax, model, X_orig, orig_feat, color, unit,
            ice_max_samples=80,
            show_ylabel=(ci == 0),
            fontsize_label=10.5, fontsize_tick=9.5, fontsize_title=11,
            linewidth_avg=2.0,
            panel_label=f'({panel_letters[plot_i]})',
        )
        if rec is not None:
            pdp_records.append(rec)

    for hide_i in range(n_feat, nrows * ncols):
        ri, ci = divmod(hide_i, ncols)
        axes[ri, ci].axis('off')

    # 取消大图标题
    # fig.suptitle(
    #     f'Partial Dependence Plots — {cfg["label"]} (Top-{n_feat} Features)',
    #     fontsize=11, fontweight='bold', y=1.02
    # )

    # 修改图片命名：three_stage → Condition1, four_stage → Condition2, serial → Condition3
    new_key = key.replace('three_stage', 'Condition1').replace('four_stage', 'Condition2').replace('serial', 'Condition3')
    out_combined = os.path.join(OUT_ROOT, 'pdp', f'pdp_{new_key}.png')
    fig.savefig(out_combined, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'    ✓ PDP combined {key}: {out_combined}')

    # ── ② 每个特征的独立单图 ────────────────────────────────────────────────
    # 修改目录命名：three_stage → Condition1, four_stage → Condition2, serial → Condition3
    new_key = key.replace('three_stage', 'Condition1').replace('four_stage', 'Condition2').replace('serial', 'Condition3')
    indiv_dir = os.path.join(OUT_ROOT, 'pdp', 'individual', new_key)
    os.makedirs(indiv_dir, exist_ok=True)

    for plot_i, orig_feat in enumerate(top_feats):
        # 单图尺寸：3.5 × 2.625 英寸 (4:3)，适合嵌入论文单栏
        fig_s, ax_s = plt.subplots(figsize=(3.5, 2.625))
        fig_s.subplots_adjust(left=0.16, right=0.95, bottom=0.18, top=0.88)

        _draw_pdp_ax(
            ax_s, model, X_orig, orig_feat, color, unit,
            ice_max_samples=100,
            show_ylabel=True,
            fontsize_label=10.5, fontsize_tick=9.5, fontsize_title=11,
            linewidth_avg=2.2,
            panel_label=f'({panel_letters[plot_i]})',
        )

        fname_safe = disp_plain(orig_feat).replace('$', '').replace('{', '').replace('}', '')
        # 修改文件名：three_stage → Condition1, four_stage → Condition2, serial → Condition3
        new_key = key.replace('three_stage', 'Condition1').replace('four_stage', 'Condition2').replace('serial', 'Condition3')
        out_s = os.path.join(indiv_dir, f'pdp_{new_key}_{fname_safe}.png')
        fig_s.savefig(out_s, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close(fig_s)
        print(f'      ✓ PDP individual [{disp(orig_feat)}]: {out_s}')

    # ── Excel：每个特征一个 sheet （基于合并图收集的 pdp_records）──
    if pdp_records:
        sheets = {}
        for rec in pdp_records:
            fname = rec['feature']
            # PDP 曲线（grid + avg_pred）
            df_curve = pd.DataFrame({
                f'{fname}_x':     rec['grid_values'],
                f'PartialDep_{unit}': rec['avg_pred'],
            })
            # Rug 分布（特征实际值散点，用于底部短竖线）
            df_rug = pd.DataFrame({f'{fname}_rug': rec['rug_values']})
            # 均值参考线
            df_mean = pd.DataFrame({f'{fname}_mean': [rec['feat_mean']]})

            # 合并到一张宽表（行数不等，用 NaN 填充）
            max_len = max(len(df_curve), len(df_rug))
            df_out = pd.DataFrame(index=range(max_len))
            # 处理曲线数据（grid + avg_pred）
            x_arr = np.full(max_len, np.nan)
            x_arr[:len(df_curve)] = df_curve[f'{fname}_x'].values
            df_out[f'{fname}_x'] = x_arr
            pred_arr = np.full(max_len, np.nan)
            pred_arr[:len(df_curve)] = df_curve[f'PartialDep_{unit}'].values
            df_out[f'PartialDep_{unit}'] = pred_arr
            # rug 值追加在右边（长度不同，截断或填 NaN）
            rug_arr = np.full(max_len, np.nan)
            rug_arr[:len(rec['rug_values'])] = rec['rug_values']
            df_out[f'{fname}_rug'] = rug_arr
            df_out[f'{fname}_mean'] = np.nan
            df_out.loc[0, f'{fname}_mean'] = rec['feat_mean']

            sheets[fname[:31]] = df_out

        # 修改Excel文件名：three_stage → Condition1, four_stage → Condition2, serial → Condition3
        new_key = key.replace('three_stage', 'Condition1').replace('four_stage', 'Condition2').replace('serial', 'Condition3')
        save_excel(
            sheets,
            os.path.join(EXCEL_ROOT, 'pdp', f'pdp_{new_key}.xlsx'),
            f'PDP single {new_key}'
        )


def plot_pdp_comparison(all_pdp_data, target_group, stage_keys_map):
    """同一预测目标下三工况 PDP 比较图（选共有重要特征）"""
    from collections import Counter
    color_map  = {'three_stage': '#2166AC', 'four_stage': '#D6604D', 'serial': '#1A9641'}
    stage_label = {'three_stage': 'Three-Stage', 'four_stage': 'Four-Stage', 'serial': 'Series'}
    tgt_label   = {'voltage': 'Voltage (V)', 'cu': 'Cu (g/L)', 'as': 'As (g/L)'}[target_group]

    common = []
    for stg, key in stage_keys_map.items():
        if key not in all_pdp_data: continue
        model, X_orig, cfg = all_pdp_data[key]
        top3 = [X_orig.columns[i]
                for i in np.argsort(model.feature_importances_)[::-1][:3]]
        common.extend(top3)
    cnt    = Counter(common)
    shared = [f for f, c in cnt.most_common() if c >= 2][:4]
    if not shared:
        shared = list(dict.fromkeys(common))[:3]

    nf = len(shared)
    if nf == 0:
        print(f'  ✗ PDP comparison {target_group}: no shared features')
        return

    stage_list = ['three_stage', 'four_stage', 'serial']
    fig, axes = plt.subplots(len(stage_list), nf,
                             figsize=(nf * 4.0, len(stage_list) * 3.0),
                             constrained_layout=True)
    if len(stage_list) == 1: axes = axes[np.newaxis, :]
    if nf == 1:              axes = axes[:, np.newaxis]

    # Excel 数据：sheet = '{stage}_{feature}'
    excel_sheets = {}

    for si, stg in enumerate(stage_list):
        key = stage_keys_map.get(stg)
        sc  = color_map[stg]
        for fi, orig_feat in enumerate(shared):
            ax = axes[si, fi]
            if key not in all_pdp_data:
                ax.axis('off'); continue
            model, X_orig, cfg = all_pdp_data[key]
            if orig_feat not in X_orig.columns:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9)
                ax.axis('off'); continue

            feat_idx = list(X_orig.columns).index(orig_feat)
            try:
                pdp_r = partial_dependence(model, X_orig, [feat_idx],
                                           kind='average', grid_resolution=60)
                gv = pdp_r['grid_values'][0]
                av = pdp_r['average'][0]
            except Exception as e:
                ax.text(0.5, 0.5, f'err\n{str(e)[:40]}',
                        ha='center', va='center', transform=ax.transAxes, fontsize=6)
                continue

            ax.fill_between(gv, av.min(), av, alpha=0.18, color=sc)
            ax.plot(gv, av, color=sc, linewidth=2.0)
            ax.set_xlabel(disp(orig_feat), fontsize=10.5, fontweight='bold')
            ax.set_ylabel(tgt_label if fi == 0 else '', fontsize=10.5, fontweight='bold')
            ax.tick_params(labelsize=9.5)
            ax.grid(linestyle='--', linewidth=0.4, alpha=0.45, color='#cccccc')
            # 取消子图标题
            # if si == 0:
            #     ax.set_title(disp(orig_feat), fontsize=11, fontweight='bold', pad=5)

        axes[si, 0].annotate(
            stage_label[stg],
            xy=(-0.38, 0.5), xycoords='axes fraction',
            ha='right', va='center', fontsize=9,
            fontweight='bold', color=sc, rotation=90
        )

        # 收集该工况所有共有特征的 PDP 数据
        if key in all_pdp_data:
            model, X_orig, cfg = all_pdp_data[key]
            for orig_feat in shared:
                if orig_feat not in X_orig.columns:
                    continue
                feat_idx = list(X_orig.columns).index(orig_feat)
                try:
                    pdp_r = partial_dependence(model, X_orig, [feat_idx],
                                               kind='average', grid_resolution=60)
                    gv = pdp_r['grid_values'][0]
                    av = pdp_r['average'][0]
                    fname = disp_plain(orig_feat)
                    sheet_name = f'{stg[:5]}_{fname}'[:31]
                    excel_sheets[sheet_name] = pd.DataFrame({
                        f'{fname}_x':           gv,
                        f'PartialDep_{tgt_label.split()[0]}': av,
                        'Stage':                stage_label[stg],
                        'Feature':              fname,
                    })
                except Exception:
                    pass

    # 取消大图标题
    # fig.suptitle(f'PDP Comparison Across Operation Modes — {tgt_label}',
    #              fontsize=11, fontweight='bold')
    out = os.path.join(OUT_ROOT, 'pdp', f'pdp_comparison_{target_group}.png')
    fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  ✓ PDP comparison ({target_group}): {out}')

    if excel_sheets:
        save_excel(
            excel_sheets,
            os.path.join(EXCEL_ROOT, 'pdp', f'pdp_comparison_{target_group}.xlsx'),
            f'PDP comparison {target_group}'
        )


# ══════════════════════════════════════════════════════════════════════════════
# 8.  主流程（与原版完全相同）
# ══════════════════════════════════════════════════════════════════════════════
def main():
    np.random.seed(42)
    ensure_dirs()

    print('\n' + '═' * 70)
    print('  Model Interpretability Analysis  —  SCI Journal Style  v2')
    print('═' * 70)

    results_dict  = {}
    all_shap_data = {}
    all_pdp_data  = {}

    for key, cfg in MODEL_CONFIGS.items():
        print(f'\n[{key}] Loading...')
        if not os.path.exists(cfg['model_path']):
            print(f'  ✗ Model not found: {cfg["model_path"]}')
            continue
        try:
            model, X_orig, y = load_data_and_model(key, cfg)
        except Exception as e:
            print(f'  ✗ Load error: {e}')
            continue

        results_dict[key] = {
            'importances':       model.feature_importances_,
            'original_features': list(X_orig.columns),
            'label':             cfg['label'],
            'target_group':      cfg['target_group'],
            'stage_group':       cfg['stage_group'],
        }
        all_pdp_data[key] = (model, X_orig, cfg)

        print(f'  → Computing SHAP...')
        try:
            sv, X_s = compute_shap(model, X_orig)
            all_shap_data[key] = (sv, X_s, cfg)
        except Exception as e:
            print(f'  ✗ SHAP error: {e}')

    # ── 1. Feature Importance ──
    print('\n' + '─' * 70)
    print('  1. Feature Importance Plots')
    print('─' * 70)
    if results_dict:
        plot_feature_importance_all(results_dict)
        export_importance_excel(results_dict)

    # ── 2. SHAP ──
    print('\n' + '─' * 70)
    print('  2. SHAP Plots')
    print('─' * 70)
    for key, (sv, X_s, cfg) in all_shap_data.items():
        plot_shap_single(key, cfg, sv, X_s)
    if all_shap_data:
        plot_shap_all_bar(all_shap_data)
        plot_shap_all_beeswarm(all_shap_data)

    # ── 3. PDP ──
    print('\n' + '─' * 70)
    print('  3. PDP Plots')
    print('─' * 70)
    for key, (model, X_orig, cfg) in all_pdp_data.items():
        plot_pdp_single(key, cfg, model, X_orig, top_n=6)

    for tgt in ['voltage', 'cu', 'as']:
        plot_pdp_comparison(
            all_pdp_data, tgt,
            {stg: f'{tgt}_{stg}' for stg in ['three_stage', 'four_stage', 'serial']}
        )

    print('\n' + '═' * 70)
    print(f'  All plot outputs saved to  : {OUT_ROOT}/')
    print(f'  All Excel outputs saved to : {EXCEL_ROOT}/')
    print('═' * 70)


if __name__ == '__main__':
    main()
