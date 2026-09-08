"""
铜电解液净化工序数据处理脚本（按工况优化版+参数异常计数）
功能：
1. 精准识别工况（三段单独/四段单独/三四段串联）
2. 按工况分组执行Hampel滤波异常值剔除（减少过度剔除）
3. 按工况分组执行KNN填充缺失值（避免跨工况干扰）
4. 修复中文显示问题（跨平台兼容）
5. 生成处理前后对比图 + 详细统计报告
6. 导出处理后的Excel文件
7. 新增：显示每个参数的区间异常值计数
业务逻辑：
- 入液：一系统脱铜后液、二系统脱铜后液、沉碲后液 → 汇合为电积前液
- 电积：三段单独/四段单独/三四段串联 → 废铜液原液罐
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import KNNImputer
import warnings
import platform
import matplotlib as mpl
warnings.filterwarnings('ignore')
# ============================================================================
# 1. 跨平台中文显示修复（核心优化）
# ============================================================================
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
sns.set_style("whitegrid")
# 按系统适配中文字体
system = platform.system()
if system == 'Windows':
    mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
elif system == 'Linux':
    mpl.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
elif system == 'Darwin':  # macOS
    mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'DejaVu Sans']
# 强制生效
mpl.rc('font', family=mpl.rcParams['font.sans-serif'][0])
# ============================================================================
# 2. 读取数据 & 预处理
# ============================================================================
file_path = 'Original_data.xlsx'
df = pd.read_excel(file_path)
print("="*70)
print("原始数据信息:")
print(f"行数: {len(df)}, 列数: {len(df.columns)}")
print(f"初始缺失值总数: {df.isnull().sum().sum()}")
print("="*70)
# 数据类型转换：object转float，非数字值转NaN
columns_to_convert = ['三段溶液温度℃', '三段电流强度A', '四段溶液温度℃', '四段电流强度A']
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# 所有非时间列转为数值型
for col in df.columns:
    if col != '日期时间':
        df[col] = pd.to_numeric(df[col], errors='coerce')
df_original = df.copy()  # 保存原始数据用于对比
print("\n数据类型转换完成")
print(f"转换后缺失值总数: {df.isnull().sum().sum()}")
# ============================================================================
# 3. 精准识别工况（核心优化）
# ============================================================================
def identify_operation_mode(row):
    """
    精准识别电积工况：
    - three_stage：仅三段开启（三段参数有效，四段无效）
    - four_stage：仅四段开启（四段参数有效，三段无效）
    - serial：三四段串联（三段+四段参数均有效）
    - unknown：未知工况（无有效参数）
    有效参数定义：非NaN + 非0
    """
    # 三段核心参数
    three_params = [row['三段溶液温度℃'], row['三段电流强度A'], row['三段电压V'], row['三段流量m3/h']]
    three_valid = sum(1 for x in three_params if pd.notna(x) and x != 0) > 0
    
    # 四段核心参数
    four_params = [row['四段溶液温度℃'], row['四段电流强度A'], row['四段电压V'], row['四段流量m3/h']]
    four_valid = sum(1 for x in four_params if pd.notna(x) and x != 0) > 0
    
    if three_valid and not four_valid:
        return 'three_stage'
    elif not three_valid and four_valid:
        return 'four_stage'
    elif three_valid and four_valid:
        return 'serial'
    else:
        return 'unknown'

# 先识别工况（用于后续区间适配）
df['operation_mode'] = df.apply(identify_operation_mode, axis=1)

# 删除未知工况的行
unknown_count = (df['operation_mode'] == 'unknown').sum()
if unknown_count > 0:
    print(f"删除 {unknown_count} 行未知工况数据")
    df = df[df['operation_mode'] != 'unknown'].copy()
print(f"保留有效工况数据行数: {len(df)}")
# ============================================================================
# 4. 参数区间异常值识别与剔除（优化版：工况差异化+参数异常计数）
# ============================================================================
# 定义预测目标
PREDICTION_TARGET = '废铜液原液罐Cu（g/L）'

# 优化：按工况差异化设置参数合理区间
mode_parameter_ranges = {
    'three_stage': {
        '电积前液Cu（g/L）': (10, 60),
        '三段溶液温度℃': (38, 68),
        '三段电流强度A': (8000, 32000),
        '三段电压V': (28, 42),
        '三段流量m3/h': (105, 125),
        '一系统脱铜后液体积m3': (0, 100),
        '一系统脱铜后液Cu（g/L）': (20, 55),
        '二系统脱铜后液体积m3': (0, 100),
        '二系统脱铜后液Cu（g/L）': (25, 52),
        '沉碲后液体积m3': (0, 50),
        '沉碲后液Cu（g/L）': (0, 75)
    },
    'four_stage': {
        '电积前液Cu（g/L）': (10, 60),
        '四段溶液温度℃': (38, 68),
        '四段电流强度A': (8000, 32000),
        '四段电压V': (28, 42),
        '四段流量m3/h': (105, 125),
        '一系统脱铜后液体积m3': (0, 100),
        '一系统脱铜后液Cu（g/L）': (20, 55),
        '二系统脱铜后液体积m3': (0, 100),
        '二系统脱铜后液Cu（g/L）': (25, 52),
        '沉碲后液体积m3': (0, 50),
        '沉碲后液Cu（g/L）': (0, 75)
    },
    'serial': {
        '电积前液Cu（g/L）': (15, 58),
        '三段溶液温度℃': (40, 65),
        '三段电流强度A': (10000, 30000),
        '三段电压V': (30, 40),
        '三段流量m3/h': (108, 122),
        '四段溶液温度℃': (40, 65),
        '四段电流强度A': (10000, 30000),
        '四段电压V': (30, 40),
        '四段流量m3/h': (108, 122),
        '一系统脱铜后液体积m3': (0, 100),
        '一系统脱铜后液Cu（g/L）': (22, 53),
        '二系统脱铜后液体积m3': (0, 100),
        '二系统脱铜后液Cu（g/L）': (28, 50),
        '沉碲后液体积m3': (0, 45),
        '沉碲后液Cu（g/L）': (0, 72)
    }
}

# 新增：统计每个参数的区间异常次数（按工况分组）
param_outlier_count = {
    'three_stage': {},
    'four_stage': {},
    'serial': {},
    'unknown': {}
}
# 初始化所有参数的异常计数为0
for mode in param_outlier_count.keys():
    # 按工况获取对应参数列表
    if mode in mode_parameter_ranges:
        params = mode_parameter_ranges[mode].keys()
    else:
        params = mode_parameter_ranges['serial'].keys()  # 未知工况用串联参数列表
    for param in params:
        param_outlier_count[mode][param] = 0

# 识别区间异常值（含参数异常计数）
print("\n开始参数区间异常值识别（优化版+参数计数）...")
out_of_range_count = 0
param_outlier_positions = {}

for idx, row in df.iterrows():
    mode = row['operation_mode']
    # 获取当前工况的参数区间
    current_ranges = mode_parameter_ranges.get(mode, mode_parameter_ranges['serial'])
    row_has_outlier = False
    
    for col, (min_val, max_val) in current_ranges.items():
        if col != PREDICTION_TARGET and pd.notna(row[col]):
            # 特殊规则：沉碲后液体积为0时，Cu浓度=0 不算异常
            if col == '沉碲后液Cu（g/L）' and row['沉碲后液体积m3'] == 0 and row[col] == 0:
                continue
            # 判定是否超出区间
            if not (min_val <= row[col] <= max_val):
                # 累加该参数的异常次数
                param_outlier_count[mode][col] += 1
                row_has_outlier = True
                # 将异常值设为NaN
                df.loc[idx, col] = np.nan
    
    if row_has_outlier:
        out_of_range_count += 1

print(f"识别到 {out_of_range_count} 行参数区间异常值（优化版）")
print(f"当前行数: {len(df)}")
print("异常值已设为空白值，等待后续KNN填充")

# 打印每个参数的区间异常值计数
print("\n各工况下每个参数的区间异常值统计:")
mode_cn_map = {'three_stage': '仅三段', 'four_stage': '仅四段', 'serial': '串联', 'unknown': '未知工况'}
for mode, param_counts in param_outlier_count.items():
    print(f"\n  {mode_cn_map[mode]}:")
    for param, count in param_counts.items():
        if count > 0:  # 只打印有异常的参数
            print(f"    {param}: {count} 个异常值")
        else:
            print(f"    {param}: 0 个异常值")

# 更新工况分布统计
mode_counts = df['operation_mode'].value_counts()
print("\n工况分布统计（剔除区间异常后）:")
for mode, count in mode_counts.items():
    mode_cn = mode_cn_map.get(mode, mode)
    print(f"  {mode_cn}: {count} 行 ({count/len(df)*100:.1f}%)")
# ============================================================================
# 5. 按工况分组的Hampel滤波（核心优化：减少过度剔除+按工况处理）
# ============================================================================
def hampel_filter_optimized(data, window_size=5, n_sigma=3, max_outlier_ratio=0.05):
    """
    优化版Hampel滤波器：
    - 缩小窗口减少过度剔除
    - 限制异常值比例上限
    - 避免MAD为0时的误判
    """
    if len(data) < window_size or len(data.dropna()) < 5:
        return data
    
    result = data.copy()
    half_window = window_size // 2
    total_valid = data.notna().sum()
    outlier_count = 0
    
    for i in range(half_window, len(data) - half_window):
        # 异常值比例超限则停止
        if outlier_count / total_valid > max_outlier_ratio:
            break
        
        window_data = data.iloc[i-half_window:i+half_window+1].dropna()
        if len(window_data) < 3:
            continue
        
        median = window_data.median()
        mad = np.median(np.abs(window_data - median))
        
        if mad == 0:  # 无波动时跳过
            continue
        
        threshold = n_sigma * 1.4826 * mad
        if abs(data.iloc[i] - median) > threshold and not pd.isna(data.iloc[i]):
            result.iloc[i] = np.nan
            outlier_count += 1
    
    return result

# 定义各工况需要处理的列（贴合业务逻辑）
mode_columns_map = {
    'three_stage': [
        '电积前液Cu（g/L）', '三段溶液温度℃', '三段电流强度A', '三段电压V', '三段流量m3/h',
        '一系统脱铜后液体积m3', '一系统脱铜后液Cu（g/L）', '二系统脱铜后液体积m3', 
        '二系统脱铜后液Cu（g/L）', '沉碲后液体积m3', '沉碲后液Cu（g/L）', '废铜液原液罐Cu（g/L）'
    ],
    'four_stage': [
        '电积前液Cu（g/L）', '四段溶液温度℃', '四段电流强度A', '四段电压V', '四段流量m3/h',
        '一系统脱铜后液体积m3', '一系统脱铜后液Cu（g/L）', '二系统脱铜后液体积m3', 
        '二系统脱铜后液Cu（g/L）', '沉碲后液体积m3', '沉碲后液Cu（g/L）', '废铜液原液罐Cu（g/L）'
    ],
    'serial': [
        '电积前液Cu（g/L）', '三段溶液温度℃', '三段电流强度A', '三段电压V', '三段流量m3/h',
        '四段溶液温度℃', '四段电流强度A', '四段电压V', '四段流量m3/h',
        '一系统脱铜后液体积m3', '一系统脱铜后液Cu（g/L）', '二系统脱铜后液体积m3', 
        '二系统脱铜后液Cu（g/L）', '沉碲后液体积m3', '沉碲后液Cu（g/L）', '废铜液原液罐Cu（g/L）'
    ]
}

print("\n开始按工况执行Hampel滤波异常值检测...")
outlier_counts = {mode: {} for mode in mode_columns_map.keys()}
# 按工况分组处理异常值
for mode, columns in mode_columns_map.items():
    mode_mask = df['operation_mode'] == mode
    mode_df = df[mode_mask].copy()
    
    if len(mode_df) < 10:
        print(f"  {mode}工况样本数不足，跳过异常值处理")
        continue
    
    for col in columns:
        valid_mask = (mode_df[col].notna()) & (mode_df[col] != 0)
        if valid_mask.sum() < 5:
            continue
        
        valid_data = mode_df[col][valid_mask].copy()
        if len(valid_data) == 0:
            continue
        
        filtered_data = hampel_filter_optimized(valid_data, window_size=7, n_sigma=3, max_outlier_ratio=0.05)
        outliers = valid_data[filtered_data.isna()].index
        
        outlier_counts[mode][col] = len(outliers)
        
        for idx in outliers:
            df.loc[idx, col] = np.nan

# 打印Hampel滤波异常值统计
print("\nHampel滤波异常值检测完成:")
for mode, col_counts in outlier_counts.items():
    if not col_counts:
        continue
    print(f"  {mode_cn_map.get(mode, mode)}工况:")
    for col, count in col_counts.items():
        if count > 0:
            print(f"    {col}: {count} 个异常值")

# ============================================================================
# 6. 按工况分组的KNN填充（核心优化）
# ============================================================================
print("\n开始按工况执行KNN填充缺失值...")
fill_feature_map = {
    'three_stage': mode_columns_map['three_stage'],
    'four_stage': mode_columns_map['four_stage'],
    'serial': mode_columns_map['serial']
}

df_imputed = df.copy()
for mode, feature_cols in fill_feature_map.items():
    mode_mask = df['operation_mode'] == mode
    mode_df = df[mode_mask].copy()
    
    if len(mode_df) < 5:
        print(f"  {mode_cn_map.get(mode, mode)}工况样本数不足，跳过填充")
        continue
    
    data_for_knn = mode_df[feature_cols].copy()
    if data_for_knn.isnull().sum().sum() == 0:
        continue
    
    # 检查废铜液原液罐Cu（g/L）是否有足够的非缺失值
    if '废铜液原液罐Cu（g/L）' in feature_cols:
        cu_col = '废铜液原液罐Cu（g/L）'
        non_missing_count = data_for_knn[cu_col].notna().sum()
        print(f"  {mode_cn_map.get(mode, mode)}工况中{cu_col}的非缺失值数量: {non_missing_count}")
    
    knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
    imputed_data = knn_imputer.fit_transform(data_for_knn)
    imputed_df = pd.DataFrame(imputed_data, columns=feature_cols, index=mode_df.index)
    
    for col in feature_cols:
        df_imputed.loc[mode_df.index, col] = imputed_df[col]
    
    # 检查废铜液原液罐Cu（g/L）的填充情况
    if '废铜液原液罐Cu（g/L）' in feature_cols:
        cu_col = '废铜液原液罐Cu（g/L）'
        missing_before = mode_df[cu_col].isnull().sum()
        missing_after = df_imputed.loc[mode_df.index, cu_col].isnull().sum()
        filled_count = missing_before - missing_after
        print(f"  {mode_cn_map.get(mode, mode)}工况中{cu_col}填充了 {filled_count} 个缺失值")
    
    # 业务规则恢复：沉碲后液体积为0时，Cu浓度强制为0
    zero_vol_mask = (mode_df['沉碲后液体积m3'] == 0) & (mode_df['沉碲后液体积m3'].notna())
    if '沉碲后液Cu（g/L）' in feature_cols:
        zero_vol_indices = mode_df[zero_vol_mask].index
        df_imputed.loc[zero_vol_indices, '沉碲后液Cu（g/L）'] = 0.0

# 按用户要求：对指定变量进行整数化处理
integer_vars = [
    '三段溶液温度℃', '三段电流强度A', '三段电压V', '三段流量m3/h',
    '四段溶液温度℃', '四段电流强度A', '四段电压V', '四段流量m3/h',
    '一系统脱铜后液体积m3', '二系统脱铜后液体积m3', '沉碲后液体积m3'
]

for var in integer_vars:
    if var in df_imputed.columns:
        df_imputed[var] = df_imputed[var].round().astype('Int64')

# 按用户要求：电流值需为整百数
current_vars = ['三段电流强度A', '四段电流强度A']
for var in current_vars:
    if var in df_imputed.columns:
        df_imputed[var] = (df_imputed[var] // 100 * 100).astype('Int64')

df = df_imputed.copy()
print("KNN填充完成")
print(f"处理后缺失值总数: {df.drop('operation_mode', axis=1).isnull().sum().sum()}")
# ============================================================================
# 7. 导出处理后的数据
# ============================================================================
df_final = df.drop('operation_mode', axis=1)
import os
# 使用outputs文件夹作为输出目录
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
output_file = f'{output_dir}/data_processed.xlsx'
df_final.to_excel(output_file, index=False, engine='openpyxl')
print(f"\n✓ 处理后的数据已导出到: {output_file}")
# ============================================================================
# 8. 生成处理前后对比图
# ============================================================================
print("\n生成处理前后对比图...")
color_before = '#1f77b4'
color_after = '#ff7f0e'
color_before_scatter = '#4C72B0'
color_after_scatter = '#DD8452'
x_indices = np.arange(len(df))
df_original = df_original.loc[df.index].copy()

figure_groups = {
    '三段参数': ['三段溶液温度℃', '三段电流强度A', '三段电压V', '三段流量m3/h'],
    '四段参数': ['四段溶液温度℃', '四段电流强度A', '四段电压V', '四段流量m3/h'],
    '入液参数': ['电积前液Cu（g/L）', '一系统脱铜后液Cu（g/L）', '二系统脱铜后液Cu（g/L）', '沉碲后液Cu（g/L）'],
    '最终参数': ['废铜液原液罐Cu（g/L）']
}

def create_comparison_figure(group_name, columns, figsize=(15, 10)):
    n_cols = 2 if len(columns) > 1 else 1
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = [axes] if (n_rows == 1 and n_cols == 1) else axes.flatten()
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        before_data = df_original[col]
        after_data = df_final[col]
        
        if '沉碲后液' in col:
            before_data_plot = before_data.copy()
            before_data_plot[before_data_plot == 0] = np.nan
            after_data_plot = after_data.copy()
            after_data_plot[after_data_plot == 0] = np.nan
            
            ax.plot(x_indices, before_data_plot, color=color_before, linewidth=1.5, alpha=0.7, label='处理前')
            ax.plot(x_indices, after_data_plot, color=color_after, linewidth=1.5, alpha=0.7, label='处理后')
        else:
            ax.plot(x_indices, before_data, color=color_before, linewidth=1.5, alpha=0.7, label='处理前')
            ax.plot(x_indices, after_data, color=color_after, linewidth=1.5, alpha=0.7, label='处理后')
        
        if '沉碲后液' in col:
            mask_before = (before_data.notna()) & (before_data != 0)
            mask_after = (after_data.notna()) & (after_data != 0)
        else:
            mask_before = before_data.notna()
            mask_after = after_data.notna()
        
        ax.scatter(x_indices[mask_before], before_data[mask_before], 
                  color=color_before_scatter, s=30, alpha=0.6, edgecolors='none')
        ax.scatter(x_indices[mask_after], after_data[mask_after],
                  color=color_after_scatter, s=30, alpha=0.6, edgecolors='none')
        
        ax.set_ylabel('数值', fontsize=12, fontweight='bold')
        ax.set_xlabel('样本序号', fontsize=11)
        ax.set_title(col, fontsize=13, fontweight='bold', pad=12)
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.6)
        ax.set_axisbelow(True)
        
        before_valid = before_data.dropna()
        after_valid = after_data.dropna()
        stats_text = (f'处理前: n={len(before_valid)}\n'
                      f'处理后: n={len(after_valid)}\n'
                      f'Δn={len(after_valid)-len(before_valid)}')
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7))
        
        ax.set_xlim(-5, len(df) + 5)
        ax.set_xticks([0, len(df)//2, len(df)-1])
        ax.set_xticklabels(['0', str(len(df)//2), str(len(df)-1)], fontsize=10)
        ax.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True)
    
    for idx in range(len(columns), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    output_figure = f'{output_dir}/data_comparison_{group_name}.png'
    plt.savefig(output_figure, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ {group_name}对比图已保存到: {output_figure}")
    plt.close()

for group_name, columns in figure_groups.items():
    figsize = (12, 8) if len(columns) == 1 else (16, 12)
    create_comparison_figure(group_name, columns, figsize=figsize)
# ============================================================================
# 9. 生成详细统计报告
# ============================================================================
print("\n" + "="*70)
print("数据处理统计报告")
print("="*70)

plot_cols = [col for col in df_final.columns if col != '日期时间']
comparison_stats = []
for col in plot_cols:
    before = df_original[col].dropna()
    after = df_final[col].dropna()
    
    comparison_stats.append({
        '列名': col,
        '处理前_缺失值': df_original[col].isnull().sum(),
        '处理后_缺失值': df_final[col].isnull().sum(),
        '缺失值减少': df_original[col].isnull().sum() - df_final[col].isnull().sum(),
        '处理前_样本数': len(before),
        '处理后_样本数': len(after),
        '样本增加': len(after) - len(before),
        '处理前_均值': before.mean() if len(before) > 0 else np.nan,
        '处理后_均值': after.mean() if len(after) > 0 else np.nan,
        '处理前_标准差': before.std() if len(before) > 0 else np.nan,
        '处理后_标准差': after.std() if len(after) > 0 else np.nan,
        '处理前_最小值': before.min() if len(before) > 0 else np.nan,
        '处理前_最大值': before.max() if len(before) > 0 else np.nan,
        '处理后_最小值': after.min() if len(after) > 0 else np.nan,
        '处理后_最大值': after.max() if len(after) > 0 else np.nan,
    })

stats_df = pd.DataFrame(comparison_stats)
stats_output = f'{output_dir}/data_processing_statistics.xlsx'
stats_df.to_excel(stats_output, index=False, engine='openpyxl')
print(f"\n✓ 统计报告已保存到: {stats_output}")

print("\n" + "-"*70)
print("处理摘要统计:")
print("-"*70)
print(f"总行数: {len(df_final)}")
print(f"总列数: {len(plot_cols)}")
print(f"处理前总缺失值: {df_original[plot_cols].isnull().sum().sum()}")
print(f"处理后总缺失值: {df_final[plot_cols].isnull().sum().sum()}")
total_reduce = df_original[plot_cols].isnull().sum().sum() - df_final[plot_cols].isnull().sum().sum()
fill_rate = 100 * total_reduce / df_original[plot_cols].isnull().sum().sum() if df_original[plot_cols].isnull().sum().sum() > 0 else 0
print(f"总缺失值减少: {total_reduce}")
print(f"缺失值填充率: {fill_rate:.1f}%")
print("\n" + "="*70)
print(f"✓ 处理完成！所有文件已保存到{output_dir}目录")
print("="*70)