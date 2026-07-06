# -*- coding: utf-8 -*-
"""
STEP 10 — Visualize TFP Trend and YoY Fluctuations
读取 FE Solow 和 DEA 两种方法计算的 TFP 数据，绘制：
  1. 上半部分：累积 TFP 增长折线图 (1981 = 0)
  2. 下半部分：逐年 TFP 波动（Year-on-Year Change）的并排柱状图

运行方式: python 10_plot_tfp_fluctuations.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 尝试导入通用设置，如果失败则使用默认字体配置
try:
    import _common as C
    C.set_cjk_font(plt)
except ImportError:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'Arial Unicode MS'] # 适配中文字体
    plt.rcParams['axes.unicode_minus'] = False

def main():
    # 1. 加载数据 (假设 CSV 文件与脚本在同级或指定的清理目录下)
    # 请根据你的实际路径调整文件读取位置
    dir_path = getattr(C, 'CLEAN_DIR', '.') if 'C' in globals() else '.'
    
    try:
        df_fe = pd.read_csv(os.path.join(dir_path, "tfp_fe_solow_4in.csv"))
        df_dea = pd.read_csv(os.path.join(dir_path, "tfp_dea_nirs_4in.csv"))
    except FileNotFoundError:
        print("未找到 CSV 文件，请确保 tfp_fe_solow_4in.csv 和 tfp_dea_nirs_4in.csv 存在。")
        return

    # 重命名列以便合并
    df_fe = df_fe.rename(columns={'lntfp': 'FE_Solow'})
    df_dea = df_dea.rename(columns={'lntfp': 'DEA_NIRS'})
    
    # 合并数据集
    df = pd.merge(df_fe, df_dea, on='year', how='inner')
    df = df.sort_values('year').reset_index(drop=True)
    
    # 转换为百分比形式 (相对 1981 的累积增长 %)
    df['FE_Solow_pct'] = df['FE_Solow'] * 100
    df['DEA_NIRS_pct'] = df['DEA_NIRS'] * 100
    
    # 计算逐年增长率 (Year-on-Year Difference)
    df['FE_YoY'] = df['FE_Solow_pct'].diff()
    df['DEA_YoY'] = df['DEA_NIRS_pct'].diff()

    # 2. 开始绘图 (双子图布局)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1.2, 1]})
    
    years = df['year'].values
    x = np.arange(len(years))  # x轴刻度位置
    
    # ================== 图 1：累积 TFP 趋势 (折线图) ==================
    ax1.plot(x, df['FE_Solow_pct'], color='#2ca02c', marker='o', lw=2.5, markersize=6, label='FE Solow Residual')
    ax1.plot(x, df['DEA_NIRS_pct'], color='#1f77b4', marker='s', lw=2.5, markersize=6, label='DEA NIRS Sequential')
    
    ax1.axhline(0, color='black', lw=1, ls='--')
    ax1.set_title('A. 中国农业全要素生产率 (TFP) 累积增长趋势 (1981=0)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('累积 TFP 增长 (%)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(years, rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(loc='upper left', fontsize=11)
    
    # 添加关键历史节点的文本标注 (以 FE 数据为例)
    if 1984 in years:
        idx_84 = np.where(years == 1984)[0][0]
        ax1.annotate('1984 联产承包红利顶峰', xy=(idx_84, df.loc[idx_84, 'FE_Solow_pct']), 
                     xytext=(idx_84-2, df.loc[idx_84, 'FE_Solow_pct']+10),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
    if 1989 in years:
        idx_89 = np.where(years == 1989)[0][0]
        ax1.annotate('1989 物价闯关与宏观冲击', xy=(idx_89, df.loc[idx_89, 'FE_Solow_pct']), 
                     xytext=(idx_89-1, df.loc[idx_89, 'FE_Solow_pct']-15),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6))

    # ================== 图 2：逐年 TFP 波动 (并排柱状图) ==================
    width = 0.35  # 柱子宽度
    
    # 绘制正负交替的柱状图
    bars1 = ax2.bar(x - width/2, df['FE_YoY'], width, label='FE Solow (YoY)', color='#2ca02c', alpha=0.8)
    bars2 = ax2.bar(x + width/2, df['DEA_YoY'], width, label='DEA NIRS (YoY)', color='#1f77b4', alpha=0.8)
    
    ax2.axhline(0, color='black', lw=1)
    ax2.set_title('B. 逐年 TFP 波动幅度 (Year-on-Year 增长率)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('YoY 增长率 (%)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(years, rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(loc='upper right', fontsize=11)
    
    # 设置波动预警区间 (学术界常见的合理农业波动区间为 ±5% 到 ±8%)
    ax2.axhspan(-8, 8, color='gray', alpha=0.1, label='Normal Agricultural Volatility (±8%)')
    
    plt.tight_layout()
    
    # 3. 保存图像
    out_path = os.path.join(getattr(C, 'FIG_DIR', '.'), "fig_tfp_fluctuations_yoy.png") if 'C' in globals() else "fig_tfp_fluctuations_yoy.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"绘图完成！图表已保存至: {out_path}")
    plt.close()

if __name__ == "__main__":
    main()