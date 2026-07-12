"""A/B 版本对比引擎：新旧版本画质指标横向对比"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def compare_versions(
    csv_a: str,
    csv_b: str,
    group_col: str = "group",
    metric_cols: Optional[List[str]] = None,
) -> Dict:
    """对比新旧两个版本的结果 CSV。

    Args:
        csv_a: 旧版本指标 CSV 路径
        csv_b: 新版本指标 CSV 路径
        group_col: 分组列名（场景类型）
        metric_cols: 要对比的指标列名，None = 自动推断

    Returns:
        {
            "summary": pd.DataFrame,   # 分组均值对比
            "deltas": pd.DataFrame,     # 每张图的 Δ 值 (B - A)
            "wins": dict                # {"metric": {"a_wins": N, "b_wins": N}}
        }
    """
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)

    if metric_cols is None:
        # 自动推断数值列
        numeric_cols = df_a.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {"defect_count"}
        metric_cols = [c for c in numeric_cols if c not in exclude and not c.startswith("exif_")]

    # 按文件名对齐
    merged = df_a[["file", group_col] + metric_cols].merge(
        df_b[["file"] + metric_cols],
        on="file",
        suffixes=("_a", "_b"),
        how="inner",
    )

    # 逐指标统计胜出次数（越高越好需要人工指定方向）
    deltas = merged.copy()
    for col in metric_cols:
        deltas[f"Δ_{col}"] = merged[f"{col}_b"] - merged[f"{col}_a"]

    # 分组均值对比
    summary_a = df_a.groupby(group_col)[metric_cols].mean()
    summary_b = df_b.groupby(group_col)[metric_cols].mean()
    summary = summary_a.join(summary_b, lsuffix="_old", rsuffix="_new")

    return {
        "summary": summary,
        "deltas": deltas,
        "metric_cols": metric_cols,
    }


def format_comparison_report(comp_result: Dict) -> str:
    """生成对比摘要文本。"""
    lines = ["=" * 60, "A/B 版本对比结果", "=" * 60]
    summary = comp_result["summary"]
    lines.append(f"\n旧版均值 vs 新版均值（按 {summary.index.name} 分组）:\n")
    lines.append(summary.to_string())
    deltas = comp_result["deltas"]
    delta_cols = [c for c in deltas.columns if c.startswith("Δ_")]
    if delta_cols:
        lines.append("\n\n指标变化方向:\n")
        for dc in delta_cols:
            mean_delta = deltas[dc].mean()
            direction = "↑ 提升" if mean_delta > 0 else "↓ 下降" if mean_delta < 0 else "→ 持平"
            lines.append(f"  {dc}: {mean_delta:+.4f}  {direction}")
    return "\n".join(lines)
