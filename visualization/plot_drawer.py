"""绘图模块：雷达图、折线图、散点图、柱状图"""

from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def draw_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    save_path: str,
) -> None:
    """分组折线图。"""
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=x_col, y=y_col, hue=group_col, marker="o")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def draw_radar(
    categories: List[str],
    values_dict: Dict[str, List[float]],
    title: str,
    save_path: str,
    max_radius: float = 100.0,
) -> None:
    """雷达对比图。"""
    n = len(categories)
    angles = [a / n * 2 * np.pi for a in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for label, values in values_dict.items():
        v = values + values[:1]
        ax.plot(angles, v, "o-", label=label, linewidth=2)
        ax.fill(angles, v, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, max_radius)
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def draw_exif_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    save_path: str,
    title: str = "",
) -> None:
    """EXIF 参数 vs 画质指标散点图。"""
    plt.figure(figsize=(8, 5))
    groups = df["group"].unique() if "group" in df.columns else ["all"]
    for g in groups:
        subset = df[df["group"] == g] if g != "all" else df
        plt.scatter(subset[x_col], subset[y_col], label=g, alpha=0.6, s=30)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def draw_defect_bar(
    stats_df: pd.DataFrame,
    group_col: str,
    defect_col: str,
    save_path: str,
) -> None:
    """缺陷统计柱状图。"""
    plt.figure(figsize=(10, 5))
    sns.barplot(data=stats_df, x=group_col, y=defect_col, palette="Reds")
    plt.title("Defect Count Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def draw_all(output_root: str, config: dict) -> None:
    """批量生成全部图表。"""
    charts_dir = Path(output_root) / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(output_root) / "metrics_csv" / "batch_results.csv"
    if not csv_path.exists():
        print("[Plot] No batch_results.csv, skip.")
        return

    df = pd.read_csv(csv_path)
    metric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                   if not c.startswith("exif_") and c != "defect_count"]

    # 分组均值雷达图
    if "group" in df.columns:
        grouped = df.groupby("group")[metric_cols].mean()
        if len(grouped) > 1 and len(metric_cols) >= 3:
            cat_short = [c[:8] for c in metric_cols]
            vals = {g: row.tolist() for g, row in grouped.iterrows()}
            draw_radar(cat_short, vals, "IQ Metrics Radar Comparison",
                       str(charts_dir / "radar_comparison.png"))

        # 折线图
        for col in metric_cols[:4]:
            draw_line_chart(df, "file", col, "group",
                            f"{col} Comparison", str(charts_dir / f"{col}_line.png"))

        # 缺陷柱状图
        if "defect_count" in df.columns:
            draw_defect_bar(df, "group", "defect_count",
                            str(charts_dir / "defect_bar.png"))

    print(f"[Plot] Charts saved to {charts_dir}")
