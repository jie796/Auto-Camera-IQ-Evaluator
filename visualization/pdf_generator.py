"""PDF 标准化评测报表生成"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image, Spacer, Table, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd


def generate_report(output_root: str, config: dict) -> str:
    """生成 PDF 标准化评测报表。"""
    report_dir = Path(output_root) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = str(report_dir / "iq_evaluation_report.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # 封面标题
    story.append(Paragraph("Camera IQ Evaluation Report", styles["Title"]))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        "基于 JPG 成品图的相机画质自动化评测系统", styles["Normal"]))
    story.append(Spacer(1, 10 * mm))

    # 配置摘要
    enabled_metrics = [k for k, v in config.get("metrics", {}).items() if v]
    story.append(Paragraph(f"启用指标: {', '.join(enabled_metrics)}", styles["Normal"]))
    defect_on = config.get("defect_detection", {}).get("enable", False)
    story.append(Paragraph(f"缺陷检测: {'开启' if defect_on else '关闭'}", styles["Normal"]))
    story.append(Spacer(1, 6 * mm))

    # --- 插入图表 ---
    charts_dir = Path(output_root) / "charts"
    chart_files = sorted(charts_dir.glob("*.png"))
    for i, chart_path in enumerate(chart_files):
        img = Image(str(chart_path), width=160 * mm, height=100 * mm)
        story.append(img)
        story.append(Spacer(1, 4 * mm))
        if i % 2 == 1:  # 每两张图分页
            story.append(PageBreak())

    # --- 数据表格 ---
    csv_path = Path(output_root) / "metrics_csv" / "batch_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        display_cols = [c for c in df.columns
                        if not c.endswith("_vis_path") and not c.startswith("defect_")]
        display_cols = display_cols[:12]
        table_data = [display_cols] + df[display_cols].values.tolist()
        t = Table(table_data, repeatRows=1)
        story.append(Spacer(1, 8 * mm))
        story.append(Paragraph("批量评测数据摘要", styles["Heading2"]))
        story.append(t)

    doc.build(story)
    print(f"[Report] PDF saved: {pdf_path}")
    return pdf_path
