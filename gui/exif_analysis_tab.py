"""EXIF 参数联动分析 Tab：ISO / 焦距对画质的影响"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox,
    QFileDialog, QTextEdit, QHBoxLayout, QGroupBox,
)
from PyQt5.QtCore import QThread, pyqtSignal
import pandas as pd
import numpy as np

from visualization.plot_drawer import draw_exif_scatter


class ExifAnalysisWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, csv_path, exif_param, metric_param, output_dir):
        super().__init__()
        self.csv_path = csv_path
        self.exif_param = exif_param
        self.metric_param = metric_param
        self.output_dir = output_dir

    def run(self):
        try:
            df = pd.read_csv(self.csv_path)
            exif_col = f"exif_{self.exif_param}"
            metric_col = self.metric_param

            if exif_col not in df.columns or metric_col not in df.columns:
                self.finished.emit(
                    f"缺少列: {exif_col} 或 {metric_col}\n"
                    f"可用列: {list(df.columns)}"
                )
                return

            df_clean = df[[exif_col, metric_col, "group"]].dropna()
            stats = df_clean.groupby("group")[[exif_col, metric_col]].mean()
            corr = df_clean[exif_col].corr(df_clean[metric_col])

            # 画散点图
            save_path = Path(self.output_dir) / "charts" / f"exif_{self.exif_param}_vs_{metric_col}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            draw_exif_scatter(df_clean, exif_col, metric_col, str(save_path),
                              title=f"{self.exif_param} vs {metric_col}")

            report = (
                f"参数: {self.exif_param} → {metric_col}\n"
                f"样本数: {len(df_clean)}\n"
                f"相关系数: {corr:.4f}\n\n"
                f"分组均值:\n{stats.to_string()}\n\n"
                f"散点图已保存: {save_path}"
            )
            self.finished.emit(report)
        except Exception as e:
            self.finished.emit(f"Error: {e}")


class ExifAnalysisTab(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layout = QVBoxLayout()

        # 数据文件选择
        file_group = QGroupBox("数据源")
        file_layout = QHBoxLayout()
        self.csv_path = QLineEdit("output/metrics_csv/batch_results.csv")
        file_layout.addWidget(self.csv_path)
        btn = QPushButton("浏览...")
        btn.clicked.connect(lambda: self._browse())
        file_layout.addWidget(btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # 参数选择
        param_group = QGroupBox("分析参数")
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("EXIF 参数:"))
        self.exif_combo = QComboBox()
        self.exif_combo.addItems(["iso", "focal_length", "f_number", "shutter"])
        param_layout.addWidget(self.exif_combo)
        param_layout.addWidget(QLabel(" vs "))
        param_layout.addWidget(QLabel("画质指标:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["snr_db", "delta_e_mean", "vignetting_ratio", "blocking_score"])
        param_layout.addWidget(self.metric_combo)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        self.run_btn = QPushButton("分析")
        self.run_btn.clicked.connect(self._run)
        layout.addWidget(self.run_btn)

        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        layout.addWidget(QLabel("分析结果:"))
        layout.addWidget(self.result_output)

        self.setLayout(layout)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 CSV", "", "CSV Files (*.csv)")
        if path:
            self.csv_path.setText(path)

    def _run(self):
        self.run_btn.setEnabled(False)
        output_dir = "output"
        self.worker = ExifAnalysisWorker(
            self.csv_path.text(),
            self.exif_combo.currentText(),
            self.metric_combo.currentText(),
            output_dir,
        )
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_finished(self, report):
        self.run_btn.setEnabled(True)
        self.result_output.setText(report)
