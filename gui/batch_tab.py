"""批量评测 Tab"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit,
    QSpinBox, QFileDialog, QTextEdit, QHBoxLayout, QGroupBox,
)
from PyQt5.QtCore import QThread, pyqtSignal
import pandas as pd

from core.batch_processor import run_pipeline


class BatchWorker(QThread):
    finished = pyqtSignal(object)
    log = pyqtSignal(str)

    def __init__(self, input_dir, meta_csv, config, workers, output_root):
        super().__init__()
        self.input_dir = input_dir
        self.meta_csv = meta_csv
        self.config = config
        self.workers = workers
        self.output_root = output_root

    def run(self):
        try:
            df = run_pipeline(
                self.input_dir, self.meta_csv, self.config,
                self.workers, self.output_root,
            )
            self.finished.emit(df)
        except Exception as e:
            self.log.emit(f"Error: {e}")
            self.finished.emit(None)


class BatchTab(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layout = QVBoxLayout()

        # 输入路径
        path_group = QGroupBox("输入设置")
        path_layout = QVBoxLayout()

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("JPG 输入目录:"))
        self.input_path = QLineEdit("data/jpg_input")
        input_row.addWidget(self.input_path)
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse_input)
        input_row.addWidget(btn_browse)
        path_layout.addLayout(input_row)

        meta_row = QHBoxLayout()
        meta_row.addWidget(QLabel("元数据 CSV:"))
        self.meta_path = QLineEdit("data/meta_info.csv")
        meta_row.addWidget(self.meta_path)
        path_layout.addLayout(meta_row)

        worker_row = QHBoxLayout()
        worker_row.addWidget(QLabel("线程数:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(4)
        worker_row.addWidget(self.workers_spin)
        path_layout.addLayout(worker_row)

        path_group.setLayout(path_layout)
        layout.addWidget(path_group)

        # 运行按钮
        self.run_btn = QPushButton("开始批量评测")
        self.run_btn.clicked.connect(self._run_batch)
        layout.addWidget(self.run_btn)

        # 日志
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(QLabel("运行日志:"))
        layout.addWidget(self.log_output)

        self.setLayout(layout)

    def _browse_input(self):
        path = QFileDialog.getExistingDirectory(self, "选择 JPG 输入目录")
        if path:
            self.input_path.setText(path)

    def _run_batch(self):
        self.run_btn.setEnabled(False)
        self.log_output.clear()
        self.log_output.append("正在运行批量评测...")

        self.worker = BatchWorker(
            self.input_path.text(),
            self.meta_path.text(),
            self.config,
            self.workers_spin.value(),
            "output",
        )
        self.worker.finished.connect(self._on_finished)
        self.worker.log.connect(self.log_output.append)
        self.worker.start()

    def _on_finished(self, df):
        self.run_btn.setEnabled(True)
        if df is not None and not df.empty:
            self.log_output.append(f"✅ 完成！共处理 {len(df)} 张图像")
            self.log_output.append(f"   结果已保存至 output/metrics_csv/batch_results.csv")
        elif df is not None:
            self.log_output.append("⚠️ 未找到可处理的 JPG 文件")
        else:
            self.log_output.append("❌ 批量处理失败，请检查日志")
