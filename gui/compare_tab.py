"""A/B 版本对比 Tab"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit,
    QFileDialog, QTextEdit, QHBoxLayout, QGroupBox,
)
from PyQt5.QtCore import QThread, pyqtSignal

from core.comparison import compare_versions, format_comparison_report


class CompareWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, csv_a, csv_b):
        super().__init__()
        self.csv_a = csv_a
        self.csv_b = csv_b

    def run(self):
        try:
            result = compare_versions(self.csv_a, self.csv_b)
            report = format_comparison_report(result)
            self.finished.emit(report)
        except Exception as e:
            self.finished.emit(f"Error: {e}")


class CompareTab(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layout = QVBoxLayout()

        path_group = QGroupBox("对比文件")
        path_layout = QVBoxLayout()

        row_a = QHBoxLayout()
        row_a.addWidget(QLabel("旧版本 CSV:"))
        self.csv_a = QLineEdit("")
        row_a.addWidget(self.csv_a)
        btn_a = QPushButton("浏览...")
        btn_a.clicked.connect(lambda: self._browse(self.csv_a))
        row_a.addWidget(btn_a)
        path_layout.addLayout(row_a)

        row_b = QHBoxLayout()
        row_b.addWidget(QLabel("新版本 CSV:"))
        self.csv_b = QLineEdit("")
        row_b.addWidget(self.csv_b)
        btn_b = QPushButton("浏览...")
        btn_b.clicked.connect(lambda: self._browse(self.csv_b))
        row_b.addWidget(btn_b)
        path_layout.addLayout(row_b)

        path_group.setLayout(path_layout)
        layout.addWidget(path_group)

        self.run_btn = QPushButton("开始对比")
        self.run_btn.clicked.connect(self._run_compare)
        layout.addWidget(self.run_btn)

        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        layout.addWidget(QLabel("对比结果:"))
        layout.addWidget(self.result_output)

        self.setLayout(layout)

    def _browse(self, target):
        path, _ = QFileDialog.getOpenFileName(self, "选择 CSV", "", "CSV Files (*.csv)")
        if path:
            target.setText(path)

    def _run_compare(self):
        if not self.csv_a.text() or not self.csv_b.text():
            self.result_output.setText("请先选择两个版本的 CSV 文件")
            return
        self.run_btn.setEnabled(False)
        self.worker = CompareWorker(self.csv_a.text(), self.csv_b.text())
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_finished(self, report):
        self.run_btn.setEnabled(True)
        self.result_output.setText(report)
