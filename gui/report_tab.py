"""报表生成 Tab"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QTextEdit, QHBoxLayout, QGroupBox, QMessageBox,
)
from PyQt5.QtCore import QThread, pyqtSignal

from visualization.pdf_generator import generate_report


class ReportWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, output_root, config):
        super().__init__()
        self.output_root = output_root
        self.config = config

    def run(self):
        try:
            path = generate_report(self.output_root, self.config)
            self.finished.emit(path)
        except Exception as e:
            self.finished.emit(f"Error: {e}")


class ReportTab(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layout = QVBoxLayout()

        path_group = QGroupBox("报表设置")
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("输出目录:"))
        self.output_path = QLineEdit("output")
        path_layout.addWidget(self.output_path)
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)

        self.generate_btn = QPushButton("生成 PDF 报表")
        self.generate_btn.clicked.connect(self._generate)
        layout.addWidget(self.generate_btn)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(QLabel("日志:"))
        layout.addWidget(self.log_output)

        self.setLayout(layout)

    def _generate(self):
        self.generate_btn.setEnabled(False)
        self.log_output.setText("正在生成报表...")
        self.worker = ReportWorker(self.output_path.text(), self.config)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_finished(self, result):
        self.generate_btn.setEnabled(True)
        if result.startswith("Error"):
            self.log_output.setText(f"❌ 报表生成失败:\n{result}")
        else:
            self.log_output.setText(f"✅ 报表已生成:\n{result}")
            QMessageBox.information(self, "完成", f"报表已保存至:\n{result}")
