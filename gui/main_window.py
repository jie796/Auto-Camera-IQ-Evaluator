"""PyQt5 主窗口入口"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QMessageBox
from PyQt5.QtCore import Qt

from gui.batch_tab import BatchTab
from gui.compare_tab import CompareTab
from gui.exif_analysis_tab import ExifAnalysisTab
from gui.report_tab import ReportTab
from config import load_config


class MainWindow(QMainWindow):
    def __init__(self, config_path: str = None):
        super().__init__()
        self.setWindowTitle("JPG IQ Evaluator — 相机画质自动化评测系统")
        self.resize(1200, 800)

        # 加载配置
        try:
            self.config = load_config(config_path)
        except Exception as e:
            QMessageBox.critical(self, "配置错误", str(e))
            self.config = {}

        # 选项卡
        self.tabs = QTabWidget()
        self.tabs.addTab(BatchTab(self.config), "批量评测")
        self.tabs.addTab(CompareTab(self.config), "版本对比")
        self.tabs.addTab(ExifAnalysisTab(self.config), "EXIF 分析")
        self.tabs.addTab(ReportTab(self.config), "报表生成")
        self.setCentralWidget(self.tabs)


def launch_gui(config_path: str = None):
    app = QApplication(sys.argv)
    window = MainWindow(config_path)
    window.show()
    sys.exit(app.exec_())
