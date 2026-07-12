#!/usr/bin/env python3
"""JPG IQ Evaluator — 统一入口"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JPG IQ Evaluator — 基于成品 JPG 的相机画质自动化评测系统"
    )
    parser.add_argument("--gui", action="store_true",
                        help="启动 PyQt5 图形界面")
    parser.add_argument("--input", type=str, default="data/jpg_input",
                        help="JPG 输入目录 (按场景分文件夹)")
    parser.add_argument("--meta", type=str, default="data/meta_info.csv",
                        help="拍摄元数据 CSV")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="全局配置路径")
    parser.add_argument("--output", type=str, default="output",
                        help="输出根目录")
    parser.add_argument("--workers", type=int, default=4,
                        help="线程数")
    parser.add_argument("--compare", nargs=2, metavar=("CSV_A", "CSV_B"),
                        help="对比两个版本的结果 CSV")
    return parser.parse_args()


def main():
    args = parse_args()

    # GUI 模式
    if args.gui:
        from gui.main_window import launch_gui
        launch_gui(args.config)
        return

    from config import load_config
    cfg = load_config(args.config)

    # A/B 对比模式
    if args.compare:
        from core.comparison import compare_versions, format_comparison_report
        result = compare_versions(args.compare[0], args.compare[1])
        print(format_comparison_report(result))
        return

    # 批量评测
    from core.batch_processor import run_pipeline
    from visualization.plot_drawer import draw_all
    from visualization.pdf_generator import generate_report

    df = run_pipeline(args.input, args.meta, cfg, args.workers, args.output)
    if not df.empty:
        draw_all(args.output, cfg)
        generate_report(args.output, cfg)
        print(f"[Done] 全流程完成，结果目录: {args.output}")
    else:
        print("[Done] 无待处理图像")


if __name__ == "__main__":
    main()
