"""统一日志：同时输出到终端 + 文件"""

import logging
import sys
from pathlib import Path


_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
_DATE_FMT = "%H:%M:%S"


def setup_logger(name: str = "evaluator", log_dir: str = "output",
                 level: int = logging.INFO) -> logging.Logger:
    """初始化日志器：终端 + 文件双输出。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # 终端 handler（彩色级别）
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_FORMAT, _DATE_FMT))
    logger.addHandler(console)

    # 文件 handler
    log_path = Path(log_dir) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(_FORMAT, _DATE_FMT))
    logger.addHandler(fh)

    return logger
