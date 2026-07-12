"""文件 I/O：目录遍历、CSV 读写"""

from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


def list_jpg_files(input_dir: str, extensions: Optional[List[str]] = None) -> Dict[str, List[Path]]:
    """按场景子文件夹列出所有 JPG 文件。"""
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ext_set = {e.lower() for e in extensions}

    groups: Dict[str, List[Path]] = {}
    root = Path(input_dir)
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        files = sorted(p for p in subdir.iterdir()
                       if p.suffix.lower() in ext_set)
        if files:
            groups[subdir.name] = files
    return groups


def read_meta_csv(path: str) -> pd.DataFrame:
    """读取拍摄元数据 CSV。"""
    return pd.read_csv(path)


def append_metrics_csv(csv_path: str, row: dict) -> None:
    """追加单行到指标 CSV。"""
    df = pd.DataFrame([row])
    df.to_csv(csv_path, mode="a",
              header=not Path(csv_path).exists(), index=False)


def read_metrics_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
