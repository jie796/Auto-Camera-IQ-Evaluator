"""输出目录管理、路径工具"""

from pathlib import Path
from typing import Dict, List, Optional


def ensure_output_dirs(output_root: str, subdirs: List[str]) -> Dict[str, Path]:
    """创建并返回输出子目录的路径映射。

    Args:
        output_root: 输出根目录
        subdirs: 子目录名列表，如 ["metrics_csv", "charts", "report", "defect_vis"]

    Returns:
        {"metrics_csv": Path("output/metrics_csv"), ...}
    """
    root = Path(output_root)
    paths = {}
    for name in subdirs:
        p = root / name
        p.mkdir(parents=True, exist_ok=True)
        paths[name] = p
    return paths


def resolve_abs(path: str, base_dir: Optional[str] = None) -> Path:
    """将相对路径解析为绝对路径。

    若 path 是相对路径且 base_dir 提供，则拼接到 base_dir 下。
    """
    p = Path(path)
    if p.is_absolute():
        return p
    if base_dir:
        return Path(base_dir) / p
    return p.resolve()


def get_csv_path(output_root: str, name: str = "batch_results") -> Path:
    """返回指标 CSV 路径。"""
    return Path(output_root) / "metrics_csv" / f"{name}.csv"


def list_image_files(
    input_dir: str,
    extensions: Optional[List[str]] = None,
) -> Dict[str, List[Path]]:
    """按子文件夹分组列出图片文件。

    Returns:
        {"scene_name": [Path, ...], ...}
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ext_set = {e.lower() for e in extensions}

    groups: Dict[str, List[Path]] = {}
    root = Path(input_dir)
    if not root.exists():
        return groups

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        files = sorted(p for p in subdir.iterdir() if p.suffix.lower() in ext_set)
        if files:
            groups[subdir.name] = files

    # 若输入目录本身就是图片文件夹（没有子目录），将所有文件归为 "default"
    if not groups:
        files = sorted(p for p in root.iterdir() if p.suffix.lower() in ext_set)
        if files:
            groups[root.name] = files

    return groups
