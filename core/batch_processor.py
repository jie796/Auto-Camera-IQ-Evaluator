"""ProcessPool 多进程并行评测 Pipeline

优化点：
1. ThreadPoolExecutor → ProcessPoolExecutor（绕开 GIL，真正多核并行）
2. _init_worker 预加载 YOLO 模型（每进程只加载一次）
3. 删除跨进程传 ThreadPoolExecutor 对象（不可 pickle）
4. defect_count 取值为缺陷实例总数（原为类别数）
5. 结果按文件名排序后输出
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── 每个 worker 子进程独享的全局变量（由 _init_worker 设置） ──
_config = None          # config dict
_detector = None        # DefectDetector 实例
_output_root = None     # 输出根目录

# 标准 24 色卡 sRGB 参考值 (Macbeth ColorChecker)
_MACBETH_REF = np.array([
    [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67],
    [133, 128, 177], [103, 189, 170], [214, 126, 44], [80, 91, 166],
    [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
    [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31],
    [187, 86, 149], [8, 133, 161], [243, 243, 242], [200, 200, 200],
    [160, 160, 160], [122, 122, 122], [85, 85, 85], [52, 52, 52],
], dtype=np.float32)


def _init_worker(config_path: str, output_root: str):
    """每个 worker 进程启动时执行一次。

    1. 把项目根目录加入 sys.path（子进程默认不包含项目目录）
    2. 加载配置
    3. 预创建 DefectDetector（含 YOLO 模型加载，只需一次）
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    global _config, _detector, _output_root
    _output_root = output_root

    from config import load_config
    _config = load_config(config_path)

    if _config.get("defect_detection", {}).get("enable"):
        from core.defect_detect import DefectDetector
        _detector = DefectDetector(_config)


def process_one(img_path: Path) -> Dict:
    """单张图片完整评测（运行在 worker 子进程中）。

    返回 plain dict（可 pickle），包含所有指标和错误信息。
    """
    # ── 所有项目内部依赖在函数内导入（模块级导入时 sys.path 尚未设好） ──
    from core.image_loader import load_jpg, extract_exif
    from core.roi_detector import detect_color_checker
    from core.iq_metrics import (
        compute_delta_e, compute_snr, compute_vignetting,
        compute_blocking_artifacts,
    )
    from core.defect_detect import visualize_defects
    from utils.exceptions import EvaluatorError

    row = {"file": img_path.name, "group": img_path.parent.name}
    try:
        img = load_jpg(str(img_path))
        exif = extract_exif(str(img_path))
        row.update({f"exif_{k}": v for k, v in exif.items() if v is not None})

        h, w = img.shape[:2]

        # ── 客观指标 ──
        if _config.get("metrics", {}).get("noise_snr"):
            roi = (w // 4, h // 4, w // 2, h // 2)
            row["snr_db"] = compute_snr(img, roi)

        if _config.get("metrics", {}).get("color_accuracy"):
            centers = detect_color_checker(img)
            if centers and len(centers) == 24:
                detected = np.array([
                    img[cy-5:cy+5, cx-5:cx+5].mean(axis=(0, 1))
                    for cx, cy in centers
                ])
                de = compute_delta_e(img, _MACBETH_REF, detected)
                row.update(de)

        if _config.get("metrics", {}).get("vignetting"):
            row.update(compute_vignetting(img))

        if _config.get("metrics", {}).get("blocking_artifacts"):
            row.update(compute_blocking_artifacts(img))

        # ── 缺陷检测 ──
        if _detector is not None:
            detections = _detector.detect(img)
            total = sum(len(v) for v in detections.values())
            row["defect_count"] = total
            for cls_name, boxes in detections.items():
                row[f"defect_{cls_name}"] = len(boxes)
                row[f"defect_{cls_name}_conf"] = (
                    round(np.mean([b["confidence"] for b in boxes]), 3)
                    if boxes else 0.0
                )
            if total > 0:
                vis_dir = Path(_output_root) / "defect_vis"
                vis_dir.mkdir(parents=True, exist_ok=True)
                vis_path = vis_dir / f"{img_path.stem}_defects.jpg"
                visualize_defects(img, detections, str(vis_path))
                row["defect_vis_path"] = str(vis_path)

    except EvaluatorError as e:
        row["error"] = str(e)
    except Exception as e:
        row["error"] = str(e)

    return row


def run_pipeline(
    input_dir: str,
    meta_csv: str,
    config: dict,
    num_workers: int = None,
    output_root: str = "output",
) -> pd.DataFrame:
    """全流程入口：主进程扫描 → ProcessPool 并行计算 → 排序 → CSV。

    Args:
        num_workers: 进程数，默认 = CPU 核心数
    """
    # ── 主进程内导入（已有 sys.path，无需 worker 初始化） ──
    from utils.file_io import list_image_files, get_csv_path
    from utils.logger import setup_logger
    import os
    import yaml

    logger = setup_logger()

    # 扫描 JPG
    groups = list_image_files(input_dir)
    if not groups:
        logger.warning("No image files found under %s", input_dir)
        return pd.DataFrame()

    # 拍平 + 按文件名排序（保证两次运行结果顺序一致）
    jpg_files = sorted(
        [f for files in groups.values() for f in files],
        key=lambda p: p.name,
    )
    total = len(jpg_files)
    logger.info("Found %d images in %d groups", total, len(groups))

    if num_workers is None:
        num_workers = os.cpu_count() or 4
    logger.info("Process workers: %d | IO workers: 0 (in-process) | Images: %d",
                num_workers, total)

    # 把 config dict 写出到临时文件，worker 子进程读取
    cfg_path = str(Path(output_root) / "_pipeline_config.yaml")
    Path(cfg_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    all_rows: List[Dict] = []

    # ── 进程池：真多核并行 ──
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(cfg_path, output_root),
    ) as proc_pool:
        futures = {proc_pool.submit(process_one, f): f for f in jpg_files}
        for future in tqdm(as_completed(futures), total=total, desc="Processing"):
            all_rows.append(future.result())

    # 按原文件顺序重排（as_completed 返回顺序不固定）
    img_order = {f.name: i for i, f in enumerate(jpg_files)}
    all_rows.sort(key=lambda r: img_order.get(r.get("file", ""), 999999))

    # 清理临时配置
    try:
        Path(cfg_path).unlink(missing_ok=True)
    except Exception:
        pass

    df = pd.DataFrame(all_rows)
    csv_path = get_csv_path(output_root)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Results saved: %s (%d images)", csv_path, len(df))
    return df
