"""多线程批量调度 Pipeline"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict
import pandas as pd
import numpy as np
from tqdm import tqdm

from core.image_loader import load_jpg, extract_exif
from core.roi_detector import detect_color_checker
from core.iq_metrics import (
    compute_delta_e, compute_snr, compute_vignetting,
    compute_blocking_artifacts,
)
from core.defect_detect import DefectDetector, visualize_defects
from utils.file_io import list_image_files, get_csv_path
from utils.logger import setup_logger
from utils.exceptions import EvaluatorError


logger = setup_logger()


# 标准 24 色卡 sRGB 参考值 (Macbeth ColorChecker)
_MACBETH_REF = np.array([
    [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67],
    [133, 128, 177], [103, 189, 170], [214, 126, 44], [80, 91, 166],
    [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
    [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31],
    [187, 86, 149], [8, 133, 161], [243, 243, 242], [200, 200, 200],
    [160, 160, 160], [122, 122, 122], [85, 85, 85], [52, 52, 52],
], dtype=np.float32)


def run_pipeline(
    input_dir: str,
    meta_csv: str,
    config: dict,
    num_workers: int = 4,
    output_root: str = "output",
) -> pd.DataFrame:
    """全流程批量调度：加载 → ROI → 指标 → CSV → 缺陷可视化。"""

    # 输出目录（通过 utils.file_io 统一管理）
    from utils.file_io import ensure_output_dirs
    dirs = ensure_output_dirs(output_root,
                              ["metrics_csv", "defect_vis"])

    # 扫描 JPG
    groups = list_image_files(input_dir)
    if not groups:
        logger.warning("No image files found under %s", input_dir)
        return pd.DataFrame()

    # 展开所有文件
    jpg_files = []
    group_map = {}
    for group_name, files in groups.items():
        for f in files:
            jpg_files.append(f)
            group_map[f] = group_name
    logger.info("Found %d images in %d groups", len(jpg_files), len(groups))

    # 缺陷检测器
    detector = (DefectDetector(config)
                if config.get("defect_detection", {}).get("enable")
                else None)

    all_rows = []

    def process_one(img_path: Path) -> Dict:
        row = {"file": img_path.name}
        # group 优先从分组映射取，fallback 到父目录名
        row["group"] = group_map.get(img_path, img_path.parent.name)
        try:
            img = load_jpg(str(img_path))
            exif = extract_exif(str(img_path))
            row.update({f"exif_{k}": v for k, v in exif.items() if v is not None})

            h, w = img.shape[:2]

            # --- 客观指标 ---
            if config.get("metrics", {}).get("noise_snr"):
                roi = (w // 4, h // 4, w // 2, h // 2)
                row["snr_db"] = compute_snr(img, roi)

            if config.get("metrics", {}).get("color_accuracy"):
                centers = detect_color_checker(img)
                if centers and len(centers) == 24:
                    detected = np.array([
                        img[cy-5:cy+5, cx-5:cx+5].mean(axis=(0, 1))
                        for cx, cy in centers
                    ])
                    de = compute_delta_e(img, _MACBETH_REF, detected)
                    row.update(de)

            if config.get("metrics", {}).get("vignetting"):
                row.update(compute_vignetting(img))

            if config.get("metrics", {}).get("blocking_artifacts"):
                row.update(compute_blocking_artifacts(img))

            # --- 缺陷检测 ---
            if detector:
                detections = detector.detect(img)
                total = sum(len(v) for v in detections.values())
                row["defect_count"] = total
                for cls_name, boxes in detections.items():
                    row[f"defect_{cls_name}"] = len(boxes)
                    row[f"defect_{cls_name}_conf"] = (
                        round(np.mean([b["confidence"] for b in boxes]), 3)
                        if boxes else 0.0
                    )
                if total > 0:
                    vis_path = str(dirs["defect_vis"] / f"{img_path.stem}_defects.jpg")
                    visualize_defects(img, detections, vis_path)
                    row["defect_vis_path"] = vis_path

        except EvaluatorError as e:
            row["error"] = str(e)
            logger.error("Failed on %s: %s", img_path.name, e)
        except Exception as e:
            row["error"] = str(e)
            logger.error("Unexpected error on %s: %s", img_path.name, e)

        return row

    # 多线程
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one, f): f for f in jpg_files}
        for future in tqdm(as_completed(futures), total=len(jpg_files), desc="Processing"):
            all_rows.append(future.result())

    df = pd.DataFrame(all_rows)
    csv_path = get_csv_path(output_root)
    df.to_csv(csv_path, index=False)
    logger.info("Results saved: %s (%d images)", csv_path, len(df))
    return df
