"""多线程批量调度 Pipeline"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Callable
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

from core.image_loader import load_jpg, extract_exif
from core.roi_detector import detect_color_checker
from core.iq_metrics import (
    compute_delta_e, compute_snr, compute_vignetting,
    compute_blocking_artifacts,
)
from core.defect_detect import DefectDetector


def run_pipeline(
    input_dir: str,
    meta_csv: str,
    config: dict,
    num_workers: int = 4,
    output_root: str = "output",
) -> pd.DataFrame:
    """全流程批量调度：加载 → ROI → 指标 → CSV → 缺陷可视化。"""
    metrics_dir = Path(output_root) / "metrics_csv"
    defect_vis_dir = Path(output_root) / "defect_vis"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    defect_vis_dir.mkdir(parents=True, exist_ok=True)

    # 扫描 JPG
    input_root = Path(input_dir)
    jpg_files = sorted(input_root.rglob("*.jpg")) + sorted(input_root.rglob("*.jpeg"))
    if not jpg_files:
        print(f"[Batch] No JPG files found under {input_dir}")
        return pd.DataFrame()

    # 缺陷检测器
    detector = DefectDetector(config) if config.get("defect_detection", {}).get("enable") else None

    all_rows = []
    lock = None  # stdlib threading lock not needed with sequential append

    def process_one(img_path: Path) -> Dict:
        row = {"file": img_path.name, "group": img_path.parent.name}
        try:
            img = load_jpg(str(img_path))
            exif = extract_exif(str(img_path))
            row.update({f"exif_{k}": v for k, v in exif.items() if v is not None})

            h, w = img.shape[:2]

            # --- 客观指标 ---
            if config.get("metrics", {}).get("noise_snr"):
                # 中心平坦 ROI
                roi = (w // 4, h // 4, w // 2, h // 2)
                row["snr_db"] = compute_snr(img, roi)

            if config.get("metrics", {}).get("color_accuracy"):
                centers = detect_color_checker(img)
                if centers is not None and len(centers) == 24:
                    detected = []
                    for cx, cy in centers:
                        patch = img[cy-5:cy+5, cx-5:cx+5]
                        detected.append(patch.mean(axis=(0, 1)))
                    # 标准 24 色卡 sRGB 参考值（Macbeth ColorChecker）
                    ref = _get_macbeth_reference()
                    de = compute_delta_e(img, ref, np.array(detected))
                    row.update(de)

            if config.get("metrics", {}).get("vignetting"):
                row.update(compute_vignetting(img))

            if config.get("metrics", {}).get("blocking_artifacts"):
                row.update(compute_blocking_artifacts(img))

            # --- 缺陷检测 ---
            if detector:
                detections = detector.detect(img)
                total_defects = sum(len(v) for v in detections.values())
                row["defect_count"] = total_defects
                for cls_name, boxes in detections.items():
                    row[f"defect_{cls_name}"] = len(boxes)
                    row[f"defect_{cls_name}_conf"] = round(
                        np.mean([b["confidence"] for b in boxes]), 3
                    ) if boxes else 0.0

                # 保存可视化
                if total_defects > 0:
                    vis_path = str(defect_vis_dir / f"{img_path.stem}_defects.jpg")
                    from core.defect_detect import visualize_defects
                    visualize_defects(img, detections, vis_path)
                    row["defect_vis_path"] = vis_path

        except Exception as e:
            row["error"] = str(e)

        return row

    # 多线程
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one, f): f for f in jpg_files}
        for future in tqdm(as_completed(futures), total=len(jpg_files), desc="Processing"):
            all_rows.append(future.result())

    df = pd.DataFrame(all_rows)
    csv_path = metrics_dir / "batch_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Batch] 结果已保存: {csv_path} ({len(df)} images)")
    return df


def _get_macbeth_reference() -> np.ndarray:
    """标准 24 色卡 sRGB 参考值 [0, 255] (6x4 布局，rows first)。"""
    return np.array([
        [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67],
        [133, 128, 177], [103, 189, 170], [214, 126, 44], [80, 91, 166],
        [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
        [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31],
        [187, 86, 149], [8, 133, 161], [243, 243, 242], [200, 200, 200],
        [160, 160, 160], [122, 122, 122], [85, 85, 85], [52, 52, 52],
    ], dtype=np.float32)
