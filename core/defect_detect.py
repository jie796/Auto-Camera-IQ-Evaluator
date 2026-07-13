"""缺陷检测：YOLOv8 AI 检测 + 传统视觉辅助检测

支持缺陷类型：
  - chromatic_aberration  色散/紫边
  - dirt                  脏点/污渍
  - blur                  模糊/失焦
  - ghost                 鬼影/眩光
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

# YOLOv8 可选加载
_ULTRALYTICS_AVAILABLE = False
try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    pass


class DefectDetector:
    """缺陷检测器，聚合 YOLOv8 模型 + 传统图像处理方法。"""

    def __init__(self, config: dict):
        self.enabled = config.get("defect_detection", {}).get("enable", False)
        self.confidence = config.get("defect_detection", {}).get("confidence", 0.35)
        self.iou = config.get("defect_detection", {}).get("iou", 0.5)
        self.class_names = config.get("defect_detection", {}).get("classes", [])
        self.model = None

        if not self.enabled:
            return

        model_path = config.get("defect_detection", {}).get("model_path", "models/yolov8_defect.pt")
        if Path(model_path).exists() and _ULTRALYTICS_AVAILABLE:
            self.model = YOLO(model_path)
        else:
            print(f"[Defect] YOLOv8 model not found at {model_path}, "
                  f"using traditional methods only. "
                  f"Ultralytics available: {_ULTRALYTICS_AVAILABLE}")

    def detect(self, img: np.ndarray) -> Dict[str, list]:
        """对单张图像执行全部缺陷检测。

        Returns:
            {"chromatic_aberration": [bbox...], "dirt": [...], "blur": [...], "ghost": [...]}
            每个 bbox = {"bbox": [x1,y1,x2,y2], "confidence": float}
        """
        results: Dict[str, list] = {cls: [] for cls in self.class_names}

        # --- YOLOv8 检测 ---
        if self.model is not None:
            preds = self.model(img, conf=self.confidence, iou=self.iou, verbose=False)
            if preds and preds[0].boxes is not None:
                for box in preds[0].boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    if cls_name in results:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        results[cls_name].append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(box.conf[0]),
                        })

        # --- 传统方法辅助检测 ---
        # 色散（基于柱状色差检测）
        if "chromatic_aberration" in results:
            ca_detections = _detect_chromatic_aberration_traditional(img)
            results["chromatic_aberration"].extend(ca_detections)

        # 模糊（拉普拉斯方差检测）
        if "blur" in results:
            blur_detections = _detect_blur_traditional(img)
            results["blur"].extend(blur_detections)

        return results


def _detect_chromatic_aberration_traditional(img: np.ndarray) -> List[Dict]:
    """传统方法检测紫边/色散：高饱和度紫/青色区域。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # 紫色色相 260°~320° ≈ [130, 160] in OpenCV HSV (H/2)
    purple_mask = cv2.inRange(hsv, (130, 50, 50), (160, 255, 255))
    # 青色色相 170°~200°
    cyan_mask = cv2.inRange(hsv, (85, 50, 50), (100, 255, 255))
    mask = cv2.bitwise_or(purple_mask, cyan_mask)

    # 形态学开运算去噪（腐蚀 → 膨胀）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:  # 过滤极小噪点
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        detections.append({
            "bbox": [x, y, x + w, y + h],
            "confidence": min(area / 500, 0.85),
            "method": "traditional_ca",
        })
    return detections


def _detect_blur_traditional(img: np.ndarray) -> List[Dict]:
    """拉普拉斯方差法检测模糊：低方差 = 模糊。"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 50:  # 经验阈值，越小越模糊
        return [{"bbox": [0, 0, img.shape[1], img.shape[0]],
                 "confidence": max(0.5, 1.0 - lap_var / 50),
                 "method": "traditional_blur"}]
    return []


def visualize_defects(img: np.ndarray, detections: Dict[str, list],
                      save_path: str) -> None:
    """在图像上绘制检测框并保存。"""
    color_map = {
        "chromatic_aberration": (255, 0, 255),   # 紫色
        "dirt": (0, 0, 255),                      # 红色
        "blur": (0, 255, 255),                    # 黄色
        "ghost": (255, 165, 0),                   # 橙色
    }
    vis = img.copy()
    for cls_name, boxes in detections.items():
        color = color_map.get(cls_name, (0, 255, 0))
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            conf = b.get("confidence", 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(vis, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
