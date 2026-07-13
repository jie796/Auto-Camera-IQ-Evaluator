"""通用图像与数据处理工具"""

from typing import Optional, Tuple
import numpy as np
import cv2


def safe_roi(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """安全截取 ROI，越界自动裁剪到图像边界。"""
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    return img[y1:y2, x1:x2]


def auto_crop_center(img: np.ndarray, ratio: float = 0.5) -> Tuple[int, int, int, int]:
    """返回图像中心区域 ROI (x, y, w, h)，ratio 控制区域占比。"""
    h, w = img.shape[:2]
    cw, ch = int(w * ratio), int(h * ratio)
    x = (w - cw) // 2
    y = (h - ch) // 2
    return x, y, cw, ch


def linear_stretch(img: np.ndarray, low_pct: float = 1, high_pct: float = 99) -> np.ndarray:
    """百分比线性拉伸增强对比度。"""
    if img.dtype != np.uint8:
        return img
    stretched = img.astype(np.float32)
    for c in range(3 if img.ndim == 3 else 1):
        channel = stretched[:, :, c] if img.ndim == 3 else stretched
        lo = np.percentile(channel, low_pct)
        hi = np.percentile(channel, high_pct)
        if hi > lo:
            channel[:] = np.clip((channel - lo) / (hi - lo) * 255, 0, 255)
    return stretched.astype(np.uint8)


def compute_sharpness_metric(img: np.ndarray) -> float:
    """拉普拉斯方差法：图像锐度评分（值越高越清晰）。"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
