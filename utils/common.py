"""通用工具：图像归一化、颜色空间转换"""

import numpy as np
import cv2


def normalize_uint8(img: np.ndarray) -> np.ndarray:
    """线性拉伸到 uint8 [0, 255]。"""
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - lo) / (hi - lo) * 255).astype(np.uint8)


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def compute_roi_mean(img: np.ndarray, cx: int, cy: int, size: int = 10) -> np.ndarray:
    """ROI 中心像素均值。"""
    half = size // 2
    return img[cy-half:cy+half, cx-half:cx+half].mean(axis=(0, 1))
