"""ROI 自动检测：24 色卡、ISO12233 分辨率卡、灰阶 ROI"""

from typing import Optional, Tuple, List
import numpy as np
import cv2


def detect_color_checker(img: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """检测 24 色卡，返回 4x6 色块中心坐标列表 [(x,y), ...]。

    策略：LAB 空间色块聚类 → 6x4 网格拟合。
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # 1. 自适应阈值分割彩色区域
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    color_mask = cv2.inRange(cv2.merge([a_channel, b_channel, np.zeros_like(a_channel)]),
                             (10, 10, 0), (255, 255, 0))

    # 2. 形态学闭运算连通
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    # 3. 找外接矩形
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # 4. 划分为 6x4 网格
    rows, cols = 4, 6
    centers = []
    for r in range(rows):
        for c in range(cols):
            cx = int(x + (c + 0.5) * w / cols)
            cy = int(y + (r + 0.5) * h / rows)
            centers.append((cx, cy))
    return centers


def detect_iso12233(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """检测 ISO 12233 分辨率测试卡 ROI (x, y, w, h)。

    策略：Canny 边缘 + 霍夫直线 → 定位倾斜边缘区域。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is None:
        return None

    # 找最长的近似 5° 倾斜线
    best_line = None
    best_len = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if 3 < angle < 10 and length > best_len:
            best_len = length
            best_line = (x1, y1, x2, y2)

    if best_line is None:
        return None

    x1, y1, x2, y2 = best_line
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half_w = best_len // 2 + 20
    half_h = 60
    return (cx - half_w, cy - half_h, best_len + 40, half_h * 2)


def detect_gray_step(img: np.ndarray) -> Optional[List[Tuple[int, int, int, int]]]:
    """检测灰阶卡，返回各阶 ROI 列表 [(x, y, w, h), ...]。"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 10:
        return None

    # 找最宽的矩形作为灰阶条
    rects = [cv2.boundingRect(c) for c in contours]
    rects.sort(key=lambda r: r[2], reverse=True)
    x, y, w, h = rects[0]

    # 垂直等分 (22 阶)
    patches = []
    for i in range(22):
        px = x + i * w // 22
        patches.append((px, y, w // 22, h))
    return patches
