"""画质指标算子：SFR MTF、ΔE、SNR、暗角、畸变、JPG 块效应"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from scipy import ndimage, signal
from skimage import color, exposure


# ========== SFR / MTF ==========

def compute_mtf50(img: np.ndarray, roi: Tuple[int, int, int, int]) -> Dict[str, float]:
    """基于 ISO 12233 倾斜边缘法的 MTF50 / MTF50P 计算。

    Args:
        img: RGB uint8
        roi: (x, y, w, h) 包含倾斜边缘的区域

    Returns:
        {"mtf50": float (cycles/pixel), "mtf50p": float, "mtf_value": float}
    """
    x, y, w, h = roi
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    patch = gray[y:y+h, x:x+w]

    # 1. 边缘检测 + 亚像素定位
    edges = cv2.Canny(patch, 30, 100)
    coords = np.column_stack(np.where(edges > 0))
    if len(coords) < 10:
        return {"mtf50": 0.0, "mtf50p": 0.0, "mtf_value": 0.0}

    # 2. 最小二乘拟合边缘直线 y = mx + b
    A = np.vstack([coords[:, 1], np.ones(len(coords))]).T
    m, b = np.linalg.lstsq(A, coords[:, 0], rcond=None)[0]

    # 3. 沿垂直方向投影得 ESF（边缘扩散函数）
    esf_length = patch.shape[0]
    esf = np.zeros(esf_length)
    count = np.zeros(esf_length)
    for row in range(patch.shape[0]):
        edge_col = int(m * row + b)
        if 0 <= edge_col < patch.shape[1]:
            esf[row] = patch[row, edge_col]
            count[row] = 1
    esf = np.where(count > 0, esf / np.maximum(count, 1), 0)
    esf = esf[count > 0]
    if len(esf) < 10:
        return {"mtf50": 0.0, "mtf50p": 0.0, "mtf_value": 0.0}

    # 4. 差分得 LSF（线扩散函数）
    lsf = np.diff(esf)

    # 5. 汉宁窗 + FFT 得 MTF
    window = np.hanning(len(lsf))
    mtf = np.abs(np.fft.fft(lsf * window))
    mtf = mtf[:len(mtf)//2] / mtf[0]  # 归一化

    # 6. 插值求 MTF50 (0.5) 和 MTF50P (0.5 峰值)
    freq = np.arange(len(mtf)) / len(mtf)
    mtf50 = _interp_freq(freq, mtf, 0.5)
    mtf50p = _interp_freq(freq, mtf, 0.5 * mtf.max())

    return {"mtf50": round(mtf50, 4), "mtf50p": round(mtf50p, 4),
            "mtf_value": round(float(mtf[0] if len(mtf) > 0 else 0), 4)}


def _interp_freq(freq: np.ndarray, mtf: np.ndarray, level: float) -> float:
    """线性插值求 MTF 曲线在 level 处的频率值。"""
    idx = np.where(mtf <= level)[0]
    if len(idx) == 0:
        return freq[-1] if len(freq) > 0 else 0.0
    i = idx[0]
    if i == 0:
        return freq[0]
    return freq[i-1] + (level - mtf[i-1]) * (freq[i] - freq[i-1]) / (mtf[i] - mtf[i-1] + 1e-10)


# ========== 色彩准确度 ΔE ==========

def compute_delta_e(
    img: np.ndarray,
    ref_colors_srgb: np.ndarray,
    detected_colors: np.ndarray,
) -> Dict[str, float]:
    """计算 24 色卡 ΔE (CIE76 ΔEab).

    Args:
        ref_colors_srgb: 参考色 (Nx3) in sRGB [0, 255]
        detected_colors: 检测色块均值 (Nx3) in sRGB [0, 255]

    Returns:
        {"delta_e_mean": float, "delta_e_max": float, "delta_e_median": float}
    """
    ref_lab = color.rgb2lab(ref_colors_srgb / 255.0)
    det_lab = color.rgb2lab(detected_colors / 255.0)
    delta = np.sqrt(np.sum((ref_lab - det_lab) ** 2, axis=1))
    return {
        "delta_e_mean": round(float(delta.mean()), 2),
        "delta_e_max": round(float(delta.max()), 2),
        "delta_e_median": round(float(np.median(delta)), 2),
    }


# ========== SNR ==========

def compute_snr(img: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
    """平坦区域 SNR (dB)。"""
    x, y, w, h = roi
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    patch = gray[y:y+h, x:x+w].astype(np.float32)
    mean = patch.mean()
    std = patch.std()
    if std < 1e-6:
        return float("inf")
    return round(20 * np.log10(mean / std), 2)


def compute_texture_noise(img: np.ndarray, roi: Tuple[int, int, int, int]) -> Dict[str, float]:
    """纹理噪声: 基于 dead leaves 或高频方差。"""
    x, y, w, h = roi
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    patch = gray[y:y+h, x:x+w].astype(np.float32)

    # 高通滤波提取纹理能量
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]]) / 8
    high = cv2.filter2D(patch, -1, kernel)
    texture_energy = float(np.std(high))
    return {"texture_noise": round(texture_energy, 4)}


# ========== 暗角 ==========

def compute_vignetting(img: np.ndarray, center: Optional[Tuple[int, int]] = None) -> Dict[str, float]:
    """计算暗角率：四角亮度 / 中心亮度。

    Returns:
        {"vignetting_ratio": float (越小暗角越严重),
         "vignetting_db": float (dB)}
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if center is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = center

    # 中心 ROI（10% 图像尺寸）
    r = min(h, w) // 10
    center_mean = gray[cy-r:cy+r, cx-r:cx+r].mean()

    # 四角 ROI
    corner_r = min(h, w) // 15
    corners = [
        gray[0:corner_r, 0:corner_r],
        gray[0:corner_r, w-corner_r:w],
        gray[h-corner_r:h, 0:corner_r],
        gray[h-corner_r:h, w-corner_r:w],
    ]
    corner_mean = np.mean([c.mean() for c in corners])

    ratio = corner_mean / (center_mean + 1e-6)
    db = 20 * np.log10(ratio + 1e-10)
    return {"vignetting_ratio": round(ratio, 4), "vignetting_db": round(db, 2)}


# ========== JPG 块效应 ==========

def compute_blocking_artifacts(img: np.ndarray) -> Dict[str, float]:
    """检测 JPG 8x8 块边界伪影。

    基于块边界灰度差 vs 块内部灰度差的比例。
    Returns:
        {"blocking_score": float, "blocking_db": float}
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    # 水平块边界（8 像素间距）
    h_diff = np.abs(np.diff(gray, axis=1))
    h_boundary = h_diff[:, 7::8].mean()
    h_inner = h_diff[:, 0:7].mean()

    # 垂直块边界
    v_diff = np.abs(np.diff(gray, axis=0))
    v_boundary = v_diff[7::8, :].mean()
    v_inner = v_diff[0:7, :].mean()

    boundary_mean = (h_boundary + v_boundary) / 2
    inner_mean = (h_inner + v_inner) / 2 + 1e-6

    score = boundary_mean / inner_mean
    return {"blocking_score": round(float(score), 4),
            "blocking_db": round(20 * np.log10(score), 2)}


# ========== 畸变 ==========

def compute_distortion(img: np.ndarray, grid_size: int = 8) -> Dict[str, float]:
    """基于棋盘格角点的畸变率 (%)。"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, (grid_size, grid_size),
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return {"distortion_pct": 0.0}

    # 理想直线 vs 实际角点偏移
    ideal = np.zeros((grid_size * grid_size, 2))
    for i in range(grid_size):
        for j in range(grid_size):
            ideal[i * grid_size + j] = [j * 50, i * 50]

    actual = corners.reshape(-1, 2)
    errors = np.sqrt(np.sum((actual - ideal) ** 2, axis=1))
    return {"distortion_pct": round(float(errors.mean()) / actual.mean() * 100, 2)}
