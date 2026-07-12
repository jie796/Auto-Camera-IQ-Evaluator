"""JPG 图像加载 + EXIF 元数据提取"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import piexif


def load_jpg(path: str) -> np.ndarray:
    """加载 JPG 为 RGB uint8 numpy array (HxWx3)。"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to load: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract_exif(path: str) -> Dict[str, Optional[float]]:
    """提取拍摄参数 EXIF。

    Returns:
        {"iso": int, "focal_length": mm, "f_number": float,
         "shutter": sec, "cct": int or None, "datetime": str}
    """
    exif_dict = piexif.load(path)
    ifd0 = exif_dict.get("0th", {})
    exif = exif_dict.get("Exif", {})

    # ISO
    iso_raw = exif.get(piexif.ExifIFD.ISOSpeedRatings)
    if iso_raw is None:
        iso_raw = ifd0.get(piexif.ImageIFD.ISOSpeedRatings)
    iso = int(iso_raw) if iso_raw is not None else None

    # 焦距 (mm)
    focal_raw = exif.get(piexif.ExifIFD.FocalLength)
    focal = float(focal_raw[0] / focal_raw[1]) if focal_raw else None

    # 光圈
    fnum_raw = exif.get(piexif.ExifIFD.FNumber)
    f_number = float(fnum_raw[0] / fnum_raw[1]) if fnum_raw else None

    # 快门速度 (秒)
    shutter_raw = exif.get(piexif.ExifIFD.ExposureTime)
    shutter = float(shutter_raw[0] / shutter_raw[1]) if shutter_raw else None

    # 时间
    dt = ifd0.get(piexif.ImageIFD.DateTime, "").decode("utf-8", errors="ignore")

    return {
        "iso": iso,
        "focal_length": focal,
        "f_number": f_number,
        "shutter": shutter,
        "datetime": dt,
    }


def load_exif_batch(paths: list) -> Dict[str, Dict]:
    """批量提取 EXIF，返回 {文件名: exif_dict}。"""
    result = {}
    for p in paths:
        try:
            result[p.name] = extract_exif(str(p))
        except Exception as e:
            result[p.name] = {"error": str(e)}
    return result
