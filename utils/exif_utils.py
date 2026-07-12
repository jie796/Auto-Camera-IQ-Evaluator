"""EXIF 解析辅助工具"""

from typing import Dict, Optional
import piexif


def get_exif_dict(path: str) -> dict:
    """标准化的 EXIF 数据提取。"""
    exif_dict = piexif.load(path)
    ifd0 = exif_dict.get("0th", {})
    exif = exif_dict.get("Exif", {})

    def _rational(val):
        if val and isinstance(val, tuple) and len(val) == 2:
            return float(val[0]) / val[1] if val[1] != 0 else None
        return None

    return {
        "make": ifd0.get(piexif.ImageIFD.Make, b"").decode("utf-8", errors="ignore"),
        "model": ifd0.get(piexif.ImageIFD.Model, b"").decode("utf-8", errors="ignore"),
        "iso": exif.get(piexif.ExifIFD.ISOSpeedRatings),
        "f_number": _rational(exif.get(piexif.ExifIFD.FNumber)),
        "focal_length": _rational(exif.get(piexif.ExifIFD.FocalLength)),
        "shutter": _rational(exif.get(piexif.ExifIFD.ExposureTime)),
        "datetime": ifd0.get(piexif.ImageIFD.DateTime, b"").decode("utf-8", errors="ignore"),
    }
