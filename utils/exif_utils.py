"""EXIF 批量提取与元数据合并"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import piexif

from utils.exceptions import ImageLoadError


def batch_extract_exif(jpg_paths: List[Path]) -> pd.DataFrame:
    """批量提取 EXIF，返回 DataFrame。

    每行包含：file, iso, focal_length, f_number, shutter, make, model, datetime
    提取失败的字段记为 None。
    """
    rows = []
    for p in jpg_paths:
        row = {"file": p.name}
        try:
            exif = piexif.load(str(p))
            ifd0 = exif.get("0th", {})
            exif_data = exif.get("Exif", {})

            def _rational(val):
                if val and isinstance(val, tuple) and len(val) == 2:
                    return float(val[0]) / val[1] if val[1] != 0 else None
                return None

            iso_raw = exif_data.get(piexif.ExifIFD.ISOSpeedRatings)
            row["iso"] = int(iso_raw) if iso_raw is not None else None
            row["focal_length"] = _rational(exif_data.get(piexif.ExifIFD.FocalLength))
            row["f_number"] = _rational(exif_data.get(piexif.ExifIFD.FNumber))
            row["shutter"] = _rational(exif_data.get(piexif.ExifIFD.ExposureTime))
            row["make"] = ifd0.get(piexif.ImageIFD.Make, b"").decode("utf-8", errors="ignore")
            row["model"] = ifd0.get(piexif.ImageIFD.Model, b"").decode("utf-8", errors="ignore")
            row["datetime"] = ifd0.get(piexif.ImageIFD.DateTime, b"").decode("utf-8", errors="ignore")
        except Exception:
            pass  # 提取失败不影响整体
        rows.append(row)

    return pd.DataFrame(rows)


def merge_meta_csv(exif_df: pd.DataFrame, meta_path: Optional[str] = None) -> pd.DataFrame:
    """将 EXIF 数据与外部 meta_info.csv 合并。

    meta_info.csv 的字段优先级更高，会覆盖 EXIF 的对应字段。
    若 meta_path 不存在，直接返回 EXIF 数据。
    """
    if not meta_path or not Path(meta_path).exists():
        return exif_df

    meta_df = pd.read_csv(meta_path)
    if "file" not in meta_df.columns:
        return exif_df

    merged = exif_df.merge(meta_df, on="file", how="left", suffixes=("", "_meta"))
    # 用 meta 覆盖 EXIF 值
    for col in meta_df.columns:
        if col == "file":
            continue
        meta_col = f"{col}_meta"
        if meta_col in merged.columns:
            merged[col] = merged[meta_col].fillna(merged[col])
            merged.drop(columns=[meta_col], inplace=True)

    return merged
