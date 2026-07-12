"""Tests for defect_detect"""

import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.defect_detect import _detect_chromatic_aberration_traditional, _detect_blur_traditional


def test_ca_detection():
    """合成紫边图像应被检测到。"""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    # 在左上角画紫色区域
    img[:20, :20, :] = [255, 0, 255]  # purple
    detections = _detect_chromatic_aberration_traditional(img)
    assert len(detections) > 0, "Should detect purple region"


def test_blur_detection_sharp():
    """清晰图像不应被检测为模糊。"""
    np.random.seed(42)
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    detections = _detect_blur_traditional(img)
    assert len(detections) == 0, "Sharp image should not be blur"


def test_blur_detection_blurry():
    """均匀模糊图像应被检测。"""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    detections = _detect_blur_traditional(img)
    assert len(detections) > 0, "Uniform image should be detected as blurry"
