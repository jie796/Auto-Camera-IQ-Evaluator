"""Tests for iq_metrics"""

import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.iq_metrics import (
    compute_snr, compute_vignetting,
    compute_blocking_artifacts, compute_delta_e,
)


def test_snr_uniform_image():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    snr = compute_snr(img, (10, 10, 80, 80))
    assert np.isinf(snr), f"Expected inf, got {snr}"


def test_snr_noisy_image():
    np.random.seed(42)
    img = (np.random.randn(100, 100, 3) * 10 + 128).clip(0, 255).astype(np.uint8)
    snr = compute_snr(img, (10, 10, 80, 80))
    assert 10 < snr < 50, f"SNR={snr} out of range"


def test_vignetting_uniform():
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    result = compute_vignetting(img)
    assert abs(result["vignetting_ratio"] - 1.0) < 0.01


def test_blocking_no_block():
    np.random.seed(42)
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    result = compute_blocking_artifacts(img)
    assert result["blocking_score"] > 0


def test_delta_e_identical():
    ref = np.array([[128, 128, 128]], dtype=np.float32)
    det = np.array([[128, 128, 128]], dtype=np.float32)
    result = compute_delta_e(None, ref, det)
    assert result["delta_e_mean"] == 0.0
