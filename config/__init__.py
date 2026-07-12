"""Config loader — YAML 配置读取 + 校验"""

from pathlib import Path
from typing import Any, Dict
import yaml



_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: str = None) -> Dict[str, Any]:
    p = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def _validate(cfg: Dict[str, Any]):
    for k in ("scene_groups", "metrics", "defect_detection"):
        if k not in cfg:
            raise ValueError(f"Missing required config key: {k}")
