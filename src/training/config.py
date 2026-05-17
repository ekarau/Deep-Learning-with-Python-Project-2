"""YAML config loader with 'extends' inheritance.

Ablation configs (configs/ablation/*.yaml) inherit from full_model.yaml
and override just the fields they ablate. This loader resolves the
inheritance chain into a single merged dict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, child: dict) -> dict:
    """Recursively merge child into base; child wins on conflict."""
    out = dict(base)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML at *path*, resolving any 'extends' parent chain."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if "extends" in cfg:
        parent_name = cfg.pop("extends")
        # Look in same dir, then in configs/ (one level up)
        for candidate in (path.parent / parent_name, path.parent.parent / parent_name):
            if candidate.exists():
                parent_cfg = load_config(candidate)
                cfg = deep_merge(parent_cfg, cfg)
                break
        else:
            raise FileNotFoundError(f"extends target not found: {parent_name}")
    return cfg
