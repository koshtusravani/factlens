"""Shared argument handling for the numbered pipeline scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from factlens.data.fever import load_claims
from factlens.data.splits import apply_split, read_ids


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="Path to a YAML config (defaults to configs/default.yaml)")
    return p


def load_split(cfg: dict, split: str, limit: int | None = None):
    """Loads train, val, or test claims using the committed ID files in splits/."""
    paths = cfg["paths"]
    if split == "test":
        claims = load_claims(paths["fever_dev"])
    else:
        claims = load_claims(paths["fever_train"])
        val_ids = read_ids(Path(paths["splits_dir"]) / "val_ids.txt")
        claims = apply_split(claims, val_ids, exclude=(split == "train"))
    if limit:
        claims = claims[:limit]
    return claims


def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"Wrote {path}")