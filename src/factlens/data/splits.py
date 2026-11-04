"""Reproducible train/validation/test splits.

FEVER test labels are hidden, so the labeled shared-task dev file serves as TEST, and a
stratified validation set is carved out of train. Only claim IDs are committed (splits/).
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from factlens.data.fever import Claim


def stratified_val_ids(claims: list[Claim], val_fraction: float, seed: int) -> set[int]:
    """Samples val_fraction of the claims per label with a fixed seed."""
    by_label: dict[str, list[int]] = defaultdict(list)
    for c in claims:
        by_label[str(c.label)].append(c.id)
    rng = random.Random(seed)
    val: set[int] = set()
    for label in sorted(by_label):
        ids = sorted(by_label[label])
        rng.shuffle(ids)
        val.update(ids[: round(len(ids) * val_fraction)])
    return val


def write_ids(path: str | Path, ids) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(i) for i in sorted(ids)) + "\n", encoding="utf-8")


def read_ids(path: str | Path) -> set[int]:
    return {int(x) for x in Path(path).read_text(encoding="utf-8").split()}


def apply_split(claims: list[Claim], ids: set[int], exclude: bool = False) -> list[Claim]:
    """Keeps claims whose id is in ids, or the complement when exclude is True."""
    return [c for c in claims if (c.id in ids) != exclude]