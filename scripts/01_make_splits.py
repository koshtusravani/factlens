"""Writes splits/val_ids.txt, a stratified 5% validation split of FEVER train.

The ID file is committed so every machine uses identical splits. The claim files are not.
"""

from pathlib import Path

from _common import base_parser

from factlens.config import load_config
from factlens.data.fever import load_claims
from factlens.data.splits import stratified_val_ids, write_ids


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    train = load_claims(cfg["paths"]["fever_train"])
    val = stratified_val_ids(train, cfg["splits"]["val_fraction"], cfg["splits"]["seed"])
    out = Path(cfg["paths"]["splits_dir"]) / "val_ids.txt"
    write_ids(out, val)
    print(f"train={len(train) - len(val)} val={len(val)} -> {out}")


if __name__ == "__main__":
    main()