"""Writes a copy of a gold pair file whose NEI evidence counts match the gold evidence counts.

Example:
    python scripts/04b_match_nei_length.py --pairs data/pairs/gold_train_nei-retrieved.jsonl
writes data/pairs/gold_train_nei-retrieved-matched.jsonl next to the input.
"""

from pathlib import Path

from _common import base_parser

from factlens.config import load_config
from factlens.models.inputs import match_nei_evidence_counts
from factlens.utils import read_jsonl, write_jsonl


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--pairs", required=True, help="Gold pair file to match")
    args = p.parse_args()

    cfg = load_config(args.config)
    pairs = list(read_jsonl(args.pairs))
    matched = match_nei_evidence_counts(pairs, cfg["splits"]["seed"])
    src = Path(args.pairs)
    out = src.with_name(src.stem + "-matched.jsonl")
    write_jsonl(out, matched)
    for label in sorted({m["label"] for m in matched}):
        counts = [len(m["evidence"]) for m in matched if m["label"] == label]
        print(f"{label}: mean evidence sentences {sum(counts) / len(counts):.2f}")
    print(f"Wrote {len(matched)} pairs -> {out}")


if __name__ == "__main__":
    main()
