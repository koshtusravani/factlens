"""Builds gold-evidence pairs for a split. NEI claims get sampled evidence.

Example:
    python scripts/04_build_gold_pairs.py --split train --nei-strategy retrieved
"""

from pathlib import Path

from _common import base_parser, load_split

from factlens.config import load_config
from factlens.models.inputs import build_gold_pairs
from factlens.retrieval import Retriever, WikiStore
from factlens.utils import write_jsonl


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--nei-strategy", choices=["retrieved", "random"], default=None)
    p.add_argument("--limit", type=int, default=None, help="Only the first N claims")
    args = p.parse_args()

    cfg = load_config(args.config)
    strategy = args.nei_strategy or cfg["nei_sampling"]["strategy"]
    claims = load_split(cfg, args.split, args.limit)
    store = WikiStore(cfg["paths"]["wiki_db"])
    retriever = Retriever.from_config(cfg, store=store)
    pairs, skipped = build_gold_pairs(claims, store, retriever, strategy,
                                      cfg["nei_sampling"]["num_sentences"], cfg["splits"]["seed"])
    suffix = f"_n{args.limit}" if args.limit else ""
    out = Path(cfg["paths"]["pairs_dir"]) / f"gold_{args.split}_nei-{strategy}{suffix}.jsonl"
    n = write_jsonl(out, pairs)
    print(f"Wrote {n} pairs, skipped {skipped} claims with gold sentences missing -> {out}")
    store.close()


if __name__ == "__main__":
    main()