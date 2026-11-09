"""Runs the retriever on a split and writes retrieved pairs plus recall metrics.

Usage (from the repo root):
    python scripts/03_run_retrieval.py --split val --limit 2000                    # R1 on val 2000
    python scripts/03_run_retrieval.py --split val --limit 2000 --no-title-match   # R0 ablation
    python scripts/03_run_retrieval.py --split test --sent-ranker hybrid           # R3 on test

Writes data/pairs/retrieved_{split}_{name}.jsonl (one row per claim) and
outputs/retrieval/{split}_{name}.json, where name is the ranker plus optional
suffixes such as _notitle and _n2000.
"""

from __future__ import annotations

import time
from pathlib import Path

from _common import base_parser, load_split, save_json

from factlens.config import load_config
from factlens.retrieval import Retriever, recall_report
from factlens.retrieval.pairs import pair_row, run_name
from factlens.retrieval.retriever import SENT_RANKERS
from factlens.utils import progress, write_jsonl


def main() -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--limit", type=int, default=None, help="Only the first N claims")
    p.add_argument("--sent-ranker", choices=SENT_RANKERS, default=None, help="Overrides config")
    p.add_argument("--no-title-match", action="store_true", help="BM25 pages only (R0)")
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    val_ids = Path(paths["splits_dir"]) / "val_ids.txt"
    if args.split != "test" and not val_ids.exists():
        raise SystemExit(f"{val_ids} is missing. Run scripts/01_make_splits.py first.")
    overrides = {}
    if args.sent_ranker:
        overrides["sent_ranker"] = args.sent_ranker
    if args.no_title_match:
        overrides["use_title_match"] = False
    retriever = Retriever.from_config(cfg, **overrides)

    claims = load_split(cfg, args.split, args.limit)
    name = run_name(retriever.sent_ranker, retriever.use_title_match, args.limit)
    print(
        f"{len(claims):,} {args.split} claims, ranker={retriever.sent_ranker}, "
        f"title_match={retriever.use_title_match}"
    )

    start = time.time()
    rows, pages, sents = [], [], []
    for c in progress(claims, desc="retrieving", unit="claim"):
        res = retriever.retrieve(c.claim)
        pages.append(res.pages)
        sents.append([e.key for e in res.sentences])
        rows.append(pair_row(c, res))
    minutes = (time.time() - start) / 60

    pairs_path = Path(paths["pairs_dir"]) / f"retrieved_{args.split}_{name}.jsonl"
    write_jsonl(pairs_path, rows)
    print(f"Wrote {pairs_path}")

    metrics = recall_report(claims, pages, sents, sent_k=retriever.sent_top_k)
    metrics.update(
        split=args.split,
        claims=len(claims),
        sent_ranker=retriever.sent_ranker,
        use_title_match=retriever.use_title_match,
        minutes=round(minutes, 1),
    )
    save_json(Path(paths["outputs_dir"]) / "retrieval" / f"{args.split}_{name}.json", metrics)
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    retriever.store.close()


if __name__ == "__main__":
    main()
