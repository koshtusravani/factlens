"""Runs the retriever on a split and writes retrieved pairs plus recall metrics.

Usage (from the repo root):
    python scripts/03_run_retrieval.py --split val --limit 2000                    # R1 on val 2000
    python scripts/03_run_retrieval.py --split val --limit 2000 --no-title-match   # R0 ablation
    python scripts/03_run_retrieval.py --split test --sent-ranker hybrid           # R3 on test

Long runs can be split into shards that run side by side, then joined:
    python scripts/03_run_retrieval.py --split train --shard 1/4    # and 2/4, 3/4, 4/4
    python scripts/03_run_retrieval.py --split train --merge-shards 4

Writes data/pairs/retrieved_{split}_{name}.jsonl (one row per claim) and
outputs/retrieval/{split}_{name}.json, where name is the ranker plus optional
suffixes such as _notitle, _n2000 and _shard1of4.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from _common import base_parser, load_split, save_json

from factlens.config import load_config
from factlens.retrieval import Retriever, recall_report
from factlens.retrieval.pairs import pair_row, parse_shard, run_name, shard_bounds
from factlens.retrieval.retriever import SENT_RANKERS
from factlens.utils import progress, read_jsonl, write_jsonl


def report(metrics: dict, args, retriever_info: dict, n_claims: int, minutes: float, path: Path):
    metrics.update(
        split=args.split,
        claims=n_claims,
        **retriever_info,
        minutes=round(minutes, 1),
    )
    save_json(path, metrics)
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


def merge(args, cfg: dict, ranker: str, use_title_match: bool, sent_k: int) -> None:
    paths = cfg["paths"]
    n = args.merge_shards
    claims = load_split(cfg, args.split, args.limit)
    name = run_name(ranker, use_title_match, args.limit)
    pairs_dir = Path(paths["pairs_dir"])
    metrics_dir = Path(paths["outputs_dir"]) / "retrieval"

    rows, minutes = [], 0.0
    for i in range(1, n + 1):
        shard_name = run_name(ranker, use_title_match, args.limit, (i, n))
        shard_path = pairs_dir / f"retrieved_{args.split}_{shard_name}.jsonl"
        if not shard_path.exists():
            raise SystemExit(f"{shard_path} is missing. Run --shard {i}/{n} first.")
        rows.extend(read_jsonl(shard_path))
        shard_metrics = metrics_dir / f"{args.split}_{shard_name}.json"
        if shard_metrics.exists():
            minutes += json.loads(shard_metrics.read_text())["minutes"]

    # the shards must line up with the split, claim for claim
    if [r["id"] for r in rows] != [c.id for c in claims]:
        raise SystemExit(
            f"Shard rows ({len(rows):,}) don't match the {len(claims):,} {args.split} claims. "
            "Were the shards run with the same settings?"
        )

    pairs_path = pairs_dir / f"retrieved_{args.split}_{name}.jsonl"
    write_jsonl(pairs_path, rows)
    print(f"Wrote {pairs_path} from {n} shards")

    pages = [r["retrieved_pages"] for r in rows]
    sents = [[(p, line) for p, line, _ in r["evidence"]] for r in rows]
    metrics = recall_report(claims, pages, sents, sent_k=sent_k)
    info = dict(sent_ranker=ranker, use_title_match=use_title_match, shards=n)
    # minutes here is the sum over shards, not wall-clock time
    report(metrics, args, info, len(claims), minutes, metrics_dir / f"{args.split}_{name}.json")


def main() -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--limit", type=int, default=None, help="Only the first N claims")
    p.add_argument("--sent-ranker", choices=SENT_RANKERS, default=None, help="Overrides config")
    p.add_argument("--no-title-match", action="store_true", help="BM25 pages only (R0)")
    p.add_argument("--shard", default=None, help="Run one piece of the split, e.g. 2/4")
    p.add_argument(
        "--merge-shards",
        type=int,
        default=None,
        metavar="N",
        help="Join the N shard files into one pair file and score it",
    )
    args = p.parse_args()
    if args.shard and args.merge_shards:
        raise SystemExit("Use --shard or --merge-shards, not both.")
    try:
        shard = parse_shard(args.shard) if args.shard else None
    except ValueError as e:
        raise SystemExit(str(e)) from None

    cfg = load_config(args.config)
    paths = cfg["paths"]
    val_ids = Path(paths["splits_dir"]) / "val_ids.txt"
    if args.split != "test" and not val_ids.exists():
        raise SystemExit(f"{val_ids} is missing. Run scripts/01_make_splits.py first.")
    r = cfg["retrieval"]
    ranker = args.sent_ranker or r.get("sent_ranker", "bm25")
    use_title_match = r["use_title_match"] and not args.no_title_match

    if args.merge_shards:
        merge(args, cfg, ranker, use_title_match, r["sent_top_k"])
        return

    overrides = {"sent_ranker": ranker, "use_title_match": use_title_match}
    retriever = Retriever.from_config(cfg, **overrides)

    claims = load_split(cfg, args.split, args.limit)
    total = len(claims)
    if shard:
        start, end = shard_bounds(total, shard)
        claims = claims[start:end]
    name = run_name(retriever.sent_ranker, retriever.use_title_match, args.limit, shard)
    where = f" (shard {shard[0]}/{shard[1]} of {total:,})" if shard else ""
    print(
        f"{len(claims):,} {args.split} claims{where}, ranker={retriever.sent_ranker}, "
        f"title_match={retriever.use_title_match}"
    )

    start_time = time.time()
    rows, pages, sents = [], [], []
    for c in progress(claims, desc="retrieving", unit="claim"):
        res = retriever.retrieve(c.claim)
        pages.append(res.pages)
        sents.append([e.key for e in res.sentences])
        rows.append(pair_row(c, res))
    minutes = (time.time() - start_time) / 60

    pairs_path = Path(paths["pairs_dir"]) / f"retrieved_{args.split}_{name}.jsonl"
    write_jsonl(pairs_path, rows)
    print(f"Wrote {pairs_path}")

    metrics = recall_report(claims, pages, sents, sent_k=retriever.sent_top_k)
    info = dict(sent_ranker=retriever.sent_ranker, use_title_match=retriever.use_title_match)
    path = Path(paths["outputs_dir"]) / "retrieval" / f"{args.split}_{name}.json"
    report(metrics, args, info, len(claims), minutes, path)
    retriever.store.close()


if __name__ == "__main__":
    main()
