"""R4: dense-only retrieval with FAISS over a subcorpus, scored like R0 to R3.

Usage (from the repo root, after the R3 val run):
    python scripts/08_faiss_retrieval.py --split val --limit 2000

The subcorpus is the gold pages of the claims, the pages R3 retrieved for them (hard
distractors), and --random-pages random pages. Gold pages are always in it, so R4 searches
an easier corpus than R1 to R3 do; the comparison is about ranking, not corpus size.

Writes data/pairs/retrieved_{split}_faiss_n{limit}.jsonl and
outputs/retrieval/{split}_faiss_n{limit}.json with the same metric names as scripts/03.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from _common import base_parser, load_split, save_json

from factlens.config import load_config
from factlens.retrieval import WikiStore, recall_report
from factlens.retrieval.dense_index import DenseIndex, subcorpus_pages
from factlens.retrieval.pairs import pair_row, run_name
from factlens.utils import read_jsonl, write_jsonl


def load_encoder(name: str, batch_size: int):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name)  # uses the Mac GPU (mps) when there is one
    print(f"encoder {name} on {model.device}")

    def encode(texts: list[str]) -> np.ndarray:
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    return encode


def main() -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--limit", type=int, default=2000, help="Only the first N claims")
    p.add_argument("--random-pages", type=int, default=20000)
    p.add_argument(
        "--distractors",
        default=None,
        help="Pair file whose retrieved pages join the subcorpus (default: the R3 run)",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    paths, r = cfg["paths"], cfg["retrieval"]
    pairs_dir = Path(paths["pairs_dir"])
    distractors = Path(
        args.distractors
        or pairs_dir / f"retrieved_{args.split}_{run_name('hybrid', True, args.limit)}.jsonl"
    )
    if not distractors.exists():
        raise SystemExit(
            f"{distractors} is missing. Run scripts/03_run_retrieval.py "
            f"--split {args.split} --limit {args.limit} --sent-ranker hybrid first."
        )

    claims = load_split(cfg, args.split, args.limit)
    store = WikiStore(paths["wiki_db"])
    extra = {pg for row in read_jsonl(distractors) for pg in row["retrieved_pages"]}
    pages = subcorpus_pages(claims, store, extra, args.random_pages, cfg["splits"]["seed"])
    print(f"{len(claims):,} {args.split} claims, subcorpus of {len(pages):,} pages")

    encode = load_encoder(r["encoder"], r["encode_batch_size"])
    t0 = time.time()
    index = DenseIndex.build(store, pages, encode)
    build_min = (time.time() - t0) / 60
    print(f"indexed {len(index):,} sentences in {build_min:.1f} min")

    t0 = time.time()
    results = index.search([c.claim for c in claims], encode, r["doc_top_k"], r["sent_top_k"])
    search_min = (time.time() - t0) / 60

    name = run_name("faiss", True, args.limit)
    pairs_path = pairs_dir / f"retrieved_{args.split}_{name}.jsonl"
    write_jsonl(pairs_path, [pair_row(c, res) for c, res in zip(claims, results, strict=True)])
    print(f"Wrote {pairs_path}")

    metrics = recall_report(
        claims,
        [res.pages for res in results],
        [[e.key for e in res.sentences] for res in results],
        sent_k=r["sent_top_k"],
    )
    metrics.update(
        split=args.split,
        claims=len(claims),
        sent_ranker="faiss",
        use_title_match=False,
        subcorpus_pages=len(pages),
        subcorpus_sentences=len(index),
        random_pages=args.random_pages,
        build_minutes=round(build_min, 1),
        minutes=round(search_min, 1),
    )
    save_json(Path(paths["outputs_dir"]) / "retrieval" / f"{args.split}_{name}.json", metrics)
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    store.close()


if __name__ == "__main__":
    main()
