"""Builds indexes/wiki.db from the Wikipedia dump in data/wiki-pages.

Usage (from the repo root):
    python scripts/02_build_wiki_db.py
    python scripts/02_build_wiki_db.py --force                       # rebuild over an old db
    python scripts/02_build_wiki_db.py --config configs/smoke.yaml   # tiny fixture db
"""

from __future__ import annotations

import time
from pathlib import Path

from _common import base_parser

from factlens.config import load_config
from factlens.retrieval import build_wiki_db


def main() -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--force", action="store_true", help="Overwrite an existing wiki.db")
    args = p.parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    db_path = Path(paths["wiki_db"])

    if db_path.exists() and not args.force:
        print(f"{db_path} already exists. Use --force to rebuild it.")
        return

    start = time.time()
    stats = build_wiki_db(paths["wiki_dir"], db_path, cfg["retrieval"]["lead_sentences"])
    minutes = (time.time() - start) / 60
    size_gb = db_path.stat().st_size / 1e9

    # these go into docs/data_notes.md
    print(f"Pages indexed:     {stats['pages']:,}")
    print(f"Sentences indexed: {stats['sentences']:,}")
    print(f"Lines dropped:     {stats['dropped_lines']:,} (stray or out-of-order entries)")
    print(f"wiki.db size:      {size_gb:.2f} GB")
    print(f"Build time:        {minutes:.1f} min")


if __name__ == "__main__":
    main()
