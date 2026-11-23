"""Samples model errors with their evidence into a Markdown sheet for manual error analysis.

Example:
    python scripts/11_sample_errors.py --errors outputs/eval/E7_hybrid_errors.jsonl --pairs data/pairs/retrieved_test_hybrid.jsonl --n 30
"""

import random
from collections import Counter
from pathlib import Path

from _common import base_parser

from factlens.config import load_config
from factlens.data.wiki import title_to_text
from factlens.utils import read_jsonl


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--errors", required=True, help="An errors JSONL file written by scripts/07")
    p.add_argument("--pairs", required=True, help="The pair file the model was evaluated on")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = load_config(args.config)
    errors = list(read_jsonl(args.errors))
    pairs = {r["id"]: r for r in read_jsonl(args.pairs)}
    print("errors:", len(errors))
    print("by reason:", dict(Counter(e["reason"] for e in errors)))
    print("by gold -> pred:", dict(Counter(f"{e['gold']} -> {e['pred']}" for e in errors).most_common()))

    rng = random.Random(args.seed)
    sample = rng.sample(errors, min(args.n, len(errors)))
    lines = [f"# Error sample: {Path(args.errors).stem}", "",
             "Categories: retrieval miss, numbers or dates, polarity, NEI confusion, label noise, other.", ""]
    for i, e in enumerate(sample, 1):
        lines += [f"## {i}. {e['claim']}", "",
                  f"Gold {e['gold']}, predicted {e['pred']}, bucket {e['reason']}", ""]
        for page, line, text in pairs.get(e["id"], {}).get("evidence", [])[:3]:
            lines.append(f"* {title_to_text(page)} ({line}): {text}")
        lines += ["", "Category:", "", "Note:", ""]
    out = Path(cfg["paths"]["outputs_dir"]) / "eval" / f"{Path(args.errors).stem}_sample.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
