"""Fine-tunes DeBERTa-v3-small on claim-evidence pairs. Intended for a Kaggle or Colab GPU.

Examples:
    python scripts/06_train_transformer.py --train PAIRS/gold_train_nei-retrieved.jsonl --val PAIRS/gold_val_nei-retrieved.jsonl --out checkpoints/E5 --subsample 30000 --epochs 1
    python scripts/06_train_transformer.py --train PAIRS/gold_train_nei-retrieved.jsonl --val PAIRS/gold_val_nei-retrieved.jsonl --out checkpoints/E4 --mode claim_only
    python scripts/06_train_transformer.py ... --base cross-encoder/nli-deberta-v3-small   (MNLI initialization ablation)

Checkpoints trained on FEVER-NLI (model names containing "fever") have seen FEVER claims and
would leak test information, so they are never used as a starting point.
"""

import random

from _common import base_parser

from factlens.config import load_config
from factlens.models.train import train
from factlens.utils import read_jsonl


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--train", required=True, nargs="+", help="One or more pair files, concatenated")
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["claim_only", "evidence"], default="evidence")
    p.add_argument("--base", default=None, help="Overrides model.base in the config")
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--subsample", type=int, default=None, help="Random subset of train for fast first runs")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--push-to-hub", default=None, help="Hub repo id, for example user/factlens-e5")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["model"]["epochs"] = args.epochs
    train_pairs = [row for path in args.train for row in read_jsonl(path)]
    val_pairs = list(read_jsonl(args.val))
    if args.subsample and args.subsample < len(train_pairs):
        train_pairs = random.Random(cfg["model"]["seed"]).sample(train_pairs, args.subsample)
    info = train(train_pairs, val_pairs, args.out, cfg, args.mode, args.class_weights, args.base)
    print(info["val_metrics"])
    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, exist_ok=True, private=True)
        api.upload_folder(folder_path=args.out, repo_id=args.push_to_hub,
                          ignore_patterns=["checkpoint-*", "*.pt"])
        print(f"Pushed to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
