"""Trains a TF-IDF and Logistic Regression baseline and reports test metrics.

Example:
    python scripts/05_train_baselines.py --train data/pairs/gold_train_nei-retrieved.jsonl --test data/pairs/gold_test_nei-retrieved.jsonl --mode claim_only --name E1_tfidf_claim_only
"""

from pathlib import Path

from _common import base_parser, save_json

from factlens.config import load_config
from factlens.evaluation.metrics import classification_report
from factlens.models.baseline import TfidfBaseline
from factlens.utils import read_jsonl


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--mode", choices=["claim_only", "evidence"], default="evidence")
    p.add_argument("--name", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    train, test = list(read_jsonl(args.train)), list(read_jsonl(args.test))
    model = TfidfBaseline(mode=args.mode).fit(train)
    preds = model.predict(test)
    report = classification_report([p["label"] for p in test], preds)
    report.update({"name": args.name, "mode": args.mode, "train": args.train, "test": args.test,
                   "n_train": len(train), "n_test": len(test),
                   "top_claim_features": model.top_claim_features()})
    model.save(Path(cfg["paths"]["checkpoints_dir"]) / f"{args.name}.joblib")
    save_json(Path(cfg["paths"]["outputs_dir"]) / "baselines" / f"{args.name}.json", report)
    print(f"{args.name}: acc={report['accuracy']:.4f} macro_f1={report['macro_f1']:.4f}")


if __name__ == "__main__":
    main()