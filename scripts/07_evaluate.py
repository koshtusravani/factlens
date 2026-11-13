"""Evaluates a verifier on a pair file: classification metrics, FEVER score, and error buckets.

Pass a transformer folder or Hub ID, or a TF-IDF baseline as tfidf:PATH_TO_JOBLIB.

Example:
    python scripts/07_evaluate.py --model tfidf:checkpoints/E2_tfidf_gold_nei-random.joblib --pairs data/pairs/retrieved_test_bm25.jsonl --name E3_tfidf_gold_to_retrieved
"""

from pathlib import Path

from _common import base_parser, load_split, save_json

from factlens.config import load_config
from factlens.evaluation.error_analysis import bucket_errors, summarize
from factlens.evaluation.metrics import classification_report, fever_score
from factlens.utils import read_jsonl, write_jsonl


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"],
                   help="Split the pairs came from, used to look up gold evidence")
    p.add_argument("--mode", choices=["claim_only", "evidence"], default="evidence")
    p.add_argument("--name", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    pairs = list(read_jsonl(args.pairs))
    if args.model.startswith("tfidf:"):
        from factlens.models.baseline import TfidfBaseline

        labels = TfidfBaseline.load(args.model.split(":", 1)[1]).predict(pairs)
    else:
        from factlens.models.verifier import Verifier

        v = Verifier(args.model, mode=args.mode, max_length=cfg["model"]["max_length"])
        labels = [lab for lab, _ in v.predict_pairs(pairs)]

    claims_by_id = {c.id: c for c in load_split(cfg, args.split)}
    claims = [claims_by_id[p["id"]] for p in pairs]
    evidence = [[(e[0], e[1]) for e in p["evidence"]] for p in pairs]
    sources = sorted({p.get("evidence_source", "?") for p in pairs})

    report = classification_report([c.label for c in claims], labels)
    report["fever"] = fever_score(claims, labels, evidence)
    if "gold" in sources:
        report["fever"]["note"] = "Gold evidence supplied, so this FEVER score is an oracle upper bound."
    errors = bucket_errors(claims, labels, evidence)
    report["errors"] = summarize(errors)
    report.update({"name": args.name, "model": args.model, "pairs": args.pairs,
                   "evidence_sources": sources})

    out = Path(cfg["paths"]["outputs_dir"]) / "eval"
    save_json(out / f"{args.name}.json", report)
    write_jsonl(out / f"{args.name}_errors.jsonl", errors)
    print(f"{args.name}: acc={report['accuracy']:.4f} macro_f1={report['macro_f1']:.4f} "
          f"fever={report['fever']['fever_score']:.4f}")


if __name__ == "__main__":
    main()