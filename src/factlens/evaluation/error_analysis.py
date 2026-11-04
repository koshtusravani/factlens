"""Error analysis helpers that bucket mistakes by cause, so the report can explain failures."""

from __future__ import annotations

from collections import Counter

from factlens.data.fever import Claim
from factlens.labels import NEI

NEGATION_CUES = ("not", "never", "only", "no ", "didn't", "isn't", "wasn't", "failed", "refused")


def bucket_errors(claims: list[Claim], pred_labels: list[str],
                  pred_evidence: list[list[tuple[str, int]]]) -> list[dict]:
    """Tags each wrong prediction with its most likely cause.

    retrieval_miss   gold evidence never reached the classifier (a retrieval problem)
    nei_confusion    the model confused NEI with a verdict, or the reverse
    polarity         SUPPORTS and REFUTES were swapped while gold evidence was present
    """
    rows = []
    for c, label, ev in zip(claims, pred_labels, pred_evidence, strict=True):
        if label == c.label:
            continue
        ev_set = {(p, int(line)) for p, line in ev[:5]}
        has_gold = any(all(k in ev_set for k in s) for s in c.evidence_sets)
        if c.verifiable and not has_gold:
            reason = "retrieval_miss"
        elif NEI in (label, c.label):
            reason = "nei_confusion"
        else:
            reason = "polarity"
        rows.append({
            "id": c.id, "claim": c.claim, "gold": c.label, "pred": label, "reason": reason,
            "has_negation_cue": any(cue in f" {c.claim.lower()} " for cue in NEGATION_CUES),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    """Counts errors by cause and by gold and predicted label."""
    return {
        "n_errors": len(rows),
        "by_reason": dict(Counter(r["reason"] for r in rows)),
        "by_gold_pred": dict(Counter(f"{r['gold']} -> {r['pred']}" for r in rows)),
        "negation_cue_share": (sum(r["has_negation_cue"] for r in rows) / len(rows)) if rows else 0.0,
    }