"""Classification metrics and the FEVER score.

The FEVER score follows the official definition (Thorne et al., 2018): a prediction counts only
when the label is correct and, for SUPPORTS or REFUTES, at least one complete gold evidence set
appears among the first five predicted sentences.
"""

from __future__ import annotations

from factlens.data.fever import Claim
from factlens.labels import LABELS, NEI


def classification_report(y_true: list[str], y_pred: list[str]) -> dict:
    """Returns accuracy, macro and per-class precision, recall, F1, and the confusion matrix."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf),
        "per_class": {
            lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, lab in enumerate(LABELS)
        },
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def _covers(pred: list[tuple[str, int]], ev_set: list[tuple[str, int]]) -> bool:
    """Returns True when every sentence of a gold evidence set is among the predictions."""
    return all(item in pred for item in ev_set)


def fever_score(claims: list[Claim], pred_labels: list[str],
                pred_evidence: list[list[tuple[str, int]]], max_evidence: int = 5) -> dict:
    """Computes FEVER score, label accuracy, and evidence precision, recall, and F1."""
    assert len(claims) == len(pred_labels) == len(pred_evidence)
    strict = acc = 0
    prec_sum = prec_n = rec_sum = rec_n = 0.0
    for c, label, ev in zip(claims, pred_labels, pred_evidence, strict=True):
        ev = [(p, int(line)) for p, line in ev[:max_evidence]]
        correct_label = label == c.label
        acc += correct_label
        if c.label == NEI:
            strict += correct_label
            continue
        all_gold = {k for s in c.evidence_sets for k in s}
        covered = any(_covers(ev, s) for s in c.evidence_sets)
        strict += correct_label and covered
        prec_sum += (sum(k in all_gold for k in ev) / len(ev)) if ev else 1.0
        prec_n += 1
        rec_sum += 1.0 if covered else 0.0
        rec_n += 1
    n = len(claims) or 1
    precision = prec_sum / prec_n if prec_n else 0.0
    recall = rec_sum / rec_n if rec_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "fever_score": strict / n,
        "label_accuracy": acc / n,
        "evidence_precision": precision,
        "evidence_recall": recall,
        "evidence_f1": f1,
    }