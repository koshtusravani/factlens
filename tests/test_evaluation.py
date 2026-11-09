"""Tests for the FEVER score, classification metrics, error buckets, and explanations."""

from factlens.evaluation.error_analysis import bucket_errors, summarize
from factlens.evaluation.metrics import classification_report, fever_score
from factlens.explain.templates import explain
from factlens.labels import NEI, REFUTES, SUPPORTS


def _claim(claims, cid):
    return next(c for c in claims if c.id == cid)


def test_fever_score_requires_a_full_evidence_set(claims):
    c2 = _claim(claims, 2)
    assert fever_score([c2], [SUPPORTS], [[("Barack_Obama", 1)]])["fever_score"] == 1.0
    r = fever_score([c2], [SUPPORTS], [[("Hawaii", 1)]])
    assert r["label_accuracy"] == 1.0 and r["fever_score"] == 0.0


def test_fever_score_nei_needs_only_the_label(claims):
    c7 = _claim(claims, 7)
    assert fever_score([c7], [NEI], [[]])["fever_score"] == 1.0
    assert fever_score([c7], [REFUTES], [[]])["fever_score"] == 0.0


def test_fever_score_counts_only_the_first_five(claims):
    c4 = _claim(claims, 4)
    ev = [("Paris", i) for i in range(5)] + [("Eiffel_Tower", 2)]
    assert fever_score([c4], [REFUTES], [ev])["fever_score"] == 0.0


def test_classification_report_perfect(claims):
    gold = [c.label for c in claims]
    r = classification_report(gold, gold)
    assert r["accuracy"] == 1.0 and r["macro_f1"] == 1.0
    assert sum(map(sum, r["confusion_matrix"])) == len(claims)


def test_error_buckets(claims):
    c4, c7 = _claim(claims, 4), _claim(claims, 7)
    rows = bucket_errors([c4, c4, c7], [SUPPORTS, SUPPORTS, SUPPORTS],
                         [[("Eiffel_Tower", 2)], [], []])
    assert [r["reason"] for r in rows] == ["polarity", "retrieval_miss", "nei_confusion"]
    assert summarize(rows)["n_errors"] == 3


def test_explanations_quote_evidence():
    ev = [{"page_id": "Eiffel_Tower", "text": "Constructed from 1887 to 1889."}]
    assert "contradicts" in explain(REFUTES, 0.9, ev)
    assert "agrees" in explain(SUPPORTS, 0.9, ev)
    assert "neither" in explain(NEI, 0.9, ev)
    assert "caution" in explain(SUPPORTS, 0.4, ev)
    assert "No relevant evidence" in explain(NEI, 0.9, [])
    assert '"Eiffel Tower"' in explain(SUPPORTS, 0.9, ev)