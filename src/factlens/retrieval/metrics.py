"""Retrieval recall for the R0 to R4 rows in docs/experiments.md.

A claim counts as a hit only when one full gold evidence set is covered, the same rule
the FEVER score uses. NEI claims have no gold evidence, so they are left out.
"""

from __future__ import annotations

from collections.abc import Sequence

from factlens.data.fever import Claim, EvidenceKey


def _recall(claims: Sequence[Claim], hit) -> float:
    scored = [c for c in claims if c.verifiable and c.evidence_sets]
    if not scored:
        return 0.0
    return sum(hit(c) for c in scored) / len(scored)


def doc_recall_at_k(claims: Sequence[Claim], pages: Sequence[Sequence[str]], k: int) -> float:
    """Share of claims where every page of some gold set is in the top k pages."""
    preds = {c.id: set(p[:k]) for c, p in zip(claims, pages, strict=True)}
    return _recall(
        claims, lambda c: any({pid for pid, _ in s} <= preds[c.id] for s in c.evidence_sets)
    )


def sent_recall_at_k(
    claims: Sequence[Claim], sentences: Sequence[Sequence[EvidenceKey]], k: int
) -> float:
    """Share of claims where some full gold set is in the top k sentences."""
    preds = {c.id: set(map(tuple, s[:k])) for c, s in zip(claims, sentences, strict=True)}
    return _recall(claims, lambda c: any(set(s) <= preds[c.id] for s in c.evidence_sets))


def recall_report(
    claims: Sequence[Claim],
    pages: Sequence[Sequence[str]],
    sentences: Sequence[Sequence[EvidenceKey]],
    doc_ks: Sequence[int] = (1, 3, 5),
    sent_k: int = 5,
) -> dict[str, float]:
    """All the numbers for one experiments.md row."""
    out = {f"doc_recall@{k}": doc_recall_at_k(claims, pages, k) for k in doc_ks}
    out[f"sent_recall@{sent_k}"] = sent_recall_at_k(claims, sentences, sent_k)
    return out
