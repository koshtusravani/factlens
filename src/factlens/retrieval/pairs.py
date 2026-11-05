"""Converts retrieval results into the pair-file rows that the models read."""

from __future__ import annotations

from factlens.data.fever import Claim
from factlens.retrieval.retriever import RetrievalResult

RETRIEVED = "retrieved"


def pair_row(claim: Claim, result: RetrievalResult) -> dict:
    """Returns one pair-file row for a claim. Evidence is raw sentence text, best first."""
    return {
        "id": claim.id,
        "claim": claim.claim,
        "label": claim.label,
        "evidence": [[e.page_id, e.line, e.text] for e in result.sentences],
        "evidence_source": RETRIEVED,
        "retrieved_pages": list(result.pages),
    }


def run_name(sent_ranker: str, use_title_match: bool = True, limit: int | None = None) -> str:
    """Returns the file-name stem for a run, for example bm25, bm25_notitle, or bm25_n2000."""
    name = sent_ranker
    if not use_title_match:
        name += "_notitle"
    if limit:
        name += f"_n{limit}"
    return name
