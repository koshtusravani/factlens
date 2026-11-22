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


def run_name(
    sent_ranker: str,
    use_title_match: bool = True,
    limit: int | None = None,
    shard: tuple[int, int] | None = None,
) -> str:
    """Returns the file-name stem for a run, for example bm25, bm25_notitle, bm25_n2000,
    or hybrid_shard2of4."""
    name = sent_ranker
    if not use_title_match:
        name += "_notitle"
    if limit:
        name += f"_n{limit}"
    if shard:
        name += f"_shard{shard[0]}of{shard[1]}"
    return name


def parse_shard(text: str) -> tuple[int, int]:
    """'2/4' -> (2, 4). Shards are numbered from 1."""
    try:
        i, n = (int(x) for x in text.split("/"))
    except ValueError:
        raise ValueError(f"shard should look like 2/4, got {text!r}") from None
    if not 1 <= i <= n:
        raise ValueError(f"shard {i} is out of range for {n} shards")
    return i, n


def shard_bounds(n_items: int, shard: tuple[int, int]) -> tuple[int, int]:
    """Start and end of one contiguous shard, so the shard files joined in order
    give the same rows as a single run."""
    i, n = shard
    return n_items * (i - 1) // n, n_items * i // n
