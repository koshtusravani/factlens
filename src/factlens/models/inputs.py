"""Claim-evidence pair construction for baselines, transformer training, and inference.

Pairs follow the team schema:
    {"id", "claim", "label", "evidence": [[page_id, line, sentence], ...], "evidence_source"}
evidence_source records how the evidence was obtained: gold, nei_retrieved, or nei_random.
"""

from __future__ import annotations

import random

from factlens.data.fever import Claim, gold_evidence_keys
from factlens.data.wiki import format_evidence
from factlens.retrieval import Retriever, WikiStore, sample_nei_evidence
from factlens.utils import progress

EVIDENCE_SEP = " | "
INPUT_MODES = ("claim_only", "evidence")


def evidence_text(evidence: list, max_sentences: int = 5) -> str:
    """Joins title-prefixed evidence sentences into a single string."""
    return EVIDENCE_SEP.join(format_evidence(p, t) for p, _, t in evidence[:max_sentences])


def model_inputs(pair: dict, mode: str) -> tuple[str, str | None]:
    """Returns (text_a, text_b) for a tokenizer. Claim-only mode drops the evidence."""
    if mode == "claim_only":
        return pair["claim"], None
    if mode == "evidence":
        return pair["claim"], evidence_text(pair["evidence"]) or "[NO EVIDENCE]"
    raise ValueError(f"Unknown input mode {mode!r}; expected one of {INPUT_MODES}")


def gold_pair(claim: Claim, store: WikiStore, retriever: Retriever, nei_strategy: str,
              num_sentences: int, seed: int) -> dict | None:
    """Builds one pair from gold evidence, or from sampled evidence for an NEI claim.

    Returns None when a gold sentence is missing from the store, so the caller can count skips.
    NEI sampling uses a per-claim generator, random.Random(seed + claim.id), which keeps samples
    identical when subsets or ordering change.
    """
    if claim.verifiable:
        evidence = []
        for page_id, line in gold_evidence_keys(claim, num_sentences):
            text = store.sentence(page_id, line)
            if text is None:
                return None
            evidence.append([page_id, line, text])
        source = "gold"
    else:
        rng = random.Random(seed + claim.id)
        sampled = sample_nei_evidence(claim.claim, retriever, strategy=nei_strategy,
                                      num_sentences=num_sentences, rng=rng)
        evidence = [[e.page_id, e.line, e.text] for e in sampled]
        source = f"nei_{nei_strategy}"
    return {"id": claim.id, "claim": claim.claim, "label": claim.label,
            "evidence": evidence, "evidence_source": source}


def build_gold_pairs(claims: list[Claim], store: WikiStore, retriever: Retriever,
                     nei_strategy: str, num_sentences: int, seed: int) -> tuple[list[dict], int]:
    """Builds gold pairs for many claims and returns (pairs, number of skipped claims)."""
    pairs, skipped = [], 0
    for c in progress(claims, desc="gold pairs", unit="claim"):
        pair = gold_pair(c, store, retriever, nei_strategy, num_sentences, seed)
        if pair is None:
            skipped += 1
        else:
            pairs.append(pair)
    return pairs, skipped