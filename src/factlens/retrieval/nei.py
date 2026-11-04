"""Evidence for NOT ENOUGH INFO claims, which have no gold sentences.

retrieved: the retriever's top sentences (hard negatives, the default)
random:    random sentences from the whole dump (the ablation)
"""

from __future__ import annotations

import random

from factlens.retrieval.retriever import Evidence, Retriever

NEI_STRATEGIES = ("retrieved", "random")


def sample_nei_evidence(
    claim: str,
    retriever: Retriever,
    strategy: str = "retrieved",
    num_sentences: int = 5,
    rng: random.Random | None = None,
) -> list[Evidence]:
    if strategy == "retrieved":
        return retriever.retrieve(claim).sentences[:num_sentences]
    if strategy == "random":
        # make one rng and reuse it for every claim; a fresh Random(42) per call
        # would give every NEI claim the same sentences
        if rng is None:
            raise ValueError("random strategy needs rng=random.Random(seed), shared across claims")
        rows = retriever.store.random_sentences(num_sentences, rng)
        return [Evidence(pid, line, text) for pid, line, text in rows]
    raise ValueError(f"strategy must be one of {NEI_STRATEGIES}, got {strategy!r}")
