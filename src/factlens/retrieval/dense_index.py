"""Dense-only retrieval with FAISS over a subcorpus (R4).

No title match and no BM25: every sentence in the subcorpus is encoded once, and each
claim is searched against all of them. Pages are ranked by their best sentence.
The full dump (25M sentences) is too big to encode on a laptop, hence the subcorpus.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

import numpy as np

from factlens.data.fever import Claim
from factlens.data.wiki import format_evidence
from factlens.retrieval.retriever import Encoder, Evidence, RetrievalResult
from factlens.retrieval.store import WikiStore, nfc
from factlens.utils import progress


def subcorpus_pages(
    claims: Sequence[Claim],
    store: WikiStore,
    extra_pages: Iterable[str] = (),
    n_random: int = 0,
    seed: int = 42,
) -> list[str]:
    """Gold pages of the claims, plus extra pages (for example what R3 retrieved, as hard
    distractors), plus n_random random pages from the dump."""
    pages = {nfc(pid) for c in claims for s in c.evidence_sets for pid, _ in s}
    pages.update(nfc(p) for p in extra_pages)
    pages.update(store.random_pages(n_random, random.Random(seed)))
    return sorted(pages)


class DenseIndex:
    def __init__(self, rows: list[tuple[str, int, str]], vectors: np.ndarray):
        import faiss  # not in CI, so import only when used

        # torch and faiss each bring their own OpenMP on macOS, and a threaded faiss
        # search then crashes. One thread is still about 1 s for 2,000 claims
        faiss.omp_set_num_threads(1)
        self.rows = rows
        # vectors are L2-normalized, so inner product is cosine similarity
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    @classmethod
    def build(
        cls, store: WikiStore, pages: Iterable[str], encode: Encoder, chunk: int = 8192
    ) -> DenseIndex:
        rows = [(p, line, text) for p in pages for line, text in store.sentences(p)]
        texts = [format_evidence(p, t) for p, _, t in rows]  # "Title: sentence", as in R2/R3
        parts = [
            encode(texts[i : i + chunk])
            for i in progress(range(0, len(texts), chunk), desc="encoding", unit="chunk")
        ]
        return cls(rows, np.vstack(parts))

    def __len__(self) -> int:
        return len(self.rows)

    def search(
        self,
        claims: list[str],
        encode: Encoder,
        doc_k: int = 5,
        sent_k: int = 5,
        pool: int = 200,
    ) -> list[RetrievalResult]:
        """Top sent_k sentences per claim, and the first doc_k distinct pages among the
        top pool sentences."""
        scores, ids = self.index.search(encode(claims), min(pool, len(self.rows)))
        out = []
        for row_scores, row_ids in zip(scores, ids, strict=True):
            ev = [
                Evidence(*self.rows[i], score=float(s))
                for s, i in zip(row_scores, row_ids, strict=True)
                if i >= 0
            ]
            pages = list(dict.fromkeys(e.page_id for e in ev))[:doc_k]
            out.append(RetrievalResult(pages=pages, sentences=ev[:sent_k]))
        return out
