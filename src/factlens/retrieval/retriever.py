"""Two-stage retriever.

Stage 1 picks pages: exact title matches on names in the claim, then BM25 to fill the rest.
Stage 2 ranks the sentences inside those pages with bm25, dense, or hybrid scoring.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from factlens.data.wiki import format_evidence
from factlens.retrieval.bm25 import BM25
from factlens.retrieval.store import WikiStore, nfc
from factlens.retrieval.text import is_disambiguated, title_candidates, tokenize

SENT_RANKERS = ("bm25", "dense", "hybrid")

# takes a list of texts, returns one L2-normalized vector per text
Encoder = Callable[[list[str]], np.ndarray]


@dataclass
class Evidence:
    page_id: str
    line: int
    text: str
    score: float = 0.0

    @property
    def key(self) -> tuple[str, int]:
        return (self.page_id, self.line)

    @property
    def formatted(self) -> str:
        return format_evidence(self.page_id, self.text)


@dataclass
class RetrievalResult:
    pages: list[str]
    sentences: list[Evidence] = field(default_factory=list)


def _minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min() if len(x) else 0.0
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


class Retriever:
    def __init__(
        self,
        store: WikiStore,
        doc_top_k: int = 5,
        sent_top_k: int = 5,
        use_title_match: bool = True,
        title_weight: float = 2.0,
        lead_weight: float = 1.0,
        sent_ranker: str = "bm25",
        hybrid_alpha: float = 0.5,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encode_batch_size: int = 64,
        encoder: Encoder | None = None,
    ):
        if sent_ranker not in SENT_RANKERS:
            raise ValueError(f"sent_ranker must be one of {SENT_RANKERS}, got {sent_ranker!r}")
        self.store = store
        self.doc_top_k = doc_top_k
        self.sent_top_k = sent_top_k
        self.use_title_match = use_title_match
        self.title_weight = title_weight
        self.lead_weight = lead_weight
        self.sent_ranker = sent_ranker
        self.hybrid_alpha = hybrid_alpha
        self.encoder_name = encoder_name
        self.encode_batch_size = encode_batch_size
        self._encoder = encoder

    @classmethod
    def from_config(cls, cfg: dict, store: WikiStore | None = None, **overrides) -> Retriever:
        """Builds a retriever from the retrieval section of a config. Keyword args win."""
        r = cfg["retrieval"]
        kwargs = dict(
            doc_top_k=r["doc_top_k"],
            sent_top_k=r["sent_top_k"],
            use_title_match=r["use_title_match"],
            title_weight=r["bm25_title_weight"],
            lead_weight=r["bm25_lead_weight"],
            sent_ranker=r.get("sent_ranker", "bm25"),
            hybrid_alpha=r.get("hybrid_alpha", 0.5),
            encoder_name=r["encoder"],
            encode_batch_size=r["encode_batch_size"],
        )
        kwargs.update(overrides)
        return cls(store or WikiStore(cfg["paths"]["wiki_db"]), **kwargs)

    # stage 1: pages

    def title_matches(self, claim: str) -> list[str]:
        """Pages whose title appears in the claim. Once a long name matches, its
        shorter pieces are skipped (so no "Barack" page after "Barack Obama")."""
        taken: set[int] = set()
        found: list[str] = []
        for start, end, key in title_candidates(nfc(claim)):
            if taken.intersection(range(start, end)):
                continue
            pages = self.store.pages_by_title_key(key)
            if pages:
                taken.update(range(start, end))
                found.extend(p for p in pages if p not in found)
        return found

    def retrieve_pages(self, claim: str, k: int | None = None) -> list[str]:
        k = k or self.doc_top_k
        weights = dict(title_weight=self.title_weight, lead_weight=self.lead_weight)
        hits = self.store.search(claim, k=max(k * 2, 10), **weights)
        if not self.use_title_match:
            return [p for p, _ in hits[:k]]

        matched = self.title_matches(claim)
        scores = dict(hits)
        scores.update(
            self.store.score_pages(claim, [p for p in matched if p not in scores], **weights)
        )
        ranked = sorted(
            dict.fromkeys([p for p, _ in hits] + matched), key=lambda p: -scores.get(p, 0)
        )

        # exact title matches always get a slot, but the final order is by BM25 score,
        # so "ABC" (the TV network) doesn't push "ABC islands" down
        must = [p for p in matched if not is_disambiguated(p)][:k]
        rest = [p for p in ranked if p not in must][: k - len(must)]
        return sorted(must + rest, key=lambda p: -scores.get(p, 0))

    # stage 2: sentences

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer  # heavy, so load only when needed

            model = SentenceTransformer(self.encoder_name)
            self._encoder = lambda t: model.encode(
                t,
                batch_size=self.encode_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.asarray(self._encoder(texts), dtype=np.float32)

    def rank_sentences(self, claim: str, pages: list[str], k: int | None = None) -> list[Evidence]:
        k = k or self.sent_top_k
        pool = [Evidence(p, line, text) for p in pages for line, text in self.store.sentences(p)]
        if not pool:
            return []
        texts = [e.formatted for e in pool]  # "Title: sentence", same as the model sees

        if self.sent_ranker in ("bm25", "hybrid"):
            lex = np.array(BM25([tokenize(t) for t in texts]).scores(tokenize(claim)))
        if self.sent_ranker in ("dense", "hybrid"):
            vecs = self._encode([claim] + texts)
            dense = vecs[1:] @ vecs[0]

        if self.sent_ranker == "bm25":
            scores = lex
        elif self.sent_ranker == "dense":
            scores = dense
        else:
            a = self.hybrid_alpha
            scores = a * _minmax(dense) + (1 - a) * _minmax(lex)

        # stable sort, so ties keep page order and line order
        order = sorted(range(len(pool)), key=lambda i: -scores[i])
        out = []
        for i in order[:k]:
            pool[i].score = float(scores[i])
            out.append(pool[i])
        return out

    def retrieve(self, claim: str) -> RetrievalResult:
        claim = nfc(claim)
        pages = self.retrieve_pages(claim)
        return RetrievalResult(pages=pages, sentences=self.rank_sentences(claim, pages))
