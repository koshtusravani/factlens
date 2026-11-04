"""Evidence retrieval: SQLite store, title match + BM25 pages, bm25/dense/hybrid sentences."""

from factlens.retrieval.metrics import doc_recall_at_k, recall_report, sent_recall_at_k
from factlens.retrieval.nei import sample_nei_evidence
from factlens.retrieval.retriever import Evidence, RetrievalResult, Retriever
from factlens.retrieval.store import WikiStore, build_wiki_db

__all__ = [
    "Evidence",
    "RetrievalResult",
    "Retriever",
    "WikiStore",
    "build_wiki_db",
    "doc_recall_at_k",
    "recall_report",
    "sample_nei_evidence",
    "sent_recall_at_k",
]
