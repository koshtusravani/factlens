"""Tests for the retrieval package, run on the tiny fixture dump. No model downloads."""

import json
import random
import re
import shutil
import unicodedata
import zlib
from pathlib import Path

import numpy as np
import pytest

from factlens.retrieval import (
    Retriever,
    WikiStore,
    build_wiki_db,
    doc_recall_at_k,
    recall_report,
    sample_nei_evidence,
    sent_recall_at_k,
)
from factlens.retrieval.bm25 import BM25
from factlens.retrieval.store import drop_stray_lines
from factlens.retrieval.text import title_candidates, title_key, tokenize

FIXTURES = Path(__file__).parent / "fixtures"


def fake_encoder(texts):
    # bag of words hashed into 64 dims; enough to test the dense code path offline
    vecs = np.zeros((len(texts), 64), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"\w+", t.lower()):
            vecs[i, zlib.crc32(w.encode()) % 64] += 1
    return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("wiki") / "wiki.db"
    build_wiki_db(FIXTURES / "wiki_mini.jsonl", path)
    return path


@pytest.fixture(scope="session")
def store(db_path):
    with WikiStore(db_path) as s:
        yield s


@pytest.fixture
def retriever(store):
    return Retriever(store, doc_top_k=3, sent_top_k=5, encoder=fake_encoder)


# text helpers


def test_title_key_drops_disambiguation():
    assert title_key("Eiffel Tower (film)") == title_key("Eiffel Tower") == "eiffel tower"


def test_title_candidates_longest_first():
    keys = [k for _, _, k in title_candidates("The Great Wall of China is visible from the Moon.")]
    assert keys[0] == "the great wall of china"
    assert "great wall of china" in keys and "moon" in keys
    assert "of" not in keys and "the" not in keys


def test_title_candidates_strip_possessive():
    keys = [k for _, _, k in title_candidates("Obama's wife is Michelle Obama.")]
    assert "obama" in keys and "michelle obama" in keys


def test_tokenize_drops_stopwords():
    assert tokenize("The Eiffel Tower is in Paris.") == ["eiffel", "tower", "paris"]


def test_bm25_prefers_matching_doc():
    bm = BM25([["eiffel", "tower"], ["moon", "satellite"]])
    a, b = bm.scores(["eiffel"])
    assert a > 0 and b == 0
    assert bm.scores([]) == [0.0, 0.0]


# store


def test_build_counts_and_skips_empty_page(store):
    assert store.count_pages() == 8  # the fixture's empty-id line is skipped
    assert store.count_sentences() == 17


def test_build_from_folder_skips_mac_junk(tmp_path):
    src = tmp_path / "wiki-pages"
    src.mkdir()
    shutil.copy(FIXTURES / "wiki_mini.jsonl", src / "wiki-001.jsonl")
    (src / "._wiki-001.jsonl").write_bytes(b"\x00\x05\x16\x07junk")
    stats = build_wiki_db(src, tmp_path / "wiki.db")
    assert stats == {"pages": 8, "sentences": 17, "dropped_lines": 0}
    assert not (tmp_path / "wiki.db.tmp").exists()


def test_stray_numbered_lines_are_dropped(tmp_path):
    # shaped like the real Duluth_Kelleys/Eskimos page
    lines = "0\tFirst .\n1927\t1927 in sports\n1\tSecond .\n1927\t1927 in sports\n2\t"
    assert drop_stray_lines(lines) == ("0\tFirst .\n1\tSecond .\n2\t", 2)
    src = tmp_path / "wiki.jsonl"
    src.write_text(json.dumps({"id": "Team", "text": "", "lines": lines}) + "\n")
    assert build_wiki_db(src, tmp_path / "wiki.db") == {
        "pages": 1,
        "sentences": 2,
        "dropped_lines": 2,
    }


def test_accented_ids_match_in_either_form(tmp_path):
    nfc_id, nfd_id = "Björk", unicodedata.normalize("NFD", "Björk")
    assert nfc_id != nfd_id
    lines = "0\tBjörk is an Icelandic singer .\n1\tShe was born in Reykjavík ."
    src = tmp_path / "wiki.jsonl"
    # write the page id in NFD, the build stores it as NFC
    src.write_text(json.dumps({"id": nfd_id, "text": "", "lines": lines}) + "\n", encoding="utf-8")
    db = tmp_path / "wiki.db"
    build_wiki_db(src, db)
    with WikiStore(db) as s:
        for pid in (nfc_id, nfd_id):
            assert s.sentence(pid, 1) == "She was born in Reykjavík."
            assert len(s.sentences(pid)) == 2
        r = Retriever(s, doc_top_k=1)
        claim_nfd = unicodedata.normalize("NFD", "Björk is a singer.")
        assert r.title_matches(claim_nfd) == [nfc_id]
        assert r.retrieve(claim_nfd).sentences[0].page_id == nfc_id


def test_store_lookups(store):
    assert store.sentence("Gustave_Eiffel", 1) == "He was born in Dijon in 1832."
    assert store.sentence("Gustave_Eiffel", 99) is None
    assert store.pages_by_title_key("eiffel tower") == [
        "Eiffel_Tower",
        "Eiffel_Tower_-LRB-film-RRB-",
    ]
    assert store.search("natural satellite", k=1)[0][0] == "Moon"
    assert store.search("the of and") == []  # only stopwords


def test_missing_db_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="02_build_wiki_db"):
        WikiStore(tmp_path / "nope.db")


def test_random_sentences_are_reproducible(store):
    a = store.random_sentences(4, random.Random(1))
    b = store.random_sentences(4, random.Random(1))
    assert a == b and len(set(a)) == 4
    assert all(store.sentence(pid, line) == text for pid, line, text in a)


# retriever


def test_title_match_finds_both_eiffel_pages(retriever):
    pages = retriever.retrieve_pages("The Eiffel Tower film won an award.")
    assert set(pages[:2]) == {"Eiffel_Tower", "Eiffel_Tower_-LRB-film-RRB-"}


def test_title_match_always_kept(store):
    # with one slot, an exact title match still gets it
    r = Retriever(store, doc_top_k=1)
    assert r.retrieve_pages("Paris is home to the Eiffel Tower.") == [
        "Eiffel_Tower"
    ]  # longest name first
    assert r.retrieve_pages("Hawaii is a state.") == ["Hawaii"]


def test_punctuation_titles_never_match():
    assert title_key(".abc") == "" and title_key("ABC") == "abc"


def test_score_pages_matches_search(store):
    hits = dict(store.search("natural satellite", k=3))
    assert store.score_pages("natural satellite", ["Moon"]) == {"Moon": hits["Moon"]}
    assert store.score_pages("natural satellite", []) == {}


def test_long_name_blocks_short_pieces(retriever):
    # "Barack Obama" matches, so "Barack" and "Obama" alone are not tried
    assert retriever.title_matches("Barack Obama was born in Hawaii.") == ["Barack_Obama", "Hawaii"]


def test_bm25_only_still_finds_pages(store):
    r = Retriever(store, doc_top_k=3, use_title_match=False)
    assert "Great_Wall_of_China" in r.retrieve_pages(
        "The Great Wall of China is visible from the Moon."
    )


@pytest.mark.parametrize("ranker", ["bm25", "dense", "hybrid"])
def test_sentence_rankers_find_gold(store, claims, ranker):
    r = Retriever(store, doc_top_k=3, sent_top_k=5, sent_ranker=ranker, encoder=fake_encoder)
    c = next(c for c in claims if c.id == 3)  # Gustave Eiffel was a French engineer.
    sents = r.retrieve(c.claim).sentences
    assert ("Gustave_Eiffel", 0) in [s.key for s in sents]
    assert len(sents) <= 5
    assert [s.score for s in sents] == sorted((s.score for s in sents), reverse=True)


def test_evidence_has_title_prefix(retriever):
    top = retriever.retrieve("Gustave Eiffel was a French engineer.").sentences[0]
    assert top.formatted.startswith("Gustave Eiffel: ")


def test_bad_ranker_raises(store):
    with pytest.raises(ValueError):
        Retriever(store, sent_ranker="magic")


def test_from_config(cfg, store):
    r = Retriever.from_config(cfg, store=store, doc_top_k=2)
    assert r.doc_top_k == 2 and r.sent_ranker == cfg["retrieval"]["sent_ranker"]


def test_full_recall_on_fixtures(retriever, claims):
    results = [retriever.retrieve(c.claim) for c in claims]
    pages = [r.pages for r in results]
    sents = [[s.key for s in r.sentences] for r in results]
    report = recall_report(claims, pages, sents, doc_ks=(3,))
    assert report["doc_recall@3"] == 1.0
    assert report["sent_recall@5"] == 1.0


# NEI sampling


def test_nei_retrieved_matches_retriever(retriever):
    claim = "Gustave Eiffel liked opera."
    got = sample_nei_evidence(claim, retriever, "retrieved", num_sentences=3)
    assert [e.key for e in got] == [e.key for e in retriever.retrieve(claim).sentences[:3]]


def test_nei_random_is_seeded(retriever):
    a = sample_nei_evidence("x", retriever, "random", 3, random.Random(7))
    b = sample_nei_evidence("x", retriever, "random", 3, random.Random(7))
    assert [e.key for e in a] == [e.key for e in b] and len(a) == 3


def test_nei_random_varies_across_claims(retriever):
    rng = random.Random(7)
    a = sample_nei_evidence("x", retriever, "random", 3, rng)
    b = sample_nei_evidence("y", retriever, "random", 3, rng)
    assert [e.key for e in a] != [e.key for e in b]


def test_nei_bad_strategy_or_missing_rng(retriever):
    with pytest.raises(ValueError):
        sample_nei_evidence("x", retriever, "nope")
    with pytest.raises(ValueError, match="rng"):
        sample_nei_evidence("x", retriever, "random")


# metrics


def test_recall_needs_full_evidence_set(claims):
    c2 = [c for c in claims if c.id == 2]  # sets: {Obama 1} or {Obama 1, Hawaii 1}
    assert doc_recall_at_k(c2, [["Hawaii", "Barack_Obama"]], k=1) == 0.0
    assert doc_recall_at_k(c2, [["Hawaii", "Barack_Obama"]], k=2) == 1.0
    assert sent_recall_at_k(c2, [[("Hawaii", 1)]], k=5) == 0.0
    assert sent_recall_at_k(c2, [[("Hawaii", 1), ("Barack_Obama", 1)]], k=5) == 1.0


def test_recall_ignores_nei(claims):
    nei = [c for c in claims if not c.verifiable]
    assert doc_recall_at_k(nei, [[] for _ in nei], k=5) == 0.0
    mixed = [c for c in claims if c.id in (1, 7)]
    assert doc_recall_at_k(mixed, [["Eiffel_Tower"], []], k=5) == 1.0
