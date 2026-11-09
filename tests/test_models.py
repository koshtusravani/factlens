"""Tests for pair construction and the TF-IDF baseline, run on the fixture dump."""

from pathlib import Path

import pytest

from factlens.labels import LABELS
from factlens.models.baseline import TfidfBaseline
from factlens.models.inputs import build_gold_pairs, evidence_text, model_inputs
from factlens.retrieval import Retriever, WikiStore, build_wiki_db

FIXTURES = Path(__file__).parent / "fixtures"
PAIR_KEYS = {"id", "claim", "label", "evidence", "evidence_source"}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    db = tmp_path_factory.mktemp("models_db") / "wiki.db"
    build_wiki_db(FIXTURES / "wiki_mini.jsonl", db)
    s = WikiStore(db)
    yield s
    s.close()


@pytest.fixture(scope="module")
def retriever(store, cfg):
    return Retriever.from_config(cfg, store=store)


def test_model_inputs_modes():
    pair = {"claim": "c", "evidence": [["Paris", 0, "Paris is a city."]]}
    assert model_inputs(pair, "claim_only") == ("c", None)
    assert model_inputs(pair, "evidence") == ("c", "Paris: Paris is a city.")
    assert model_inputs({"claim": "c", "evidence": []}, "evidence")[1] == "[NO EVIDENCE]"
    with pytest.raises(ValueError):
        model_inputs(pair, "bogus")


def test_gold_pairs_follow_schema(claims, store, retriever):
    pairs, skipped = build_gold_pairs(claims, store, retriever, "retrieved", 5, seed=42)
    assert skipped == 0 and len(pairs) == 9
    assert all(set(p) == PAIR_KEYS for p in pairs)
    by_id = {p["id"]: p for p in pairs}
    assert by_id[4]["evidence_source"] == "gold"
    assert by_id[4]["evidence"][0][:2] == ["Eiffel_Tower", 2]
    assert by_id[4]["evidence"][0][2].startswith("Constructed from 1887 to 1889")
    assert by_id[7]["evidence_source"] == "nei_retrieved" and by_id[7]["evidence"]


def test_random_nei_is_reproducible_across_orderings(claims, store, retriever):
    nei = [c for c in claims if not c.verifiable]
    a, _ = build_gold_pairs(nei, store, retriever, "random", 3, seed=42)
    b, _ = build_gold_pairs(list(reversed(nei)), store, retriever, "random", 3, seed=42)
    assert {p["id"]: p["evidence"] for p in a} == {p["id"]: p["evidence"] for p in b}
    assert all(p["evidence_source"] == "nei_random" for p in a)


def test_tfidf_baseline_fits_and_predicts(claims, store, retriever):
    pairs, _ = build_gold_pairs(claims, store, retriever, "retrieved", 5, seed=42)
    for mode in ("claim_only", "evidence"):
        model = TfidfBaseline(mode=mode, min_df=1).fit(pairs)
        preds = model.predict(pairs)
        assert len(preds) == len(pairs) and set(preds) <= set(LABELS)
    assert set(model.top_claim_features(3)) == set(LABELS)
    assert "Paris" in evidence_text(pairs[0]["evidence"])