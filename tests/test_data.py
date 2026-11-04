"""Unit tests for Wikipedia normalization, FEVER loading, and split generation."""

from factlens.data.fever import gold_evidence_keys
from factlens.data.splits import apply_split, stratified_val_ids
from factlens.data.wiki import format_evidence, normalize_text, parse_lines, title_to_text
from factlens.labels import NEI


def test_normalize_ptb_tokens():
    assert normalize_text("tower -LRB- French : tour -RRB- in Paris , France .") == \
        "tower (French: tour) in Paris, France."
    assert normalize_text("the 1889 World 's Fair") == "the 1889 World's Fair"


def test_title_to_text_handles_disambiguation():
    assert title_to_text("Eiffel_Tower_-LRB-film-RRB-") == "Eiffel Tower (film)"


def test_parse_lines_skips_empty_and_drops_links():
    lines = parse_lines("0\tFirst sentence .\tlinkA\n1\t\n2\tThird one .\tlinkB")
    assert lines == [(0, "First sentence."), (2, "Third one.")]


def test_format_evidence_prefixes_title():
    assert format_evidence("Barack_Obama", "He was born in Honolulu.") == \
        "Barack Obama: He was born in Honolulu."


def test_load_claims_dedupes_sets_and_handles_nei(claims):
    by_id = {c.id: c for c in claims}
    assert len(claims) == 9
    assert by_id[1].evidence_sets == [[("Eiffel_Tower", 0)]]  # Duplicate annotation collapsed
    assert by_id[2].evidence_sets[1] == [("Barack_Obama", 1), ("Hawaii", 1)]
    assert by_id[7].label == NEI and by_id[7].evidence_sets == [] and not by_id[7].verifiable


def test_gold_keys_keep_first_set_first(claims):
    c = next(c for c in claims if c.id == 2)
    assert gold_evidence_keys(c) == [("Barack_Obama", 1), ("Hawaii", 1)]
    assert gold_evidence_keys(c, max_sentences=1) == [("Barack_Obama", 1)]


def test_stratified_split_is_deterministic_and_balanced(claims):
    a = stratified_val_ids(claims, 1 / 3, seed=42)
    b = stratified_val_ids(claims, 1 / 3, seed=42)
    assert a == b and len(a) == 3
    labels = {c.label for c in apply_split(claims, a)}
    assert len(labels) == 3  # One claim per class
    assert len(apply_split(claims, a, exclude=True)) == 6