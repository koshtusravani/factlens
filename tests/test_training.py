"""Tests for training utilities that need neither torch nor a model download."""

from factlens.labels import LABEL2ID, NEI, REFUTES, SUPPORTS
from factlens.models.train import PairDataset, class_weights


def test_class_weights_upweight_rare_classes():
    pairs = [{"label": SUPPORTS}] * 6 + [{"label": REFUTES}] * 2 + [{"label": NEI}] * 2
    w = class_weights(pairs)
    assert abs(sum(w) / 3 - 1.0) < 1e-9
    assert w[0] < w[1] and w[1] == w[2]


def test_pair_dataset_builds_title_prefixed_inputs():
    calls = []

    def fake_tokenizer(a, b, truncation, max_length):
        calls.append((a, b, truncation, max_length))
        return {"input_ids": [1, 2, 3]}

    pairs = [{"claim": "c", "label": REFUTES, "evidence": [["Paris", 0, "Paris is a city."]]}]
    ds = PairDataset(pairs, fake_tokenizer, "evidence", 64)
    item = ds[0]
    assert len(ds) == 1 and item["labels"] == LABEL2ID[REFUTES]
    assert calls[0] == ("c", "Paris: Paris is a city.", "only_second", 64)