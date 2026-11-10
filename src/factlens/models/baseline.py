"""TF-IDF and Logistic Regression baselines, claim-only and claim plus evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from factlens.models.inputs import evidence_text


class TfidfBaseline:
    """Separate TF-IDF spaces for claim and evidence, concatenated, then a linear classifier.

    Separate vocabularies let the model weigh claim artifacts (for example "not" or "only") and
    evidence terms independently, which keeps the claim-only comparison clean.
    """

    def __init__(self, mode: str = "evidence", max_features: int = 100_000, C: float = 4.0,
                 min_df: int = 2, seed: int = 42):
        if mode not in ("claim_only", "evidence"):
            raise ValueError(f"Unknown mode {mode!r}")
        self.mode = mode
        self.claim_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df,
                                         max_features=max_features, sublinear_tf=True)
        self.ev_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df,
                                      max_features=max_features, sublinear_tf=True)
        self.clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)

    def _features(self, pairs: list[dict], fit: bool):
        claims = [p["claim"] for p in pairs]
        xc = self.claim_vec.fit_transform(claims) if fit else self.claim_vec.transform(claims)
        if self.mode == "claim_only":
            return xc
        evs = [evidence_text(p["evidence"]) or "noevidence" for p in pairs]
        xe = self.ev_vec.fit_transform(evs) if fit else self.ev_vec.transform(evs)
        return hstack([xc, xe]).tocsr()

    def fit(self, pairs: list[dict]) -> TfidfBaseline:
        """Fits the vectorizers and the classifier on training pairs."""
        self.clf.fit(self._features(pairs, fit=True), [p["label"] for p in pairs])
        return self

    def predict(self, pairs: list[dict]) -> list[str]:
        """Returns one predicted label per pair."""
        return list(self.clf.predict(self._features(pairs, fit=False)))

    def top_claim_features(self, n: int = 15) -> dict[str, list[str]]:
        """Lists the claim n-grams with the largest weight per class, which exposes artifacts."""
        names = self.claim_vec.get_feature_names_out()
        out = {}
        for i, label in enumerate(self.clf.classes_):
            w = self.clf.coef_[i][: len(names)]
            out[str(label)] = [names[j] for j in np.argsort(-w)[:n]]
        return out

    def save(self, path: str | Path) -> None:
        """Saves the fitted model with joblib."""
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> TfidfBaseline:
        """Loads a model saved with save."""
        import joblib

        return joblib.load(path)