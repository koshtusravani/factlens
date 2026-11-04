"""Plain BM25 for ranking the small pool of sentences pulled from the top pages."""

from __future__ import annotations

import math
from collections import Counter


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg_len = sum(self.lens) / len(docs) if docs else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        # +1 inside the log keeps idf positive for very common words
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, length in zip(self.tfs, self.lens, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
            s = 0.0
            for t in query:
                if t in tf:
                    s += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
            out.append(s)
        return out
