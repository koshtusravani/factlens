"""Tokenizing and title-matching helpers shared by the store and the rankers."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does for from had has have he
    her him his how i if in into is it its may me more most my no nor not of on only or other
    our she so some such than that the their them then there these they this those to too us
    very was we were what when where which while who whom why will with would you your""".split()
)

# words that can sit inside a name, like "Great Wall of China"
CONNECTORS = frozenset("of the and de la le du des del von van da di".split())

_WORD = re.compile(r"\w+")
_CLAIM_TOKEN = re.compile(r"\w+(?:['’.-]\w+)*")
_DISAMBIG = re.compile(r"\s*\([^)]*\)\s*$")


def tokenize(text: str) -> list[str]:
    """Lowercase words with stopwords dropped."""
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]


def title_key(title: str) -> str:
    """Lookup key for a title: drop the "(film)" style suffix, lowercase, keep only words.

    Titles starting with punctuation (".abc", ".one") get no key, or they would match
    plain words in claims.
    """
    if not _WORD.match(title):
        return ""
    return " ".join(_WORD.findall(_DISAMBIG.sub("", title).lower()))


def is_disambiguated(page_id: str) -> bool:
    return page_id.endswith("-RRB-")


def title_candidates(claim: str, max_len: int = 6) -> list[tuple[int, int, str]]:
    """Capitalized n-grams in the claim that could be page titles.

    Returns (start, end, key) spans, longest first, so "Barack Obama" is
    tried before "Barack" or "Obama".
    """
    tokens = []
    for t in _CLAIM_TOKEN.findall(claim):
        if t.endswith(("'s", "’s")):
            t = t[:-2]
        tokens.append(t)

    def is_cap(t: str) -> bool:
        return t[:1].isupper() or t[:1].isdigit()

    # split the claim into runs of capitalized words (connectors allowed in the middle)
    runs, run = [], []
    for i, t in enumerate(tokens):
        if is_cap(t) or (run and t.lower() in CONNECTORS):
            run.append(i)
        else:
            if run:
                runs.append(run)
            run = []
    if run:
        runs.append(run)

    spans = []
    for run in runs:
        for a in run:
            for b in run:
                # a name has to start with a capital letter and end on a capitalized word
                if b < a or b - a >= max_len:
                    continue
                if not tokens[a][:1].isupper() or not is_cap(tokens[b]):
                    continue
                key = title_key(" ".join(tokens[a : b + 1]))
                if key and key not in STOPWORDS:
                    spans.append((a, b + 1, key))
    spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
    return spans
