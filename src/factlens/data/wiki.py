"""Parsing and normalization of the FEVER Wikipedia dump (wiki-pages/*.jsonl).

Each dump line has the form {"id": page_title, "text": ..., "lines": "0\\tSentence\\tlink...\\n1\\t..."}.
Page titles use underscores and PTB bracket tokens such as -LRB- for "(".
"""

from __future__ import annotations

import re

_PTB = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LCB-": "{",
    "-RCB-": "}",
    "-COLON-": ":",
    "``": '"',
    "''": '"',
}
_PTB_RE = re.compile("|".join(re.escape(k) for k in _PTB))
_SPACE_RE = re.compile(r"\s+")
# Removes the spaces the tokenizer inserted inside brackets and before punctuation.
_OPEN_RE = re.compile(r"([(\[{]) ")
_CLOSE_RE = re.compile(r" ([)\]},.;:!?%])")


def normalize_text(text: str) -> str:
    """Converts PTB tokens back to plain characters and tidies tokenizer spacing."""
    text = _PTB_RE.sub(lambda m: _PTB[m.group(0)], text)
    text = _SPACE_RE.sub(" ", text)
    text = _OPEN_RE.sub(r"\1", text)
    text = _CLOSE_RE.sub(r"\1", text)
    text = text.replace(" 's", "'s")
    return text.strip()


def title_to_text(page_id: str) -> str:
    """Turns a FEVER page id such as 'Eiffel_Tower_-LRB-film-RRB-' into 'Eiffel Tower (film)'."""
    return normalize_text(page_id.replace("_", " ").replace("-LRB-", " -LRB- ").replace("-RRB-", " -RRB- "))


def parse_lines(lines_field: str) -> list[tuple[int, str]]:
    """Parses the lines field into (line_number, normalized_sentence) pairs, skipping empty lines."""
    out: list[tuple[int, str]] = []
    for raw in lines_field.split("\n"):
        if not raw:
            continue
        parts = raw.split("\t")
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        sentence = normalize_text(parts[1])
        if sentence:
            out.append((int(parts[0]), sentence))
    return out


def format_evidence(page_id: str, sentence: str) -> str:
    """Prepends the page title so pronouns in the sentence have a resolvable referent."""
    return f"{title_to_text(page_id)}: {sentence}"