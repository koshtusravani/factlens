"""Loading FEVER claim files and extracting gold evidence."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from factlens.labels import LABELS, NEI
from factlens.utils import read_jsonl

EvidenceKey = tuple[str, int]  # (page_id, line_number)


@dataclass
class Claim:
    """One FEVER claim with its label and gold evidence sets."""

    id: int
    claim: str
    label: str | None = None
    evidence_sets: list[list[EvidenceKey]] = field(default_factory=list)

    @property
    def verifiable(self) -> bool:
        return self.label is not None and self.label != NEI


def _parse_evidence(raw_evidence) -> list[list[EvidenceKey]]:
    """Converts FEVER evidence annotations into a deduplicated list of evidence sets."""
    sets: list[list[EvidenceKey]] = []
    seen: set[tuple[EvidenceKey, ...]] = set()
    for annotation in raw_evidence or []:
        # claim files spell accents as separate marks (NFD) but the wiki dump uses NFC,
        # so "Björk" would not match without this
        items = [
            (unicodedata.normalize("NFC", e[2]), int(e[3]))
            for e in annotation
            if e[2] is not None and e[3] is not None
        ]
        key = tuple(sorted(set(items)))
        if items and key not in seen:
            seen.add(key)
            sets.append(list(dict.fromkeys(items)))
    return sets


def load_claims(path: str | Path) -> list[Claim]:
    """Loads a FEVER JSONL file. Unlabeled files (the hidden test set) yield label=None."""
    claims = []
    for row in read_jsonl(path):
        label = row.get("label")
        if label is not None and label not in LABELS:
            raise ValueError(f"Unexpected label {label!r} in claim {row.get('id')}")
        claims.append(
            Claim(
                id=int(row["id"]),
                claim=row["claim"],
                label=label,
                evidence_sets=_parse_evidence(row.get("evidence")),
            )
        )
    return claims


def gold_evidence_keys(claim: Claim, max_sentences: int = 5) -> list[EvidenceKey]:
    """Collects gold sentences set by set, so the first complete set always comes first.

    Keeping whole sets together means a multi-hop claim does not lose half of its proof.
    """
    keys: list[EvidenceKey] = []
    for ev_set in claim.evidence_sets:
        for key in ev_set:
            if key not in keys:
                keys.append(key)
    return keys[:max_sentences]