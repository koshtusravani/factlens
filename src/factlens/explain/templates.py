"""Evidence-grounded, template-based explanations.

An explanation only restates the verdict and quotes retrieved evidence, so it cannot introduce
facts that the system did not retrieve.
"""

from __future__ import annotations

from factlens.data.wiki import title_to_text
from factlens.labels import DISPLAY, NEI, REFUTES, SUPPORTS

LOW_CONFIDENCE = 0.55


def explain(label: str, confidence: float, evidence: list[dict]) -> str:
    """Builds a short explanation. Each evidence item needs page_id and text keys, best first."""
    verdict = DISPLAY[label]
    hedge = (" The model is not confident, so this verdict should be treated with caution."
             if confidence < LOW_CONFIDENCE else "")
    if not evidence:
        return f"No relevant evidence was retrieved, so the claim cannot be verified.{hedge}"
    top = evidence[0]
    source = title_to_text(top["page_id"])
    if label == SUPPORTS:
        body = f'The Wikipedia article "{source}" states: "{top["text"]}" This agrees with the claim.'
    elif label == REFUTES:
        body = f'The Wikipedia article "{source}" states: "{top["text"]}" This contradicts the claim.'
    elif label == NEI:
        body = (f'The closest evidence came from "{source}", but it neither confirms nor '
                "contradicts the claim.")
    else:
        raise ValueError(f"Unknown label {label!r}")
    return f"Verdict: {verdict} ({confidence:.0%} confidence). {body}{hedge}"