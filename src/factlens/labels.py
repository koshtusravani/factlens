"""Label definitions used consistently across baselines, transformers, and evaluation."""

SUPPORTS = "SUPPORTS"
REFUTES = "REFUTES"
NEI = "NOT ENOUGH INFO"

LABELS = [SUPPORTS, REFUTES, NEI]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))

DISPLAY = {SUPPORTS: "SUPPORTED", REFUTES: "REFUTED", NEI: "NOT ENOUGH INFO"}