"""Inference wrapper around a fine-tuned sequence classification checkpoint."""

from __future__ import annotations

from factlens.labels import ID2LABEL
from factlens.models.inputs import model_inputs


class Verifier:
    """Loads a checkpoint (local folder or Hugging Face Hub ID) and predicts verdicts."""

    def __init__(self, model_path: str, mode: str = "evidence", max_length: int = 256,
                 device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device).eval()
        self.mode = mode
        self.max_length = max_length
        cfg_map = getattr(self.model.config, "id2label", None) or {}
        self.id2label = {int(k): v for k, v in cfg_map.items()} if cfg_map else ID2LABEL

    def predict_pairs(self, pairs: list[dict], batch_size: int = 32) -> list[tuple[str, dict[str, float]]]:
        """Returns (label, {label: probability}) for each pair."""
        results = []
        for i in range(0, len(pairs), batch_size):
            batch = [model_inputs(p, self.mode) for p in pairs[i : i + batch_size]]
            a = [x[0] for x in batch]
            b = None if self.mode == "claim_only" else [x[1] for x in batch]
            enc = self.tokenizer(a, b, truncation="only_second" if b else True,
                                 max_length=self.max_length, padding=True,
                                 return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
            for row in probs:
                dist = {self.id2label[j]: float(p) for j, p in enumerate(row)}
                results.append((max(dist, key=dist.get), dist))
        return results

    def predict(self, claim: str, evidence: list[list]) -> tuple[str, dict[str, float]]:
        """Predicts one claim against [page_id, line, text] evidence rows."""
        return self.predict_pairs([{"claim": claim, "evidence": evidence}])[0]