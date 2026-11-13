"""Fine-tuning of DeBERTa-v3 (or any encoder) for three-way claim verification.

Designed for a free Kaggle or Colab T4: fp16, dynamic padding, and evaluation after each epoch.
The best checkpoint by validation macro F1 is kept.
"""

from __future__ import annotations

import json
from pathlib import Path

from factlens.labels import ID2LABEL, LABEL2ID, LABELS
from factlens.models.inputs import model_inputs


class PairDataset:
    """Tokenizes pairs lazily, so 138k examples never sit in memory as tensors."""

    def __init__(self, pairs: list[dict], tokenizer, mode: str, max_length: int):
        self.pairs, self.tok, self.mode, self.max_length = pairs, tokenizer, mode, max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        a, b = model_inputs(self.pairs[i], self.mode)
        enc = self.tok(a, b, truncation="only_second" if b else True, max_length=self.max_length)
        enc["labels"] = LABEL2ID[self.pairs[i]["label"]]
        return enc


def class_weights(pairs: list[dict]) -> list[float]:
    """Returns inverse-frequency class weights, normalized to mean 1, in LABELS order."""
    counts = [sum(p["label"] == lab for p in pairs) for lab in LABELS]
    raw = [len(pairs) / (len(LABELS) * max(c, 1)) for c in counts]
    mean = sum(raw) / len(raw)
    return [w / mean for w in raw]


def train(train_pairs: list[dict], val_pairs: list[dict], output_dir: str, cfg: dict,
          mode: str = "evidence", use_class_weights: bool = False, base: str | None = None) -> dict:
    """Fine-tunes a sequence classifier and saves the best checkpoint plus run_info.json."""
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

    m = cfg["model"]
    base = base or m["base"]
    set_seed(m["seed"])
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # Allows starting from an NLI head with different labels.
    )
    weights = torch.tensor(class_weights(train_pairs), dtype=torch.float) if use_class_weights else None

    class WeightedTrainer(Trainer):
        """Applies optional class weights. The kwargs absorb arguments added in newer versions."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            w = weights.to(outputs.logits.device) if weights is not None else None
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=w)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds),
                "macro_f1": f1_score(labels, preds, average="macro")}

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=m["epochs"],
        learning_rate=m["lr"],
        per_device_train_batch_size=m["batch_size"],
        per_device_eval_batch_size=m["batch_size"] * 2,
        warmup_ratio=m["warmup_ratio"],
        weight_decay=m["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
        seed=m["seed"],
        dataloader_num_workers=2,
    )
    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=PairDataset(train_pairs, tokenizer, mode, m["max_length"]),
        eval_dataset=PairDataset(val_pairs, tokenizer, mode, m["max_length"]),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    run_info = {"base": base, "mode": mode, "class_weights": use_class_weights,
                "n_train": len(train_pairs), "n_val": len(val_pairs),
                "train_sources": sorted({p.get("evidence_source", "?") for p in train_pairs}),
                "config": m, "val_metrics": metrics}
    Path(output_dir, "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    return run_info