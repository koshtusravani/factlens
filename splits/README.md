# Splits

val_ids.txt holds the FEVER train claim IDs used for validation (5%, stratified by label, seed 42).
It is generated once with scripts/01_make_splits.py and committed, so both laptops, Colab, and
Kaggle use identical splits. Never regenerate it after experiments have started.

| Split | Source file                            | Purpose                                             |
| ----- | -------------------------------------- | --------------------------------------------------- |
| train | train.jsonl minus val_ids.txt          | Training                                            |
| val   | train.jsonl rows listed in val_ids.txt | Model selection and retrieval tuning                |
| test  | shared_task_dev.jsonl (labeled)        | Final numbers only; official test labels are hidden |
