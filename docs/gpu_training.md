# GPU training on Kaggle or Colab

Training needs only the pair files, not wiki.db.

## One-time setup

1. Zip data/pairs/\*.jsonl and upload it as a private Kaggle dataset named factlens-pairs.
2. Create a Hugging Face access token with write permission.
3. In the Kaggle notebook, open Add-ons, then Secrets, and add it as HF_TOKEN.

## Each run (Kaggle notebook, GPU T4 x1, internet on)

    from kaggle_secrets import UserSecretsClient
    import os
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    !git clone https://github.com/koshtusravani/factlens.git
    %cd factlens
    !pip install -q -e . sentencepiece protobuf
    P = "/kaggle/input/factlens-pairs"
    !python scripts/06_train_transformer.py --train {P}/gold_train_nei-retrieved.jsonl --val {P}/gold_val_nei-retrieved.jsonl --out checkpoints/deberta-gold-30k --subsample 30000 --epochs 1

For full runs, drop --subsample, use --epochs 2, and add --push-to-hub koshtusravani/factlens-deberta-gold.

## Troubleshooting

If the loss becomes NaN under fp16, lower model.lr to 1e-5 in configs/default.yaml and rerun.
