# FactLens

Evidence-grounded claim verification on FEVER. FactLens retrieves Wikipedia evidence with a
two-stage retriever (exact title match plus BM25 over SQLite FTS5, then lexical, dense, or hybrid
sentence ranking) and classifies each claim as SUPPORTED, REFUTED, or NOT ENOUGH INFO with a
fine-tuned DeBERTa-v3-small. A Streamlit app shows the verdict, class probabilities, evidence with
sources, and an extractive explanation.

**Team:** Trisha (retrieval and data), Sravani (models, evaluation, integration)

**Research question:** How much does retrieved evidence improve the accuracy and explainability
of fine-tuned transformers for claim verification, beyond what claim wording alone allows?

## Quick start

    python -m venv .venv
    .venv\Scripts\activate              (macOS: source .venv/bin/activate)
    pip install -r requirements.txt
    pip install -e .
    pytest

## Pipeline

| Step | Script                      | Owner   | Output                                         |
| ---- | --------------------------- | ------- | ---------------------------------------------- |
| 0    | 00_download_data.py         | Trisha  | data/fever, data/wiki-pages                    |
| 1    | 01_make_splits.py           | Sravani | splits/val_ids.txt (committed)                 |
| 2    | 02_build_wiki_db.py         | Trisha  | indexes/wiki.db                                |
| 3    | 03_run_retrieval.py         | Trisha  | Retrieved pairs and retrieval metrics          |
| 4    | 04_build_gold_pairs.py      | Sravani | Gold pairs with sampled NEI evidence           |
| 5    | 05_train_baselines.py       | Sravani | TF-IDF and Logistic Regression results         |
| 6    | 06_train_transformer.py     | Sravani | DeBERTa checkpoints (Kaggle or Colab GPU)      |
| 7    | 07_evaluate.py              | Sravani | Accuracy, macro F1, FEVER score, error buckets |
| 8    | 08_faiss_dense_retrieval.py | Trisha  | Dense-only FAISS comparison                    |
| 9    | 09_eval_symmetric.py        | Sravani | Symmetric FEVER robustness                     |
| 10   | 10_results_table.py         | Shared  | outputs/results.md                             |

App:

    streamlit run app/streamlit_app.py

## Design decisions

- **Splits.** Official FEVER test labels are hidden, so the labeled shared-task dev set is the
  test set, and 5% of train (stratified, seed 42) is validation. Only claim IDs are committed.
- **NEI evidence.** NOT ENOUGH INFO claims have no gold evidence, so training pairs use the
  retriever's top sentences as hard negatives. A random-sentence variant is kept as an ablation.
- **Scale.** The dump has about 5.4M pages and 25M sentences. A flat FAISS index over all of it
  would need tens of GB, so retrieval runs in two stages: documents first, then sentences inside
  them. FAISS is used on a reproducible subcorpus for the dense-only comparison.
- **Title prefixes.** Every evidence sentence is passed to the model as "Title: sentence", so
  pronouns have a referent.
- **FEVER score.** Matches the official scorer: a correct label counts only when a complete gold
  evidence set appears within the first five predicted sentences.
- **Artifacts.** Claim-only models and Symmetric FEVER measure how much performance comes from
  claim wording rather than evidence.
- **Naming.** A classifier produces the verdict, so the system is described as
  retrieval-augmented verification. Explanations are extractive and quote retrieved text only.

## Results

To be filled from outputs/results.md at v1.0.0.

## Limitations

The knowledge base is the June 2017 Wikipedia snapshot, so newer facts are out of scope. Live web
retrieval is future work.
