# Experiment matrix

All numbers are on the test split (labeled shared-task dev, 19,998 claims) unless marked val.

## Retrieval (Trisha)

| ID  | Document stage            | Sentence stage                           | doc R@5 | sent R@5 | Notes                                    |
| --- | ------------------------- | ---------------------------------------- | ------- | -------- | ---------------------------------------- |
| R0  | BM25 only, no title match | BM25 in pool                             |         |          | Ablation                                 |
| R1  | Title match plus BM25     | BM25 in pool                             |         |          | CPU baseline                             |
| R2  | Title match plus BM25     | Dense (MiniLM)                           |         |          |                                          |
| R3  | Title match plus BM25     | Hybrid                                   |         |          |                                          |
| R4  | None                      | FAISS dense-only on subcorpus (val 2000) |         |          | Compare with R1 to R3 on the same claims |

sent R@5 is the ceiling on FEVER score for SUPPORTS and REFUTES claims.

## Verification (Sravani)

| ID  | Model                      | Train evidence      | Test evidence   | Acc | Macro F1 | FEVER  | Question                                           |
| --- | -------------------------- | ------------------- | --------------- | --- | -------- | ------ | -------------------------------------------------- |
| E1  | TF-IDF + LR                | None                | None            |     |          | n/a    | How far do claim artifacts go?                     |
| E2  | TF-IDF + LR                | Gold                | Gold            |     |          | oracle | Classical model with perfect evidence              |
| E3  | TF-IDF + LR                | Gold                | Retrieved       |     |          |        | Classical model in the realistic setting           |
| E4  | DeBERTa-v3-small           | None                | None            |     |          | n/a    | Transformer on artifacts alone                     |
| E5  | DeBERTa-v3-small           | Gold                | Gold            |     |          | oracle | Upper bound of the verifier                        |
| E6  | DeBERTa-v3-small           | Gold                | Retrieved       |     |          |        | Cost of train and test evidence mismatch           |
| E7  | DeBERTa-v3-small           | Gold plus retrieved | Retrieved       |     |          |        | Does training on retrieved evidence close the gap? |
| E8  | DeBERTa-v3-small from MNLI | Best of E6, E7      | Retrieved       |     |          |        | Value of NLI pretraining                           |
| E9  | Best model and E4          | As trained          | Symmetric FEVER |     |          | n/a    | Robustness to claim artifacts                      |

## How the research question is answered

E4 versus E5 versus E7 measures how much evidence improves verification over claim cues. E5
versus E7 isolates the cost of imperfect retrieval. E9 checks that the gain reflects reasoning
over evidence rather than artifacts. The evidence panel in the app and the extractive
explanations address explainability.
