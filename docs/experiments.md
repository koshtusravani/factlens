# Experiment matrix

All numbers are on the test split (labeled shared-task dev, 19,998 claims, balanced across the
three labels) unless marked val. NEI training and test evidence is retrieved and length-matched
to gold (scripts/04b) unless stated otherwise.

## Retrieval (Trisha)

| ID  | Document stage            | Sentence stage                           | doc R@5     | sent R@5    | Notes                                                                 |
| --- | ------------------------- | ---------------------------------------- | ----------- | ----------- | --------------------------------------------------------------------- |
| R0  | BM25 only, no title match | BM25 in pool                             | 0.496 (val) | 0.429 (val) | Ablation, val 2000                                                    |
| R1  | Title match plus BM25     | BM25 in pool                             | 0.818       | 0.677       | CPU baseline, after #28; val 2000: 0.804 / 0.639                      |
| R2  | Title match plus BM25     | Dense (MiniLM)                           | 0.804 (val) | 0.701 (val) | val 2000                                                              |
| R3  | Title match plus BM25     | Hybrid                                   | 0.818       | 0.748       | alpha 0.5, chosen mode; val 2000: 0.804 / 0.709; train: 0.814 / 0.727 |
| R4  | None                      | FAISS dense-only on subcorpus (val 2000) | 0.893 (val) | 0.758 (val) | 208k pages, 985k sentences; with 20k random pages: 0.920 / 0.774      |

Title matching raises doc R@5 on val 2000 from 0.496 to 0.804. sent R@5 caps the FEVER score:
with 0.748, the best possible FEVER score is about 0.83.

R4 (scripts/08) searches a subcorpus, not the full 5.4M pages: the gold pages of the val 2000
claims, the pages R3 retrieved for them, and 200k random pages. Because the gold pages are always
in it, R4 is not a like-for-like win over R3. What it shows is that dense sentence search ranks
evidence well: growing the corpus 7x (28.5k to 208k pages) costs only 2.7 points of doc R@5.

## Verification (Sravani)

| ID  | Model                       | Train evidence              | Test evidence      | Acc   | Macro F1 | FEVER          |
| --- | --------------------------- | --------------------------- | ------------------ | ----- | -------- | -------------- |
| E1  | TF-IDF + LR                 | None                        | None               | 0.547 | 0.548    | n/a            |
| E2  | TF-IDF + LR                 | Matched gold                | Matched gold       | 0.575 | 0.573    | oracle         |
| E3  | TF-IDF + LR                 | Matched gold                | Retrieved (hybrid) | 0.460 | 0.444    | 0.423          |
| E3b | TF-IDF + LR                 | Matched gold                | Retrieved (bm25)   | 0.446 | 0.421    | 0.404          |
| E4  | DeBERTa-v3-small            | None (claim only)           | None               | 0.632 | 0.624    | n/a            |
| E5  | DeBERTa-v3-small            | Matched gold                | Matched gold       | 0.881 | 0.880    | 0.880 (oracle) |
| E6  | DeBERTa-v3-small (E5 model) | Matched gold                | Retrieved (hybrid) | 0.426 | 0.371    | 0.414          |
| E7  | DeBERTa-v3-small            | Matched gold plus retrieved | Retrieved (hybrid) | 0.753 | 0.749    | 0.634          |
| E7g | DeBERTa-v3-small (E7 model) | Matched gold plus retrieved | Matched gold       | 0.871 | 0.870    | 0.870 (oracle) |
| E8  | DeBERTa-v3-small from MNLI  |                             |                    |       |          | Not run yet    |
| E9  | Symmetric FEVER             |                             |                    |       |          | Not run yet    |

DeBERTa runs: 2 epochs, lr 2e-5, batch 32 per device, 2x T4 on Kaggle. Models on the Hugging
Face Hub: sravanikoshtu/factlens-e4, factlens-e5-matched, factlens-e7. All pair files used bm25
NEI evidence; the hybrid default from PR 32 applies only if pairs are rebuilt.

## Findings

1. Claim artifacts. Claim wording alone reaches 0.547 (TF-IDF) and 0.632 (DeBERTa) against 0.333
   chance. Top REFUTES n-grams are negation and restriction words (only, not, zero, never,
   exclusively, except), matching Schuster et al. (2019).
2. Length leak. Before matching, every NEI pair had 5 evidence sentences, while SUPPORTS and
   REFUTES gold evidence averaged 1.5. This inflated TF-IDF E2 from 0.575 to 0.731 and DeBERTa E5
   from 0.881 to 0.964 (NEI F1 0.994). scripts/04b removes the leak.
3. Evidence use. Without the length cue, gold evidence adds +25.7 macro F1 for DeBERTa
   (0.624 to 0.880) but only +2.5 for TF-IDF (0.548 to 0.573): the transformer reasons over
   evidence content, while the lexical model mostly reads the claim.
4. Source leak. A model trained on gold evidence for SUPPORTS and REFUTES and retrieved
   evidence for NEI labels most retrieved test claims as NEI (TF-IDF: 66% NEI predictions;
   DeBERTa E6: acc 0.426, below claim only).
5. Training on retrieved evidence fixes the source leak. E7 improves over E6 by +32.7 accuracy
   and +22.0 FEVER on retrieved evidence, and loses only 1 point on gold (0.871 versus 0.881).
6. Retrieval is the remaining bottleneck. Title matching lifts doc R@5 from 0.496 to 0.804, and
   hybrid ranking adds +1.9 FEVER over bm25 at the classifier level (TF-IDF E3), but E7 still
   reaches FEVER 0.634 against a retrieval ceiling of about 0.83.
7. Error analysis (E7 on retrieved evidence, 4,940 errors; random sample of 30, seed 42).
   By confusion: NEI predicted as a verdict 52% (NEI to SUPPORTS 32%), REFUTES to NEI 23%,
   REFUTES to SUPPORTS 12%. Retrieval misses (gold evidence not in the top 5) are 20% of errors.
   Manual categories in the sample: label noise 12, NEI confusion 8, retrieval miss 7, numbers or
   dates 1, polarity 1, other 1. In 12 of the 14 sampled NEI errors the retrieved evidence does
   decide the claim (for example, Basquiat "died of a heroin overdose" for a cocaine claim), so
   FEVER NEI labels understate E7. Remaining model errors need closed-world or temporal
   inference ("used in Switzerland and Liechtenstein" implies not Finland). Retrieval misses
   come from misspelled names, disambiguation pages, and missing date sentences.

## Limitations

- E7 selects its best epoch on gold validation pairs, since no retrieved validation file exists.
- Random NEI sampling (an earlier variant) inflates NEI F1 from 0.68 to 0.94 on hard negatives;
  all reported models use retrieved NEI evidence.
