# FactLens delivery plan: November 3 to December 1, 2026

Four one-week sprints, each ending in a tagged release. Trisha owns retrieval and data. Sravani
owns models, evaluation, integration, and the app.

Critical path: wiki.db (Trisha, Nov 6), then gold pairs (Sravani, Nov 8), then Kaggle training
(Sravani, Nov 12).

Daily: yesterday, today, blockers in the team chat; move cards on the board.
Weekly (Sunday evening): demo what merged, tag the release, plan the next sprint.

## Sprint 1: Nov 3 to Nov 10, v0.1.0 Data and baselines

| Day        | Trisha                                               | Sravani                                                    |
| ---------- | ---------------------------------------------------- | ---------------------------------------------------------- |
| Tue Nov 3  | clone, venv, pytest                                  | Commit foundation, branch protection, issue board          |
| Wed Nov 4  | PR feature/retrieval-core; start data download       | Review Trisha's PR; PR feature/models-core after it merges |
| Thu Nov 5  | Review Sravani's PR; build wiki.db                   | PR feature/app; generate and commit splits/val_ids.txt     |
| Fri Nov 6  | PR feature/retrieval-scripts; share wiki.db on Drive | Download wiki.db; spot-check sentences                     |
| Sat Nov 7  | Retrieval on val 2000, with and without title match  | Gold pairs for train, val, test (train overnight)          |
| Sun Nov 8  | Review Sravani's pairs                               | Train pairs with random NEI sampling                       |
| Mon Nov 9  | Full retrieval on val and test (bm25)                | Baselines E1 and E2                                        |
| Tue Nov 10 | Review baselines PR                                  | Baseline E3; tag v0.1.0                                    |

## Sprint 2: Nov 11 to Nov 17, v0.2.0 Fine-tuned verifier

| Day        | Trisha                                                 | Sravani                                             |
| ---------- | ------------------------------------------------------ | --------------------------------------------------- |
| Wed Nov 11 | Upload wiki.db as a private Kaggle dataset             | Upload pairs to Kaggle; follow docs/gpu_training.md |
| Thu Nov 12 | Dense ranking on val 2000 (Kaggle)                     | 30k subsample DeBERTa run                           |
| Fri Nov 13 | Hybrid ranking on val 2000                             | Full E5 gold-evidence run, push to Hub              |
| Sat Nov 14 | Pick best mode; update configs/default.yaml in a PR    | E4 claim-only run, push to Hub                      |
| Sun Nov 15 | Retrieved train pairs with the chosen mode (overnight) | E5 on gold test, E6 on retrieved test               |
| Mon Nov 16 | Retrieved test pairs with the chosen mode              | Rerun E6 on new retrieved test pairs                |
| Tue Nov 17 | Review evaluation PR                                   | Fill E4 to E6 in docs/experiments.md; tag v0.2.0    |

## Sprint 3: Nov 18 to Nov 24, v0.3.0 Integrated application

| Day        | Trisha                                           | Sravani                                       |
| ---------- | ------------------------------------------------ | --------------------------------------------- |
| Wed Nov 18 | FAISS dense-only experiment on val 2000 (Kaggle) | E7: train on gold plus retrieved pairs        |
| Thu Nov 19 | FAISS write-up, R4 row                           | E8: MNLI initialization ablation              |
| Fri Nov 20 | Test the app together on both laptops            | Point the app at the best Hub model           |
| Sat Nov 21 | Retrieval error analysis: 50 test misses         | Evaluate E7 and E8; choose the headline model |
| Sun Nov 22 | Review app PR                                    | App polish                                    |
| Mon Nov 23 | Draft report sections 3 and 4                    | Draft report sections 5 and 6                 |
| Tue Nov 24 | Retrieval error analysis PR                      | Tag v0.3.0                                    |

## Sprint 4: Nov 25 to Dec 1, v1.0.0 Final release

| Day        | Trisha                                    | Sravani                                            |
| ---------- | ----------------------------------------- | -------------------------------------------------- |
| Wed Nov 25 | Results table and confusion matrix figure | Model error analysis; E9 Symmetric FEVER (stretch) |
| Thu Nov 26 | Thanksgiving buffer                       | Thanksgiving buffer                                |
| Fri Nov 27 | README results section                    | Report sections 1, 7, 8                            |
| Sat Nov 28 | Code freeze                               | Report sections 9 and 10; full draft               |
| Sun Nov 29 | Slides; record a two-minute backup demo   | Review report and slides                           |
| Mon Nov 30 | Rehearse together                         | Fresh-clone reproducibility check                  |
| Tue Dec 1  | Final review                              | Tag v1.0.0 and submit                              |
