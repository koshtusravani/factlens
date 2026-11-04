# Contributing to FactLens

## One-time setup

    git clone https://github.com/koshtusravani/factlens.git
    cd factlens
    python -m venv .venv
    .venv\Scripts\activate              (macOS: source .venv/bin/activate)
    pip install -r requirements.txt
    pip install -e .
    pytest

Use Python 3.11, the same version CI runs.

## Branches

| Branch              | Purpose                                      |
| ------------------- | -------------------------------------------- |
| main                | Stable, reviewed, CI-green. Protected.       |
| feature/area-name   | New work, for example feature/retrieval-core |
| exp/name            | Experiment configs and result write-ups      |
| fix/name, docs/name | Bug fixes and documentation                  |

Keep branches short-lived (one issue, a few days). Start every branch from a fresh main:

    git switch main
    git pull
    git switch -c feature/retrieval-core
    git add -A
    git commit -m "feat(retrieval): short description"
    git push -u origin feature/retrieval-core
    gh pr create --fill

## Commit messages

Use Conventional Commits: feat, fix, test, docs, refactor, exp, chore.
Example: feat(retrieval): add hybrid lexical and dense sentence ranking

## Pull requests

1. One PR per issue. Write "Closes #N" in the description.
2. Fill in the template, including the testing checklist and any metrics.
3. The other member reviews, then approves or requests changes with specific comments.
4. Squash merge once CI is green, then delete the branch.

## What never goes in git

Data, indexes, checkpoints, outputs, virtual environments, .env files, and tokens. Share wiki.db
and pair files through Google Drive or a private Kaggle dataset, and models through the Hugging
Face Hub.

## Releases

After each milestone:

    git tag v0.1.0
    git push --tags
    gh release create v0.1.0 --generate-notes
