# Team workflow for 3 days

## Branches

- `feat/backend-devops-foundation` - backend and devops owner branch
- `feat/ml-feature-ideas` - ML engineer #1 branch (new feature generators)
- `feat/ml-feature-selection` - ML engineer #2 branch (evaluation and selection)

## Ownership map

- Backend+DevOps owner files:
  - `run.py`
  - `src/pipeline/*`
  - `src/io/*`
  - `src/core/*`
  - CI/CD and infra files
- ML #1 owner files:
  - `src/features/generators/*`
  - `src/features/registry.py`
- ML #2 owner files:
  - `src/evaluation/*`
  - experiment notes and validation scripts

## Working agreement

- Keep generated feature columns prefixed with `gen_`.
- Keep each feature set at `<= 5` columns.
- Do not write anywhere except `output/` in runtime code.
- Every PR must keep `python run.py` runnable.
- Merge to `main` only through short PR review from at least one teammate.

## 3-day execution plan

1. Day 1
- Backend+DevOps: finish pipeline orchestration, logging, output contracts.
- ML #1: add at least 2 new generators in `src/features/generators/`.
- ML #2: improve CatBoost evaluator and scoring stability.

2. Day 2
- Backend+DevOps: add container run script and quick validation checks.
- ML #1: iterate on data-readme-aware generators and remove weak features.
- ML #2: compare candidate sets and tune selection criteria.

3. Day 3
- All: freeze code by midday, run final checks, prepare zip submit.
- Backend+DevOps: verify clean run from empty `output/`.
- ML team: lock best feature set and provide final notes.

## Daily sync

- 10-minute morning sync: blockers and owner actions.
- 10-minute evening sync: merged changes and next-day priorities.
