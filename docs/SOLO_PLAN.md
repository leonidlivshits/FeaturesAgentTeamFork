# Solo execution plan (3 days)

## Objective

Ship one stable submit that passes checks and produces competitive features with minimal moving parts.

## Day 1: Reliability baseline

1. Keep pipeline deterministic and reproducible.
2. Ensure `python run.py` works from clean environment.
3. Add robust fallback generators and output validation.
4. Pass `python src/utils/check_submission.py` locally.

## Day 2: Feature quality

1. Improve relational aggregates from auxiliary tables.
2. Add or tune candidate generators.
3. Compare candidate sets using CatBoost CV.
4. Keep final set to max 5 features.

## Day 3: Final hardening

1. Run full smoke checks on both provided datasets.
2. Reduce runtime to stay safely under 600 sec.
3. Freeze code and produce final zip.
4. Submit once only after all local checks pass.

## Daily operating mode

1. Work in short loops: code -> run -> verify -> commit.
2. Commit every completed subtask with small diffs.
3. Keep one changelog note per experiment (score and idea).

