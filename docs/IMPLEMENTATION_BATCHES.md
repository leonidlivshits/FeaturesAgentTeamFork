# Implementation batches (solo mode)

## Batch 1: Stable runtime foundation (completed)

1. Keep one deterministic pipeline entrypoint in `run.py`.
2. Keep strict data contract: read only from `data/`, write only to `output/`.
3. Validate generated feature sets before evaluation.

## Batch 2: LLM-guided feature planning (completed)

1. Build optional GigaChat client from `.env`.
2. Send compact schema + `data/readme.txt` context to LLM.
3. Accept only strict JSON with allowed operations.

## Batch 3: Safe local feature execution (completed)

1. Convert LLM output into internal `FeatureSpec` with strict operation rules (arity + column types).
2. Compute features only with safe local operators (no dynamic code execution).
3. Sanitize computed features (`inf/-inf`, extreme tails) and keep top features up to project limit (5).

## Batch 4: Fallback and resiliency (completed)

1. If LLM is unavailable/disabled (`FEATURES_AGENT_DISABLE_LLM=1`) or response is invalid, build deterministic heuristic plan.
2. Keep pipeline runnable without internet and without crashing on invalid feature sets.
3. Add evaluator-level and pipeline-level fallback selectors if CatBoost evaluation fails.

## Batch 5: Quality upgrades (completed)

1. Add readme-aware join hints for auxiliary tables (completed).
2. Add experiment logging with per-feature-set CV metrics (completed).
3. Add runtime guardrails to stay well below 600 seconds (completed).

## Batch 6: Release automation (completed)

1. Add one-command pre-submit script (checks + zip packaging).
2. Pack only required project artifacts (without `.git`, `.venv`, `data`, `output`).
3. Store built archive in `dist/`.
