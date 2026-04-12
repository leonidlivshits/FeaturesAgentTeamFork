# Batch 6 + 7 Implementation

## Batch 6: LLM as Proposer (not Executor)

### Сделано
1. Self-consistency для LLM-планов (`llm_guided`):
- собираются несколько валидных candidate-планов за попытки;
- дубли планов отбрасываются;
- лучший план выбирается по proxy-score до CatBoost.

2. Кэш планов по профилю схемы:
- кэш-ключ зависит от dataset profile, join-рекомендаций, провайдера LLM и доступных колонок;
- при повторном профиле схема не дергает LLM повторно в рамках процесса.

3. Бюджет LLM-запросов:
- добавлен символьный бюджет на запуск (`llm_char_budget_per_run`);
- если бюджет заканчивается, LLM-генерация останавливается и включается fallback.

4. Расширенные метаданные в feature set:
- `llm_candidate_plans`
- `llm_cache_hit`
- `llm_selected_plan_proxy_score`
- `llm_char_budget_total`
- `llm_char_budget_used`

## Batch 7: Experiments, Ablations, Anti-Overfit

### Anti-overfit в оценке
1. В evaluator добавлены метрики стабильности CV:
- `cv_std`
- `n_splits`
- `effective_score = cv_mean_auc - penalty * cv_std`

2. Выбор лучшего candidate-набора теперь идет по `effective_score`, а не только по `cv_mean_auc`.

3. Group-aware CV:
- для repeat-purchase используется `GroupKFold` по user-группе (когда группа доступна);
- fallback на StratifiedKFold.

### Протокол экспериментов
Добавлена утилита:
- `src/utils/experiment_runner.py`

Что делает:
- прогоняет абляции по режимам (`heuristic`, `hybrid`, `llm`);
- прогоняет несколько датасетов последовательно (stress test переносимости);
- собирает отчеты:
  - `docs/EXPERIMENTS_REPORT.md`
  - `docs/EXPERIMENTS_LOG.jsonl`
- безопасно восстанавливает исходный `data/` из бэкапа после завершения.

## Основные измененные файлы

- `src/core/config.py`
- `src/features/generators/llm_guided.py`
- `src/evaluation/catboost_evaluator.py`
- `src/pipeline/agent.py`
- `src/utils/experiment_runner.py`
