# Batch 3: Feature Factory for Dataset #1 (Deposit)

## Что реализовано

1. Новый генератор `deposit_domain`:
- файл: `src/features/generators/deposit_domain.py`;
- активируется только когда `dataset_profile.dataset_type == "deposit"`;
- строит несколько независимых candidate-наборов:
  - `deposit_domain_robust`
  - `deposit_domain_interaction`
  - `deposit_domain_categorical`

2. Join-aware контекст для deposit:
- использует `schema_context.recommended_joins`;
- агрегирует aux-таблицы безопасно по ключу;
- извлекает row-count, robust numeric stats и категориальные профили.

3. Candidate diversity внутри генератора:
- отбор колонок по proxy-сигналу;
- отсечение избыточных признаков по высокой корреляции;
- фильтрация дублей категориальных колонок (полное совпадение).

4. Diversity на уровне всего pipeline:
- добавлена дедупликация эквивалентных candidate-наборов через сигнатуру данных;
- повторяющиеся наборы не попадают в оценку CatBoost.

## Измененные файлы

- `src/features/generators/deposit_domain.py` (новый)
- `src/features/registry.py`
- `src/pipeline/agent.py`

## Ожидаемый эффект

- больше качественных кандидатов именно для dataset #1;
- меньше дублирующих candidate-наборов;
- более стабильный отбор лучших 5 признаков за счет diversity.
