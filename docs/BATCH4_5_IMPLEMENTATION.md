# Batch 4 + 5: Repeat-Purchase Features and Search Engine

## Batch 4: Feature Factory for Dataset #2 (Repeat Purchase)

### Реализовано
- Новый генератор: `src/features/generators/repeat_purchase_domain.py`
- Включен в registry: `src/features/registry.py`

### Что делает генератор
1. Определяет ключи `user_id` / `product_id` из schema profile.
2. Строит join-aware признаки из таблиц:
- `users`
- `orders`
- `order_items`
- `products`
3. Формирует семейства кандидатов:
- `repeat_purchase_domain_core` (user/product/pair агрегаты)
- `repeat_purchase_domain_temporal` (time/behavior features)
4. Применяет diversity отбор:
- proxy ranking по одиночным признакам;
- отсечение избыточных (корреляция/полное совпадение).

### Анти-утечки
- В генерации не используется `target`.
- Признаки строятся только из истории/справочников.

## Batch 5: Search/Selection Engine

### Реализовано
- Обновлен evaluator: `src/evaluation/catboost_evaluator.py`
  - поддержка `GroupKFold` при наличии групп;
  - fallback на `StratifiedKFold` при отсутствии групп;
  - публичный `evaluate_features(...)`.
- Обновлен pipeline: `src/pipeline/agent.py`
  - `greedy_forward_select_feature_set(...)`;
  - пул признаков из всех candidate-наборов;
  - greedy forward selection до 5 фич;
  - diversity-фильтрация пула;
  - дедупликация эквивалентных наборов.

### Group-aware CV
- Для `repeat_purchase` автоматически включается группировка по `user_id` (или аналогу из schema profile), чтобы снизить optimistic leakage между фолдами.

### Decision Trace
- Добавлены поля:
  - `selection_strategy`
  - `group_cv_used`

## Практический эффект
- Более сильные признаки для датасета повторных покупок.
- Более стабильный выбор лучших 5 признаков из общего пула.
- Ближе к соревновательной постановке, где важен именно quality feature engineering, а не сложность модели.
