# Batch 2: Универсальный Слой Понимания Схемы

Цель партии 2: сделать один и тот же агент пригодным для двух разных задач бинарной классификации без ручного переписывания кода под конкретный датасет.

## Что добавлено

1. Новый модуль `src/data/schema.py`
- строит `SchemaContext` из `train/test + aux + readme`;
- определяет `dataset_profile` (`repeat_purchase` | `deposit` | `generic`);
- извлекает `table_schemas` (типы колонок, кандидаты ключей);
- строит `join_edges` (граф связей таблиц по ключам с coverage-score);
- формирует `recommended_joins` (рекомендованный join-ключ для каждой aux-таблицы);
- читает `data_dictionary` (если есть) и подмешивает подсказки по ролям полей.

2. Обновлен `DataBundle`
- в `src/data/loaders.py` добавлено поле `schema_context`;
- теперь все генераторы получают не только таблицы, но и структурное описание схемы.

3. Интеграция в pipeline
- в `src/pipeline/agent.py` в `decision_trace` добавлены:
  - `dataset_type`
  - `dataset_type_confidence`
  - `schema_join_edges`
  - `schema_recommended_joins`

4. Интеграция в генераторы
- `llm_guided`:
  - prompt теперь содержит `schema_summary` из `SchemaContext`;
  - aux-таблицы мержатся по рекомендованному ключу схемы (не только по `id_column`).
- `relational_aggregates`:
  - join-planner учитывает `recommended_joins`, `join_edges` и подсказки из `data_dictionary`.

## Архитектура под 2 датасета

```text
data/
  train.csv
  test.csv
  readme.txt
  ...aux tables...
        |
        v
load_data_bundle()
  -> infer id/target
  -> build_schema_context()
      - profile detection
      - schema graph
      - recommended joins
        |
        v
feature generators
  - llm_guided (schema-aware prompt + schema-aware joins)
  - relational_aggregates (schema-aware join planner)
  - heuristic generators
        |
        v
candidate evaluation (CatBoost CV)
        |
        v
output/train.csv, output/test.csv (<=5 features)
```

## Почему это покрывает оба набора

1. Датасет `deposit`:
- профиль определяется по `client_*` паттернам и соответствующим таблицам;
- рекомендованные join-ключи обычно `client_id`/аналог.

2. Датасет `repeat_purchase`:
- профиль определяется по `users/orders/order_items/products` и `user_id/product_id/order_id`;
- схема находит связи между таблицами по `*_id`;
- генераторы могут строить признаки из aux по ключам `user_id` и `product_id`, даже если `id_column = row_id`.

## Дальнейшее развитие (партии 3-4)

1. Добавить специализированные агрегаты `user-product`, `order-time` и категорийные иерархии.
2. Ввести join-path агрегации через 2 шага (`orders -> order_items -> train`).
3. Внедрить dataset-specific кандидаты признаков поверх общего schema слоя.
