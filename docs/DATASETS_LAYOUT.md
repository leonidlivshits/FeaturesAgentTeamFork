# Работа с двумя датасетами (без риска потерять data/)

Рекомендуемая структура:

```text
datasets/
  dataset_1/
    train.csv
    test.csv
    readme.txt
    ...дополнительные таблицы...
  dataset_2/
    train.csv
    test.csv
    readme.txt
    ...дополнительные таблицы...
data/
  ...активный датасет для запуска...
```

`run.py` всегда читает только `data/`, поэтому переключение делаем через безопасный скрипт с бэкапом.

## Переключить на dataset_2

```bash
uv run python src/utils/switch_dataset.py --source datasets/dataset_2
```

Скрипт:
- сначала сохраняет текущее содержимое `data/` в `datasets/_backups/...`;
- потом переключает активный датасет в `data/`.

## Переключить на dataset_1

```bash
uv run python src/utils/switch_dataset.py --source datasets/dataset_1
```

## Запуск после переключения

```bash
uv run python run.py
uv run python src/utils/check_submission.py
```

Если ты уже случайно перетер `data/`, восстанови из:
- `datasets/_backups/...` (если бэкап есть), или
- исходной папки `datasets/dataset_1` / `datasets/dataset_2`.
