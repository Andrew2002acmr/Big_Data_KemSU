# Лабораторная работа 2

**Вариант 6 — Hospital Readmissions.** Задача — агрегация по полу, направлению выписки и повторному поступлению:
число записей, сумма и средняя длительность пребывания, средний BMI.

Результат: [report.md](report.md). Воспроизводимый ноутбук: [lab_2.ipynb](lab_2.ipynb).

## Что выполнено

- Сравнены pandas, Dask и Spark на исходнике, 100 МБ и 1 ГБ.
- Измерены время, RAM, CPU и счётчики I/O; результаты проверены по всем ключам и агрегатам.
- Сохранены физические планы и реальные stage/shuffle-метрики Spark.
- Проверены salting, число shuffle-партиций, repartition, AQE и Parquet, а также их комбинация.
- Выполнен обязательный CPU-трек A: pandas, Polars и DuckDB. Таблица и команды находятся в [gpu/README.md](gpu/README.md).

Большие наборы созданы повторением синтетического исходника с уникальными тестовыми ID.
Они не представляют новых медицинских наблюдений. Исходные данные и временные файлы находятся в `tmp/` и не включаются в Git.

## Запуск

Python 3.12; для PySpark нужна Java. Замеры выполнены на Windows, 8 логических CPU, 15,85 ГиБ RAM.
Для повторения требуется около 1,3 ГБ временных данных дополнительно к окружению Python и рабочей памяти.

Из папки `lab_2`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

Можно выполнять этапы отдельно:

```powershell
python run.py prepare
python run.py bench
python run.py opt
python run.py report
```

Время запроса исключает импорт и запуск движка; они сохраняются отдельно.
Приведены медианы трёх прогонов, кэш ОС не очищался. GPU не измерялся; трек A является самостоятельным выполнением части задания.

## Файлы

| Файл | Назначение |
|---|---|
| `prepare.py` | Загрузка, очистка, Great Expectations и создание нагрузочных наборов |
| `benchmark.py` | Одинаковая задача на пяти движках и проверка агрегатов |
| `metrics.py` | Замеры процесса и потомков через psutil |
| `spark_optimize.py` | Отдельные оптимизации Spark, планы и метрики стадий |
| `report.py` | Сборка отчёта, графиков, ноутбука и таблицы трека A |
| `run.py` | Последовательный запуск этапов |
| `results/` | Сохранённые измерения и планы |
| `assets/` | Графики |

Источник: [Hospital Readmission Prediction (synthetic dataset)](https://www.kaggle.com/datasets/siddharth0935/hospital-readmission-predictionsynthetic-dataset).
