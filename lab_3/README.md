# Лабораторная работа 3

**Вариант 6**

Ежедневная витрина повторных госпитализаций по направлению выписки и полу. Источник — [Hospital Readmission Prediction](https://www.kaggle.com/datasets/siddharth0935/hospital-readmission-predictionsynthetic-dataset). В выгрузке нет дат госпитализаций, поэтому расписание задаёт момент обновления, а витрина описывает состояние последней обработанной совокупности.

Результаты и схема пайплайна находятся в [отчёте](report.md), воспроизводимая демонстрация — в [Jupyter Notebook](lab_3.ipynb).

## Запуск

Нужен Python 3.12. Из папки `lab_3` в PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python pipeline.py --docs
```

По умолчанию CSV скачивается один раз с Kaggle. Для локальной выгрузки добавь `--source путь/к/файлу.csv`. При запуске по расписанию укажи путь к актуальной выгрузке, которую обновляет поставщик данных.

База появится в `tmp/local/hospital.duckdb`. В `tmp/local/runs/<run_id>/` сохраняются метрики, логи dbt, GX Data Docs до и после загрузки. При `--docs` там же появится сайт dbt в папке `dbt/`. Открой `gx_before/index.html` и `gx_after/index.html` локально. Для dbt Docs запусти `python -m http.server 8080 --directory <путь-к-runs/run_id/dbt>` и открой `http://127.0.0.1:8080`.

```powershell
python demo.py --docs
python -m pytest -q
python release.py
```

`demo.py` запускает проверку повторной загрузки, инъекции сбоев, DLQ, несовместимой схемы и отката. Для автономного прогона на 120 тестовых строках используй `python demo.py --fixture`. `release.py` проверяет canary, переключение витрины и возврат к прежней версии после сбоя. Артефакты этих прогонов сохраняются в `tmp/`.

### Prefect UI и расписание

Запусти `prefect server start` в отдельном терминале. В рабочем терминале укажи локальный API:

```powershell
$env:PREFECT_API_URL = 'http://127.0.0.1:4200/api'
python pipeline.py --docs
```

Граф задач и логи доступны на `http://127.0.0.1:4200`. Для ежедневного запуска в 05:00 по Новосибирску выполни `python pipeline.py --serve --source путь/к/выгрузке.csv` и оставь процесс `serve` запущенным. В рабочем каталоге используется один writer для DuckDB.

### Ноутбук

Запусти `lab_3.ipynb` в ядре с установленными зависимостями из `requirements.txt`. Ядро можно зарегистрировать командой `python -m ipykernel install --user --name lab3`. Ноутбук берёт исходник с Kaggle или из CSV, указанного в переменной окружения `LAB3_SOURCE`.

## Файлы

| Файл | Назначение |
|---|---|
| `pipeline.py` | Prefect flow, ретраи, метрики и интеграция с dbt |
| `source.py`, `warehouse.py` | Контракт CSV, DLQ, UPSERT и версии |
| `quality.py` | Great Expectations и HTML Data Docs |
| `dbt/` | SQL-модели, зависимости и тесты витрины |
| `demo.py`, `tests/` | Проверки сбоев, идемпотентности, схемы и отката |
| `release.py` | Проверка локального релиза и отката |
| `results/`, `assets/` | Метрики, графики и скриншоты |
