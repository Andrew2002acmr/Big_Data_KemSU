# Лабораторная работа 1

Анализ Big Data-сценария и выбор технологий. Вариант 6 — Hospital Readmissions.
Также выполнено практическое задание по NYC Taxi и сравнению Pandas, Dask и PySpark.

Основной результат — [отчёт](report.md). Графики находятся в папке [assets](assets).

## Файлы

| Файл | Назначение |
|---|---|
| `fetch_data.py` | Загрузка исходных датасетов |
| `analyze.py` | Анализ данных, сравнение форматов и построение графиков |
| `hospital_etl.py` | Разбор давления, карантин ошибок и проверки Great Expectations |
| `hospital_resource.py` | Проверка обработки ошибок и замер медицинского ETL |
| `benchmark.py` | Сравнение трёх движков на 10 ГБ CSV |
| `make_report.py` | Генерация отчёта из шаблона и сверка агрегатов бенчмарка |
| `requirements.txt` | Версии Python-зависимостей |
| `measurements.json` | Статистика датасетов и результаты проверок |
| `hospital_resource.json` | Замеры медицинского ETL |
| `benchmark_*.json` | Замеры движков, агрегаты по зонам и результат их сверки |

## Данные

- [Hospital Readmission Prediction — синтетический датасет](https://www.kaggle.com/datasets/siddharth0935/hospital-readmission-predictionsynthetic-dataset): 30 000 строк, 12 полей.
- [NYC TLC Yellow Taxi за январь 2024](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page): 2 964 624 строки, 19 полей.

Исходные данные загружаются скриптом `fetch_data.py` в `tmp/data` и не хранятся в Git.
Файлы для теста на 10 ГБ создаются повторением записей NYC Taxi; это тестовая копия, а не новые реальные поездки.

## Запуск

Использованы Python 3.12 и Java для PySpark. Замеры выполнены на Windows с 8 логическими CPU и 15,85 ГиБ RAM.
Для теста на 10 ГБ потребуется около 11 ГБ свободного диска дополнительно к окружению Python.

Пример запуска на Windows из корня репозитория:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe fetch_data.py
.\.venv\Scripts\python.exe hospital_resource.py
.\.venv\Scripts\python.exe analyze.py
.\.venv\Scripts\python.exe benchmark.py prepare
.\.venv\Scripts\python.exe benchmark.py pandas
.\.venv\Scripts\python.exe benchmark.py dask
.\.venv\Scripts\python.exe benchmark.py spark
```

Для просмотра результатов запускать расчёты не требуется: отчёт, графики и JSON с измерениями уже сохранены.
Время выполнения на другой машине может отличаться. В медицинском датасете нет дат госпитализаций и отделений:
временной прогноз описан как проектное решение, а замеры ETL выполнены на доступных данных.

`report.md` содержит ручные правки. Скрипт `make_report.py` не входит в обычный запуск:
он перезаписывает отчёт текстом из шаблона. Его следует запускать только при намеренном пересоздании отчёта.
