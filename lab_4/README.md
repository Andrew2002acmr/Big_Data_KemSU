# Лабораторная работа 4

**Вариант 6**

Оценка риска повторной госпитализации в течение 30 дней на момент выписки. Особенность варианта — интерпретируемый прогноз. Для отдельной части задания выполнено воспроизведение аварий по ряду температуры NAB.

[Краткий отчёт](report.md) · [Jupyter Notebook](lab_4.ipynb) · [Результаты замеров](results)

## Запуск

Нужен Python 3.12. Из папки `lab_4`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py all
python -m pytest -q
```

Скрипт сам скачивает [синтетические госпитализации](https://www.kaggle.com/datasets/siddharth0935/hospital-readmission-predictionsynthetic-dataset) и [температурный ряд NAB](https://github.com/numenta/NAB/blob/master/data/realKnownCause/ambient_temperature_system_failure.csv). Для работы с локальными CSV укажи `--hospital путь.csv --sensor путь.csv`. Для отдельного этапа используй `python run.py hospital` или `python run.py sensor`. Данные, реестр, модели и графики новых прогонов хранятся в `tmp/local`. Чтобы повторить эксперимент в отдельной папке, передай `--work tmp/повтор`.

Ноутбук использует те же функции и запускается в ядре с установленными зависимостями. Для локальных копий можно задать `LAB4_HOSPITAL` и `LAB4_SENSOR`. Даты в CSV госпитализаций отсутствуют. Точка времени Feast и порядок поступления меток задаются симуляцией.

## API и интерфейс MLflow

После `python run.py hospital` запусти в отдельных терминалах:

```powershell
python api.py
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///tmp/local/mlflow.db --no-serve-artifacts
```

- Swagger UI — `http://127.0.0.1:8000/docs`. Пример тела запроса `POST /predict` — `{"patient_id": 1, "explain": true}`. Поле `explain` добавляет SHAP-вклады в логит. Для проверки маршрутизации можно запустить `python api.py --mode shadow` или `python api.py --mode canary`. Автоматическая проверка перехода и отката входит в `run.py hospital`.
- MLflow UI — `http://127.0.0.1:5000`. Эксперименты, метрики и зарегистрированные версии доступны в локальном реестре.
- В локальном прогоне предупреждения мониторинга записываются в `tmp/local/alerts.jsonl`. Функция `alert` поддерживает отдельно переданный Slack webhook.

| Файл | Назначение |
|---|---|
| `data.py`, `store.py`, `feast_repo/` | Признаки и проверка совпадения offline/online |
| `training.py` | Прототипы, batch, online, SHAP и MLflow |
| `api.py`, `serving_checks.py` | HTTP-инференс, замеры и canary |
| `monitoring.py` | PSI, KS, контроль качества и retrain |
| `sensor.py` | Прямой и рекурсивный прогноз NAB, CUSUM и журнал действий |
| `results/`, `assets/` | Метрики, графики и снимок MLflow |
