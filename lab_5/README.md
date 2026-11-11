# Лабораторная работа 5

**Вариант 6**

Потоковая оценка риска повторной госпитализации при выписке. Требование к задержке **p99 < 30 мс**. Основной поток использует Flink. Дополнительный профиль запускает Kafka → Spark Structured Streaming → ONNX.

[Краткий отчёт](report.md) · [Результаты прогонов](results)

## Запуск

Понадобится Docker с Compose. Из папки `lab_5`:

```powershell
docker compose up -d --build
docker compose logs -f bench
python tests/check_requirements.py
```

Compose поднимает Kafka KRaft, Flink, Redis, HTTP-сервис ONNX, Prometheus и Grafana. `bench` создаёт профили и события, запускает измерения при 50 и 100 событиях/с, отправляет намеренно поздние и ошибочные события и проверяет восстановление после сбоя записи. Результаты появляются в `tmp/results/`. Генератор использует фиксированные seed и синтетические профили. При повторной доставке идентификатор события и ключ пациента сохраняются. Повторить замер с проверкой p99 можно командой `docker compose run --rm --no-deps bench python benchmark.py --enforce-slo`.

- Flink UI доступен на `http://127.0.0.1:18081`.
- Grafana доступна на `http://127.0.0.1:13000/d/lab5/streaming`.
- Активные алерты Prometheus доступны на `http://127.0.0.1:19090/alerts`.
- ONNX API и схема запросов доступны на `http://127.0.0.1:18000/docs`.

### Spark и отказ узла

Запускай эти проверки после завершения основного замера:

```powershell
docker compose --profile spark up -d --build spark
docker compose run --rm --no-deps bench python benchmark.py --engine spark --rate 150
docker compose run --rm --no-deps bench python node_failure.py prepare
docker compose stop -t 1 taskmanager
docker compose run --rm --no-deps bench python node_failure.py queue
docker compose start taskmanager
docker compose run --rm --no-deps bench python node_failure.py verify
python tests/check_requirements.py --with-spark --with-node-failure
```

Интерфейс Spark доступен на `http://127.0.0.1:14040`. По окончании можно остановить стенд командой `docker compose --profile spark down`.

GitHub Actions проверяет сборку и функциональный прогон Flink. Результат p99 на раннере сохраняется в артефакте. Для строгой проверки задержки есть `python tests/check_requirements.py` и флаг `--enforce-slo` генератора.

### Обучение ONNX

Модель `model/readmission.onnx` включена в репозиторий. Скрипт переобучения читает [синтетический датасет Hospital Readmission Prediction](https://www.kaggle.com/datasets/siddharth0935/hospital-readmission-predictionsynthetic-dataset). Нужен Python 3.12:

```powershell
python -m pip install -r requirements-train.txt
python train_export.py
```

Для своей копии CSV можно передать `--csv путь.csv`. В `model/readmission.json` хранятся версия, порядок признаков и результаты сравнения sklearn с ONNX.
