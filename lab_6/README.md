# Лабораторная работа № 6

Вариант 6. Сценарий — оценка риска повторной госпитализации. Цель — снизить расчётную стоимость исполнения пакетной Spark-задачи минимум на 35 % подбором размера инстанса. Цели HTTP-сервиса на нагрузочном тесте — p95 до 50 мс и доступность от 99,9 %. Порог модели выбран в предыдущей лабораторной при FPR ниже 1 %.

## Запуск

Из каталога `lab_6` выполни команду с запущенным Docker Desktop в режиме Linux-контейнеров.

```sh
docker compose up --build --exit-code-from bootstrap
```

Bootstrap создаёт три узла kind, ставит Spark Operator, Flink Operator и KEDA, собирает образы, запускает нагрузку и сохраняет результаты в `tmp/verification.json`. Для первого запуска нужен доступ к реестрам образов и датасету Kaggle. Стенду нужно примерно 8 ГиБ RAM и 12 ГиБ свободного диска. Команда подключает сокет локального Docker для создания узлов.

Если CSV уже скачан, положи его в `lab_6/tmp/hospital.csv` перед запуском. Контрольная сумма сверяется с зафиксированной версией. ONNX-модель, клиентские функции и Java-задание Flink используются из `lab_5` и входят в архив шестой лабораторной.

При установленных Python 3.12, kind 0.29, Helm 3 и kubectl 1.33 доступны отдельные этапы.

```sh
python -m pip install -r requirements.txt
python deploy.py setup --data /path/to/hospital_readmissions_30k.csv
python deploy.py verify
python deploy.py clean
```

После прогона кластер остаётся доступным для просмотра. `clean` удаляет кластер `lab6` и его локальные данные.

| Компонент | Адрес |
| --- | --- |
| Grafana | [Панель DevOps](http://127.0.0.1:13660/d/lab6/devops) |
| Prometheus | [Активные правила](http://127.0.0.1:19660/alerts) |
| Headlamp | [Kubernetes](http://127.0.0.1:14660/) |
| HTTP-модель | [Проверка готовности](http://127.0.0.1:18660/health) |

Для входа в Headlamp нужен временный токен для просмотра ресурсов.

```sh
kubectl --kubeconfig tmp/kubeconfig -n lab6 create token lab6-viewer --duration=30m
kubectl --kubeconfig tmp/kubeconfig -n lab6 port-forward svc/readmissions-rest 18086:8081
```

Вторая команда открывает интерфейс Flink на `127.0.0.1:18086`. Если стенд создан через Compose, внешний kubeconfig можно получить командой `kind export kubeconfig --name lab6 --kubeconfig tmp/kubeconfig` с установленным kind.

## Конфигурации и проверка

- `kubernetes/spark.yml` задаёт два профиля ресурсов для одной агрегации. Расчёт использует сумму времени жизни исполнителей из Spark event log и модель одного выделенного инстанса на исполнителя. Ставки EC2 On-Demand для us-east-1 составляют $0,192/ч для `m6i.xlarge` и $0,096/ч для `m6i.large`. Драйвер, хранение, сеть и цена Spot вне этой оценки. [AWS Prescriptive Guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/optimize-costs-microsoft-workloads/right-size-selection.html)
- `yarn/` содержит профиль CapacityScheduler с очередями 60/30/10 и настройками динамического выделения исполнителей. `kubernetes/cluster-autoscaler-aws.yml` содержит values для EKS 1.33 с IAM-ролью сервисного аккаунта и помеченными AWS Auto Scaling Groups. На kind масштабирование реплик выполняет KEDA. [Документация Cluster Autoscaler](https://github.com/kubernetes/autoscaler/blob/master/cluster-autoscaler/cloudprovider/aws/README.md)
- `kubernetes/spark-gpu.yml` задаёт профиль одного NVIDIA GPU на executor и 0,25 GPU на задачу через RAPIDS. Для реального запуска требуются GPU-ноды с меткой `accelerator=nvidia`, NVIDIA device plugin и образ Spark 3.5.5 с совместимым RAPIDS. На CPU-стенде применяется обычный профиль Spark. [Документация NVIDIA RAPIDS](https://docs.nvidia.com/spark-rapids/index.html)
- `data_check.py` проверяет схему Great Expectations, восстанавливает CSV из локального кэша DVC и читает две версии Delta Lake. GitHub Actions проверяет данные и синтаксис, затем разворачивает стенд и выполняет нагрузочные проверки. Результаты также записываются в локальный MLflow.
- `report.md` содержит итоги замеров и границы проверки.
