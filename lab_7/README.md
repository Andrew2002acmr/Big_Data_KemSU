# Лабораторная работа № 7

Вариант 6. Аудиокаталог русской речи с разметкой вопросов и поиском похожих записей с ограничением по длительности. Эксперимент выполнен локально на CPU, Windows и Python 3.12.

## Запуск

Из каталога `lab_7` выполни команды PowerShell.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab lab_7.ipynb
```

Выполни все ячейки по порядку. Эквивалентный запуск без Jupyter и отдельный замер препроцессинга доступны командами ниже.

```powershell
.\.venv\Scripts\python.exe run.py
.\.venv\Scripts\python.exe benchmark_preprocess.py
```

На первом запуске скачиваются два русских шарда и метаданные Common Voice, а также аудио- и текстовые веса CLAP. Для зависимостей и данных понадобится несколько гигабайт свободного места. `run.py --download-only` заранее загружает исходные данные. Аудио, индекс и Parquet-кэш хранятся в `tmp/`. Замер полных шардов обрабатывает все их записи и занимает больше времени, чем основной прогон на 1 000 записях.

## Данные и результаты

- Публичное зеркало содержит русскоязычную версию Common Voice 17.0 с исходными MP3 и TSV. Код фиксирует ревизию зеркала. [Common Voice 17.0 на Hugging Face](https://huggingface.co/datasets/fsicoli/common_voice_17_0) · [Карточка русского корпуса](https://github.com/MontrealCorpusTools/mfa-models/blob/main/corpus/russian/common_voice_russian/17.0/README.md)
- Замороженный CLAP HTSAT используется через два INT8 ONNX-энкодера с зафиксированными ревизией и SHA-256. [Модель CLAP](https://huggingface.co/laion/clap-htsat-unfused) · [Экспорт ONNX](https://huggingface.co/Xenova/clap-htsat-unfused)
- [Отчёт](report.md) содержит итоговое сравнение. [Ноутбук](lab_7.ipynb) показывает шесть этапов и контрольные проверки.

На 1 000 записях табличная модель получила лучший ROC-AUC на обоих разбиениях по дикторам. Векторный поиск LanceDB показал recall@5 около 0,99 относительно точного перебора.
