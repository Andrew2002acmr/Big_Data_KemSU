# Трек A: сравнение на CPU

Выполнен обязательный трек A. GPU-треки B/C не запускались.

| Инструмент / CSV | Время, с | RSS, ГиБ | CPU, с | Чтения / записи ОС |
|---|---:|---:|---:|---:|
| pandas / 100mb | 1,751 | 0,237 | 1,75 | 384 / 0 |
| pandas / 1gb | 17,123 | 1,647 | 18,47 | 3 817 / 0 |
| polars / 100mb | 0,226 | 0,326 | 0,75 | 4 / 0 |
| polars / 1gb | 1,865 | 0,656 | 7,12 | 31 / 0 |
| duckdb / 100mb | 0,518 | 0,204 | 1,92 | 4 / 0 |
| duckdb / 1gb | 4,632 | 0,291 | 19,55 | 32 / 0 |

Запуск из папки `lab_2` после `prepare.py`:

```powershell
python benchmark.py pandas 100mb 3
python benchmark.py pandas 1gb 3
python benchmark.py polars 100mb 3
python benchmark.py polars 1gb 3
python benchmark.py duckdb 100mb 3
python benchmark.py duckdb 1gb 3
```

Размеры относятся к CSV. Время — медиана трёх запусков, RAM — максимальный RSS. Все агрегаты проверяются автоматически.
