"""Verify saved results and build the concise report, figures and notebook."""
from pathlib import Path
import json,statistics,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark import verify,KEYS

ROOT=Path(__file__).parent
R=ROOT/'results'
A=ROOT/'assets';A.mkdir(exist_ok=True)


def load(name):return json.loads((R/name).read_text(encoding='utf-8'))
def n(value,d=2):return f'{value:,.{d}f}'.replace(',',' ').replace('.',',')


def main():
    data=load('data.json');opt=load('spark_optimizations.json')
    engines=['pandas','dask','spark','polars','duckdb']
    sizes=['full','100mb','1gb']
    results={(e,s):load(f'{e}_{s}.json') for e in engines for s in sizes}
    for (engine,size),value in results.items():
        case=next(c for c in data['cases'] if c['name']==size)
        verify(value['result'],case['reference'])
        assert value['rows']==case['rows'] and len(value['runs'])==3
    assert all(v['success'] for v in data['checks'].values())
    profiles=opt['profiles']
    baseline=profiles['baseline']['median_seconds']
    best_name=min(profiles,key=lambda name:profiles[name]['median_seconds'])
    stage_run=profiles['baseline']['runs'][0]
    stages=stage_run['stages']
    final_stage=stages[-1]
    groups=data['cases'][0]['reference']
    counts=[int(x['count']) for x in groups]
    cv=statistics.pstdev(counts)/statistics.mean(counts)
    top=max(groups,key=lambda x:x['count'])
    salt_factor=profiles['shuffle8']['median_seconds']/profiles['salt4']['median_seconds']
    dask_percent=100*(1-results['dask','1gb']['median']['seconds']/results['pandas','1gb']['median']['seconds'])
    cpu_percent={e:100*(1-results[e,'1gb']['median']['seconds']/results['pandas','1gb']['median']['seconds'])
                 for e in ['polars','duckdb']}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    for field,title,ylabel,filename in [
        ('seconds','Время запроса','Секунды','time.png'),
        ('peak_rss_bytes','Пиковая память процесса и потомков','ГиБ RSS','memory.png')]:
        fig,ax=plt.subplots(figsize=(7.3,3.7))
        for engine in engines:
            x=[results[engine,s]['input_bytes']/1e6 for s in sizes]
            y=[(results[engine,s]['max_peak_rss_bytes']/2**30 if field=='peak_rss_bytes'
                else results[engine,s]['median'][field]) for s in sizes]
            ax.plot(x,y,marker='o',label=engine)
        ax.set(xscale='log',xlabel='Размер CSV, МБ (логарифмическая шкала)',ylabel=ylabel,title=title)
        ax.grid(alpha=.25);ax.legend(ncol=3);fig.tight_layout();fig.savefig(A/filename,dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.3,3.8))
    ordered=sorted(groups,key=lambda x:x['count'],reverse=True)
    ax.bar(range(len(ordered)),[x['count'] for x in ordered])
    ax.set(xticks=range(len(ordered)),xticklabels=['/'.join(str(x[k]) for k in KEYS) for x in ordered],
           ylabel='Число записей',title='Распределение составных ключей: исходные 30 000 строк')
    ax.tick_params(axis='x',rotation=75,labelsize=7);fig.tight_layout();fig.savefig(A/'skew.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.3,3.7))
    for name,p in profiles.items():
        cpu=statistics.median(run['cpu_seconds'] for run in p['runs'])
        gain=100*(1-p['median_seconds']/baseline)
        ax.scatter(cpu,gain,label=name,s=38)
    ax.axhline(0,color='gray',linewidth=.8)
    ax.set(xlabel='Затраты CPU, процесс + JVM, с',ylabel='Сокращение времени к baseline, %',title='Оптимизации Spark: затраты и выгода')
    ax.margins(x=.12,y=.16);ax.legend(ncol=2,loc='upper right',fontsize=8)
    ax.grid(alpha=.25);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(A/'optimization.png',dpi=150);plt.close(fig)

    def metrics_table(selected):
        lines=['| Инструмент / CSV | Время, с | RSS, ГиБ | CPU, с | Чтения / записи ОС |',
               '|---|---:|---:|---:|---:|']
        for e,s in selected:
            v=results[e,s];m=v['median']
            lines.append(f"| {e} / {s} | {n(m['seconds'],3)} | {n(v['max_peak_rss_bytes']/2**30,3)} | {n(m['cpu_seconds'])} | {n(m['read_count'],0)} / {n(m['write_count'],0)} |")
        return '\n'.join(lines)
    main_table=metrics_table([(e,s) for e in ['pandas','dask','spark'] for s in sizes])
    track_table=metrics_table([(e,s) for e in ['pandas','polars','duckdb'] for s in ['100mb','1gb']])
    stage_table='\n'.join(f"| {x['stageId']} | {x['numTasks']} | {n(x['inputBytes']/1e6,3)} | {n(x['shuffleWriteBytes']/1024,2)} | {n(x['executorRunTime'],0)} |" for x in stages)
    optimization_table='\n'.join(f"| {name} | {n(v['median_seconds'],3)} | {n(baseline/v['median_seconds'])}× | {n(statistics.median(r['shuffle_write_bytes'] for r in v['runs'])/1024,2)} | {v['runs'][0]['stage_count']} |" for name,v in profiles.items())
    cases_table='\n'.join(f"| {c['name']} | {n(c['rows'],0)} | {n(c['csv_bytes']/1e6,3)} | {n(c['parquet_bytes']/1e6,3)} |" for c in data['cases'])
    dev={'pandas':1,'dask':2,'spark':4,'polars':1.5,'duckdb':1.5}
    support={'pandas':.5,'dask':1,'spark':2,'polars':.5,'duckdb':.5}
    costs='\n'.join(f"| {e} | {n((results[e,'1gb']['startup_seconds']+results[e,'1gb']['median']['seconds'])*20/3600,4)} | {n(dev[e],1)} | {n(support[e],1)} |" for e in engines)
    text=f'''# Лабораторная работа 2. Сравнение инструментов обработки данных

**Вариант 6**

## 1. Подготовка и задача

Задача: группировка по `gender`, `discharge_destination`, `readmitted_30_days`; число записей, сумма и средняя длительность пребывания, средний BMI. Это описательная оценка нагрузки по группам. Синтетический набор не доказывает клинические закономерности.

В исходнике {data['source_rows']} строк. Удалено дубликатов: {data['duplicates_removed']}, записей с критическими пропусками: {data['critical_missing_removed']}; очищено {data['clean_rows']} строк. Все 5 проверок Great Expectations прошли: ID заполнен и уникален, длительность 1–10, BMI 0–100, исход `Yes/No`. Проверка BMI задаёт технический контракт набора, а не медицинский диагноз.

| Набор | Строки | CSV, МБ | Parquet, МБ |
|---|---:|---:|---:|
{cases_table}

Исходник меньше 100 МБ, поэтому большие наборы получены повторением очищенных записей с уникальными тестовыми ID. Это нагрузочные копии, не новые наблюдения. Размеры 100 МБ/1 ГБ относятся к CSV; Parquet содержит те же строки. Данные генерируются в `tmp/`, в Git хранятся только код и результаты.

Окружение: Python 3.12, pandas 3.1.0, Dask 2026.8.0, Spark 3.5.7, Polars 2.1.0, DuckDB 2.0.0; Windows, 8 логических CPU, {n(results['pandas','full']['ram_bytes']/2**30)} ГиБ RAM. Spark: `local[4]`, heap 2 ГиБ, исходно 200 shuffle-партиций, AQE выключен.

## 2. Сравнение производительности

Для всех движков одинаковые файлы, составной ключ и агрегаты. Импорт и запуск исключены из времени запроса и сохранены отдельно. Время включает чтение, вычисления и материализацию результата; Spark измеряется после `collect()`, Dask после `compute()`. Каждый набор — отдельный процесс, внутри 3 последовательных прогона; время и CPU — медианы, RSS — максимум. Кэш ОС не очищался.

{main_table}

RSS измерялся каждые 20 мс по процессу и потомкам, включая JVM. CPU — суммарное процессорное время, а не длительность запроса. I/O — счётчики операций ОС: включают служебный ввод-вывод, IPC и работу JVM; mmap и кэш отражаются в них иначе, поэтому это не число физических обращений к диску. Байты I/O и нормированный CPU% сохранены в `results/`.

![Время](assets/time.png)
![Память](assets/memory.png)

На 1 ГБ относительное сокращение времени Dask к pandas: **{n(dask_percent)}%** по формуле `(t_pandas − t_dask) / t_pandas`. Отрицательное значение означает замедление. Pandas успешно обработал 1 ГБ: предел отказа в этом опыте не найден, универсальную границу памяти из этих замеров вывести нельзя. Для прямого сравнения Spark дополнительно измерен на тех же 100 МБ и 1 ГБ, а не только на полном исходнике.

## 3. Физический план Spark

Базовый план: `FileScan CSV → partial HashAggregate → Exchange hashpartitioning(200) → final HashAggregate`. Catalyst оставил только 5 нужных столбцов, Tungsten/whole-stage code generation виден по меткам `*(...)`. Полные планы до/после AQE и salting: [results/plans](results/plans).

В первом измеренном baseline на 1 ГБ число выполненных стадий — **{stage_run['stage_count']}**; записано в shuffle **{n(stage_run['shuffle_write_bytes']/1024)} КиБ**, прочитано **{n(stage_run['shuffle_read_bytes']/1024)} КиБ**. Данные получены из Spark status REST API после выполнения запроса. Вход — сумма счётчиков Spark `inputBytes` по задачам, а не размер файла в каталоге.

| Stage | Задач | Вход, МБ | Запись shuffle, КиБ | Сумма executorRunTime, мс |
|---|---:|---:|---:|---:|
{stage_table}

На завершающей стадии непустых задач **{final_stage['nonempty_tasks']} из {final_stage['numTasks']}**; их медианное/max время executor — **{n(final_stage['task_runtime_ms']['median'],1)} / {n(final_stage['task_runtime_ms']['max'],1)} мс**. Пустые задачи исключены из этой статистики, но участвуют в накладных расходах планирования.

Первая стадия читает файл и предварительно агрегирует; следующая объединяет частичные суммы. Сумма времени задач не равна времени стадии из-за параллелизма. Основные гипотезы: уменьшить накладные расходы 200 задач для 18 групп и ускорить чтение форматом Parquet. Для этой агрегации map-side combine резко уменьшает shuffle относительно размера входного CSV.

## 4. Skew и salting

В исходнике {len(groups)} составных ключей. Самый частый: `{' / '.join(str(top[k]) for k in KEYS)}` — {n(top['count']/data['clean_rows']*100)}% строк. Коэффициент вариации частот `std/mean` = **{n(cv,3)}**; max/mean = **{n(max(counts)/statistics.mean(counts),3)}**.

![Ключи](assets/skew.png)

Реализована соль `patient_id % 4`: сначала группировка по ключу с солью, затем объединение сумм и счётчиков; средние считаются через сумму/число, не как среднее средних. На компактных частотах исходника оценена загрузка 8 hash-партиций: CV **{n(opt['hash8_load_estimate']['before']['cv'],3)} → {n(opt['hash8_load_estimate']['salt4']['cv'],3)}**. Это оценка распределения исходных строк, а не измеренный shuffle Spark.

На 1 ГБ salting по отношению к варианту `shuffle8` дал **{n(salt_factor)}×** (`t_shuffle8/t_salt4`). Значение меньше 1 означает ухудшение. Перекос частот сам по себе не доказывает узкое место reducer: частичная агрегация уже сжимает группы, а соль добавляет вторую перегруппировку. Фактические task-метрики и объём shuffle сохранены для каждого прогона.

## 5. Оптимизации Spark

По 3 прогона на 1 ГБ, с разогревом JVM вне замеров и сменой порядка вариантов. Таблица сравнивает все варианты с CSV/200/AQE-off; `combined` включает Parquet, 8 партиций и AQE. Для `salt4` дополнительно рассчитано сравнение с `shuffle8` в разделе 4.

| Вариант | Время, с | baseline / время | Запись shuffle, КиБ | Стадий, 1-й прогон |
|---|---:|---:|---:|---:|
{optimization_table}

`shuffle8`: только 200 → 8; `repartition8`: дополнительное распределение сырых строк по ключу; `aqe`: только AQE; `parquet`: только формат. Для Parquet API дополнительно учитывает служебную стадию чтения метаданных, поэтому 3 стадии не означают 2 shuffle. Вариант с минимальной медианой среди проверенных: **{best_name}**, **{n(profiles[best_name]['median_seconds'],3)} с**. Это лучший из измеренных вариантов, не полный поиск всех настроек. Время предварительного преобразования CSV → Parquet не включено: выигрыш окупается при повторных чтениях.

![Затраты и выгода](assets/optimization.png)

## 6. CPU-трек A: pandas, Polars, DuckDB

Тот же запрос и те же CSV, по 3 прогона. Polars использует ленивое streaming-выполнение и 4 потока, DuckDB — SQL и 4 потока; pandas — C-парсер и группировку DataFrame.

{track_table}

На 1 ГБ сокращение времени против pandas: Polars **{n(cpu_percent['polars'])}%**, DuckDB **{n(cpu_percent['duckdb'])}%**. Возможные причины различий — параллельная обработка, векторизация и продвижение проекции. Отдельного GPU-замера нет: выполнен обязательный CPU-трек. На коротких запросах накладные расходы GPU могут перекрыть выигрыш; утверждать GPU-ускорение по CPU-таблице нельзя. Команды и таблица трека: [gpu/README.md](gpu/README.md).

## 7. Выбор инструмента и стоимость

| Условие | Выбор | Компромисс |
|---|---|---|
| Исходные 30 000 строк, простые проверки | pandas | Ограничение памятью одного процесса |
| Аналитический SQL на одном узле | DuckDB | Не полноценная распределённая платформа |
| Большие локальные таблицы и ленивые вычисления | Polars | Другой API и модель выполнения |
| Существующий pandas-пайплайн, нужно масштабирование | Dask | Планирование задач и контроль размера партиций |
| Данных больше ресурсов узла, сложные распределённые ETL | Spark | JVM, shuffle и эксплуатация кластера |

Учебная модель стоимости: одинаковая машина 8 CPU/16 ГиБ за **20 ₽/час**; один холодный запуск стоит `(startup + median_query) × 20 / 3600`. Разработка/поддержка ниже — оценочные трудозатраты, не фактически засечённое время; при 1 000 ₽/час годовая трудовая часть = `(разработка + 12 × поддержка) × 1000`. Хранение и сеть в эту частичную модель не входят.

| Инструмент | Вычисления 1 ГБ, ₽/запуск | Разработка, ч | Поддержка, ч/месяц |
|---|---:|---:|---:|
{costs}

Для текущего медицинского набора достаточно pandas; Polars/DuckDB стоит рассмотреть при росте локальных выгрузок. Spark обоснован при требованиях распределённого ETL, а не только размером CSV. Переход требует контроля схемы, корректности групп и сроков обработки. Данные без дат не позволяют обещать SLA обученного временного прогноза.

**Итог:** результаты всех пяти движков проверены по всем ключам, числу записей, суммам и средним; целочисленные агрегаты совпадают точно, для BMI разрешена погрешность вычислений с плавающей точкой. Измерены реальные стадии, shuffle, CPU и I/O; отрицательный эффект оптимизации сохраняется как результат.

МБ = 10⁶ байт, ГБ = 10⁹, ГиБ = 2³⁰. Локальные замеры не заменяют тестирование в распределённом кластере.
'''
    (ROOT/'report.md').write_text(text,encoding='utf-8')
    gpu=ROOT/'gpu';gpu.mkdir(exist_ok=True)
    (gpu/'README.md').write_text('# Трек A: сравнение на CPU\n\nВыполнен обязательный трек A. GPU-треки B/C не запускались.\n\n'+track_table+
        '\n\nЗапуск из папки `lab_2` после `prepare.py`:\n\n```powershell\n'+
        '\n'.join(f'python benchmark.py {e} {s} 3' for e in ['pandas','polars','duckdb'] for s in ['100mb','1gb'])+
        '\n```\n\nРазмеры относятся к CSV. Время — медиана трёх запусков, RAM — максимальный RSS. Все агрегаты проверяются автоматически.\n',encoding='utf-8')
    # Notebook contains actual saved conclusions plus reproducible commands, without fake executions.
    cells=[{'cell_type':'markdown','metadata':{},'source':text.split('## 1.')[0].splitlines(True)},
        {'cell_type':'markdown','metadata':{},'source':['Выполняйте из папки `lab_2`. Полный повтор экспериментов занимает время и создаёт около 1,3 ГБ временных данных. Готовые измерения находятся в `results/`.\n']},
        {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
         'source':['# Повторить все этапы последовательно; используются те же скрипты, что в отчёте.\n','import subprocess, sys\n','subprocess.run([sys.executable, "run.py"], check=True)\n']},
        {'cell_type':'markdown','metadata':{},'source':('## 1.'+text.split('## 1.',1)[1]).splitlines(True)}]
    for i,cell in enumerate(cells):cell['id']=f'lab2-{i}'
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
    (ROOT/'lab_2.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    assert opt['verified'] and all(len(v['runs'])==3 for v in profiles.values())
    (R/'verification.json').write_text(json.dumps({'comparisons':len(results),'runs_per_case':3,
        'spark_optimization_runs':sum(len(v['runs']) for v in profiles.values()),
        'all_keys_counts_sums_verified':True,'groups':len(groups),'source_sha256':data['source_sha256']},indent=2),encoding='utf-8')
    print('Report created:',len(text.split()),'words',flush=True)


if __name__=='__main__':main()
