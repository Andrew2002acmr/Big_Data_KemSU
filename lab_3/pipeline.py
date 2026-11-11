"""Ежедневное обновление витрины повторных госпитализаций."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import duckdb
import pandas as pd
from filelock import FileLock
from prefect import flow, get_run_logger, task
from prefect.client.schemas.schedules import CronSchedule
from prefect.runtime import flow_run, task_run
from prefect.tasks import exponential_backoff

from quality import input_checks, numeric_input, output_checks, validate
from source import DataError, TemporaryError, extract, inspect_batch, json_text
from warehouse import FIELDS, load_raw, warehouse_summary, write_dlq

ROOT = Path(__file__).resolve().parent
STAGES = ('extract', 'validate', 'load', 'dbt', 'quality')


def retry_temporary(_task, _run, state):
    return isinstance(state.result(raise_on_failure=False), (TemporaryError, duckdb.IOException))


RETRY = dict(retries=3, retry_delay_seconds=exponential_backoff(1), retry_condition_fn=retry_temporary)


def fail_once(stage, enabled):
    if stage in enabled and task_run.run_count == 1:
        get_run_logger().warning('Введён временный сбой на этапе %s', stage)
        raise TemporaryError(f'Тестовый сбой: {stage}')


@task(**RETRY, persist_result=False)
def extract_batch(source_path, cache, faults):
    fail_once('extract', faults)
    return extract(source_path, cache)


@task(**RETRY, persist_result=False)
def validate_batch(data, db_path, run_dir, dlq_alert, faults):
    logger = get_run_logger()
    fail_once('validate', faults)
    batch_id, records, rejected, info = inspect_batch(data)
    write_dlq(db_path, rejected)
    if len(rejected) >= dlq_alert:
        logger.warning('DLQ threshold: %s ошибок в выгрузке %s', len(rejected), batch_id[:12])
    if info.get('schema_drift'):
        logger.warning('Schema drift: совместимое приведение к v1 через %s', info['schema'])
    if info['blocked']:
        logger.error('Schema/CSV alert: загрузка остановлена, подробности в ops.dlq')
        raise DataError('Выгрузка не соответствует контракту')
    if not records:
        raise DataError('Нет пригодных к загрузке строк')

    frame = pd.DataFrame(records, columns=FIELDS)
    result = validate(numeric_input(frame), 'before_load', input_checks(), run_dir / 'gx_before')
    if result['failed']:
        write_dlq(db_path, [(batch_id, 0, 'gx_before_failed', json_text(result))])
        raise DataError(f"Входной контроль GX: {result['failed']}")
    logger.info('Вход: %s строк, приняты %s, отклонены %s', info['input_rows'], len(frame), len(rejected))
    return batch_id, frame, {**info, 'accepted_rows': len(frame), 'dlq_rows': len(rejected), 'gx_before': result}


@task(**RETRY, persist_result=False)
def load_batch(db_path, batch_id, frame, faults):
    result = load_raw(db_path, batch_id, frame, lambda: fail_once('load', faults))
    get_run_logger().info('DuckDB: %s', result)
    return result


def dbt_command(command, db_path, run_dir):
    args = [sys.executable, '-c', 'from dbt.cli.main import cli; cli()', '--no-use-colors',
            '--no-send-anonymous-usage-stats', *command, '--project-dir', str(ROOT / 'dbt'),
            '--profiles-dir', str(ROOT / 'dbt'), '--target-path', str(run_dir / 'dbt'),
            '--log-path', str(run_dir / 'dbt_logs')]
    env = {**os.environ, 'LAB3_DB': str(db_path), 'PYTHONIOENCODING': 'utf-8'}
    try:
        result = subprocess.run(args, env=env, capture_output=True, text=True, encoding='utf-8', timeout=180)
    except subprocess.TimeoutExpired as exc:
        raise TemporaryError('Превышено время ожидания dbt') from exc
    name = '_'.join(command)
    (run_dir / f'dbt_{name}.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        raise DataError(f'dbt {name}: код {result.returncode}, лог в {run_dir}')


@task(**RETRY, persist_result=False)
def build_models(db_path, run_dir, faults):
    dbt_command(['build'], db_path, run_dir)
    fail_once('dbt', faults)
    get_run_logger().info('dbt build завершён: 4 модели, 8 тестов')


@task(**RETRY, persist_result=False)
def verify_mart(db_path, batch_id, run_dir, faults):
    fail_once('quality', faults)
    with duckdb.connect(str(db_path), read_only=True) as db:
        mart = db.execute('SELECT * FROM analytics.mart_readmissions ORDER BY destination_id, gender').fetchdf()
        destinations = [value for value, in db.execute('SELECT destination_id FROM analytics.dim_destination').fetchall()]
    result = validate(mart, 'after_dbt', output_checks(destinations), run_dir / 'gx_after')
    if result['failed']:
        write_dlq(db_path, [(batch_id, 0, 'gx_after_failed', json_text(result))])
        raise DataError(f"Выходной контроль GX: {result['failed']}")
    return result, mart.to_dict(orient='records')


@flow(name='Hospital readmissions ELT', persist_result=False)
def run_pipeline(source_path=None, work='tmp/local', faults=(), dlq_alert=1, duration_alert=900, docs=False):
    started = perf_counter()
    logger = get_run_logger()
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    run_dir = work / 'runs' / str(flow_run.id)
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = work / 'hospital.duckdb'
    durations = {}
    previous = None

    def timed(stage, function, *args):
        nonlocal previous
        before = perf_counter()
        previous = function.submit(*args, wait_for=[previous] if previous is not None else [])
        result = previous.result()
        durations[stage] = round(perf_counter() - before, 3)
        return result

    # Один writer на файл. Параллельные окружения имеют отдельные базы.
    with FileLock(str(work / 'writer.lock'), timeout=60):
        data = timed('extract', extract_batch, source_path, work / 'source.csv', faults)
        batch_id, frame, inspection = timed('validate', validate_batch, data, db_path, run_dir, dlq_alert, faults)
        load = timed('load', load_batch, db_path, batch_id, frame, faults)
        timed('dbt', build_models, db_path, run_dir, faults)
        gx_after, mart = timed('quality', verify_mart, db_path, batch_id, run_dir, faults)
        summary = warehouse_summary(db_path)
        if docs:
            dbt_command(['docs', 'generate'], db_path, run_dir)

    duration = round(perf_counter() - started, 3)
    if duration > duration_alert:
        logger.warning('SLA alert: %.2f секунд при лимите %s', duration, duration_alert)
    quality = (inspection['gx_before']['passed'] + gx_after['passed']) / (inspection['gx_before']['total'] + gx_after['total'])
    metrics = {
        **inspection, **load, 'gx_after': gx_after, 'warehouse': summary, 'mart': mart,
        'batch_id': batch_id, 'run_id': str(flow_run.id), 'stage_seconds': durations,
        'duration_sec': duration, 'data_quality_score': quality,
        'accepted_share': len(frame) / inspection['input_rows'],
    }
    (run_dir / 'metrics.json').write_text(json_text(metrics) + '\n', encoding='utf-8')
    logger.info('Пайплайн: %s строк, %.2f с, GX %.0f%%', summary['rows'], duration, 100 * quality)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Локальный CSV. По умолчанию выгрузка Kaggle.')
    parser.add_argument('--work', type=Path, default=Path('tmp/local'))
    parser.add_argument('--docs', action='store_true', help='Дополнительно сформировать dbt Docs.')
    parser.add_argument('--serve', action='store_true', help='Развернуть в Prefect с ежедневным расписанием.')
    args = parser.parse_args()
    source_path = str(args.source.resolve()) if args.source else None
    work = str(args.work.resolve())
    if args.serve:
        run_pipeline.serve(name='daily-readmissions', parameters={'source_path': source_path, 'work': work},
                           schedules=[CronSchedule(cron='0 5 * * *', timezone='Asia/Novosibirsk')])
    else:
        result = run_pipeline(source_path, work, docs=args.docs)
        print(f"Результат: {args.work / 'runs' / result['run_id'] / 'metrics.json'}")


if __name__ == '__main__':
    main()
