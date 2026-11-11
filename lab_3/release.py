"""Проверка релиза в staging и локальное переключение версии витрины."""

import argparse
import csv
import json
from itertools import islice
from pathlib import Path
from uuid import uuid4

import duckdb
from filelock import FileLock

from demo import fixture, write_csv
from pipeline import run_pipeline
from source import json_text


def smoke(db_path):
    with duckdb.connect(str(db_path), read_only=True) as db:
        raw, distinct_keys = db.execute('SELECT count(*), count(DISTINCT idempotency_key) FROM raw.encounters').fetchone()
        mart, readmitted, groups = db.execute('''
            SELECT sum(encounter_count), sum(readmitted_count), count(*) FROM analytics.mart_readmissions
        ''').fetchone()
        if not (raw > 0 and raw == distinct_keys == mart and 0 <= readmitted <= mart and 1 <= groups <= 9):
            raise RuntimeError('Smoke test: несогласованные итоги raw и витрины')
    return {'rows': raw, 'groups': groups, 'readmission_rate': readmitted / mart}


def write_pointer(path, version):
    temp = path.with_suffix('.part')
    temp.write_text(json_text(version) + '\n', encoding='utf-8')
    temp.replace(path)


def release(source, work, fail_after_publish=False):
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    current = work / 'current.json'
    with FileLock(str(work / 'release.lock'), timeout=60):
        previous = json.loads(current.read_text(encoding='utf-8')) if current.exists() else None
        version = uuid4().hex[:12]
        candidate = work / 'versions' / version
        candidate.mkdir(parents=True)
        with Path(source).open(encoding='utf-8-sig', newline='') as input_file:
            reader = csv.reader(input_file)
            sample = write_csv(candidate / 'sample.csv', next(reader, []), islice(reader, 1000))

        run_pipeline(str(sample), str(candidate / 'canary'))
        canary = smoke(candidate / 'canary' / 'hospital.duckdb')
        production = candidate / 'full'
        metrics = run_pipeline(str(Path(source).resolve()), str(production))
        verified = smoke(production / 'hospital.duckdb')
        published = {'version': version, **verified, 'duration_sec': metrics['duration_sec'],
                     'accepted_share': metrics['accepted_share']}
        events = {'canary': canary, 'before': previous, 'candidate': published}
        if previous:
            events['readmission_drift_pp'] = (published['readmission_rate'] - previous['readmission_rate']) * 100
            events['accepted_share_drift_pp'] = (published['accepted_share'] - previous['accepted_share']) * 100
            events['duration_change_sec'] = published['duration_sec'] - previous['duration_sec']
            if abs(events['readmission_drift_pp']) > 5 or abs(events['accepted_share_drift_pp']) > 5:
                events['drift_alert'] = True
        write_pointer(current, published)
        try:
            if fail_after_publish:
                raise RuntimeError('Введён сбой проверки опубликованной версии')
            active = json.loads(current.read_text(encoding='utf-8'))
            smoke(work / 'versions' / active['version'] / 'full' / 'hospital.duckdb')
        except Exception as exc:
            if previous is None:
                current.unlink()
            else:
                write_pointer(current, previous)
            events.update(status='rolled_back', error=str(exc))
        else:
            events['status'] = 'published'
        (candidate / 'release.json').write_text(json_text(events) + '\n', encoding='utf-8')
        return events


def verify_release(work, source):
    first = release(source, work)
    assert first['status'] == 'published'
    with Path(source).open(encoding='utf-8-sig', newline='') as input_file:
        header, *rows = csv.reader(input_file)
    outcome = header.index('readmitted_30_days')
    shifted = 'Yes' if first['candidate']['readmission_rate'] < 0.5 else 'No'
    for row in rows:
        row[outcome] = shifted
    drift_source = write_csv(work / 'drift.csv', header, rows)
    failed = release(drift_source, work, fail_after_publish=True)
    assert failed['status'] == 'rolled_back'
    assert failed['drift_alert']
    assert json.loads((work / 'current.json').read_text(encoding='utf-8')) == first['candidate']
    result = {'successful_release': first, 'failed_release': failed}
    (work / 'verification.json').write_text(json_text(result) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('tmp/release'))
    parser.add_argument('--source', type=Path, help='По умолчанию локальная тестовая выгрузка из 120 строк.')
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    source = args.source or fixture(args.work / 'fixture.csv')
    result = verify_release(args.work, source)
    print(f"Релиз: {result['successful_release']['status']}; сбой: {result['failed_release']['status']}")


if __name__ == '__main__':
    main()
