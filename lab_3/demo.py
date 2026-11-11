"""Воспроизводимая проверка идемпотентности, сбоев, DLQ и истории."""

import argparse
import csv
from pathlib import Path
from uuid import uuid4

import pandas as pd

from pipeline import STAGES, dbt_command, run_pipeline
from quality import input_checks, validate
from source import COLUMNS, extract, json_text
from warehouse import read_as_of, restore_version, warehouse_summary


def write_csv(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def fixture(path):
    genders = ('Female', 'Male', 'Other')
    destinations = ('Home', 'Rehab', 'Nursing_Facility')
    rows = [(index + 1, 40 + index % 30, genders[index % 3], '120/80', 190, 24.5, 'No', 'No',
             2, 1 + index % 10, destinations[(index // 3) % 3], 'Yes' if index % 8 == 0 else 'No')
            for index in range(120)]
    return write_csv(path, COLUMNS, rows)


def print_metrics(result):
    print('## METRICS ##')
    print(f"IDEMPOTENCY_TEST_PASSED={result['idempotent']}")
    print(f"DLQ_ERROR_COUNT={result['dlq_error_count']}")
    print(f"DATA_QUALITY_SCORE={result['initial']['data_quality_score']:.2f}")
    print(f"PIPELINE_DURATION_SEC={result['initial']['duration_sec']:.2f}")


def demonstrate(source_path, work, docs=False):
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if source_path is None:
        source_path = work / 'source.csv'
        extract(None, source_path)
    source_path = Path(source_path).resolve()
    database = work / 'hospital.duckdb'
    assert not database.exists(), 'Для демонстрации нужен новый каталог'
    with source_path.open(encoding='utf-8-sig', newline='') as input_file:
        header, *rows = csv.reader(input_file)

    initial = run_pipeline(str(source_path), str(work), docs=docs)
    repeated = run_pipeline(str(source_path), str(work))
    count = initial['raw_rows']
    assert count == repeated['raw_rows'] == repeated['warehouse']['distinct_keys']
    assert repeated['changed'] == 0 and repeated['warehouse'] == initial['warehouse']
    assert repeated['mart'] == initial['mart']

    age_index = header.index('age')
    old_age = int(rows[0][age_index])
    new_age = old_age - 1 if old_age == 120 else old_age + 1
    rows[0][age_index] = str(new_age)
    invalid = rows[0].copy()
    invalid[header.index('blood_pressure')] = 'bad-pressure'
    changed_header = ['patient_age' if name == 'age' else name for name in header]
    changed = write_csv(work / 'changed.csv', changed_header, [*rows, rows[0], invalid])
    repaired = run_pipeline(str(changed), str(work), faults=STAGES)
    assert repaired['changed'] == 1 and repaired['raw_rows'] == count and repaired['version'] == 2
    assert repaired['schema'] == 'v2' and repaired['duplicate_rows'] == 1
    assert repaired['warehouse']['dlq']['invalid_pressure_format'] >= 1

    current = warehouse_summary(database)
    incompatible = write_csv(work / 'incompatible.csv', ['patient_id'], [[1]])
    blocked = run_pipeline(str(incompatible), str(work), return_state=True)
    assert blocked.is_failed()
    after_block = warehouse_summary(database)
    assert after_block['rows'] == current['rows'] and after_block['versions'] == current['versions']
    assert after_block['dlq']['incompatible_schema'] == 1

    # Сохраняем реальный пример провала GX без обхода входного контроля в пайплайне.
    bad_data = pd.DataFrame([{'patient_id': '1', 'age': 30, 'length_of_stay': -1,
                              'discharge_destination': 'Home', 'readmitted_30_days': 'No'}])
    gx_failure = validate(bad_data, 'invalid_fixture', input_checks(), work / 'gx_invalid')
    assert gx_failure['passed'] == gx_failure['total'] - 1

    patient_id = rows[0][header.index('patient_id')]
    ages = []
    for version in (1, 2):
        snapshot = read_as_of(database, version)
        ages.append(int(snapshot.loc[snapshot['patient_id'] == patient_id, 'age'].item()))
    assert ages == [old_age, new_age]
    restored_version = restore_version(database, 1)
    assert restored_version == 3
    assert read_as_of(database, 1).equals(read_as_of(database, restored_version))
    restored_dir = work / 'restored'
    restored_dir.mkdir()
    dbt_command(['build'], database, restored_dir)

    result = {
        'initial': initial, 'repeated': repeated, 'repaired': repaired,
        'idempotent': True, 'dlq_error_count': sum(after_block['dlq'].values()),
        'dlq_by_reason': after_block['dlq'], 'incompatible_blocked': True,
        'gx_negative_test': gx_failure, 'injected_stages': list(STAGES),
        'history': {'old_age': old_age, 'updated_age': new_age, 'restored_version': restored_version},
    }
    (work / 'verification.json').write_text(json_text(result) + '\n', encoding='utf-8')
    print_metrics(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--source', type=Path)
    inputs.add_argument('--fixture', action='store_true')
    parser.add_argument('--work', type=Path, default=Path('tmp') / f'demo-{uuid4().hex[:8]}')
    parser.add_argument('--docs', action='store_true')
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    path = fixture(args.work / 'fixture.csv') if args.fixture else args.source
    demonstrate(path, args.work, docs=args.docs)


if __name__ == '__main__':
    main()
