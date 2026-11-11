"""Проверка CSV, фиксация версии DVC и чтение двух версий Delta."""

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

import great_expectations as gx
import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'tmp'
SOURCE = ('https://api.kaggle.com/v1/datasets/download/siddharth0935/'
          'hospital-readmission-predictionsynthetic-dataset')
CSV_SHA256 = '8bc0c352a0990d7089b1f9035aa3a50dc24711bc56a9d93c0fdf38877733daf3'


def validate(frame):
    context = gx.get_context(mode='ephemeral')
    asset = context.data_sources.add_pandas(name='hospital').add_dataframe_asset(name='csv')
    batch = asset.add_batch_definition_whole_dataframe(name='batch')
    suite = context.suites.add(gx.ExpectationSuite(name='readmission'))
    for expectation in [
        gx.expectations.ExpectTableRowCountToEqual(value=30000),
        gx.expectations.ExpectColumnValuesToBeUnique(column='patient_id'),
        gx.expectations.ExpectColumnValuesToNotBeNull(column='patient_id'),
        gx.expectations.ExpectColumnValuesToBeBetween(column='length_of_stay', min_value=1, max_value=365),
        gx.expectations.ExpectColumnValuesToBeInSet(column='readmitted_30_days', value_set=['Yes', 'No']),
    ]:
        suite.add_expectation(expectation)
    definition = context.validation_definitions.add(gx.ValidationDefinition(name='csv', data=batch, suite=suite))
    good = definition.run(batch_parameters={'dataframe': frame})
    broken = frame.copy()
    broken.loc[0, 'readmitted_30_days'] = 'Invalid'
    bad = definition.run(batch_parameters={'dataframe': broken})
    if not good.success or bad.success:
        raise ValueError('Нарушена проверка качества входных данных')
    return {'passed': int(good.statistics['successful_expectations']),
            'checked': int(good.statistics['evaluated_expectations']), 'invalid_rejected': not bad.success}


def dvc_check(path):
    env = os.environ.copy()
    if env.get('LAB6_DVC_PYTHONPATH'):
        env['PYTHONPATH'] = env['LAB6_DVC_PYTHONPATH']
    env['DVC_NO_ANALYTICS'] = 'true'
    with tempfile.TemporaryDirectory(dir=WORK) as directory:
        local = Path(directory) / 'hospital.csv'
        shutil.copyfile(path, local)
        for args in [('init', '--no-scm'), ('config', 'cache.type', 'copy'), ('add', local.name)]:
            subprocess.run([sys.executable, '-m', 'dvc', *args], cwd=directory, env=env, check=True, stdout=subprocess.DEVNULL)
        pointer = local.with_suffix('.csv.dvc').read_text()
        expected = (ROOT / 'data/hospital.csv.dvc').read_text()
        if pointer != expected:
            raise ValueError('Указатель DVC расходится с данными')
        local.unlink()
        subprocess.run([sys.executable, '-m', 'dvc', 'checkout'], cwd=directory, env=env, check=True, stdout=subprocess.DEVNULL)
        if hashlib.sha256(local.read_bytes()).hexdigest() != CSV_SHA256:
            raise ValueError('DVC восстановил другую версию CSV')
    return True


def delta_check(frame):
    groups = frame.groupby(['discharge_destination', 'gender'], as_index=False).size()
    with tempfile.TemporaryDirectory(dir=WORK) as directory:
        write_deltalake(directory, pa.Table.from_pandas(groups, preserve_index=False))
        updated = groups.copy()
        updated.loc[0, 'size'] += 1
        write_deltalake(directory, pa.Table.from_pandas(updated, preserve_index=False), mode='overwrite')
        totals = [int(DeltaTable(directory, version=version).to_pandas()['size'].sum()) for version in (0, 1)]
    if totals != [30000, 30001]:
        raise ValueError(f'Ошибка проверки Delta time travel {totals}')
    return {'v0_rows': totals[0], 'v1_rows': totals[1]}


def prepare(source=None):
    WORK.mkdir(exist_ok=True)
    target = WORK / 'hospital.csv'
    if source and Path(source).resolve() != target:
        shutil.copyfile(source, target)
    elif not target.is_file():
        with urlopen(SOURCE, timeout=90) as response:
            with ZipFile(io.BytesIO(response.read())) as archive:
                target.write_bytes(archive.read('hospital_readmissions_30k.csv'))
    if hashlib.sha256(target.read_bytes()).hexdigest() != CSV_SHA256:
        raise ValueError('SHA-256 входного CSV не совпадает с версией лабораторной')
    frame = pd.read_csv(target)
    results = {'rows': len(frame), 'sha256': CSV_SHA256, 'validation': validate(frame),
               'dvc_restore': dvc_check(target), 'delta': delta_check(frame)}
    (WORK / 'data.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    return results


if __name__ == '__main__':
    print(json.dumps(prepare(sys.argv[1] if len(sys.argv) > 1 else None), ensure_ascii=False))
