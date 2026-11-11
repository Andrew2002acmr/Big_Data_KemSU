"""Загрузка CSV, контракт входных данных и классификация ошибок."""

import csv
import hashlib
import io
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from zipfile import ZipFile

DATA_URL = ('https://api.kaggle.com/v1/datasets/download/siddharth0935/'
            'hospital-readmission-predictionsynthetic-dataset')
DATA_FILE = 'hospital_readmissions_30k.csv'
SOURCE = 'hospital_readmissions'
COLUMNS = (
    'patient_id', 'age', 'gender', 'blood_pressure', 'cholesterol', 'bmi',
    'diabetes', 'hypertension', 'medication_count', 'length_of_stay',
    'discharge_destination', 'readmitted_30_days',
)
SCHEMAS = {'v1': {}, 'v2': {'patient_age': 'age'}}
DESTINATIONS = {'Home', 'Rehab', 'Nursing_Facility'}
RANGES = {
    'patient_id': (1, 2**63 - 1), 'age': (0, 120), 'cholesterol': (1, 1000),
    'medication_count': (0, 100), 'length_of_stay': (1, 365),
}
CATEGORIES = {
    'gender': {'Female', 'Male', 'Other'}, 'discharge_destination': DESTINATIONS,
    **{name: {'Yes', 'No'} for name in ('diabetes', 'hypertension', 'readmitted_30_days')},
}


class TemporaryError(Exception):
    """Ошибка инфраструктуры, при которой есть смысл повторить задачу."""


class DataError(Exception):
    """Невосстановимая ошибка данных текущей выгрузки."""


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def extract(source_path, cache):
    if source_path is not None:
        return Path(source_path).read_bytes()
    if not cache.is_file():
        try:
            with urlopen(DATA_URL, timeout=30) as response:
                blob = response.read()
        except HTTPError as exc:
            if exc.code == 429 or 500 <= exc.code < 600:
                raise TemporaryError(f'Kaggle HTTP {exc.code}') from exc
            raise
        except (URLError, TimeoutError) as exc:
            raise TemporaryError('Kaggle недоступен') from exc
        with ZipFile(io.BytesIO(blob)) as archive:
            data = archive.read(DATA_FILE)
        temp = cache.with_suffix('.part')
        temp.write_bytes(data)
        temp.replace(cache)
    return cache.read_bytes()


def normalize(row):
    values = dict(zip(COLUMNS, (item.strip() for item in row), strict=True))
    for name, (lower, upper) in RANGES.items():
        try:
            value = int(values[name])
            if not lower <= value <= upper:
                raise ValueError
        except ValueError:
            return None, f'invalid_{name}'
        values[name] = str(value)

    try:
        bmi = float(values['bmi'])
        if not 0 < bmi <= 100:
            raise ValueError
    except ValueError:
        return None, 'invalid_bmi'
    values['bmi'] = str(bmi)

    for name, allowed in CATEGORIES.items():
        if values[name] not in allowed:
            return None, f'invalid_{name}'
    pressure = re.fullmatch(r'([0-9]{1,3})/([0-9]{1,3})', values['blood_pressure'])
    if not pressure:
        return None, 'invalid_pressure_format'
    systolic, diastolic = map(int, pressure.groups())
    if not 0 < diastolic < systolic:
        return None, 'invalid_pressure_order'
    values['blood_pressure'] = f'{systolic}/{diastolic}'
    return values, None


def inspect_batch(data):
    batch_id = digest(data)
    accepted = defaultdict(dict)
    rejected = []

    def reject(line, reason, payload):
        rejected.append((batch_id, line, reason, json_text(payload)))

    try:
        reader = csv.reader(io.StringIO(data.decode('utf-8-sig'), newline=''), strict=True)
        header = next(reader, [])
    except (UnicodeError, csv.Error) as exc:
        reject(0, 'invalid_csv', {'error': str(exc)})
        return batch_id, [], rejected, {'blocked': True, 'input_rows': 0}

    # Добавочные поля можно отбросить, известный псевдоним учитывает версию v2.
    version = 'v1' if set(COLUMNS).issubset(header) else 'v2'
    mapping = SCHEMAS[version]
    mapped = [mapping.get(name, name) for name in header]
    missing = sorted(set(COLUMNS) - set(mapped))
    if missing or len(set(mapped)) != len(mapped):
        reject(0, 'incompatible_schema', {'header': header, 'missing': missing})
        return batch_id, [], rejected, {'blocked': True, 'input_rows': 0}

    indexes = [mapped.index(name) for name in COLUMNS]
    metrics = {
        'input_rows': 0, 'duplicate_rows': 0, 'blocked': False, 'schema': version,
        'schema_drift': version != 'v1' or set(header) != set(COLUMNS),
    }
    try:
        for line, row in enumerate(reader, start=2):
            metrics['input_rows'] += 1
            if len(row) != len(header):
                reject(line, 'wrong_column_count', row)
                continue
            values, reason = normalize([row[index] for index in indexes])
            if reason:
                reject(line, reason, row)
                continue
            key = digest(f"{SOURCE}:{values['patient_id']}".encode())
            payload_hash = digest(json_text(values).encode())
            accepted[key].setdefault(payload_hash, []).append((line, row, values))
    except csv.Error as exc:
        reject(reader.line_num, 'invalid_csv', {'error': str(exc)})
        metrics['blocked'] = True
        return batch_id, [], rejected, metrics

    records = []
    for key, variants in accepted.items():
        if len(variants) > 1:
            for rows in variants.values():
                for line, row, _ in rows:
                    reject(line, 'conflicting_patient_id', row)
            continue
        payload_hash, rows = next(iter(variants.items()))
        metrics['duplicate_rows'] += len(rows) - 1
        records.append((key, payload_hash, *rows[0][2].values()))
    return batch_id, records, rejected, metrics
