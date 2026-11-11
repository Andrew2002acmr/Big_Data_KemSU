"""Источники, преобразование признаков и разбиение без утечки целевого поля."""

import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
HOSPITAL_URL = ('https://api.kaggle.com/v1/datasets/download/siddharth0935/'
                'hospital-readmission-predictionsynthetic-dataset')
SENSOR_URL = ('https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/'
              'ambient_temperature_system_failure.csv')
HOSPITAL_FILE = 'hospital_readmissions_30k.csv'
ID = 'patient_id'
TARGET = 'readmitted_30_days'
EVENT_TIME = pd.Timestamp('2024-01-01T00:00:00Z')
SEED = 42
FEATURES = (
    'age', 'bmi', 'systolic', 'diastolic', 'cholesterol', 'medication_count', 'length_of_stay',
    'diabetes_flag', 'hypertension_flag', 'discharge_rehab', 'discharge_nursing',
)
FEATURE_RANGES = dict(zip(FEATURES, ((0, 120), (1, 100), (1, 300), (1, 200), (1, 1000),
                                     (0, 100), (1, 365), (0, 1), (0, 1), (0, 1), (0, 1)), strict=True))


def download(path, url, member=None):
    path = Path(path)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=45) as response:
            payload = response.read()
        if member:
            with ZipFile(io.BytesIO(payload)) as archive:
                payload = archive.read(member)
        temporary = path.with_suffix('.part')
        temporary.write_bytes(payload)
        temporary.replace(path)
    return path


def hospital_features(path):
    raw = pd.read_csv(path)
    if not raw[ID].is_unique or not raw[ID].gt(0).all() or not raw[TARGET].isin(['Yes', 'No']).all():
        raise ValueError('Ожидаются уникальные положительные идентификаторы и известная метка')
    frame = raw[[ID, 'age', 'bmi', 'cholesterol', 'medication_count', 'length_of_stay']].copy()
    pressure = raw['blood_pressure'].str.extract(r'^([0-9]{1,3})/([0-9]{1,3})$').astype(float)
    valid = pressure[0] > pressure[1]
    frame['systolic'] = pressure[0].where(valid)
    frame['diastolic'] = pressure[1].where(valid)
    for column in ('diabetes', 'hypertension'):
        frame[f'{column}_flag'] = raw[column].map({'Yes': 1., 'No': 0.})
    for key, value in (('discharge_rehab', 'Rehab'), ('discharge_nursing', 'Nursing_Facility')):
        destination = raw['discharge_destination']
        frame[key] = destination.eq(value).astype(float).where(destination.isin(['Home', 'Rehab', 'Nursing_Facility']))
    for name, (lower, upper) in FEATURE_RANGES.items():
        frame[name] = pd.to_numeric(frame[name], errors='coerce').astype(float).where(lambda values: values.between(lower, upper))
    frame['event_timestamp'] = EVENT_TIME
    labels = raw.set_index(ID)[TARGET].eq('Yes').astype(int)
    eda = {
        'rows': len(raw), 'duplicate_ids': int(raw[ID].duplicated().sum()),
        'positive': int(labels.sum()), 'negative': int(len(labels) - labels.sum()),
        'missing_raw': {name: int(count) for name, count in raw.isna().sum().items()},
        'missing_features': {name: int(count) for name, count in frame[list(FEATURES)].isna().sum().items()},
        'feature_target_correlation': frame.set_index(ID)[list(FEATURES)].corrwith(labels).fillna(0).to_dict(),
        'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }
    return frame, labels, eda


def split_dataset(features, labels):
    train, rest = train_test_split(features.index, test_size=.4, random_state=SEED, stratify=labels)
    valid, test = train_test_split(rest, test_size=.5, random_state=SEED, stratify=labels.loc[rest])
    return [(features.loc[ids], labels.loc[ids]) for ids in (train, valid, test)]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
