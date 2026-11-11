"""Общий контракт признаков, ONNX и ключей Redis."""

import json
import os
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

import numpy as np
import onnxruntime as ort
from redis import Redis

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'model' / 'readmission.onnx'
FEATURES = ('age', 'bmi', 'diabetes', 'hypertension', 'length_of_stay', 'discharge_rehab', 'discharge_nursing')
DESTINATIONS = ('Home', 'Rehab', 'Nursing_Facility')
MODEL_VERSION = 'readmission-v1'
HTTP = build_opener(ProxyHandler({}))


def redis_client():
    return Redis(host=os.getenv('REDIS_HOST', 'redis'), port=int(os.getenv('REDIS_PORT', '6379')),
                 decode_responses=True, socket_timeout=5)


def feature_key(patient):
    return f'lab5:feature:v1:{patient}'


def key(run, category, event_id):
    return f'lab5:{run}:{category}:{event_id}'


def build_features(profile, event):
    return [*profile, event['stay'], float(event['destination'] == 'Rehab'),
            float(event['destination'] == 'Nursing_Facility')]


def session():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(MODEL_PATH), sess_options=options, providers=['CPUExecutionProvider'])


def scores(model, rows):
    return model.run(None, {'input': np.asarray(rows, dtype=np.float32)})[0].ravel()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
