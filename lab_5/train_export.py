"""Обучение на CSV варианта 6 и экспорт логистической модели в ONNX."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import FEATURES, MODEL_PATH, MODEL_VERSION, build_features, scores, session

DATA_URL = ('https://api.kaggle.com/v1/datasets/download/siddharth0935/'
            'hospital-readmission-predictionsynthetic-dataset')
DATA_MEMBER = 'hospital_readmissions_30k.csv'


def read_data(path):
    if path is None:
        with urlopen(DATA_URL, timeout=45) as response, ZipFile(io.BytesIO(response.read())) as archive:
            payload = archive.read(DATA_MEMBER)
    else:
        payload = path.read_bytes()
    rows = csv.DictReader(io.StringIO(payload.decode('utf-8-sig')))
    x, y = [], []
    for row in rows:
        profile = [float(row['age']), float(row['bmi']), float(row['diabetes'] == 'Yes'),
                   float(row['hypertension'] == 'Yes')]
        x.append(build_features(profile, {'stay': int(row['length_of_stay']), 'destination': row['discharge_destination']}))
        y.append(int(row['readmitted_30_days'] == 'Yes'))
    return np.asarray(x, dtype=np.float32), np.asarray(y), hashlib.sha256(payload).hexdigest()


def export(model):
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    weights = (classifier.coef_[0] / scaler.scale_).reshape(-1, 1)
    bias = classifier.intercept_ - scaler.mean_ @ weights
    nodes = [helper.make_node('MatMul', ['input', 'weights'], ['linear']),
             helper.make_node('Add', ['linear', 'bias'], ['logit']),
             helper.make_node('Sigmoid', ['logit'], ['score'])]
    graph = helper.make_graph(nodes, MODEL_VERSION,
                              [helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, len(FEATURES)])],
                              [helper.make_tensor_value_info('score', TensorProto.FLOAT, [None, 1])],
                              [numpy_helper.from_array(weights.astype(np.float32), 'weights'),
                               numpy_helper.from_array(bias.astype(np.float32), 'bias')])
    artifact = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
    onnx.checker.check_model(artifact)
    onnx.save_model(artifact, MODEL_PATH)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path)
    args = parser.parse_args()
    x, y, fingerprint = read_data(args.csv)
    x_train, rest, y_train, y_rest = train_test_split(x, y, test_size=.4, stratify=y, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(rest, y_rest, test_size=.5, stratify=y_rest, random_state=42)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(x_train, y_train)
    export(model)
    runtime = session()
    actual = scores(runtime, x_test)
    expected = model.predict_proba(x_test)[:, 1]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    neg = np.sort(scores(runtime, x_val)[y_val == 0])
    threshold = float(np.nextafter(neg[-max(1, int(len(neg) * .007))], np.float32(np.inf)))
    flagged = actual >= threshold
    result = {'model': MODEL_VERSION, 'features': FEATURES, 'rows': len(x), 'source_sha256': fingerprint,
              'onnx_sha256': hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
              'test_auc': float(roc_auc_score(y_test, actual)), 'test_ap': float(average_precision_score(y_test, actual)),
              'threshold': threshold, 'test_fpr': float(flagged[y_test == 0].mean()),
              'test_recall': float(flagged[y_test == 1].mean()),
              'max_onnx_difference': float(np.max(np.abs(actual - expected)))}
    MODEL_PATH.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
