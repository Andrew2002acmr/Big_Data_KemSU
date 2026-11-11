"""Обучение, тестовое развёртывание модели и упражнение NAB."""

import argparse
import logging
from pathlib import Path
from time import perf_counter

import mlflow
import numpy as np
from threadpoolctl import threadpool_limits

from data import FEATURES, HOSPITAL_FILE, HOSPITAL_URL, download, hospital_features, save_json, split_dataset
from figures import plot_hospital
from monitoring import monitor
from sensor import run_sensor
from serving_checks import acceptance, benchmark
from store import build_store
from training import (MODEL, assess, baseline, compare_models, configure_tracking, get_version,
                      metrics, register, replay_online, shap_linear)


def preflight(model, validation, score):
    example = validation[0].iloc[[0]]
    samples = []
    for _ in range(80):
        start = perf_counter()
        model.predict_proba(example)
        samples.append((perf_counter() - start) * 1000)
    latency = {'p95_ms': float(np.percentile(samples, 95)), 'errors': 0}
    if not acceptance(score, latency, score['auc']):
        raise ValueError(f'Модель не прошла предварительный допуск: {score}, {latency}')
    return latency


def interpretation(model, x, train):
    sample = x.iloc[:512]
    base, values = shap_linear(model, sample, baseline(model, train))
    logits = model.decision_function(sample)
    np.testing.assert_allclose(base + values.sum(axis=1), logits, atol=1e-10)
    return {'mean_abs_shap': dict(zip(FEATURES, np.abs(values).mean(axis=0).tolist(), strict=True)),
            'max_logit_error': float(np.max(np.abs(base + values.sum(axis=1) - logits))),
            'base_log_odds': base, 'example_log_odds': float(logits[0]),
            'example_shap': dict(zip(FEATURES, values[0].tolist(), strict=True))}


def run_hospital(source, work, request_count=240):
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    path = Path(source) if source else download(work / HOSPITAL_FILE, HOSPITAL_URL, HOSPITAL_FILE)
    frame, labels, eda = hospital_features(path)
    features, match = build_store(work, frame)
    train, validation, test = split_dataset(features, labels.loc[features.index])
    client = configure_tracking(work)
    with threadpool_limits(limits=2):
        batch, comparison = compare_models(train, validation)
        online, hours, warm_time = replay_online(train, validation, batch)
        batch_threshold, batch_valid = assess(batch, validation)
        online_threshold, online_valid = assess(online, validation)
        initial_gate = preflight(batch, validation, batch_valid)
        batch_version = register(client, batch, train[0], batch_threshold, 'Пакетная модель', batch_valid,
                                 comparison['Логистическая регрессия']['fit_sec'])
        online_version = register(client, online, train[0], online_threshold, 'Онлайн-модель', online_valid,
                                  warm_time + sum(hour['update_sec'] for hour in hours))
        client.set_registered_model_alias(MODEL, 'champion', batch_version)
        client.set_registered_model_alias(MODEL, 'candidate', online_version)
        x_test, y_test = test
        final = {'batch': metrics(y_test, batch.predict_proba(x_test)[:, 1], batch_threshold),
                 'online': metrics(y_test, online.predict_proba(x_test)[:, 1], online_threshold)}
        with mlflow.start_run(run_name='Batch и online: отложенная выборка'):
            mlflow.log_metrics({f'{strategy}_{name}': value for strategy, scores in final.items()
                                for name, value in scores.items()})
        interpretable = interpretation(batch, x_test, train[0])
        serving = benchmark(work, x_test.index[:request_count].tolist(), int(features.index.max()) + 1,
                            batch_valid, online_valid)
        if serving['baseline']['p95_ms'] >= 50:
            raise RuntimeError('Порог p95 < 50 мс не выполнен')
        drift = monitor(client, get_version(client, 'champion'), train, validation, test, work)
    assert drift['healthy']['max_psi'] < .30 < drift['injected']['max_psi'] and drift.get('retrain_version')
    result = {'eda': eda, 'feature_match': match, 'sizes': [len(split[0]) for split in (train, validation, test)],
              'models_validation': comparison, 'online_validation': online_valid, 'batch_threshold': batch_threshold,
              'online_threshold': online_threshold, 'final_test': final, 'online_hours': hours,
              'warm_online_fit_sec': warm_time, 'versions': {'batch': batch_version, 'online': online_version},
              'preflight': initial_gate, 'interpretation': interpretable, 'serving': serving, 'monitoring': drift}
    plot_hospital(work, eda, interpretable['mean_abs_shap'], hours)
    save_json(work / 'hospital_metrics.json', result)
    print_metrics(result)
    return result


def print_metrics(result):
    p95 = result['serving']['baseline']['p95_ms']
    auc = result['final_test']
    explanation = result['serving']['explanation']['p95_ms']
    print('## METRICS ##')
    print(f"FEATURE_MATCH_TEST_PASSED={result['feature_match']['passed']}")
    print(f"DATA_DRIFT_SCORE={result['monitoring']['healthy']['max_psi']:.4f}")
    print(f'INFERENCE_LATENCY_MS={p95:.2f}')
    print(f"ONLINE_BATCH_DIFFERENCE={auc['online']['auc'] - auc['batch']['auc']:.4f}")
    print(f'UNIQUE_REQUIREMENT_IMPACT={explanation - p95:.4f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', nargs='?', choices=['hospital', 'sensor', 'all'], default='all')
    parser.add_argument('--hospital', type=Path)
    parser.add_argument('--sensor', type=Path)
    parser.add_argument('--work', type=Path, default=Path('tmp/local'))
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)
    if args.part in ('hospital', 'all'):
        run_hospital(args.hospital, args.work)
    if args.part in ('sensor', 'all'):
        result = run_sensor(args.sensor, args.work)
        print('NAB:', result['test'])


if __name__ == '__main__':
    main()
