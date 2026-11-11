"""Окна дрейфа, контроль качества и решение о переобучении."""

import json
import logging
from pathlib import Path
from urllib.request import Request, urlopen

import mlflow
import numpy as np
from scipy.stats import ks_2samp

from data import FEATURE_RANGES, SEED
from training import MODEL, assess, fit_batch, metrics, register

LOG = logging.getLogger(__name__)


def psi(reference, current):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if not len(reference) or not len(current):
        return None
    values = np.unique(reference)
    if len(values) == 1:
        return float((current != values[0]).mean() * 10)
    if len(values) <= 10:
        inner = (values[:-1] + values[1:]) / 2
    else:
        inner = np.unique(np.quantile(reference, np.arange(1, 10) / 10))
    bins = np.r_[-np.inf, inner, np.inf]
    left = np.maximum(np.histogram(reference, bins)[0] / len(reference), 1e-5)
    right = np.maximum(np.histogram(current, bins)[0] / len(current), 1e-5)
    return float(((right - left) * np.log(right / left)).sum())


def drift_report(reference, current, baseline_probs, probs, baseline_quality, quality, weekly=False):
    features = {}
    for name, (lower, upper) in FEATURE_RANGES.items():
        left, right = reference[name], current[name]
        samples = left.dropna(), right.dropna()
        ks = ks_2samp(*samples) if all(len(sample) for sample in samples) else None
        features[name] = {'psi': psi(left, right), 'ks_pvalue': float(ks.pvalue) if ks else None,
                          'missing': int(right.isna().sum()), 'outside_range': int((right.lt(lower) | right.gt(upper)).sum())}
    score = max((row['psi'] for row in features.values() if row['psi'] is not None), default=0.)
    accuracy_drop = baseline_quality['accuracy'] - quality['accuracy']
    auc_drop = baseline_quality['auc'] - quality['auc']
    prediction_psi = psi(baseline_probs, probs)
    messages = []
    if score > .2:
        messages.append('Сдвиг входных признаков: PSI > 0,2')
    if prediction_psi is not None and prediction_psi > .2:
        messages.append('Сдвиг распределения прогнозов')
    if any(row['outside_range'] or row['missing'] / len(current) > .05 for row in features.values()):
        messages.append('Нарушены диапазоны или порог пропусков')
    if accuracy_drop > .10 or auc_drop > .10:
        messages.append('Падение качества по полученным меткам')
    return {'max_psi': score, 'prediction_psi': prediction_psi, 'features': features, 'accuracy_drop': accuracy_drop,
            'auc_drop': auc_drop, 'alerts': messages, 'retrain': bool(score > .30 or accuracy_drop > .15 or weekly)}


def alert(work, messages, webhook=None):
    if not messages:
        return
    with (Path(work) / 'alerts.jsonl').open('a', encoding='utf-8') as output:
        output.write(json.dumps({'alerts': messages}, ensure_ascii=False) + '\n')
    LOG.warning('%s', ' | '.join(messages))
    if webhook:
        payload = json.dumps({'text': '\n'.join(messages)}).encode()
        with urlopen(Request(webhook, data=payload, headers={'Content-Type': 'application/json'}), timeout=10):
            pass


def monitor(client, champion, train, validation, test, work):
    model, version, meta = champion
    x, y = test
    half = len(x) // 2
    reference, current = x.iloc[:half], x.iloc[half:]
    y_ref, y_current = y.iloc[:half], y.iloc[half:]
    baseline_probs = model.predict_proba(reference)[:, 1]
    baseline = metrics(y_ref, baseline_probs, meta['threshold'])
    healthy_probs = model.predict_proba(current)[:, 1]
    healthy = drift_report(reference, current, baseline_probs, healthy_probs, baseline,
                           metrics(y_current, healthy_probs, meta['threshold']))

    shifted = current.assign(age=(current['age'] + 25).clip(upper=120), bmi=current['bmi'] + 8)
    shifted.loc[shifted.index[0], 'age'] = 150
    shifted.loc[shifted.index[:3], 'cholesterol'] = np.nan
    # Инъекция сдвига зависимости с полностью созревшими метками.
    corrupted = y_current.copy()
    flip = corrupted.sample(frac=.32, random_state=SEED).index
    corrupted.loc[flip] = 1 - corrupted.loc[flip]
    drift_probs = model.predict_proba(shifted)[:, 1]
    injected = drift_report(reference, shifted, baseline_probs, drift_probs, baseline,
                            metrics(corrupted, drift_probs, meta['threshold']))
    result = {'champion': version, 'healthy': healthy, 'injected': injected}
    alert(work, injected['alerts'])
    with mlflow.start_run(run_name='Мониторинг: контрольное окно и инъекция дрейфа'):
        for key, data in [('healthy', healthy), ('injected', injected)]:
            mlflow.log_metrics({f'{key}_psi': data['max_psi'], f'{key}_accuracy_drop': data['accuracy_drop']})
        mlflow.log_dict(result, 'drift.json')
    if injected['retrain']:
        # Отложенное окно остаётся проверочным. Для теста обрабатываем отдельный поток из train.
        x_train, y_train = train
        refreshed = x_train.assign(age=(x_train['age'] + 25).clip(upper=120), bmi=x_train['bmi'] + 8)
        new_model = fit_batch(refreshed, y_train)
        threshold, evaluation = assess(new_model, validation)
        retrained = register(client, new_model, refreshed, threshold, 'Автоматический retrain', evaluation)
        client.set_registered_model_alias(MODEL, 'retrain', retrained)
        result['retrain_version'] = retrained
    return result
