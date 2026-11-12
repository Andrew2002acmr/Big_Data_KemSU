"""Оценка полезности, статистического сходства и эмпирического риска."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from mock_insurance import AGES, CHANNELS, DISTRICTS, PRODUCTS, STATUSES, TARGET

DOMAINS = dict(age_band=AGES, district=DISTRICTS, product=PRODUCTS, channel=CHANNELS, status=STATUSES)
BOUNDS = dict(tenure=(0, 6), premium=(1500, 29000), prior_claims=(0, 4), risk_score=(0, 1))
THRESHOLD = .30
FAIRNESS_LIMIT = .20


def features(frame):
    blocks = []
    for column, categories in DOMAINS.items():
        values = frame[column].to_numpy()
        blocks.append(np.column_stack([values == category for category in categories]))
    for column, (low, high) in BOUNDS.items():
        values = frame[column].to_numpy(dtype='float32')
        blocks.append((2 * np.clip((values - low) / (high - low), 0, 1) - 1)[:, None])
    return np.concatenate(blocks, axis=1, dtype='float32')


def score(frame, probabilities):
    target = frame[TARGET].to_numpy()
    return {'auc': float(roc_auc_score(target, probabilities)),
            'f1': float(f1_score(target, probabilities >= THRESHOLD, zero_division=0))}


def utility(train, test):
    if train[TARGET].nunique() < 2:
        prior = float(train[TARGET].mean())
        probabilities = np.full(len(test), prior)
    else:
        model = LogisticRegression(C=3, max_iter=250).fit(features(train), train[TARGET])
        probabilities = model.predict_proba(features(test))[:, 1]
    return score(test, probabilities), probabilities


def fairness(frame, probabilities):
    groups = {}
    positive = frame[TARGET].eq(1).to_numpy()
    for group in AGES:
        members = frame.age_band.eq(group).to_numpy() & positive
        groups[group] = {'positives': int(members.sum()), 'tpr': float((probabilities[members] >= THRESHOLD).mean())}
    rates = [row['tpr'] for row in groups.values()]
    return {'groups': groups, 'tpr_gap': max(rates) - min(rates),
            'threshold': THRESHOLD, 'limit': FAIRNESS_LIMIT}


def sdtypes(frame):
    return {'columns': {name: {'sdtype': 'categorical' if name in DOMAINS or name == TARGET else 'numerical'}
                        for name in frame}}


def fidelity(original, synthetic):
    from sdmetrics.reports.single_table import QualityReport

    report = QualityReport()
    report.generate(original, synthetic, sdtypes(original), verbose=False)
    values = report.get_properties().set_index('Property')['Score']
    columns = report.get_details('Column Shapes')
    pairs = report.get_details('Column Pair Trends')
    return {'overall': float(report.get_score()), 'column_shapes': float(values['Column Shapes']),
            'column_pair_trends': float(values['Column Pair Trends']),
            'columns': [{'column': row['Column'], 'score': float(row['Score'])}
                        for row in columns.to_dict('records') if pd.notna(row['Score'])],
            'evaluated_pairs': int(pairs['Score'].notna().sum())}


def distance_features(frame):
    base = features(frame).astype('float64')
    one_hot = sum(len(categories) for categories in DOMAINS.values())
    # Один несовпавший категориальный признак даёт 1, как полный диапазон числового.
    base[:, :one_hot] /= 2
    base[:, one_hot:] = (base[:, one_hot:] + 1) / 2
    return np.column_stack([base, (frame.policy_year.to_numpy() - 2019) / 6, frame[TARGET].to_numpy()])


def nearest(queries, gallery):
    out = []
    for start in range(0, len(queries), 128):
        block = cdist(queries[start:start + 128], gallery, metric='cityblock') / (len(BOUNDS) + len(DOMAINS) + 2)
        out.append(np.partition(block, 1, axis=1)[:, :2])
    return np.concatenate(out)


def dcr_audit(train, control, synthetic):
    n = min(1500, len(control), len(synthetic))
    base, heldout, released = (distance_features(frame.sample(n, random_state=seed))
                               for frame, seed in ((train, 71), (control, 72), (synthetic, 73)))
    dcr = {}
    for name, queries in [('synthetic', released), ('control', heldout)]:
        distances = nearest(queries, base)
        first, second = distances.T
        ratio = np.divide(first, second, out=np.zeros_like(first), where=second > 0)
        dcr[name] = {'dcr_p05': float(np.quantile(first, .05)), 'dcr_median': float(np.median(first)),
                     'nndr_median': float(np.median(ratio)), 'exact_rows': int((first == 0).sum())}
    # Простейшая атака: более близкая к выпуску запись считается участником обучения.
    scores = -np.concatenate([nearest(base, released)[:, 0], nearest(heldout, released)[:, 0]])
    target = np.repeat([1, 0], n)
    fpr, tpr, _ = roc_curve(target, scores)
    return {'n_per_split': n, **dcr,
            'membership_auc': float(roc_auc_score(target, scores)), 'membership_advantage': float(np.max(tpr - fpr))}


def wilson(successes, n):
    z = float(norm.ppf(.975))
    rate = successes / n
    center = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0, float(center - half)), min(1, float(center + half))]


def attack_audit(train, control, synthetic):
    from anonymeter.evaluators import InferenceEvaluator, LinkabilityEvaluator, SinglingOutEvaluator

    size = min(1800, len(control), len(synthetic))
    original, control, synthetic = (df.sample(size, random_state=seed).reset_index(drop=True)
                                    for df, seed in ((train, 81), (control, 82), (synthetic, 83)))
    columns = ['age_band', 'district', 'product', 'channel', 'status', 'tenure', 'prior_claims',
               'risk_score', 'premium', TARGET]
    # Для атак на опубликованный профиль используем одинаковую схему и размер трёх таблиц.
    original, control, synthetic = (frame[columns].copy() for frame in (original, control, synthetic))
    count = 250
    common = dict(ori=original, syn=synthetic, control=control, n_attacks=count)
    attacks = {
        'singling_out': SinglingOutEvaluator(**common, n_cols=4).evaluate(mode='multivariate'),
        'linkability': LinkabilityEvaluator(**common, aux_cols=(['age_band', 'district', 'product'],
                                                                ['channel', 'status', 'tenure', 'premium']),
                                            n_neighbors=1).evaluate(n_jobs=2),
        'inference': InferenceEvaluator(**common, aux_cols=['age_band', 'district', 'product', 'prior_claims',
                                                           'risk_score', 'premium'],
                                        secret=TARGET, regression=False).evaluate(n_jobs=2),
    }
    out = {}
    for name, evaluator in attacks.items():
        results = evaluator.results(confidence_level=.95)
        attack = int(results.n_success)
        baseline = int(results.n_baseline)
        control = int(results.n_control)
        risk = evaluator.risk(confidence_level=.95)
        out[name] = {'n': count, 'main': attack / count, 'baseline': baseline / count,
                     'control': control / count, 'main_ci95': wilson(attack, count),
                     'control_ci95': wilson(control, count), 'risk': float(risk.value),
                     'risk_ci95': list(map(float, risk.ci))}
    return out
