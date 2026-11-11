import numpy as np
import pandas as pd
from scipy.special import expit

from api import in_canary
from data import EVENT_TIME, FEATURES, ID
from monitoring import drift_report, psi
from sensor import alarm_events, event_metrics
from serving_checks import acceptance
from store import build_store
from training import baseline, fit_batch, shap_linear


def feature_frame(size):
    generator = np.random.default_rng(42)
    return pd.DataFrame(generator.normal(size=(size, len(FEATURES))), columns=FEATURES)


def test_store_matches_direct_features(tmp_path):
    frame = feature_frame(16).assign(**{ID: np.arange(1, 17), 'event_timestamp': EVENT_TIME})
    historical, result = build_store(tmp_path, frame)
    assert result['passed'] and result['checked'] == 16
    np.testing.assert_allclose(historical.to_numpy(), frame[list(FEATURES)].to_numpy())


def test_explanations_sum_to_logit():
    x = feature_frame(100)
    y = (x['age'] + x['bmi'] > 0).astype(int)
    model = fit_batch(x, y)
    base, values = shap_linear(model, x.iloc[:10], baseline(model, x))
    np.testing.assert_allclose(expit(base + values.sum(axis=1)), model.predict_proba(x.iloc[:10])[:, 1])


def test_drift_and_retrain_gate():
    ref = feature_frame(100).abs()
    shifted = ref.assign(age=ref['age'] + 130)
    probs = np.linspace(.1, .3, len(ref))
    assert psi(ref['age'], ref['age']) == 0
    result = drift_report(ref, shifted, probs, probs, {'accuracy': .9, 'auc': .8}, {'accuracy': .65, 'auc': .5})
    assert result['max_psi'] > .3 and result['retrain']
    assert result['features']['age']['outside_range'] == len(ref)


def test_release_gate():
    assert 300 < sum(in_canary(str(index)) for index in range(10000)) < 700
    score = {'auc': .60, 'fpr': .007}
    assert acceptance(score, {'p95_ms': 10, 'errors': 0}, .60)
    assert not acceptance(score, {'p95_ms': 50, 'errors': 0}, .60)
    assert not acceptance(score, {'p95_ms': 10, 'errors': 1}, .60)


def test_sensor_action_ladder_and_cooldown():
    times = pd.date_range('2024-01-01', periods=10, freq='h')
    policy = {'critical': 75., 'min_probability': .1, 'persistence': 2, 'cooldown_hours': 12,
              'cusum_allowance': .25, 'cusum_limit': 1.}
    events = alarm_events(times, np.full(10, 80.), np.full((10, 12, 4), 81.), np.full(10, 2.), policy)
    assert [event['action'] for event in events] == ['notify', 'throttle', 'shutdown']
    measured = event_metrics(events, (times[3], times[-1]), 1)
    assert measured['recall'] == 1 and measured['lead_hours'] == 2
