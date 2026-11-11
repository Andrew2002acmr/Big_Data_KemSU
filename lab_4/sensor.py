"""Многошаговый прогноз NAB и воспроизведение контура автоотключения."""

from itertools import product
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss

from data import SENSOR_URL, download, save_json

H = 12
LAGS = np.array([0, 1, 2, 3, 6, 12, 23, 24, 47, 48, 167])
QUANTILES = (.50, .75, .90, .95)
FIT_END = pd.Timestamp('2013-10-31 23:00')
CALIBRATION_START = pd.Timestamp('2013-11-01')
HEALTHY_END = pd.Timestamp('2013-12-14 23:00')
TUNE_END = pd.Timestamp('2013-12-31 23:00')
TEST_START = pd.Timestamp('2014-03-01')
INCIDENTS = [(pd.Timestamp('2013-12-15'), pd.Timestamp('2013-12-30 23:00')),
             (pd.Timestamp('2014-03-29'), pd.Timestamp('2014-04-20 23:00'))]
TOLERANCE = pd.Timedelta(hours=24)


def lag_frame(history, hours):
    phase = 2 * np.pi * hours / 24
    return np.column_stack([history, np.sin(phase), np.cos(phase)])


def fit_quantile(x, y, q):
    model = lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=95, learning_rate=.065,
                              num_leaves=19, min_child_samples=25, n_jobs=2, verbosity=-1, random_state=42)
    model.fit(x, y)
    return model


def forecasts(series):
    actual = series.to_numpy()
    inputs = series.ffill(limit=3).to_numpy()
    origins = np.arange(LAGS.max(), len(series) - H)
    times = series.index[origins]
    future = origins[:, None] + np.arange(1, H + 1)
    targets = actual[future]
    x = lag_frame(inputs[origins[:, None] - LAGS], times.hour.to_numpy())
    direct = np.empty((len(origins), H, len(QUANTILES)))
    for index in range(H):
        y = targets[:, index]
        use = (series.index[future[:, index]] <= FIT_END) & np.isfinite(y) & np.isfinite(x[:, 0])
        for q_index, q in enumerate(QUANTILES):
            model = fit_quantile(x[use], y[use], q)
            direct[:, index, q_index] = model.booster_.predict(x, num_threads=2)
            if index == q_index == 0:
                one_step = model

    direct = np.maximum.accumulate(direct, axis=2)
    maximum = LAGS.max()
    history = np.empty((len(origins), maximum + 1 + H))
    history[:, :maximum + 1] = inputs[origins[:, None] - np.arange(maximum, -1, -1)]
    hours = times.hour.to_numpy()
    for step in range(H):
        matrix = lag_frame(history[:, maximum + step - LAGS], (hours + step) % 24)
        history[:, maximum + step + 1] = one_step.booster_.predict(matrix, num_threads=2)
    recursive = history[:, -H:]
    calibration = (times >= CALIBRATION_START) & (times + pd.Timedelta(hours=H) <= HEALTHY_END)
    residual = targets[calibration] - recursive[calibration]
    recursive_90 = recursive + np.maximum(np.nanquantile(residual, .9, axis=0), 0)
    return times, actual[origins], targets, direct, recursive, recursive_90


def event_metrics(events, incident, days):
    notices = [event for event in events if event['action'] == 'notify']
    start, end = incident
    hits = [event for event in notices if start - TOLERANCE <= pd.Timestamp(event['time']) <= end + TOLERANCE]
    false = len(notices) - len(hits)
    lead = (start - pd.Timestamp(hits[0]['time'])).total_seconds() / 3600 if hits else None
    return {'notifications': len(notices), 'true_notifications': len(hits), 'false_notifications': false,
            'precision': len(hits) / len(notices) if notices else None, 'recall': int(bool(hits)),
            'lead_hours': lead, 'false_per_day': false / days, 'missed_incidents': int(not hits),
            'false_shutdowns': sum(event['action'] == 'shutdown' and not start - TOLERANCE <= pd.Timestamp(event['time']) <= end + TOLERANCE
                                   for event in events)}


def alarm_events(times, values, prediction, residual, params):
    threshold = params['critical']
    risk = np.max([np.where(prediction[:, :, i] > threshold, 1 - quantile, 0.)
                   for i, quantile in enumerate(QUANTILES)], axis=(0, 2))
    cusum, count, blocked_until = 0., 0, -1
    events = []
    for position, (timestamp, value, error, probability) in enumerate(zip(times, values, residual, risk, strict=True)):
        if not np.isfinite(value) or not np.isfinite(error):
            cusum, count = 0., 0
            continue
        cusum = max(0., cusum + error - params['cusum_allowance'])
        eligible = (probability > params['min_probability'] and cusum > params['cusum_limit']
                    and position >= blocked_until)
        count = count + 1 if eligible else 0
        if params['persistence'] <= count <= params['persistence'] + 2:
            action = ('notify', 'throttle', 'shutdown')[count - params['persistence']]
            events.append({'time': timestamp.isoformat(), 'action': action, 'risk_lower_bound': float(probability),
                           'cusum': float(cusum), 'temperature': float(value)})
            if action == 'shutdown':
                blocked_until = position + params['cooldown_hours']
                cusum, count = 0., 0
    return events


def choose_policy(times, values, direct, residual, series):
    healthy = residual[(times >= CALIBRATION_START) & (times <= HEALTHY_END)]
    sigma = float(np.nanstd(healthy))
    historical = series.loc[:FIT_END].dropna()
    window = (times >= CALIBRATION_START) & (times <= TUNE_END)
    subset = (times[window], values[window], direct[window], residual[window])
    days = (TUNE_END - CALIBRATION_START).total_seconds() / 86400
    ranked = []
    for critical, p, k, cooldown, multiplier in product((.95, .975, .99), (.05, .10, .20), (2, 3, 4), (12, 24, 48), (2, 4)):
        policy = {'critical': float(historical.quantile(critical)), 'min_probability': p, 'persistence': k,
                  'cooldown_hours': cooldown, 'cusum_allowance': sigma * .5, 'cusum_limit': sigma * multiplier}
        journal = alarm_events(*subset, policy)
        evaluation = event_metrics(journal, INCIDENTS[0], days)
        # Один ложный сигнал стоит 1, ложное отключение дополнительно 3, пропуск аварии 8.
        cost = evaluation['false_notifications'] + 3 * evaluation['false_shutdowns'] + 8 * evaluation['missed_incidents']
        lead = evaluation['lead_hours'] if evaluation['lead_hours'] is not None else -10000
        ranked.append(((cost, -lead, len(journal)), policy, evaluation, journal))
    _, policy, evaluation, journal = min(ranked, key=lambda option: option[0])
    return policy, evaluation, journal


def step_metrics(times, targets, point, upper, denominator):
    output = []
    for step in range(H):
        selected = (times >= TEST_START) & np.isfinite(targets[:, step])
        y = targets[selected, step]
        mae = float(np.mean(np.abs(y - point[selected, step])))
        output.append({'step': step + 1, 'mae': mae, 'mase': mae / denominator,
                       'pinball_p90': float(mean_pinball_loss(y, upper[selected, step], alpha=.9)), 'n': int(selected.sum())})
    return output


def plot_sensor(out, series, times, prediction, journals, comparison, threshold):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.8), gridspec_kw={'height_ratios': [1.2, 1.2, 1]})
    windows = [(CALIBRATION_START, TUNE_END, 'Калибровка'), (TEST_START, series.index[-1], 'Отложенная проверка')]
    for axis, (start, end, title), incident, journal in zip(axes[:2], windows, INCIDENTS, journals, strict=True):
        horizon_times = times + pd.Timedelta(hours=H)
        selected = (horizon_times >= start) & (horizon_times <= end)
        point = prediction[selected, H - 1, 0]
        band = prediction[selected, H - 1, 2]
        axis.plot(series.loc[start:end], color='#687983', linewidth=.75, label='Факт')
        axis.plot(horizon_times[selected], point, color='#175e75', linewidth=.9, label='Медиана, +12 ч')
        axis.fill_between(horizon_times[selected], point, band, color='#69b4ba', alpha=.35, label='Медиана — q0,90')
        axis.axvspan(*incident, color='#cf7936', alpha=.20, label='Авария')
        axis.axhline(threshold, linestyle='--', linewidth=.8, color='#952d37', label='Порог')
        for event in journal:
            if event['action'] == 'notify':
                axis.axvline(pd.Timestamp(event['time']), color='#952d37', alpha=.5)
        axis.set(ylabel='Температура, ед. ряда', title=f'NAB · {title}')
        axis.legend(ncol=3, loc='upper right', fontsize=8)
    for name, label, color in (('direct', 'Прямой', '#175e75'), ('recursive', 'Рекурсивный', '#be6329')):
        axes[2].plot([p['step'] for p in comparison[name]], [p['mae'] for p in comparison[name]],
                     marker='o', markersize=3, label=label, color=color)
    axes[2].set(xlabel='Горизонт, часы', ylabel='MAE', xticks=[1, 3, 6, 9, 12])
    axes[2].grid(axis='y', alpha=.2)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out / 'sensor.png', dpi=150)
    plt.close(fig)


def run_sensor(source, out):
    path = Path(source) if source else download(out / 'sensor.csv', SENSOR_URL)
    raw = pd.read_csv(path, parse_dates=['timestamp'])
    series = raw.set_index('timestamp')['value'].sort_index().asfreq('h')
    times, values, targets, direct, recursive, rec_90 = forecasts(series)
    residual = np.r_[np.nan, values[1:] - direct[:-1, 0, 0]]
    policy, calibration, calibration_log = choose_policy(times, values, direct, residual, series)
    test = times >= TEST_START
    journal = alarm_events(times[test], values[test], direct[test], residual[test], policy)
    days = (times[test][-1] - times[test][0]).total_seconds() / 86400
    seasonal_error = float(series.loc[:FIT_END].diff(24).abs().mean())
    comparison = {'direct': step_metrics(times, targets, direct[:, :, 0], direct[:, :, 2], seasonal_error),
                  'recursive': step_metrics(times, targets, recursive, rec_90, seasonal_error)}
    result = {'source_rows': len(raw), 'hourly_gaps': int(series.isna().sum()), 'horizon': H,
              'fit_end': FIT_END.isoformat(), 'mase_denominator': seasonal_error, 'policy': policy,
              'calibration': calibration, 'calibration_journal': calibration_log, 'test_days': days,
              'test_incident_max': float(series.loc[INCIDENTS[1][0]:INCIDENTS[1][1]].max()),
              'test': event_metrics(journal, INCIDENTS[1], days), 'test_journal': journal, 'forecast': comparison}
    plot_sensor(out, series, times, direct, (calibration_log, journal), comparison, policy['critical'])
    save_json(out / 'sensor_metrics.json', result)
    return result
