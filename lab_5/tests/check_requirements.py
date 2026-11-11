"""Автоматическая проверка результатов Docker-стенда."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, nargs='?', default=Path('tmp/results'))
    parser.add_argument('--with-spark', action='store_true')
    parser.add_argument('--with-node-failure', action='store_true')
    args = parser.parse_args()
    flink = json.loads((args.results / 'flink.json').read_text(encoding='utf-8'))
    assert flink['late']['accepted'] and flink['late']['duplicate_removed']
    assert flink['late']['side_output']['reason'] == 'window_closed'
    assert flink['late']['dlq']['reason'] == 'invalid_features'
    assert flink['late']['window']['count'] == 3 and flink['late']['window']['late'] == 1
    assert flink['fault']['passed'] and flink['fault']['lost'] == 0
    assert flink['data_drift_psi'] > .2 and flink['model_http_parity']
    assert flink['feature_audit']['feature_max_delta'] <= 1e-6
    for name in ('normal', 'peak'):
        assert flink[name]['parity_max_delta'] <= 1e-6
        assert flink[name]['p99_ms'] < 30, f"{name}: p99 {flink[name]['p99_ms']:.2f} ms, expected < 30"
    if args.with_spark:
        spark = json.loads((args.results / 'spark.json').read_text(encoding='utf-8'))
        assert spark['feature_audit']['feature_max_delta'] <= 1e-6
        assert spark['normal']['parity_max_delta'] <= 1e-6 and spark['fault']['passed']
    if args.with_node_failure:
        node = json.loads((args.results / 'node.json').read_text(encoding='utf-8'))
        assert node['queued'] == node['recovered'] and node['lost'] == 0 and node['kafka_lag_after'] == 0
    print('Все проверки стенда пройдены')


if __name__ == '__main__':
    main()
