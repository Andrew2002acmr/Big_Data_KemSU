"""Исполнение SmartNoise и MOSTLY AI в отдельных окружениях."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from dp_experiment import BLOCKS, DELTA, DOMAINS_DP, dimensions, ledger_row, sample_block_codes


def mwem(train, epsilon):
    from snsynth.mwem import MWEMSynthesizer

    tables, ledger = [], []
    for columns, fraction in BLOCKS:
        budget = epsilon * fraction
        shape = dimensions(columns)
        # Два публичных якоря фиксируют домены MWEM без чтения приватных min/max.
        bounded = np.vstack([train[list(columns)].to_numpy(dtype=int), np.zeros(len(columns), dtype=int),
                             np.asarray(shape, dtype=int) - 1])
        # При замене слота чувствительность вектора ячеек 2.
        model = MWEMSynthesizer(epsilon=budget / 2, iterations=8, q_count=16,
                                marginal_width=len(columns), measure_only=True, mult_weights_iterations=6)
        model.fit(bounded, categorical_columns=list(range(len(columns))), preprocessor_eps=0)
        estimates, _, _ = model.synthetic_histograms[0]
        assert estimates.shape == shape and np.isclose(model.spent * 2, budget)
        tables.append(estimates)
        row = ledger_row('SmartNoise MWEM', '+'.join(columns), epsilon, budget, 0,
                         'журнал MWEM × 2; чистая DP при публичном размере и фиксированных доменах')
        row['neighbor'] = 'замена одного слота клиента; отсутствие кодируется публичной заглушкой'
        ledger.append(row)
    return sample_block_codes(tables, len(train), np.random.default_rng()), {
        'version': version('smartnoise-synth'), 'iterations': 8, 'measure_only': True,
        'public_slots': len(train), 'ledger': ledger,
    }


def mostlyai(train, epsilon, directory):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['HF_DATASETS_CACHE'] = str(directory / 'dataset_cache')
    import torch
    from mostlyai.sdk import MostlyAI

    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    protection = .20 * epsilon
    training = epsilon - protection
    noise = {1: 2.5, 3: 1.25, 10: .75}[epsilon]
    source = train.astype(str)
    sdk = MostlyAI(local=True, local_dir=directory / 'sdk')
    config = {'name': f'insurance-dp-{epsilon}', 'tables': [{
        'name': 'profiles', 'data': source,
        'columns': [{'name': name, 'model_encoding_type': 'TABULAR_CATEGORICAL'} for name in source],
        'tabular_model_configuration': {
            'model': 'MOSTLY_AI/Small', 'max_epochs': 8, 'batch_size': 128,
            'enable_model_report': False, 'enable_flexible_generation': False, 'value_protection': True,
            'differential_privacy': {'max_epsilon': epsilon, 'delta': DELTA, 'noise_multiplier': noise,
                                     'max_grad_norm': 1.0, 'value_protection_epsilon': protection},
        },
    }]}
    trained = sdk.train(config=config)
    result = sdk.probe(trained, size=len(train))
    cleaned = {}
    repairs = 0
    for name, domain in DOMAINS_DP.items():
        raw = pd.to_numeric(result[name], errors='coerce')
        invalid = raw.isna() | ~raw.isin(range(len(domain)))
        repairs += int(invalid.sum())
        cleaned[name] = raw.mask(invalid, 0).astype(int)
    rows = [ledger_row('MOSTLY AI TabularARGN', 'защита категорий', epsilon, protection, 0,
                       'DP-анализ; бюджет делится между столбцами'),
            ledger_row('MOSTLY AI TabularARGN', 'учёт обучения в SDK', epsilon, training, DELTA,
                       'Opacus RDP; номинальная часть совокупного max_epsilon')]
    return pd.DataFrame(cleaned), {'version': version('mostlyai'), 'engine': version('mostlyai-engine'),
                                  'epochs_cap': 8, 'noise_multiplier': noise, 'public_replacements': repairs,
                                  'validation_selection_separately_accounted': False,
                                  'ledger': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=['mwem', 'mostlyai'])
    parser.add_argument('epsilon', type=int, choices=[1, 3, 10])
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    original = pd.read_parquet(args.directory / 'train.parquet')
    start = perf_counter()
    if args.method == 'mwem':
        sample, metrics = mwem(original, args.epsilon)
    else:
        sample, metrics = mostlyai(original, args.epsilon, args.directory)
    sample.to_parquet(args.directory / 'sample.parquet', index=False)
    metrics['fit_and_sample_s'] = perf_counter() - start
    (args.directory / 'audit.json').write_text(json.dumps(metrics, ensure_ascii=False), encoding='utf-8')
