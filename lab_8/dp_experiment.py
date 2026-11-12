"""OpenDP-гистограммы с публичными доменами, сравнение DP-генераторов и DP-SGD."""

import json
from math import prod
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from evaluation import DOMAINS, features, score, utility
from mock_insurance import PROFILE_COLUMNS, TARGET, YEARS

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'tmp'
EPSILONS = (1, 3, 10)
DELTA = 1e-5
PUBLIC_SAMPLE_SIZE = 10_500
UNIT = 'клиент: один профиль со всеми связанными данными'
NEIGHBOR = 'добавление/удаление одного клиента'
RISK_EDGES = (0, .2, .4, .55, .7, .85, 1)
PREMIUM_EDGES = (1500, 4500, 7000, 9500, 12000, 15000, 19000, 29000)
DOMAINS_DP = {**DOMAINS, 'risk_bin': tuple(range(len(RISK_EDGES) - 1)),
              'premium_bin': tuple(range(len(PREMIUM_EDGES) - 1)), 'tenure': tuple(range(7)),
              'prior_claims': tuple(range(5)), 'policy_year': YEARS, TARGET: (0, 1)}
BLOCKS = [
    (('risk_bin', 'prior_claims', 'product', TARGET), .55),
    (('risk_bin', TARGET, 'age_band'), .07),
    (('product', 'risk_bin', 'premium_bin'), .12),
    ((TARGET, 'prior_claims', 'tenure'), .10),
    (('district', 'channel', 'status', 'policy_year'), .16),
]


def encode_domains(frame):
    work = frame.copy()
    work['risk_bin'] = np.digitize(work.pop('risk_score'), RISK_EDGES[1:-1])
    work['premium_bin'] = np.digitize(work.pop('premium'), PREMIUM_EDGES[1:-1])
    codes = {column: pd.Categorical(work[column], categories=domain).codes.astype(int)
             for column, domain in DOMAINS_DP.items()}
    table = pd.DataFrame(codes)
    assert (table >= 0).all().all()
    return table


def decode_domains(codes, rng):
    frame = pd.DataFrame({name: np.asarray(domain)[codes[name].to_numpy(dtype=int)] for name, domain in DOMAINS_DP.items()})
    for field, output, edges in [('risk_bin', 'risk_score', RISK_EDGES), ('premium_bin', 'premium', PREMIUM_EDGES)]:
        values = frame.pop(field).to_numpy(dtype=int)
        boundaries = np.asarray(edges)
        frame[output] = rng.uniform(boundaries[values], boundaries[values + 1]).round(5 if output == 'risk_score' else 2)
    return frame[PROFILE_COLUMNS]


def dimensions(columns):
    return tuple(len(DOMAINS_DP[column]) for column in columns)


def histogram(table, columns):
    shape = dimensions(columns)
    coordinates = table[list(columns)].to_numpy(dtype=int).T
    linear = np.ravel_multi_index(coordinates, shape)
    return np.bincount(linear, minlength=prod(shape)).reshape(shape)


def sample_block_codes(tables, n, rng):
    def choose(weights, count):
        flattened = np.maximum(np.asarray(weights, dtype=float).ravel(), 0) + .01
        return rng.choice(len(flattened), size=count, p=flattened / flattened.sum())

    data = {}
    for index, ((columns, _), weights) in enumerate(zip(BLOCKS, tables, strict=True)):
        if index in (0, 4):
            values = np.unravel_index(choose(weights, n), weights.shape)
            data.update(dict(zip(columns, values, strict=True)))
        else:
            parents = np.column_stack([data[column] for column in columns[:-1]])
            result = np.empty(n, dtype=int)
            for group in np.unique(parents, axis=0):
                mask = np.all(parents == group, axis=1)
                result[mask] = choose(weights[tuple(group)], int(mask.sum()))
            data[columns[-1]] = result
    return pd.DataFrame(data)


def sample_blocks(tables, n, rng):
    return decode_domains(sample_block_codes(tables, n, rng), rng)


def ledger_row(method, operation, budget, epsilon, delta, accountant):
    return {'method': method, 'operation': operation, 'requested_epsilon': budget, 'epsilon_upper': float(epsilon),
            'delta': delta, 'unit': UNIT, 'neighbor': NEIGHBOR, 'accountant': accountant}


def opendp_sample(train, epsilon, n, label='main'):
    import opendp.prelude as dp

    dp.enable_features('contrib')
    codes = encode_domains(train)
    tables, ledger = [], []
    for columns, fraction in BLOCKS:
        budget = epsilon * fraction
        counts = histogram(codes, columns)
        # Для публичных ячеек добавление одного профиля меняет ровно один счётчик на 1.
        laplace = dp.m.make_laplace(dp.vector_domain(dp.atom_domain(T=int)), dp.l1_distance(T=int),
                                    scale=float(np.nextafter(1 / budget, np.inf)))
        assert laplace.check(1, budget)
        protected = np.asarray(laplace(counts.ravel().tolist())).reshape(counts.shape)
        tables.append(protected)
        ledger.append(ledger_row('OpenDP', f'{label}: ' + '+'.join(columns), epsilon, budget, 0,
                                 'дискретный Laplace, L1=1, последовательная композиция'))
    return sample_blocks(tables, n, np.random.default_rng()), ledger


def external(train, epsilon, method, reuse=False):
    directory = WORK / 'dp' / method / str(epsilon)
    directory.mkdir(parents=True, exist_ok=True)
    source, output, audit = (directory / name for name in ('train.parquet', 'sample.parquet', 'audit.json'))
    encoded = encode_domains(train)
    if reuse and output.is_file() and audit.is_file() and source.is_file() and encoded.equals(pd.read_parquet(source)):
        return decode_domains(pd.read_parquet(output), np.random.default_rng()), json.loads(audit.read_text(encoding='utf-8'))
    encoded.to_parquet(source, index=False)
    env = os.environ.copy()
    prefix = 'LAB8_MWEM' if method == 'mwem' else 'LAB8_MOSTLY'
    local = ROOT / ('.venv-mwem' if method == 'mwem' else '.venv-mostly') / (
        'Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    interpreter = env.get(prefix + '_PYTHON', str(local) if local.is_file() else sys.executable)
    if env.get(prefix + '_PATH'):
        env['PYTHONPATH'] = env[prefix + '_PATH']
    subprocess.run([interpreter, str(ROOT / 'dp_worker.py'), method, str(epsilon), str(directory)], env=env, check=True)
    details = json.loads(audit.read_text(encoding='utf-8'))
    return decode_domains(pd.read_parquet(output), np.random.default_rng()), details


def dp_sgd(train, test):
    import torch
    from opacus import PrivacyEngine
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.set_num_threads(3)
    x = torch.from_numpy(features(train))
    y = torch.from_numpy(train[TARGET].to_numpy(dtype='float32'))
    testing = torch.from_numpy(features(test))
    dataset = TensorDataset(x, y)
    results = {}
    for private in (False, True):
        model = nn.Linear(x.shape[1], 1)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
        optimizer = torch.optim.SGD(model.parameters(), lr=.34)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        noise = 1.2
        if private:
            engine = PrivacyEngine(accountant='rdp', secure_mode=False)
            model, optimizer, loader = engine.make_private(module=model, optimizer=optimizer, data_loader=loader,
                                                           noise_multiplier=noise, max_grad_norm=1.0)
        model.train()
        for _ in range(12):
            for batch, targets in loader:
                if not len(batch):
                    continue
                optimizer.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(model(batch).squeeze(1), targets)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            probabilities = model(testing).squeeze(1).sigmoid().numpy()
        results['private' if private else 'non_private'] = score(test, probabilities)
        if private:
            results['privacy'] = ledger_row('Opacus DP-SGD', '12 эпох', None, engine.get_epsilon(DELTA), DELTA, 'Opacus RDP')
            results['settings'] = {'noise_multiplier': noise, 'max_grad_norm': 1, 'batch_size': 256,
                                   'epochs': 12, 'secure_rng': False}
    return results


def sweep(train, test, baseline, methods=('opendp', 'mwem', 'mostlyai'), include_low_volume=True, reuse=False):
    rows, ledger, samples = [], [], {}
    for method in methods:
        for budget in EPSILONS:
            start = perf_counter()
            if method == 'opendp':
                sample, additions = opendp_sample(train, budget, PUBLIC_SAMPLE_SIZE)
                detail = {}
            else:
                sample, detail = external(train, budget, method, reuse=reuse)
                additions = detail.pop('ledger')
            ledger.extend(additions)
            measured, _ = utility(sample, test)
            print(f'{method}: epsilon={budget}, AUC={measured["auc"]:.3f}', flush=True)
            rows.append({'method': method, 'requested_epsilon': budget, 'epsilon_upper': sum(row['epsilon_upper'] for row in additions),
                         'delta': sum(row['delta'] for row in additions), 'tstr': measured,
                         'relative_auc': measured['auc'] / baseline, 'seconds': perf_counter() - start, 'details': detail})
            samples[(method, budget)] = sample
    if not include_low_volume:
        return {'runs': rows, 'ledger': ledger}, samples
    small = train.sample(300, random_state=102)
    small_synthetic, additions = opendp_sample(small, 1, PUBLIC_SAMPLE_SIZE, label='300 клиентов')
    ledger.extend(additions)
    measured, _ = utility(small_synthetic, test)
    return {'runs': rows, 'low_volume': {'train_n': 300, 'tstr': measured}, 'ledger': ledger}, samples
