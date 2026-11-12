"""Сравнение GaussianCopula и CTGAN с внутренним подбором CTGAN."""

import random
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluation import BOUNDS, DOMAINS, fidelity, sdtypes, utility
from mock_insurance import PROFILE_COLUMNS, SEED, TARGET, YEARS, relational, validate_relations


def clean(sample):
    out = sample[PROFILE_COLUMNS].copy()
    if out.isna().any().any():
        raise ValueError('Генератор вернул пропуски в обязательных полях')
    for name, categories in DOMAINS.items():
        if not out[name].isin(categories).all():
            raise ValueError(f'Неизвестные значения в {name}')
    for name, (lower, upper) in BOUNDS.items():
        out[name] = pd.to_numeric(out[name]).clip(lower, upper)
    for name in ('tenure', 'prior_claims'):
        out[name] = out[name].round().astype(int)
    out['policy_year'] = pd.to_numeric(out['policy_year']).clip(min(YEARS), max(YEARS)).round().astype(int)
    out[TARGET] = pd.to_numeric(out[TARGET]).round().astype(int)
    assert out[TARGET].isin([0, 1]).all()
    return out


def train_sdv(train, model, n, params=None):
    import torch
    from sdv.metadata import Metadata
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    metadata = Metadata()
    metadata.add_table('profiles')
    for name, spec in sdtypes(train)['columns'].items():
        metadata.add_column(column_name=name, table_name='profiles', **spec)
    if model == 'gaussian_copula':
        generator = GaussianCopulaSynthesizer(metadata)
    else:
        generator = CTGANSynthesizer(metadata, verbose=False, enable_gpu=False, generator_dim=(64, 64),
                                     discriminator_dim=(64, 64), **params)
    start = perf_counter()
    generator.fit(train)
    seconds = perf_counter() - start
    return clean(generator.sample(n)), seconds


def verify_synthesis(sample, raw_tables, prefix):
    tables = relational(sample, prefix, SEED + 51)
    pii = set()
    for column in raw_tables['clients'].columns.intersection(['name', 'phone', 'address', 'snils', 'inn', 'passport']):
        pii.update(raw_tables['clients'][column].astype(str))
    overlap = sum(len(set(values.dropna().astype(str)) & pii)
                  for table in tables.values() for _, values in table.items())
    assert overlap == 0
    for name, key in (('clients', 'client_id'), ('policies', 'policy_id'), ('agents', 'agent_id'),
                      ('claims', 'claim_id'), ('payments', 'payment_id')):
        assert not set(tables[name][key]) & set(raw_tables[name][key])
    return {'tables': validate_relations(tables), 'pii_overlap': overlap,
            'fk_errors': 0, 'one_policy_per_subject': True}


def synthesis(train, test, raw_tables):
    candidates = [{'epochs': 12, 'batch_size': 500}, {'epochs': 24, 'batch_size': 1000},
                  {'epochs': 60, 'batch_size': 500}]
    fitting, validation = train_test_split(train, test_size=.20, stratify=train[TARGET], random_state=SEED)
    search = []
    for config in candidates:
        sample, seconds = train_sdv(fitting, 'ctgan', len(fitting), config)
        measured, _ = utility(sample, validation)
        search.append({'parameters': config, 'validation_auc': measured['auc'], 'fit_s': seconds})
    best = max(search, key=lambda result: result['validation_auc'])['parameters']
    scores, samples = {}, {}
    for name, params in [('gaussian_copula', None), ('ctgan', best)]:
        sample, seconds = train_sdv(train, name, len(train), params)
        measured, _ = utility(sample, test)
        scores[name] = {'fit_s': seconds, 'parameters': params, 'tstr': measured,
                        'fidelity': fidelity(train, sample),
                        'relations': verify_synthesis(sample, raw_tables, name)}
        samples[name] = sample
    return {'ctgan_tuning': search, 'models': scores}, samples
