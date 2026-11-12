"""Воспроизводимая учебная BI-схема страхования, вариант 6."""

import numpy as np
import pandas as pd
from faker import Faker
from sklearn.model_selection import train_test_split

SEED = 42
TARGET = 'claim'
DISTRICTS = ('Центр', 'Сибирь', 'Урал', 'Юг')
REGIONS = {
    'Центр': ('Москва', 'Тула', 'Калуга'), 'Сибирь': ('Кемерово', 'Томск', 'Омск'),
    'Урал': ('Тюмень', 'Курган', 'Екатеринбург'), 'Юг': ('Краснодар', 'Ростов', 'Астрахань'),
}
PRODUCTS = ('авто', 'имущество', 'поездка')
CHANNELS = ('онлайн', 'агент')
STATUSES = ('действует', 'закрыт')
AGES = ('18–29', '30–44', '45–59', '60–74')
YEARS = (2019, 2020, 2021, 2022, 2023, 2024, 2025)
CATEGORICAL = ['age_band', 'district', 'product', 'channel', 'status']
NUMERICAL = ['tenure', 'premium', 'prior_claims', 'risk_score']
FEATURES = CATEGORICAL + NUMERICAL
PROFILE_COLUMNS = FEATURES + ['policy_year', TARGET]
PII_COLUMNS = ['name', 'phone', 'address', 'snils', 'inn', 'passport']


def check_digit(number, weights, mod=11):
    return sum(int(digit) * weight for digit, weight in zip(number, weights, strict=True)) % mod % 10


def identifiers(rng):
    def digits(count):
        return ''.join(str(digit) for digit in rng.integers(0, 10, count))

    snils = digits(9)
    control = sum(int(digit) * weight for digit, weight in zip(snils, range(9, 0, -1), strict=True)) % 101
    snils = f'{snils[:3]}-{snils[3:6]}-{snils[6:]} {0 if control == 100 else control:02}'
    inn = '42' + digits(8)
    inn += str(check_digit(inn, (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)))
    inn += str(check_digit(inn, (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)))
    ogrn = '12442' + digits(7)
    ogrn += str(int(ogrn) % 11 % 10)
    passport = f'42 {digits(2)} {digits(6)}'
    phone = f'+7 (9{digits(2)}) {digits(3)}-{digits(2)}-{digits(2)}'
    return {'snils': snils, 'inn': inn, 'ogrn': ogrn, 'passport': passport, 'phone': phone}


def profile(n=15000, seed=SEED):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 75, n)
    district = rng.choice(DISTRICTS, n, p=[.33, .29, .21, .17])
    city = np.asarray([rng.choice(REGIONS[value]) for value in district])
    product = rng.choice(PRODUCTS, n, p=[.55, .30, .15])
    channel = rng.choice(CHANNELS, n, p=[.64, .36])
    tenure = rng.integers(0, 7, n)
    prior = np.minimum(rng.poisson(.38 + .33 * (product == 'авто') + .05 * tenure), 4)
    risk = np.clip(rng.beta(2.2, 3.8, n) + .06 * prior + .07 * (product == 'авто'), 0, 1)
    closed = rng.random(n) < (.1 + .12 * (tenure < 2))
    premium = np.clip(3300 + 6300 * (product == 'авто') + 1100 * (product == 'имущество')
                      + 7600 * risk + 460 * prior + rng.normal(0, 1100, n), 1500, 29000).round(2)
    logits = -3.9 + 4.1 * risk + .48 * prior + .36 * (product == 'авто') - .10 * tenure + .48 * (age < 30)
    logits += .21 * (channel == 'агент') - .24 * closed
    target = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    age_band = np.asarray(AGES)[np.digitize(age, [30, 45, 60])]
    return pd.DataFrame({
        'age_band': age_band, 'district': district, 'product': product, 'channel': channel,
        'status': np.where(closed, STATUSES[1], STATUSES[0]), 'tenure': tenure, 'premium': premium,
        'prior_claims': prior, 'risk_score': risk.round(5), 'policy_year': rng.choice(YEARS, n),
        TARGET: target, 'birth_year': 2026 - age, 'region': city,
    })


def relational(rows, prefix, seed, include_pii=False):
    """Один клиент — один полис. Страховые случаи и платежи подчинены полисам."""
    rng = np.random.default_rng(seed)
    frame = rows.reset_index(drop=True)
    n = len(frame)
    client_id = [f'{prefix}-c-{i:06}' for i in range(n)]
    policy_id = [f'{prefix}-p-{i:06}' for i in range(n)]
    clients = frame[[column for column in ('age_band', 'district', 'birth_year', 'region') if column in frame]].copy()
    clients.insert(0, 'client_id', client_id)
    if include_pii:
        fake = Faker('ru_RU')
        fake.seed_instance(seed)
        people = []
        for _ in range(n):
            person = identifiers(rng)
            person.pop('ogrn')
            person.update(name=fake.name(), address=fake.address().replace('\n', ', '))
            people.append(person)
        clients = pd.concat([clients, pd.DataFrame(people, columns=PII_COLUMNS)], axis=1)
    agents = pd.DataFrame({'agent_id': [f'{prefix}-a-{i}' for i in range(len(DISTRICTS) * 2)],
                           'district': np.repeat(DISTRICTS, 2)})
    agent_pool = {name: agents.loc[agents.district.eq(name), 'agent_id'].tolist() for name in DISTRICTS}
    policies = frame.drop(columns=[column for column in ('age_band', 'district', 'birth_year', 'region') if column in frame])
    policies.insert(0, 'policy_id', policy_id)
    policies.insert(1, 'client_id', client_id)
    policies.insert(2, 'agent_id', [rng.choice(agent_pool[name]) for name in frame['district']])
    claims, payments = [], []
    for index, policy in enumerate(policy_id):
        payments.append((f'{prefix}-x-{len(payments):06}', policy, None, 'взнос', float(frame.loc[index, 'premium'])))
        if frame.loc[index, TARGET] != 1:
            continue
        count = 1 + int(rng.random() < .10 + .18 * frame.loc[index, 'risk_score'])
        for _ in range(count):
            claim_id = f'{prefix}-s-{len(claims):06}'
            approved = rng.random() >= .15
            amount = round(float(rng.uniform(0.6, 3.0) * frame.loc[index, 'premium']), 2) if approved else 0.0
            claims.append((claim_id, policy, 'выплачен' if approved else 'отказ', amount))
            if approved:
                payments.append((f'{prefix}-x-{len(payments):06}', policy, claim_id, 'выплата', amount))
    tables = {'clients': clients, 'policies': policies, 'agents': agents,
              'claims': pd.DataFrame(claims, columns=['claim_id', 'policy_id', 'decision', 'paid_amount']),
              'payments': pd.DataFrame(payments, columns=['payment_id', 'policy_id', 'claim_id', 'kind', 'amount'])}
    validate_relations(tables)
    return tables


def validate_relations(tables):
    keys = dict(clients='client_id', policies='policy_id', agents='agent_id', claims='claim_id', payments='payment_id')
    for table, key in keys.items():
        assert tables[table][key].is_unique and tables[table][key].notna().all()
    relations = [('policies', 'clients', 'client_id'), ('policies', 'agents', 'agent_id'),
                 ('claims', 'policies', 'policy_id'), ('payments', 'policies', 'policy_id'),
                 ('payments', 'claims', 'claim_id')]
    for child, parent, key in relations:
        assert set(tables[child][key].dropna()) <= set(tables[parent][key])
    policies = tables['policies']
    assert policies['client_id'].is_unique
    assert set(tables['claims']['policy_id']) == set(policies.loc[policies[TARGET].eq(1), 'policy_id'])
    paid = tables['payments'].loc[tables['payments']['kind'].eq('выплата')]
    parent = tables['claims'].set_index('claim_id')
    assert paid['claim_id'].map(parent['policy_id']).eq(paid['policy_id']).all()
    assert paid['claim_id'].map(parent['paid_amount']).eq(paid['amount']).all()
    return {name: len(table) for name, table in tables.items()}


def partition(frame):
    train, other = train_test_split(frame, test_size=.30, random_state=SEED, stratify=frame[TARGET])
    test, control = train_test_split(other, test_size=.50, random_state=SEED, stratify=other[TARGET])
    assert len(set(train.index) | set(test.index) | set(control.index)) == len(frame)
    return tuple(part[PROFILE_COLUMNS].reset_index(drop=True) for part in (train, test, control))
