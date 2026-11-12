"""Инъекции PII, проверка детекторов и перебор обобщений квазиидентификаторов."""

from dataclasses import dataclass
from itertools import product
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

from faker import Faker
import numpy as np
import pandas as pd
from presidio_analyzer import Pattern, PatternRecognizer

from mock_insurance import PII_COLUMNS, REGIONS, SEED, TARGET, check_digit, identifiers

KINDS = ('name', 'phone', 'address', 'snils', 'inn', 'ogrn', 'passport')
TAGS = dict(name='ФИО', phone='Телефон', address='Адрес', snils='СНИЛС', inn='ИНН', ogrn='ОГРН', passport='Паспорт')
PATTERNS = {
    'name': r'(?<=ФИО: )[А-ЯЁ][а-яё-]+(?: [А-ЯЁ][а-яё-]+){1,2}',
    'phone': r'(?<!\d)(?:\+7|8)[ -]?\(?9\d{2}\)?[ -]?\d{3}[ -]?\d{2}[ -]?\d{2}(?!\d)',
    'address': r'г\. [А-ЯЁ][А-Яа-яЁё -]+, ул\. [А-ЯЁ][А-Яа-яЁё -]+, д\. \d+[А-Яа-яЁё]?',
    'snils': r'(?<!\d)\d{3}[- ]\d{3}[- ]\d{3}[- ]\d{2}(?!\d)',
    'inn': r'(?<!\d)(?:\d{12}|\d{10})(?!\d)',
    'ogrn': r'(?<!\d)(?:\d{15}|\d{13})(?!\d)',
    'passport': r'(?<=Паспорт: )\d{2} ?\d{2} \d{6}(?!\d)',
}
QI = ['birth_year', 'region', 'product', 'policy_year']


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int
    kind: str


def valid_checksum(kind, value):
    value = re.sub(r'\D', '', value)
    if kind == 'snils':
        number = sum(int(digit) * weight for digit, weight in zip(value[:9], range(9, 0, -1), strict=True)) % 101
        return (0 if number == 100 else number) == int(value[9:])
    if kind == 'inn':
        if len(value) == 10:
            return check_digit(value[:9], (2, 4, 10, 3, 5, 9, 4, 6, 8)) == int(value[9])
        return (check_digit(value[:10], (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)) == int(value[10])
                and check_digit(value[:11], (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)) == int(value[11]))
    if kind == 'ogrn':
        return int(value[:-1]) % (11 if len(value) == 13 else 13) % 10 == int(value[-1])
    return True


class Detector:
    def __init__(self):
        from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter

        self.patterns = {
            kind: PatternRecognizer(supported_entity=kind, supported_language='ru',
                                    patterns=[Pattern(kind, pattern, .85)])
            for kind, pattern in PATTERNS.items()
        }
        self.segmenter = Segmenter()
        self.tagger = NewsNERTagger(NewsEmbedding())
        self.document = Doc

    def detect(self, text):
        spans = set()
        for kind, recognizer in self.patterns.items():
            for hit in recognizer.analyze(text, [kind]):
                if valid_checksum(kind, text[hit.start:hit.end]):
                    spans.add(Span(hit.start, hit.end, kind))
        if re.search('[А-ЯЁ]', text):
            doc = self.document(text)
            doc.segment(self.segmenter)
            doc.tag_ner(self.tagger)
            for hit in doc.spans:
                if hit.type == 'PER' and ' ' in hit.text and not any(
                        span.kind == 'name' and span.start <= hit.start < span.end for span in spans):
                    spans.add(Span(hit.start, hit.stop, 'name'))
        return spans


def inject(n=48):
    fake = Faker('ru_RU')
    fake.seed_instance(SEED + 2)
    rng = np.random.default_rng(SEED + 2)
    texts = []
    for index in range(n):
        entries = identifiers(rng)
        entries.update(name=fake.name(), address=f'г. Томск, ул. Учебная, д. {int(rng.integers(1, 250))}')
        if index % 12 == 0:
            entries['snils'] = re.sub(r'\D', '', entries['snils'])
            entries['address'] = f'Томск, Учебная {index + 1}'
        for kind, value in entries.items():
            prefix = f'{TAGS[kind]}: ' if index % 8 else 'Из служебной заметки: '
            texts.append((prefix + value + '; обращение принято.', {Span(len(prefix), len(prefix) + len(value), kind)}))
    for index in range(36):
        texts.append((f'Контроль: номер операции {20260000 + index}, полис действителен.', set()))
    # Правдоподобный контрольный номер договора формально совпал с шаблоном СНИЛС.
    texts.append((f"Номер договора: {identifiers(rng)['snils']}; проверка завершена.", set()))
    return texts


def mask(text, spans):
    keep = np.ones(len(text), dtype=bool)
    for span in spans:
        keep[span.start:span.end] = False
    return re.sub('█+', '[СКРЫТО]', ''.join(char if keep[i] else '█' for i, char in enumerate(text)))


def pii_audit(clients):
    detector = Detector()
    counters = {kind: dict(tp=0, fp=0, fn=0, residual=0) for kind in KINDS}
    for text, truth in inject():
        found = detector.detect(text)
        masked = mask(text, found)
        for kind in KINDS:
            expected, detected = ({hit for hit in spans if hit.kind == kind} for spans in (truth, found))
            counts = counters[kind]
            counts['tp'] += len(expected & detected)
            counts['fp'] += len(detected - expected)
            counts['fn'] += len(expected - detected)
            counts['residual'] += sum(text[span.start:span.end] in masked for span in expected)
    redacted = clients.copy()
    redacted[PII_COLUMNS] = '[СКРЫТО]'
    overlap = sum(len(set(clients[column]) & set(redacted[column])) for column in PII_COLUMNS)
    assert overlap == 0
    for counts in counters.values():
        counts['precision'] = counts['tp'] / max(1, counts['tp'] + counts['fp'])
        counts['recall'] = counts['tp'] / (counts['tp'] + counts['fn'])
    return {'by_type': counters, 'injected': sum(c['tp'] + c['fn'] for c in counters.values()),
            'structured_rows': len(clients), 'structured_pii_overlap': overlap, 'free_text_in_release': False}


def hierarchies(frame):
    city_to_district = {city: district for district, cities in REGIONS.items() for city in cities}
    levels = [
        [frame.birth_year // width * width for width in (1, 5, 10, 20)],
        [frame.region, frame.region.map(city_to_district)],
        [frame['product']],
        [(frame.policy_year - 2019) // width * width + 2019 for width in (1, 2, 3)],
    ]
    constant = pd.Series('*', index=frame.index)
    return [values + [constant] for values in levels]


def group_stats(codes):
    keys = np.zeros(len(codes[0]), dtype='int64')
    for column in codes:
        keys = keys * 128 + column
    sizes = np.unique(keys, return_counts=True)[1]
    return {'k_min': int(sizes.min()), 'singletons': int((sizes == 1).sum()),
            'groups_le_2': int((sizes <= 2).sum()), 'classes': len(sizes)}


def k_audit(frame):
    options = hierarchies(frame)
    codes = [[pd.factorize(column)[0] for column in values] for values in options]
    penalties = []
    for groupings in codes:
        raw = groupings[0]
        domain = len(np.unique(raw))
        losses = []
        for groups in groupings:
            membership = pd.DataFrame({'raw': raw, 'group': groups}).drop_duplicates().groupby('group').size()
            losses.append(float(np.mean((membership.to_numpy()[groups] - 1) / max(1, domain - 1))))
        penalties.append(losses)
    results = []
    for selected in product(*(range(len(values)) for values in options)):
        stats = group_stats([codes[i][level] for i, level in enumerate(selected)])
        loss = float(np.mean([penalties[i][level] for i, level in enumerate(selected)]))
        results.append({'levels': list(selected), 'information_loss': loss, **stats})
    baseline = results[0]
    choices = [dict(min((item for item in results if item['k_min'] >= target),
                        key=lambda item: (item['information_loss'], item['k_min'], item['levels'])), requested_k=target)
               for target in (2, 5, 10)]
    demo = frame.loc[~frame.duplicated(QI, keep=False), QI + [TARGET]].iloc[0].to_dict()
    observed = pycanon_verify(frame, options, [baseline, *choices])
    assert observed == [item['k_min'] for item in [baseline, *choices]]
    return {'baseline': baseline, 'choices': choices, 'linkage_example': demo,
            'pycanon_version': '1.3.6', 'pycanon_k': observed, 'quasi_identifiers': QI}


def pycanon_verify(frame, options, policies):
    root = Path(__file__).resolve().parent
    local = root / '.venv-k' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    interpreter = os.environ.get('LAB8_K_PYTHON', str(local) if local.is_file() else sys.executable)
    env = os.environ.copy()
    if env.get('LAB8_K_PATH'):
        env['PYTHONPATH'] = env['LAB8_K_PATH']
    with TemporaryDirectory(dir=root / 'tmp') as temporary:
        paths = []
        for index, policy in enumerate(policies):
            path = Path(temporary) / f'{index}.csv'
            pd.DataFrame({name: options[i][level] for i, (name, level) in enumerate(zip(QI, policy['levels'], strict=True))},
                         index=frame.index).to_csv(path, index=False)
            paths.append(str(path))
        check = ('import json,sys,pandas as pd; from pycanon import anonymity; '
                 'print(json.dumps([int(anonymity.k_anonymity(pd.read_csv(p), '
                 f'{QI!r})) for p in sys.argv[1:]]))')
        return json.loads(subprocess.check_output([interpreter, '-c', check, *paths], env=env, text=True).splitlines()[-1])
