"""Загрузка русского Common Voice 17.0 и построение выборки без повторов фраз."""

from hashlib import file_digest as hash_stream, sha256
from http.client import HTTPException
from pathlib import Path
import re
import shutil
import tarfile
from time import sleep
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'tmp'
CV_REVISION = '8262c16bf297c87a9cd88c51997c4758ed7a8ba2'
CV_URL = f'https://huggingface.co/datasets/fsicoli/common_voice_17_0/resolve/{CV_REVISION}'
PARTS = ('dev', 'test')
FEATURES = ['word_count', 'letter_count', 'mean_word_length', 'unique_word_share']
WORDS = re.compile(r'[^\W_]+', re.UNICODE)


def fetch(url, path, checksum=None):
    """Сохраняет полную загрузку атомарно и при необходимости сверяет SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def matches(file):
        if not checksum:
            return True
        with file.open('rb') as stream:
            return hash_stream(stream, 'sha256').hexdigest() == checksum

    if path.is_file() and matches(path):
        return path
    partial = path.with_suffix(path.suffix + '.part')
    for attempt in range(3):
        try:
            request = Request(url, headers={'User-Agent': 'lab7-cpu/1.0'})
            with urlopen(request, timeout=90) as source, partial.open('wb') as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
                if source.headers.get('Content-Length') and target.tell() != int(source.headers['Content-Length']):
                    raise OSError(f'Неполная загрузка: {url}')
            if not matches(partial):
                raise OSError(f'Контрольная сумма не совпала: {path.name}')
            partial.replace(path)
            return path
        except (OSError, HTTPException):
            if attempt == 2:
                raise
            sleep(2 ** attempt)


def transcript_fields(sentence):
    words = WORDS.findall(sentence.casefold())
    letters = sum(map(len, words))
    return ' '.join(words), (len(words), letters, letters / len(words), len(set(words)) / len(words)) if words else None


def select(limit, seed=42):
    tables = []
    source_counts = {}
    for split in PARTS:
        path = fetch(f'{CV_URL}/transcript/ru/{split}.tsv', WORK / 'cv' / f'{split}.tsv')
        rows = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, quoting=3)
        source_counts[split] = len(rows)
        tables.append(rows[['path', 'client_id', 'sentence', 'age', 'gender', 'locale']].assign(source_split=split))
    data = pd.concat(tables, ignore_index=True)
    data = data[data['client_id'].ne('') & data['locale'].eq('ru')].copy()
    data['clip_key'] = data['path'].map(lambda path: Path(path).stem)
    data['target'] = data['sentence'].str.rstrip(' \t»”\"\'').str.endswith('?').astype('int8')
    fields = data['sentence'].map(transcript_fields)
    data['phrase'] = fields.map(lambda row: row[0])
    data[FEATURES] = pd.DataFrame(fields.map(lambda row: row[1] or (np.nan,) * len(FEATURES)).tolist(),
                                  columns=FEATURES, index=data.index)
    data = (data.dropna(subset=FEATURES).sample(frac=1, random_state=seed)
            .drop_duplicates('clip_key').drop_duplicates('phrase'))
    counts = data['target'].value_counts().to_dict()
    if set(counts) != {0, 1}:
        raise ValueError('В корпусе нужны оба класса')
    eligible = data[data.groupby(['client_id', 'target']).cumcount().lt(12)]
    count = min(limit // 2, int(eligible['target'].value_counts().min()))
    if count < 25:
        raise ValueError('Нужно не меньше 25 записей каждого класса')
    selected = pd.concat([eligible[eligible['target'].eq(target)].head(count) for target in (0, 1)])
    selected = selected.sample(frac=1, random_state=seed).reset_index(drop=True)
    assert selected['clip_key'].is_unique and selected['phrase'].is_unique
    assert selected['path'].map(lambda name: re.fullmatch(r'common_voice_ru_\d+\.mp3', name) is not None).all()
    return selected, {'source_rows': source_counts, 'unique_candidates': len(data), 'class_candidates': counts,
                      'selected_rows': len(selected), 'speakers': int(selected['client_id'].nunique()),
                      'missing_age': int(selected['age'].eq('').sum()), 'missing_gender': int(selected['gender'].eq('').sum())}


def prepare(limit=1000):
    selected, audit = select(limit)
    audio = WORK / 'audio'
    audio.mkdir(parents=True, exist_ok=True)
    for split in PARTS:
        missing = set(selected.loc[selected['source_split'].eq(split), 'path'])
        missing = {name for name in missing if not (audio / name).is_file()}
        if not missing:
            continue
        path = audio_archive(split)
        with tarfile.open(path, 'r|') as archive:
            for entry in archive:
                clip = Path(entry.name).name
                if entry.isfile() and clip in missing:
                    target = audio / clip
                    partial = target.with_suffix('.part')
                    with archive.extractfile(entry) as source, partial.open('wb') as output:
                        shutil.copyfileobj(source, output)
                    partial.replace(target)
                    missing.remove(clip)
                    if not missing:
                        break
        if missing:
            raise ValueError(f'Аудиошард {path.name} не содержит {len(missing)} файлов')
    return selected, audit


def audio_archive(split):
    name = f'ru_{split}_0.tar'
    return fetch(f'{CV_URL}/audio/ru/{split}/{name}?download=true', WORK / 'cv' / name)


def file_digest(path):
    return sha256(path.read_bytes()).hexdigest()
