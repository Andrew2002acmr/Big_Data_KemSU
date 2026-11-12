"""Единый препроцессинг и замороженные аудио- и текстовый энкодеры CLAP."""

import json
from hashlib import sha256
from math import gcd
from time import perf_counter

import numpy as np
import onnxruntime as ort
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import resample_poly
import soundfile as sf
from transformers import AutoTokenizer, ClapFeatureExtractor

from common_voice import WORK, fetch, file_digest

MODEL_REVISION = 'c28f2883575e590e04d3146ff0713c2448d691ba'
MODEL_URL = f'https://huggingface.co/Xenova/clap-htsat-unfused/resolve/{MODEL_REVISION}'
MODEL_DIR = WORK / 'clap'
SAMPLE_RATE = 48000
SAMPLES = SAMPLE_RATE * 10
DIM = 512
AUDIO_MODEL = 'audio_model_quantized.onnx'
TEXT_MODEL = 'text_model_quantized.onnx'
CHECKSUMS = {
    AUDIO_MODEL: '3fcff2c8824e7bcb83a983f2a49edab3b60cbcf4872ac70efee517355173bd1f',
    TEXT_MODEL: '1a3df8b197e249816e08415fd040434c44762b2eea7eb7bf8a48a0f0bf3c14e5',
}


def model_files(text=False):
    config = ['preprocessor_config.json', 'config.json']
    if text:
        config += ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
    for name in config:
        fetch(f'{MODEL_URL}/{name}?download=true', MODEL_DIR / name)
    name = TEXT_MODEL if text else AUDIO_MODEL
    return fetch(f'{MODEL_URL}/onnx/{name}?download=true', MODEL_DIR / name, CHECKSUMS[name])


def session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def load_clip(path):
    frames, rate = sf.read(path, dtype='float32', always_2d=True)
    if not len(frames) or not np.isfinite(frames).all():
        raise ValueError(f'Пустая или повреждённая запись: {getattr(path, "name", "из архива")}')
    duration = len(frames) / rate
    signal = frames.mean(axis=1, dtype=np.float32)
    # Детерминированно берём центральное окно до ресемплинга длинных записей.
    window = rate * 10
    if len(signal) > window:
        start = (len(signal) - window) // 2
        signal = signal[start:start + window]
    if rate != SAMPLE_RATE:
        factor = gcd(rate, SAMPLE_RATE)
        signal = resample_poly(signal, SAMPLE_RATE // factor, rate // factor).astype('float32')
    return signal[:SAMPLES], duration


def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or (norms <= 0).any():
        raise ValueError('Некорректный CLAP-вектор')
    result = (vectors / norms).astype('float32')
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-3, rtol=0)
    return result


def feature_extractor():
    name = 'preprocessor_config.json'
    fetch(f'{MODEL_URL}/{name}?download=true', MODEL_DIR / name)
    return ClapFeatureExtractor.from_pretrained(str(MODEL_DIR), local_files_only=True)


def feature_batch(processor, clips):
    features = processor(list(clips), sampling_rate=SAMPLE_RATE, padding='repeatpad',
                         truncation='rand_trunc', return_tensors='np')['input_features']
    assert features.shape == (len(clips), 1, 1001, 64), features.shape
    return features.astype('float32')


class AudioEncoder:
    def __init__(self, threads=4):
        self.runtime = session(model_files(), threads)
        self.processor = feature_extractor()

    def preprocess(self, paths):
        clips, durations = zip(*(load_clip(path) for path in paths), strict=True)
        return feature_batch(self.processor, clips), durations

    def encode(self, features):
        vectors, = self.runtime.run(['audio_embeds'], {'input_features': features})
        return normalize(vectors)


def text_vectors(prompts, threads=4):
    encoder = session(model_files(text=True), threads)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    # Экспорт принимает input_ids. Отдельные запросы обходятся без паддинга токенов.
    vectors = [encoder.run(['text_embeds'], {'input_ids': tokenizer(prompt, return_tensors='np')['input_ids']})[0]
               for prompt in prompts]
    return normalize(np.concatenate(vectors))


def arrow_vectors(vectors):
    return pa.FixedSizeListArray.from_arrays(pa.array(vectors.reshape(-1), type=pa.float32()), DIM)


def embeddings(data, batch_size=8, threads=4, progress=False):
    if batch_size < 1 or threads < 1:
        raise ValueError('Размер батча и число потоков должны быть положительными')
    paths = [WORK / 'audio' / name for name in data['path']]
    hashes = [file_digest(path) for path in paths]
    if len(set(hashes)) != len(hashes):
        raise ValueError('В выборке есть побайтно одинаковые аудиозаписи')
    key = sha256(f'{MODEL_REVISION}/48k/repeatpad/center/v1\n'.encode())
    for clip, digest in zip(data['clip_key'], hashes, strict=True):
        key.update(f'{clip}:{digest}\n'.encode())
    cache = WORK / 'embeddings' / f'{key.hexdigest()}.parquet'
    cache.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    if cache.is_file():
        table = pq.read_table(cache)
        stats = json.loads(table.schema.metadata[b'measurements'])
        stats.update(cache_hit=True, cache_read_s=perf_counter() - start)
        return table.to_pandas(), stats

    encoder = AudioEncoder(threads)
    blocks, durations = [], []
    prep_s = encode_s = sample_prep_s = 0.0
    sample_rows = min(len(data), max(1, 64 // batch_size) * batch_size)
    for start_index in range(0, len(paths), batch_size):
        start = perf_counter()
        features, seconds = encoder.preprocess(paths[start_index:start_index + batch_size])
        prep_s += perf_counter() - start
        start = perf_counter()
        blocks.append(encoder.encode(features))
        encode_s += perf_counter() - start
        durations.extend(seconds)
        if len(durations) <= sample_rows:
            sample_prep_s = prep_s
        if progress and (len(durations) % (8 * batch_size) == 0 or len(durations) == len(paths)):
            print(f'CLAP {len(durations)}/{len(paths)}', flush=True)
    vectors = np.concatenate(blocks)
    assert vectors.shape == (len(data), DIM)
    stats = {'rows': len(data), 'dimensions': DIM, 'batch_size': batch_size, 'threads': threads,
             'preprocess_s': prep_s, 'encode_s': encode_s, 'sample_rows': sample_rows, 'sample_preprocess_s': sample_prep_s,
             'norm_max_error': float(np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1))), 'cache_hit': False}
    table = pa.table({'clip_key': data['clip_key'].tolist(), 'duration_s': durations, 'vector': arrow_vectors(vectors)})
    table = table.replace_schema_metadata({b'measurements': json.dumps(stats).encode()})
    partial = cache.with_suffix('.part')
    pq.write_table(table, partial, compression='zstd')
    partial.replace(cache)
    return table.to_pandas(), stats
