"""Проверка эмбеддингов, сравнение моделей и оценка выдачи LanceDB."""

from hashlib import sha256
from time import perf_counter

import lancedb
import lightgbm as lgb
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from clap_audio import DIM, arrow_vectors, text_vectors
from common_voice import FEATURES, WORK

LABELS = {0: 'не вопрос', 1: 'вопрос'}
SEEDS = (42, 2026)
PROMPTS = (
    'A speaker reads a statement aloud in Russian with declarative intonation.',
    'A speaker asks a question aloud in Russian with interrogative intonation.',
)


def join(data, encoded):
    keys = data['clip_key']
    if not keys.is_unique or not encoded['clip_key'].is_unique or set(keys) != set(encoded['clip_key']):
        raise ValueError('Ключи табличной и аудиочасти не совпадают один к одному')
    combined = data.merge(encoded, on='clip_key', how='left', validate='one_to_one', sort=False)
    assert len(combined) == len(data) == len(encoded)
    vectors = np.stack(combined.pop('vector')).astype('float32')
    assert vectors.shape == (len(data), DIM)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3, rtol=0)
    return combined, vectors


def split(frame, seed):
    target = frame['target'].to_numpy()
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates = [(train, test) for train, test in folds.split(frame, target, frame['client_id'])
                  if len(np.unique(target[train])) == len(np.unique(target[test])) == 2]
    train, test = min(candidates, key=lambda pair: abs(len(pair[1]) / len(frame) - .2)
                      + abs(target[pair[1]].mean() - target.mean()))
    assert not set(frame.iloc[train]['client_id']) & set(frame.iloc[test]['client_id'])
    assert not set(frame.iloc[train]['phrase']) & set(frame.iloc[test]['phrase'])
    assert len(train) + len(test) == len(frame)
    return train, test


def metrics(target, scores):
    prediction = scores >= .5
    return {'roc_auc': roc_auc_score(target, scores), 'accuracy': accuracy_score(target, prediction),
            'f1': f1_score(target, prediction, zero_division=0)}


def compare(frame, vectors):
    table = frame[FEATURES].astype('float32')
    media = pd.DataFrame(vectors, columns=[f'e_{column}' for column in range(DIM)])
    configs = {'table': table, 'audio': media, 'both': pd.concat([table, media], axis=1)}
    target = frame['target'].to_numpy()
    trials = []
    for seed in SEEDS:
        train, test = split(frame, seed)
        boundary = float(table.iloc[train]['word_count'].median())
        short = table.iloc[test]['word_count'].to_numpy() <= boundary
        scores, segments = {}, {}
        for name, features in configs.items():
            model = lgb.LGBMClassifier(n_estimators=120, learning_rate=.05, num_leaves=15, min_child_samples=12,
                                       random_state=seed, n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
            model.fit(features.iloc[train], target[train])
            prediction = model.predict_proba(features.iloc[test])[:, 1]
            scores[name] = metrics(target[test], prediction)
            segments[name] = {
                group: {'n': int(mask.sum()), 'roc_auc': roc_auc_score(target[test][mask], prediction[mask])}
                for group, mask in [('short', short), ('long', ~short)] if len(np.unique(target[test][mask])) == 2
            }
        assert set(scores) == {'table', 'audio', 'both'}
        best = max(scores[name]['roc_auc'] for name in ('table', 'audio'))
        trials.append({'seed': seed, 'train': len(train), 'test': len(test), 'test_questions': int(target[test].sum()),
                       'train_speakers': int(frame.iloc[train]['client_id'].nunique()),
                       'test_speakers': int(frame.iloc[test]['client_id'].nunique()), 'models': scores,
                       'gain_over_best_single': scores['both']['roc_auc'] - best,
                       'word_boundary_from_train': boundary, 'segments': segments})
    return trials


def pca_plot(frame, vectors, train, test, output):
    pca = PCA(n_components=2, svd_solver='randomized', random_state=42).fit(vectors[train])
    coordinates = pca.transform(vectors[test])
    target = frame.iloc[test]['target'].to_numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(8.5, 5.3), layout='constrained')
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for label, color in [(0, '#30629b'), (1, '#b74c32')]:
        ax.scatter(*coordinates[target == label].T, s=28, alpha=.68, color=color,
                   label=f'{LABELS[label]} · {int((target == label).sum())}', linewidths=0)
    variance = pca.explained_variance_ratio_
    ax.set(title='CLAP · проекция аудиозаписей тестовой выборки', xlabel=f'PC1 · {variance[0]:.1%}',
           ylabel=f'PC2 · {variance[1]:.1%}')
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=160)
    return variance.tolist()


def sample_queries(frame, test, per_label=20):
    target = frame.iloc[test]['target'].to_numpy()
    random = np.random.default_rng(42)
    return np.concatenate([random.permutation(test[target == label])[:per_label] for label in (0, 1)])


def short_id(clip):
    return sha256(clip.encode()).hexdigest()[:10]


def top_exact(vectors, query, mask, k):
    candidates = np.flatnonzero(mask)
    similarities = vectors[candidates] @ query
    order = np.argsort(-similarities, kind='stable')[:k]
    return candidates[order], similarities[order]


def quality(frame, vectors, train, test, output, threads=4):
    variance = pca_plot(frame, vectors, train, test, output)
    gallery = frame.iloc[train].reset_index(drop=True)
    labels = gallery['target'].to_numpy()
    examples, precision = [], []
    queries = sample_queries(frame, test)
    for index in queries:
        query = frame.iloc[index]
        top, similarity = top_exact(vectors[train], vectors[index], np.ones(len(train), dtype=bool), 5)
        label = int(query['target'])
        precision.append(float((labels[top] == label).mean()))
        examples.append({'query': short_id(query['clip_key']), 'label': LABELS[label],
                         'neighbors': [{'label': LABELS[int(labels[pos])], 'cosine': float(score)}
                                       for pos, score in zip(top, similarity, strict=True)]})
    texts = []
    for label, vector in enumerate(text_vectors(PROMPTS, threads)):
        top, similarity = top_exact(vectors[train], vector, np.ones(len(train), dtype=bool), 5)
        texts.append({'prompt': PROMPTS[label], 'expected': LABELS[label],
                      'precision_at_5': float((labels[top] == label).mean()),
                      'top_labels': [LABELS[int(value)] for value in labels[top]], 'cosine': similarity.tolist()})
    return {'pca_explained_variance': variance, 'n_queries': len(queries),
            'neighbor_precision_at_5': float(np.mean(precision)),
            'examples': [examples[0], examples[len(examples) // 2]], 'text_queries': texts}


def lance_catalog(frame, vectors, train):
    gallery = frame.iloc[train].reset_index(drop=True)
    boundary = float(gallery['duration_s'].median())
    bands = np.where(gallery['duration_s'] <= boundary, 'short', 'long')
    rows = pa.table({'clip_key': gallery['clip_key'].tolist(), 'target': gallery['target'].tolist(),
                     'duration_band': bands.tolist(), 'vector': arrow_vectors(vectors[train])})
    db = lancedb.connect(WORK / 'lance')
    table = db.create_table('audio_catalog', rows, mode='overwrite')
    table.create_index(vector_column_name='vector', metric='cosine', index_type='IVF_FLAT', num_partitions=8)
    return table, gallery, boundary, bands


def top_lance(table, query, duration_band=None, k=5, exact=False):
    if duration_band not in (None, 'short', 'long'):
        raise ValueError('Допустимые диапазоны длительности: short и long')
    search = table.search(query, vector_column_name='vector').metric('cosine').nprobes(4)
    if duration_band is not None:
        search = search.where(f"duration_band = '{duration_band}'", prefilter=True)
    if exact:
        search = search.bypass_vector_index()
    return search.select(['clip_key', 'target', 'duration_band', '_distance']).limit(k).to_list()


def retrieval(frame, vectors, train, test):
    table, gallery, boundary, bands = lance_catalog(frame, vectors, train)
    media = vectors[train]
    query_rows = sample_queries(frame, test)
    all_examples = []
    measurements = {'unfiltered': [], 'prefiltered': []}
    for query_number, index in enumerate(query_rows):
        vector = vectors[index]
        item = frame.iloc[index]
        band = 'short' if item['duration_s'] <= boundary else 'long'
        outcomes = {}
        # Чередуем порядок, чтобы прогрев не давал одному режиму преимущества.
        modes = list(measurements)
        if query_number % 2:
            modes.reverse()
        for name in modes:
            filtered = name == 'prefiltered'
            request_filter = band if filtered else None
            mask = bands == band if filtered else np.ones(len(gallery), dtype=bool)
            top, _ = top_exact(media, vector, mask, 5)
            reference = set(gallery.iloc[top]['clip_key'])
            if query_number == 0:
                exact = top_lance(table, vector, request_filter, exact=True)
                assert {row['clip_key'] for row in exact} == reference
                top_lance(table, vector, request_filter)  # прогрев вне замера
            start = perf_counter()
            rows = top_lance(table, vector, request_filter)
            elapsed = (perf_counter() - start) * 1000
            assert len(rows) == 5 and (not filtered or all(row['duration_band'] == band for row in rows))
            same_class = sum(row['target'] == item['target'] for row in rows)
            relevant = sum(row['target'] == item['target'] and row['duration_band'] == band for row in rows)
            overlap = len({row['clip_key'] for row in rows} & reference)
            measurements[name].append((elapsed, same_class / 5, relevant / 5, overlap / 5))
            outcomes[name] = [{'label': LABELS[row['target']], 'duration_band': row['duration_band'],
                               'cosine': 1 - row['_distance']} for row in rows]
        all_examples.append({'query': short_id(item['clip_key']), 'label': LABELS[int(item['target'])],
                             'duration_band': band, **outcomes})
    stats = {}
    for name, raw in measurements.items():
        latency, semantic, combined, recall = np.asarray(raw).T
        stats[name] = {'median_ms': float(np.median(latency)), 'p95_ms': float(np.percentile(latency, 95)),
                       'semantic_precision_at_5': float(semantic.mean()), 'joint_precision_at_5': float(combined.mean()),
                       'ann_recall_at_5': float(recall.mean())}
    return {'gallery': len(gallery), 'queries': len(query_rows), 'duration_boundary_s': boundary,
            'index': 'IVF_FLAT', 'partitions': 8, 'nprobes': 4, **stats,
            'examples': [all_examples[0], all_examples[len(all_examples) // 2]]}
