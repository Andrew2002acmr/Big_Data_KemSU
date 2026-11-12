"""Полный локальный прогон лабораторной или скачивание исходных файлов."""

import argparse
import json
from pathlib import Path
import platform

from common_voice import CV_REVISION, FEATURES, WORK, prepare


def save_result(directory, audit, stats, quality, trials, search):
    from clap_audio import DIM, MODEL_REVISION

    result = {
        'source_revision': CV_REVISION, 'model_revision': MODEL_REVISION,
        'runtime': {'platform': platform.system(), 'python': platform.python_version(), 'provider': 'CPUExecutionProvider'},
        'dataset': audit, 'embeddings': stats, 'quality': quality, 'classification': trials,
        'features': {'table': len(FEATURES), 'audio': DIM, 'both': DIM + len(FEATURES)},
        'retrieval': search,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'measurements.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                                                  encoding='utf-8', newline='\n')
    return result


def run(limit, batch_size, threads, directory):
    from assessment import SEEDS, compare, join, quality, retrieval, split
    from clap_audio import embeddings

    data, audit = prepare(limit)
    print(audit, flush=True)
    encoded, stats = embeddings(data, batch_size, threads, progress=True)
    print(stats, flush=True)
    frame, vectors = join(data, encoded)
    train, test = split(frame, SEEDS[0])
    diagnostics = quality(frame, vectors, train, test, directory / 'pca.png', threads)
    trials = compare(frame, vectors)
    search = retrieval(frame, vectors, train, test)
    return save_result(directory, audit, stats, diagnostics, trials, search)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=1000, help='максимальный размер сбалансированной выборки')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, default=WORK / 'run')
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    if args.download_only:
        print(prepare(args.limit)[1])
    else:
        run(args.limit, args.batch_size, args.threads, args.output)
        print(f'Результаты: {args.output.resolve()}')
