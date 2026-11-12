"""Измерение препроцессинга двух полных русских шардов без запуска энкодера."""

from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
import json
import os
from pathlib import Path
import tarfile
from time import perf_counter

from common_voice import PARTS, WORK, audio_archive


def measure(task):
    from clap_audio import feature_batch, feature_extractor, load_clip

    split, partition = task
    archive_path = audio_archive(split)
    processor = feature_extractor()
    rows = seen = skipped = 0
    clips = []
    start = perf_counter()
    with tarfile.open(archive_path, 'r|') as archive:
        for member in archive:
            if not member.isfile() or Path(member.name).suffix.lower() != '.mp3':
                continue
            seen += 1
            if (seen - 1) % 2 == partition:
                with archive.extractfile(member) as source:
                    try:
                        signal, _ = load_clip(BytesIO(source.read()))
                    except (RuntimeError, ValueError):
                        skipped += 1
                        continue
                clips.append(signal)
                rows += 1
                if len(clips) == 8:
                    feature_batch(processor, clips)
                    clips.clear()
                if rows % 2000 == 0:
                    print(f'{split}/{partition}: {rows}', flush=True)
    if clips:
        feature_batch(processor, clips)
    return {'split': split, 'partition': partition, 'rows': rows, 'skipped': skipped, 'seconds': perf_counter() - start}


if __name__ == '__main__':
    # По одному потоку BLAS на процесс. Чётные и нечётные записи каждого шарда обрабатываются параллельно.
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    for split in PARTS:
        audio_archive(split)
    from clap_audio import feature_extractor

    feature_extractor()  # Проверяем конфигурацию до запуска рабочих процессов.
    tasks = [(split, partition) for split in PARTS for partition in (0, 1)]
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
        results = list(pool.map(measure, tasks))
    report = {'scope': list(PARTS), 'rows': sum(item['rows'] for item in results),
              'skipped': sum(item['skipped'] for item in results), 'workers': len(tasks),
              'wall_seconds': perf_counter() - start, 'parts': results}
    (WORK / 'source_preprocess.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(report), flush=True)
