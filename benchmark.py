"""Protocol v2: separate engine startup from an identical, fully evaluated CSV query.
Run: python benchmark.py prepare | pandas | dask | spark
"""
from pathlib import Path
import sys, os, json, time, threading, hashlib
import psutil

ROOT = Path(__file__).parent
DATA = ROOT/'tmp'/'benchmark'
MODE = sys.argv[1]
if MODE == 'prepare':
    import pandas as pd
    DATA.mkdir(exist_ok=True, parents=True)
    source = ROOT/'tmp'/'data'/'taxi.csv'
    if not source.exists():
        pd.read_parquet(source.with_name('yellow_tripdata_2024-01.parquet')).to_csv(
            source, index=False, lineterminator='\n')
    original = source.read_bytes()
    header, body = original.split(b'\n', 1)
    sample = pd.read_csv(source, nrows=20000)
    types = {c:'float64' if pd.api.types.is_numeric_dtype(sample[c]) else 'object'
             for c in sample.columns}
    target = 10_000_000_000
    remaining = target
    names = []
    while remaining > len(header)+1000:
        part = body[:min(len(body), remaining-len(header)-1)]
        part = part[:part.rfind(b'\n')+1]
        if not part: break
        path = DATA/f'part-{len(names):03}.csv'
        path.write_bytes(header+b'\n'+part)
        names.append(path.name)
        remaining -= path.stat().st_size
    manifest = {'files':names, 'input_bytes':target-remaining, 'types':types,
                'source_csv_sha256':hashlib.sha256(original).hexdigest()}
    (DATA/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'files':len(names), 'bytes':target-remaining}), flush=True)
    sys.exit()

assert MODE in ['pandas','dask','spark']
manifest = json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
files = [DATA/name for name in manifest['files']]
assert all(path.is_file() for path in files)
assert sum(path.stat().st_size for path in files) == manifest['input_bytes']
types = manifest['types']
proc = psutil.Process()
phase = ['startup']
peaks = {'startup':0, 'query':0}
stop = threading.Event()
def rss():
    value = 0
    for child in [proc]+proc.children(recursive=True):
        try: value += child.memory_info().rss
        except psutil.Error: pass
    return value
def monitor():
    while not stop.wait(.05):
        key = phase[0]
        peaks[key] = max(peaks[key], rss())
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
started = time.perf_counter()
# Pandas is needed by every engine to expose the same materialized result type.
import pandas as pd
spark = None
if MODE == 'dask':
    import dask.dataframe as dd
elif MODE == 'spark':
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
    os.environ['PYSPARK_PYTHON'] = sys.executable
    from pyspark.sql import SparkSession, functions as F, types as T
    spark = (SparkSession.builder.master('local[4]').appName('Lab10GB-v2')
             .config('spark.ui.enabled','false').config('spark.driver.memory','2g')
             .config('spark.sql.shuffle.partitions','8').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    spark.conf.set('spark.sql.csv.parser.columnPruning.enabled','false')
    schema = T.StructType([T.StructField(c, T.DoubleType() if typ == 'float64' else T.StringType())
                           for c,typ in types.items()])
startup_seconds = time.perf_counter()-started
peaks['startup'] = max(peaks['startup'], rss())
phase[0] = 'query'
peaks['query'] = rss()
started = time.perf_counter()
if MODE == 'pandas':
    result = None
    for path in files:
        for chunk in pd.read_csv(path, dtype=types, chunksize=100000):
            group = chunk.groupby('PULocationID').total_amount.agg(['sum','count'])
            result = group if result is None else result.add(group, fill_value=0)
elif MODE == 'dask':
    frame = dd.read_csv([str(path) for path in files], dtype=types, blocksize='64MiB')
    # Required columns prevent the optimizer from reading only the two grouped columns.
    frame = frame.map_partitions(lambda part: part, meta=frame._meta,
                                 required_columns=list(types))
    result = frame.groupby('PULocationID').total_amount.agg(['sum','count']).compute(
        scheduler='threads', num_workers=4)
else:
    frame = spark.read.option('header',True).schema(schema).csv([p.as_uri() for p in files])
    rows = frame.groupBy('PULocationID').agg(F.sum('total_amount').alias('sum'),
        F.count('total_amount').alias('count')).collect()
    result = pd.DataFrame([row.asDict() for row in rows]).set_index('PULocationID')
result = result.sort_index()
query_seconds = time.perf_counter()-started
peaks['query'] = max(peaks['query'], rss())
stop.set(); thread.join()
by_zone = [{'PULocationID':int(key), 'count':int(row['count']), 'sum':float(row['sum'])}
           for key,row in result.iterrows()]
out = {'protocol_version':2, 'engine':MODE, 'startup_seconds':startup_seconds,
       'query_seconds':query_seconds, 'seconds':query_seconds,
       'startup_peak_rss_bytes':peaks['startup'], 'query_peak_rss_bytes':peaks['query'],
       'input_bytes':manifest['input_bytes'],
       'source_csv_sha256':manifest['source_csv_sha256'],
       'rows':int(result['count'].sum()), 'groups':len(result),
       'total_amount_sum':float(result['sum'].sum()), 'by_zone':by_zone,
       'cpu_logical':psutil.cpu_count(), 'ram_total_bytes':psutil.virtual_memory().total}
(ROOT/f'benchmark_{MODE}.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='by_zone'}, indent=2), flush=True)
if spark is not None: spark.stop()
