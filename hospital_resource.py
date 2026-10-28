"""Small measured ETL experiment, independent of the untrained demand forecast."""
from pathlib import Path
import os
os.environ.setdefault('GX_ANALYTICS_ENABLED', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('ARROW_NUM_THREADS', '1')
import json, time, threading, io, contextlib
import psutil

ROOT = Path(__file__).parent
DATA = ROOT/'tmp'/'data'
proc = psutil.Process()
affinity = proc.cpu_affinity()
proc.cpu_affinity(affinity[:2])
startup = time.perf_counter()
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import great_expectations as gx
from hospital_etl import normalize_pressure, validate_batch
pa.set_cpu_count(2)
context = gx.get_context(mode='ephemeral')
asset = context.data_sources.add_pandas(name='resource').add_dataframe_asset(name='episodes')
definition = asset.add_batch_definition_whole_dataframe('whole')
startup_seconds = time.perf_counter()-startup

# Corruption is isolated in a probe, never inserted into the source dataset.
probe = pd.DataFrame({'patient_id':range(7), 'blood_pressure':[
    '130/72', ' 120/80 ', 'bad', None, '120/', '70/130', '999999999999999999999/80']})
clean, rejected, probe_metrics = normalize_pressure(probe)
assert clean.patient_id.tolist() == [0, 1]
assert rejected.patient_id.tolist() == [2, 3, 4, 5, 6]
assert probe_metrics['invalid_format'] == 4
assert probe_metrics['invalid_order'] == 1
assert str(clean.systolic.dtype) == 'Int64'

original = pd.read_csv(DATA/'hospital_readmissions_30k.csv')
expanded = pd.concat([original.assign(patient_id=original.patient_id+i*len(original))
                      for i in range(10)], ignore_index=True)
stress_path = DATA/'hospital_stress_300k.csv'
expanded.to_csv(stress_path, index=False, lineterminator='\n')
del original, expanded, clean, rejected

results = []
for label, path in [('source_30k', DATA/'hospital_readmissions_30k.csv'),
                    ('repeated_300k', stress_path)]:
    samples = []
    for run in range(3):
        stop = threading.Event()
        peak = [proc.memory_info().rss]
        def monitor():
            while not stop.wait(.01):
                peak[0] = max(peak[0], proc.memory_info().rss)
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        before_cpu = proc.cpu_times()
        started = time.perf_counter()
        frame = pd.read_csv(path)
        normalized, quarantine, metrics = normalize_pressure(frame)
        assert not len(quarantine)
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            checks = validate_batch(definition.get_batch(batch_parameters={'dataframe': frame}), gx)
        assert all(v['success'] for v in checks.values())
        # This is a load aggregate, not a time series or fitted forecast.
        summary = normalized.groupby('discharge_destination').length_of_stay.agg(['sum','count'])
        output = DATA/f'hospital_etl_{label}.parquet'
        pq.write_table(pa.Table.from_pandas(normalized, preserve_index=False), output,
                       compression='snappy')
        elapsed = time.perf_counter()-started
        after_cpu = proc.cpu_times()
        peak[0] = max(peak[0], proc.memory_info().rss)
        stop.set(); thread.join()
        samples.append({'seconds':elapsed, 'cpu_seconds':
            after_cpu.user+after_cpu.system-before_cpu.user-before_cpu.system,
            'peak_rss_bytes':peak[0]})
        row_count = len(frame)
        assert int(summary['count'].sum()) == row_count
        del frame, normalized, quarantine, summary
    results.append({'label':label,'rows':row_count,'runs':samples,
        'median_seconds':sorted(s['seconds'] for s in samples)[1],
        'max_seconds':max(s['seconds'] for s in samples),
        'max_cpu_seconds':max(s['cpu_seconds'] for s in samples),
        'max_peak_rss_bytes':max(s['peak_rss_bytes'] for s in samples),
        'output_bytes':output.stat().st_size})
out = {'logical_cpus_allowed':proc.cpu_affinity(), 'memory_limit_enforced':False,
       'machine_ram_bytes':psutil.virtual_memory().total,
       'startup_seconds':startup_seconds,'pressure_probe':probe_metrics,
       'source': 'synthetic Kaggle; 300k repeats source with unique synthetic IDs',
       'results':results}
(ROOT/'hospital_resource.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2), flush=True)
