"""Measured statistics used in report.md. Raw downloads remain in tmp/data."""
from pathlib import Path
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fastavro import writer
import great_expectations as gx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hospital_etl import normalize_pressure, validate_batch

ROOT = Path(__file__).parent
DATA = ROOT / 'tmp' / 'data'
ASSETS = ROOT / 'assets'
ASSETS.mkdir(exist_ok=True)
df = pd.read_csv(DATA / 'hospital_readmissions_30k.csv', low_memory=False)
df.to_csv(DATA / 'hospital_normalized.csv', index=False, lineterminator='\n')
table = pa.Table.from_pandas(df, preserve_index=False)
pq.write_table(table, DATA / 'hospital.parquet', compression='snappy')
pq.write_table(table, DATA / 'hospital_uncompressed.parquet', compression=None,
               use_dictionary=False)
schema = {'type': 'record', 'name': 'Encounter', 'fields': []}
for c in df.columns:
    typ = ('long' if pd.api.types.is_integer_dtype(df[c]) else
           'double' if pd.api.types.is_float_dtype(df[c]) else 'string')
    schema['fields'].append({'name': c, 'type': ['null', typ]})
records = df.to_dict('records')
for codec in ['null', 'deflate']:
    with (DATA / f'hospital_{codec}.avro').open('wb') as f:
        writer(f, schema, records, codec=codec)

context = gx.get_context(mode='ephemeral')
source = context.data_sources.add_pandas(name='hospital')
asset = source.add_dataframe_asset(name='encounters')
batch_definition = asset.add_batch_definition_whole_dataframe('all')
batch = batch_definition.get_batch(batch_parameters={'dataframe': df})
checks = validate_batch(batch, gx)

bed = df.length_of_stay.sort_values(ascending=False).reset_index(drop=True)
cum = bed.cumsum() / bed.sum()
before = cum.shift(1,fill_value=0)
labels = before.map(lambda v: 'A' if v < .8 else 'B' if v < .95 else 'C')
abc = {c: {'records':int(labels.eq(c).sum()),
           'bed_days':int(bed[labels.eq(c)].sum())} for c in ['A','B','C']}
clean_hospital, quarantine, pressure_metrics = normalize_pressure(df)
clean_hospital.to_parquet(DATA/'hospital_clean.parquet', index=False, compression='snappy')
quarantine.to_csv(DATA/'hospital_quarantine.csv', index=False)
examples=df[['patient_id','blood_pressure']].head(2).copy()
examples['systolic']=clean_hospital.systolic.reindex(examples.index)
examples['diastolic']=clean_hospital.diastolic.reindex(examples.index)
out = {
 'hospital': {
    'rows': len(df), 'columns': len(df.columns),
    'patients': int(df.patient_id.nunique()),
    'memory_bytes': int(df.memory_usage(deep=True).sum()),
    'arrow_bytes': table.nbytes,
    'sizes': {p.name: p.stat().st_size for p in DATA.glob('hospital*')},
    'original_csv_bytes': (DATA / 'hospital_readmissions_30k.csv').stat().st_size,
    'dtypes': {str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().items()},
    'missing': {k: int(v) for k, v in df.isna().sum().items()},
    'duplicates': int(df.duplicated().sum()),
    'duplicate_patient_ids': int(df.patient_id.duplicated().sum()),
    'readmitted': {k: int(v) for k,v in df.readmitted_30_days.value_counts().items()},
    'bed_days': int(bed.sum()), 'top_20pct_beddays_share': float(bed.iloc[:6000].sum()/bed.sum()),
    'abc': abc, 'checks': checks,
    'examples': json.loads(examples.to_json(orient='records')),
    'invalid_bp_format': pressure_metrics['invalid_format'],
    'systolic_le_diastolic':pressure_metrics['invalid_order'],
    'quarantine_rows':pressure_metrics['quarantine_rows'],
    'negative_numeric':int(df.select_dtypes('number').lt(0).any(axis=1).sum()),
    'ranges': {c:[float(df[c].min()),float(df[c].max())] for c in df.select_dtypes('number')},
 }
}
(ROOT/'measurements.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10})
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot([0]+list((cum.index+1)/len(cum)*100), [0]+list(cum*100))
ax.axhline(80, linestyle='--', color='gray')
ax.set(xlabel='Доля записей, % (по убыванию длительности)', ylabel='Доля койко-дней, %',
       title='ABC-анализ: объем данных и прокси их ценности')
ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(ASSETS/'abc.png',dpi=150); plt.close(fig)

# Forecasts are a modeled stream, not a measurement of the synthetic dataset.
parquet_per_row=(DATA/'hospital.parquet').stat().st_size/len(df)
daily=10_000
months=list(range(13))
fig,ax=plt.subplots(figsize=(7,3.2))
ax.plot(months,[m*365/12*daily*parquet_per_row/2**20 for m in months],marker='o')
ax.set(xlabel='Месяц',ylabel='Объем Parquet, МиБ',title='Модель роста: 10 000 записей в день')
ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(ASSETS/'growth.png',dpi=150); plt.close(fig)

taxi_path=DATA/'yellow_tripdata_2024-01.parquet'
if taxi_path.exists():
    t=pd.read_parquet(taxi_path)
    t.to_csv(DATA/'taxi.csv',index=False,lineterminator='\n')
    valid=t.tpep_pickup_datetime.ge('2024-01-01') & t.tpep_pickup_datetime.lt('2024-02-01')
    counts=t.loc[valid].set_index('tpep_pickup_datetime').resample('5min').size().reindex(
        pd.date_range('2024-01-01','2024-01-31 23:55',freq='5min'),fill_value=0)
    tax={
      'rows':len(t),'columns':len(t.columns),'valid_january':int(valid.sum()),
      'csv_bytes':(DATA/'taxi.csv').stat().st_size,
      'parquet_bytes':taxi_path.stat().st_size,
      'memory_bytes':int(t.memory_usage(deep=True).sum()),
      'mean_per_hour':float(valid.sum()/(31*24)),
      'mean_per_second':float(valid.sum()/(31*86400)),
      'peak_5min':int(counts.max()),'peak_time':str(counts.idxmax()),
      'cv_5min':float(counts.std(ddof=0)/counts.mean()),
      'zero_distance':int(t.trip_distance.eq(0).sum()),
      'negative_total':int(t.total_amount.lt(0).sum()),
      'wrong_pickup_month':int((~valid).sum()),
      'nonpositive_duration':int(t.tpep_dropoff_datetime.le(t.tpep_pickup_datetime).sum()),
    }
    out['taxi']=tax
    (ROOT/'measurements.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(tax,indent=2),flush=True)
    fig,ax=plt.subplots(figsize=(7,3.2))
    ax.plot(counts.index,counts.values,linewidth=.5)
    ax.set(xlabel='Январь 2024',ylabel='Поездки / 5 минут',title='NYC Taxi: фактическое распределение событий')
    ax.grid(alpha=.2); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(ASSETS/'taxi_events.png',dpi=150); plt.close(fig)
