"""Download, validate and create reproducible CSV/Parquet workloads for variant 6."""
from pathlib import Path
import json, hashlib, urllib.request, zipfile, io, contextlib
import pandas as pd
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
import great_expectations as gx

ROOT=Path(__file__).parent
TMP=ROOT/'tmp'
RESULTS=ROOT/'results'
KEYS=['gender','discharge_destination','readmitted_30_days']


def aggregate(frame):
    return frame.groupby(KEYS, observed=True).agg(
        count=('length_of_stay','size'), sum_stay=('length_of_stay','sum'),
        sum_bmi=('bmi','sum'))


def main():
    TMP.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    source=TMP/'source.csv'
    url='https://api.kaggle.com/v1/datasets/download/siddharth0935/hospital-readmission-predictionsynthetic-dataset'
    if not source.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            archive=zipfile.ZipFile(io.BytesIO(response.read()))
        source.write_bytes(archive.read('hospital_readmissions_30k.csv'))
    raw=pd.read_csv(source)
    frame=raw.drop_duplicates().copy()
    duplicates=len(raw)-len(frame)
    frame[KEYS]=frame[KEYS].fillna('Unknown')
    before=len(frame)
    frame=frame.dropna(subset=['patient_id','length_of_stay','bmi'])
    rejected=before-len(frame)
    context=gx.get_context(mode='ephemeral')
    asset=context.data_sources.add_pandas(name='hospital').add_dataframe_asset(name='source')
    batch=asset.add_batch_definition_whole_dataframe('all').get_batch(batch_parameters={'dataframe':frame})
    checks={}
    expectations={
        'id_not_null':gx.expectations.ExpectColumnValuesToNotBeNull(column='patient_id'),
        'id_unique':gx.expectations.ExpectColumnValuesToBeUnique(column='patient_id'),
        'stay_range':gx.expectations.ExpectColumnValuesToBeBetween(column='length_of_stay',min_value=1,max_value=10),
        'bmi_range':gx.expectations.ExpectColumnValuesToBeBetween(column='bmi',min_value=0,max_value=100),
        'outcome':gx.expectations.ExpectColumnValuesToBeInSet(column='readmitted_30_days',value_set=['Yes','No']),
    }
    with contextlib.redirect_stderr(io.StringIO()):
        for name, exp in expectations.items():
            result=batch.validate(exp)
            checks[name]={'success':bool(result.success),'unexpected_count':result.result.get('unexpected_count',0)}
    assert all(x['success'] for x in checks.values())
    table=pa.Table.from_pandas(frame,preserve_index=False)
    schema=table.schema
    columns=frame.columns.tolist()
    types={c:'int64' if pd.api.types.is_integer_dtype(frame[c]) else
           'float64' if pd.api.types.is_float_dtype(frame[c]) else 'string' for c in columns}
    cases=[]
    for label,target in [('full',None),('100mb',100_000_000),('1gb',1_000_000_000)]:
        path=TMP/f'{label}.csv'; parquet=TMP/f'{label}.parquet'
        rows=0; pieces=0
        with path.open('wb') as out,pq.ParquetWriter(parquet,schema,compression='snappy') as writer:
            while True:
                part=table.set_column(0,'patient_id',pa.array(frame.patient_id.to_numpy()+pieces*len(frame)))
                sink=pa.BufferOutputStream()
                csv.write_csv(part,sink,write_options=csv.WriteOptions(include_header=pieces==0))
                blob=sink.getvalue().to_pybytes()
                if target is not None and out.tell()+len(blob)>target:
                    blob=blob[:target-out.tell()]
                    blob=blob[:blob.rfind(b'\n')+1]
                    take=blob.count(b'\n')-(1 if pieces==0 else 0)
                    if take<=0:break
                    part=part.slice(0,take)
                else:take=len(frame)
                out.write(blob);writer.write_table(part)
                rows+=take;pieces+=1
                if target is None or take<len(frame):break
        q,r=divmod(rows,len(frame))
        reference=aggregate(frame)*q
        if r:reference=reference.add(aggregate(frame.iloc[:r]),fill_value=0)
        records=reference.reset_index().to_dict('records')
        cases.append({'name':label,'rows':rows,'csv_bytes':path.stat().st_size,
                      'parquet_bytes':parquet.stat().st_size,'reference':records})
        print(label,rows,path.stat().st_size,flush=True)
    salted=frame.assign(_salt=frame.patient_id%4).groupby(KEYS+['_salt']).size().reset_index(name='records')
    salted.to_json(TMP/'key_salt_counts.jsonl',orient='records',lines=True)
    metadata={'dataset_url':url,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'source_rows':len(raw),'clean_rows':len(frame),'duplicates_removed':duplicates,
              'critical_missing_removed':rejected,'missing_before':raw.isna().sum().to_dict(),
              'checks':checks,'types':types,'keys':KEYS,'cases':cases,
              'note':'100 MB/1 GB repeat the synthetic source with unique test IDs; not new clinical observations.'}
    (RESULTS/'data.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
