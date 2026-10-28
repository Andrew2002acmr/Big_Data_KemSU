"""Same composite-key aggregation on identical CSV inputs, with full result checks."""
from pathlib import Path
import os,sys,json,time,statistics,math
os.environ.setdefault('POLARS_MAX_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from metrics import Meter

ROOT=Path(__file__).parent
KEYS=['gender','discharge_destination','readmitted_30_days']
COLS=KEYS+['length_of_stay','bmi']


def verify(records,reference):
    expected={tuple(r[k] for k in KEYS):r for r in reference}
    actual={tuple(r[k] for k in KEYS):r for r in records}
    assert len(records)==len(actual)==len(expected)
    assert actual.keys()==expected.keys()
    for row in records:
        ref=expected[tuple(row[k] for k in KEYS)]
        assert int(row['count'])==int(ref['count'])
        assert int(row['sum_stay'])==int(ref['sum_stay'])
        assert math.isclose(float(row['sum_bmi']),float(ref['sum_bmi']),rel_tol=1e-11,abs_tol=1e-5)


def main(engine,size,runs=3):
    data=json.loads((ROOT/'results'/'data.json').read_text(encoding='utf-8'))
    case=next(x for x in data['cases'] if x['name']==size)
    path=ROOT/'tmp'/f'{size}.csv'
    start=time.perf_counter()
    import pandas as pd
    spark=None;con=None
    if engine=='dask':import dask.dataframe as dd
    elif engine=='polars':import polars as pl
    elif engine=='duckdb':
        import duckdb
        con=duckdb.connect();con.execute('SET threads=4');con.execute("SET memory_limit='4GB'")
    elif engine=='spark':
        os.environ['SPARK_LOCAL_IP']='127.0.0.1';os.environ['PYSPARK_PYTHON']=sys.executable
        from pyspark.sql import SparkSession,functions as F,types as T
        spark=(SparkSession.builder.master('local[4]').appName('Lab2Comparison')
            .config('spark.ui.enabled','false').config('spark.driver.memory','2g')
            .config('spark.sql.shuffle.partitions','200').config('spark.default.parallelism','4')
            .config('spark.sql.adaptive.enabled','false').getOrCreate())
        spark.sparkContext.setLogLevel('ERROR')
        schema=T.StructType([T.StructField(c,T.LongType() if t=='int64' else
             T.DoubleType() if t=='float64' else T.StringType()) for c,t in data['types'].items()])
    elif engine!='pandas':raise ValueError(engine)
    startup=time.perf_counter()-start
    measurements=[];records=[]
    for _ in range(runs):
        meter=Meter().start()
        if engine=='pandas':
            frame=pd.read_csv(path,usecols=COLS,dtype={c:data['types'][c] for c in COLS})
            result=frame.groupby(KEYS,observed=True).agg(count=('length_of_stay','size'),
                sum_stay=('length_of_stay','sum'),sum_bmi=('bmi','sum')).reset_index()
            records=result.to_dict('records')
            del frame,result
        elif engine=='dask':
            frame=dd.read_csv(str(path),usecols=COLS,dtype={c:data['types'][c] for c in COLS},blocksize='32MiB')
            result=frame.groupby(KEYS,observed=True).agg({'length_of_stay':['count','sum'],'bmi':'sum'}).compute(
                scheduler='threads',num_workers=4)
            result.columns=['count','sum_stay','sum_bmi']
            records=result.reset_index().to_dict('records');del frame,result
        elif engine=='polars':
            result=pl.scan_csv(path,schema_overrides={c:pl.Int64 if t=='int64' else
                pl.Float64 if t=='float64' else pl.String for c,t in data['types'].items()}).group_by(KEYS).agg(
                pl.len().alias('count'),pl.col('length_of_stay').sum().alias('sum_stay'),
                pl.col('bmi').sum().alias('sum_bmi')).collect(engine='streaming')
            records=result.to_dicts();del result
        elif engine=='duckdb':
            sql="""SELECT gender,discharge_destination,readmitted_30_days,COUNT(*) AS count,
                SUM(length_of_stay) AS sum_stay,SUM(bmi) AS sum_bmi
                FROM read_csv(?,header=true,columns={patient_id:'BIGINT',age:'BIGINT',gender:'VARCHAR',
                blood_pressure:'VARCHAR',cholesterol:'BIGINT',bmi:'DOUBLE',diabetes:'VARCHAR',
                hypertension:'VARCHAR',medication_count:'BIGINT',length_of_stay:'BIGINT',
                discharge_destination:'VARCHAR',readmitted_30_days:'VARCHAR'})
                GROUP BY gender,discharge_destination,readmitted_30_days"""
            cursor=con.execute(sql,[str(path)])
            names=[x[0] for x in cursor.description]
            records=[dict(zip(names,row)) for row in cursor.fetchall()]
        else:
            result=spark.read.option('header',True).schema(schema).csv(path.as_uri()).groupBy(KEYS).agg(
                F.count('*').alias('count'),F.sum('length_of_stay').alias('sum_stay'),F.sum('bmi').alias('sum_bmi'))
            records=[r.asDict() for r in result.collect()]
        measurements.append(meter.finish())
        verify(records,case['reference'])
    records.sort(key=lambda r:tuple(r[k] for k in KEYS))
    for row in records:
        row['avg_stay']=float(row['sum_stay'])/int(row['count'])
        row['avg_bmi']=float(row['sum_bmi'])/int(row['count'])
    medians={key:statistics.median(m[key] for m in measurements) for key in measurements[0]}
    output={'engine':engine,'case':size,'input_bytes':case['csv_bytes'],'rows':case['rows'],
            'startup_seconds':startup,'runs':measurements,'median':medians,
            'max_peak_rss_bytes':max(m['peak_rss_bytes'] for m in measurements),'result':records,
            'cpu_logical':__import__('psutil').cpu_count(),
            'ram_bytes':__import__('psutil').virtual_memory().total,'verified':True}
    (ROOT/'results'/f'{engine}_{size}.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(engine,size,round(medians['seconds'],3),'s',flush=True)
    if spark:spark.stop()
    if con:con.close()


if __name__=='__main__':main(sys.argv[1],sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 3)
