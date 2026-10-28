"""Isolated Spark optimizations, actual stages/shuffle metrics and salting."""
from pathlib import Path
import os,sys,json,time,statistics,urllib.request,urllib.parse
from metrics import Meter
from benchmark import KEYS,verify

ROOT=Path(__file__).parent


def main(size='1gb',runs=3,smoke=False):
    os.environ['SPARK_LOCAL_IP']='127.0.0.1';os.environ['PYSPARK_PYTHON']=sys.executable
    from pyspark.sql import SparkSession,functions as F,types as T
    data=json.loads((ROOT/'results'/'data.json').read_text(encoding='utf-8'))
    case=next(c for c in data['cases'] if c['name']==size)
    spark=(SparkSession.builder.master('local[4]').appName('Lab2Optimization')
        .config('spark.driver.memory','2g').config('spark.ui.enabled','true')
        .config('spark.ui.retainedStages','2000').config('spark.ui.retainedJobs','2000')
        .config('spark.default.parallelism','4').config('spark.sql.shuffle.partitions','200')
        .config('spark.sql.adaptive.enabled','false').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    schema=T.StructType([T.StructField(c,T.LongType() if t=='int64' else
        T.DoubleType() if t=='float64' else T.StringType()) for c,t in data['types'].items()])
    plans=ROOT/'results'/'plans';plans.mkdir(exist_ok=True)
    port=urllib.parse.urlparse(spark.sparkContext.uiWebUrl).port
    api=f'http://127.0.0.1:{port}/api/v1/applications/{spark.sparkContext.applicationId}'
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def get(path):
        with opener.open(api+path,timeout=15) as response:return json.load(response)

    def stages(group):
        tracker=spark.sparkContext.statusTracker()
        ids=set()
        for job in tracker.getJobIdsForGroup(group):
            info=tracker.getJobInfo(job)
            if info:ids.update(int(x) for x in info.stageIds)
        result=[]
        for sid in sorted(ids):
            attempts=get(f'/stages/{sid}?details=true')
            complete=[x for x in attempts if x['status']=='COMPLETE']
            if not complete:continue
            stage=max(complete,key=lambda x:x['attemptId'])
            tasks=list(stage.get('tasks',{}).values())
            nonempty=[]
            for task in tasks:
                tm=task.get('taskMetrics',{})
                if tm.get('inputMetrics',{}).get('recordsRead',0) or tm.get('shuffleReadMetrics',{}).get('recordsRead',0):
                    nonempty.append(tm.get('executorRunTime',0))
            selected={k:stage.get(k,0) for k in ['stageId','attemptId','name','numTasks','numCompleteTasks',
                'inputBytes','inputRecords','shuffleReadBytes','shuffleReadRecords','shuffleWriteBytes',
                'shuffleWriteRecords','executorRunTime','executorCpuTime','memoryBytesSpilled','diskBytesSpilled']}
            selected['nonempty_tasks']=len(nonempty)
            selected['task_runtime_ms']={'median':statistics.median(nonempty) if nonempty else 0,
                'max':max(nonempty,default=0)}
            result.append(selected)
        assert result,'No completed Spark stages captured'
        return result

    profiles={
        'baseline':{'shuffle':200,'aqe':False,'format':'csv'},
        'shuffle8':{'shuffle':8,'aqe':False,'format':'csv'},
        'repartition8':{'shuffle':200,'aqe':False,'format':'csv','repartition':True},
        'aqe':{'shuffle':200,'aqe':True,'format':'csv'},
        'parquet':{'shuffle':200,'aqe':False,'format':'parquet'},
        'combined':{'shuffle':8,'aqe':True,'format':'parquet'},
        'salt4':{'shuffle':8,'aqe':False,'format':'csv','salt':True},
    }
    if smoke:profiles={'baseline':profiles['baseline']}

    def query(config):
        spark.conf.set('spark.sql.shuffle.partitions',str(config['shuffle']))
        spark.conf.set('spark.sql.adaptive.enabled',str(config['aqe']).lower())
        path=(ROOT/'tmp'/f"{size}.{config['format']}").as_uri()
        frame=spark.read.parquet(path) if config['format']=='parquet' else spark.read.option('header',True).schema(schema).csv(path)
        if config.get('repartition'):frame=frame.repartition(8,*KEYS)
        if config.get('salt'):
            frame=frame.withColumn('_salt',F.pmod(F.col('patient_id'),F.lit(4)))
            partial=frame.groupBy(KEYS+['_salt']).agg(F.count('*').alias('count'),
                F.sum('length_of_stay').alias('sum_stay'),F.sum('bmi').alias('sum_bmi'))
            return partial.groupBy(KEYS).agg(F.sum('count').alias('count'),
                F.sum('sum_stay').alias('sum_stay'),F.sum('sum_bmi').alias('sum_bmi'))
        return frame.groupBy(KEYS).agg(F.count('*').alias('count'),
            F.sum('length_of_stay').alias('sum_stay'),F.sum('bmi').alias('sum_bmi'))

    # JVM/query warm-up is outside measurements, using only the original small file.
    spark.read.option('header',True).schema(schema).csv((ROOT/'tmp'/'full.csv').as_uri()).groupBy(KEYS).count().collect()
    collected={name:[] for name in profiles}
    for run in range(runs):
        names=list(profiles)
        # Rotate ordering so no optimization always occupies the cold first position.
        names=names[run%len(names):]+names[:run%len(names)]
        for name in names:
            group=f'{name}_{run}'
            spark.sparkContext.setJobGroup(group,group)
            meter=Meter().start()
            result=query(profiles[name])
            rows=[r.asDict() for r in result.collect()]
            measurement=meter.finish()
            verify(rows,case['reference'])
            time.sleep(.1)
            details=stages(group)
            measurement['stages']=details
            measurement['shuffle_write_bytes']=sum(x['shuffleWriteBytes'] for x in details)
            measurement['shuffle_read_bytes']=sum(x['shuffleReadBytes'] for x in details)
            measurement['stage_count']=len(details)
            collected[name].append(measurement)
            if run==0:
                (plans/f'{name}.txt').write_text(result._jdf.queryExecution().toString(),encoding='utf-8')
            print(name,run+1,round(measurement['seconds'],3),'s',len(details),'stages',flush=True)
    summary={name:{'config':profiles[name],'runs':values,
        'median_seconds':statistics.median(v['seconds'] for v in values),
        'median_cpu_seconds':statistics.median(v['cpu_seconds'] for v in values),
        'max_peak_rss_bytes':max(v['peak_rss_bytes'] for v in values)} for name,values in collected.items()}

    # Hash-partition loads are evaluated on compact counts, not by shuffling 1 GB of raw rows.
    sample=spark.read.schema(T.StructType([*[T.StructField(k,T.StringType()) for k in KEYS],
        T.StructField('_salt',T.LongType()),T.StructField('records',T.LongType())])).json((ROOT/'tmp'/'key_salt_counts.jsonl').as_uri())
    buckets={}
    for name,cols in [('before',KEYS),('salt4',KEYS+['_salt'])]:
        loads=sample.withColumn('_bucket',F.pmod(F.hash(*cols),F.lit(8))).groupBy('_bucket').agg(F.sum('records').alias('n')).collect()
        values=[0]*8
        for row in loads:values[row['_bucket']]=int(row['n'])
        mean=statistics.mean(values)
        buckets[name]={'records':values,'cv':statistics.pstdev(values)/mean,'max_to_mean':max(values)/mean}
    output={'case':size,'runs':runs,'profiles':summary,'hash8_load_estimate':buckets,
            'metrics_source':'Spark status REST API and psutil; no fabricated stage/shuffle counts',
            'verified':True}
    dest='spark_smoke.json' if smoke else 'spark_optimizations.json'
    (ROOT/'results'/dest).write_text(json.dumps(output,indent=2),encoding='utf-8')
    spark.stop()


if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else '1gb',
    int(sys.argv[2]) if len(sys.argv)>2 else 3,'--smoke' in sys.argv)
