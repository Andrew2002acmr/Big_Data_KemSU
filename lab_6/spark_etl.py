"""Одинаковая Spark-агрегация для обоих профилей ресурсов."""

import json
import pathlib
import sys
import tempfile
import time

from pyspark.sql import SparkSession, functions as F


def main():
    with tempfile.TemporaryDirectory() as directory:
        spark = (SparkSession.builder.appName('Hospital daily aggregates')
                 .config('spark.eventLog.enabled', 'true')
                 .config('spark.eventLog.compress', 'false')
                 .config('spark.eventLog.dir', pathlib.Path(directory).as_uri()).getOrCreate())
        result = aggregate(spark, sys.argv[1] if len(sys.argv) > 1 else '/opt/spark/work-dir/hospital.csv')
        spark.stop()
        event_log = next(file for file in pathlib.Path(directory).iterdir() if not file.name.startswith('.'))
        with event_log.open() as stream:
            result.update(executor_metrics(map(json.loads, stream)))
    print('LAB6_RESULT ' + json.dumps(result), flush=True)


def aggregate(spark, path):
    spark.sparkContext.setLogLevel('WARN')
    start = time.perf_counter()
    raw = spark.read.option('header', True).csv(path)
    rows = (raw.select('patient_id', 'discharge_destination', 'gender', 'readmitted_30_days',
                       F.col('length_of_stay').cast('int').alias('stay'))
            .withColumn('repeat', F.explode(F.sequence(F.lit(1), F.lit(24))))
            .repartition(4, 'patient_id'))
    result = (rows.groupBy('discharge_destination', 'gender')
              .agg(F.count('*').alias('n'), F.sum(F.when(F.col('readmitted_30_days') == 'Yes', 1).otherwise(0)).alias('readmitted'),
                   F.sum('stay').alias('stay_total')).orderBy('discharge_destination', 'gender').collect())
    groups = [row.asDict() for row in result]
    assert len(groups) == 9 and sum(row['n'] for row in groups) == 720000
    return {'rows': 720000, 'groups': groups, 'processing_sec': time.perf_counter() - start}


def executor_metrics(events):
    added, removed = {}, {}
    disk_spill = memory_spill = 0
    end = 0
    for event in events:
        name = event['Event']
        if name == 'SparkListenerExecutorAdded':
            added[event['Executor ID']] = event['Timestamp']
        elif name == 'SparkListenerExecutorRemoved':
            removed[event['Executor ID']] = event['Timestamp']
        elif name == 'SparkListenerTaskEnd':
            metrics = event.get('Task Metrics') or {}
            disk_spill += metrics.get('Disk Bytes Spilled', 0)
            memory_spill += metrics.get('Memory Bytes Spilled', 0)
        elif name == 'SparkListenerApplicationEnd':
            end = event['Timestamp']
    assert added and end
    seconds = {executor: (removed.get(executor, end) - start) / 1000 for executor, start in added.items()}
    return {'executor_seconds': seconds, 'total_executor_seconds': sum(seconds.values()),
            'disk_spill_bytes': disk_spill, 'memory_spill_bytes': memory_spill}


if __name__ == '__main__':
    main()
