"""Отдельный стенд: Spark Structured Streaming → ONNX → Redis SET NX."""

import json
import os

from pyspark.sql import SparkSession

from common import DESTINATIONS, MODEL_VERSION, build_features, dumps, feature_key, key, redis_client, scores, session


def main():
    spark = (SparkSession.builder.appName('Kafka ONNX readmission')
             .config('spark.ui.bindAddress', '0.0.0.0').config('spark.sql.shuffle.partitions', 2).getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    redis = redis_client()
    model = session()

    def predict(batch, batch_id):
        events = [json.loads(row.payload) for row in batch.collect()]
        events = [event for event in events if not event.get('tick')]
        if not events:
            return
        profiles = redis.mget([feature_key(event['patient']) for event in events])
        valid, vectors = [], []
        with redis.pipeline(transaction=False) as output:
            for event, profile in zip(events, profiles, strict=True):
                if profile is None or not 1 <= event['stay'] <= 365 or event['destination'] not in DESTINATIONS:
                    output.set(key(event['run'], 'dlq', event['id']), dumps({'id': event['id'], 'reason': 'invalid_features'}), nx=True)
                    continue
                valid.append(event)
                vectors.append(build_features(json.loads(profile), event))
            if vectors:
                for event, vector, score in zip(valid, vectors, scores(model, vectors), strict=True):
                    result = {'id': event['id'], 'score': float(score), 'model': MODEL_VERSION, 'batch_id': batch_id}
                    if event.get('audit'):
                        result['features'] = vector
                    output.set(key(event['run'], 'pred', event['id']), dumps(result), nx=True)
            output.execute()
        if any(event.get('fault') and redis.set(key(event['run'], 'spark-fault', event['id']), '1', nx=True)
               for event in events):
            raise RuntimeError('Injected Spark driver failure after writing results')

    stream = (spark.readStream.format('kafka').option('kafka.bootstrap.servers', os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092'))
              .option('subscribe', 'spark-events').option('startingOffsets', 'earliest').option('maxOffsetsPerTrigger', 2000)
              .load().selectExpr('CAST(value AS STRING) AS payload'))
    (stream.writeStream.foreachBatch(predict).option('checkpointLocation', '/state/checkpoint')
     .trigger(processingTime='200 milliseconds').start().awaitTermination())


if __name__ == '__main__':
    main()
