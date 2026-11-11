"""Трёхфазный тест восстановления после остановки TaskManager."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from benchmark import event, producer_client, seed_profiles, send, timed_load, until
from common import dumps, key, redis_client
from monitor import checkpoint, job, kafka_lag

TOPIC = 'hospital-events'
STATE = 'lab5:node-test:plan'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'queue', 'verify'])
    args = parser.parse_args()
    db = redis_client()
    producer = producer_client()
    if args.phase == 'prepare':
        seed_profiles(db)
        run = 'node-' + uuid4().hex[:10]
        stamp = int(db.get('lab5:virtual-clock') or 1_800_000_000_000) + 30_000
        db.set('lab5:virtual-clock', stamp + 10_000)
        job_id = until(job, 'running Flink job')
        _, [reference] = timed_load(db, producer, TOPIC, [event(run, 'reference', stamp)], 50)
        before = until(lambda: checkpoint(job_id), 'checkpoint before node failure')
        state = {'run': run, 'stamp': stamp, 'job': job_id, 'checkpoint_before': before['id'], 'reference': reference}
        db.set(STATE, dumps(state))
        print(f"Checkpoint {before['id']} before stopping TaskManager")
        return

    state = json.loads(db.get(STATE))
    run = state['run']
    names = [f'backlog-{index}' for index in range(80)]
    keys = [key(run, 'pred', name) for name in names]
    if args.phase == 'queue':
        for index, name in enumerate(names, 1):
            send(producer, TOPIC, event(run, name, state['stamp'] + index))
        assert producer.flush(10) == 0
        assert db.exists(*keys) == 0, 'TaskManager must be stopped before queuing the backlog'
        lag = kafka_lag()
        assert lag >= len(keys)
        db.set(STATE, dumps({**state, 'kafka_lag_during_outage': lag}))
        print(f'Queued {len(keys)} events while TaskManager was stopped, lag={lag}')
        return

    found = until(lambda: (items if all(items := db.mget(keys)) else None), 'backlog after TaskManager restart', 150)
    assert {json.loads(value)['id'] for value in found} == set(names)
    assert state['reference'] == json.loads(db.get(key(run, 'pred', 'reference')))
    after = until(lambda: (done if (done := checkpoint(state['job'])) and done['id'] > state['checkpoint_before'] else None),
                  'checkpoint after TaskManager restart')
    until(lambda: kafka_lag() == 0, 'Kafka lag returns to zero')
    result = {'checkpoint_before': state['checkpoint_before'], 'checkpoint_after': after['id'],
              'kafka_lag_during_outage': state['kafka_lag_during_outage'], 'kafka_lag_after': 0,
              'queued': len(keys), 'recovered': len(found), 'lost': 0, 'reference_unchanged': True}
    Path('/results/node.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(dumps(result))


if __name__ == '__main__':
    main()
