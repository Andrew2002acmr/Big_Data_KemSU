"""Хранилище DuckDB, UPSERT, DLQ и версии сырой таблицы."""

from contextlib import contextmanager

import duckdb
import pandas as pd

from source import COLUMNS, DataError

FIELDS = ('idempotency_key', 'payload_hash', *COLUMNS)
FIELD_SQL = ', '.join(FIELDS)


@contextmanager
def transaction(db):
    db.execute('BEGIN')
    try:
        yield
        db.execute('COMMIT')
    except BaseException:
        db.execute('ROLLBACK')
        raise


def initialize(db):
    types = ', '.join(f'{column} VARCHAR NOT NULL' for column in FIELDS[1:])
    db.execute(f'''
        CREATE SCHEMA IF NOT EXISTS raw;
        CREATE SCHEMA IF NOT EXISTS ops;
        CREATE TABLE IF NOT EXISTS raw.encounters (
            idempotency_key VARCHAR PRIMARY KEY, {types}, UNIQUE (patient_id)
        );
        CREATE TABLE IF NOT EXISTS ops.versions (
            version BIGINT PRIMARY KEY, batch_id VARCHAR NOT NULL,
            created_at TIMESTAMPTZ DEFAULT current_timestamp
        );
        CREATE TABLE IF NOT EXISTS ops.snapshots AS
            SELECT CAST(0 AS BIGINT) AS version, * FROM raw.encounters WHERE FALSE;
        CREATE TABLE IF NOT EXISTS ops.dlq (
            batch_id VARCHAR, line BIGINT, reason VARCHAR, payload VARCHAR NOT NULL,
            created_at TIMESTAMPTZ DEFAULT current_timestamp, PRIMARY KEY (batch_id, line, reason)
        );
    ''')


def write_dlq(path, rejected):
    if not rejected:
        return
    with duckdb.connect(str(path)) as db, transaction(db):
        initialize(db)
        db.register('rejected', pd.DataFrame(rejected, columns=['batch_id', 'line', 'reason', 'payload']))
        db.execute('''
            INSERT INTO ops.dlq (batch_id, line, reason, payload)
            SELECT batch_id, line, reason, payload FROM rejected ON CONFLICT DO NOTHING
        ''')


def snapshot(db, batch_id):
    version = db.execute('SELECT coalesce(max(version), 0) + 1 FROM ops.versions').fetchone()[0]
    db.execute('INSERT INTO ops.versions (version, batch_id) VALUES (?, ?)', [version, batch_id])
    db.execute('INSERT INTO ops.snapshots SELECT ?, * FROM raw.encounters', [version])
    return version


def load_raw(path, batch_id, frame, before_commit):
    with duckdb.connect(str(path)) as db, transaction(db):
        initialize(db)
        db.register('incoming', frame)
        updates = ', '.join(f'{name} = s.{name}' for name in FIELDS[1:])
        values = ', '.join(f's.{name}' for name in FIELDS)
        changed = db.execute(f'''
            MERGE INTO raw.encounters AS t USING incoming AS s
            ON t.idempotency_key = s.idempotency_key
            WHEN MATCHED AND t.payload_hash <> s.payload_hash THEN UPDATE SET {updates}
            WHEN NOT MATCHED THEN INSERT ({FIELD_SQL}) VALUES ({values})
        ''').fetchone()[0]
        version = (snapshot(db, batch_id) if changed else
                   db.execute('SELECT max(version) FROM ops.versions').fetchone()[0])
        count = db.execute('SELECT count(*) FROM raw.encounters').fetchone()[0]
        before_commit()
    return {'changed': changed, 'version': version, 'raw_rows': count}


def read_as_of(path, version):
    with duckdb.connect(str(path), read_only=True) as db:
        if db.execute('SELECT 1 FROM ops.versions WHERE version = ?', [version]).fetchone() is None:
            raise DataError(f'Версия {version} отсутствует')
        return db.execute('''
            SELECT * EXCLUDE (version) FROM ops.snapshots WHERE version = ? ORDER BY patient_id::BIGINT
        ''', [version]).fetchdf()


def restore_version(path, version):
    """Вернуть raw к историческому состоянию с сохранением текущей истории."""
    with duckdb.connect(str(path)) as db, transaction(db):
        if db.execute('SELECT 1 FROM ops.versions WHERE version = ?', [version]).fetchone() is None:
            raise DataError(f'Версия {version} отсутствует')
        db.execute('DELETE FROM raw.encounters')
        db.execute('INSERT INTO raw.encounters SELECT * EXCLUDE (version) FROM ops.snapshots WHERE version = ?', [version])
        return snapshot(db, f'restore:{version}')


def warehouse_summary(path):
    with duckdb.connect(str(path), read_only=True) as db:
        rows, distinct_keys = db.execute('SELECT count(*), count(DISTINCT idempotency_key) FROM raw.encounters').fetchone()
        versions = db.execute('SELECT count(*) FROM ops.versions').fetchone()[0]
        dlq = dict(db.execute('SELECT reason, count(*) FROM ops.dlq GROUP BY reason ORDER BY reason').fetchall())
    return {'rows': rows, 'distinct_keys': distinct_keys, 'versions': versions, 'dlq': dlq}
