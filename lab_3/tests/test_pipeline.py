import pytest
from prefect.testing.utilities import prefect_test_harness

from demo import demonstrate, fixture, write_csv
from source import COLUMNS, inspect_batch
from warehouse import warehouse_summary, write_dlq


@pytest.fixture(scope='module')
def verification(tmp_path_factory):
    work = tmp_path_factory.mktemp('pipeline')
    with prefect_test_harness():
        return demonstrate(fixture(work / 'source.csv'), work)


def test_idempotency(verification):
    assert verification['idempotent']
    assert verification['initial']['raw_rows'] == 120
    assert verification['repeated']['changed'] == 0


def test_dlq_and_recovery(verification):
    assert verification['dlq_by_reason'] == {'incompatible_schema': 1, 'invalid_pressure_format': 1}
    assert verification['incompatible_blocked']
    assert verification['repaired']['raw_rows'] == 120
    assert verification['history']['restored_version'] == 3
    assert verification['gx_negative_test']['failed'] == ['expect_column_values_to_be_between']


def test_conflicting_rows_are_not_loaded(tmp_path):
    first = ['1', '44', 'Female', '120/80', '190', '24.5', 'No', 'No', '2', '3', 'Home', 'No']
    changed = first.copy()
    changed[1] = '45'
    path = write_csv(tmp_path / 'conflict.csv', COLUMNS, [first, changed])
    _, records, rejected, _ = inspect_batch(path.read_bytes())
    assert records == []
    assert [row[2] for row in rejected] == ['conflicting_patient_id', 'conflicting_patient_id']


def test_dlq_does_not_duplicate_replayed_errors(tmp_path):
    database = tmp_path / 'test.duckdb'
    rejected = [('batch', 2, 'invalid_pressure_format', '[]')]
    write_dlq(database, rejected)
    write_dlq(database, rejected)
    assert warehouse_summary(database)['dlq'] == {'invalid_pressure_format': 1}
