"""Shared parsing and validation for the laboratory's hospital ETL."""
import pandas as pd


def normalize_pressure(frame):
    """Return clean rows, quarantined rows, and error counts without losing bad input."""
    raw = frame['blood_pressure'].astype('string').str.strip()
    valid_format = raw.str.fullmatch(r'[0-9]{1,3}/[0-9]{1,3}', na=False)
    # Mask invalid strings before conversion; nullable integers retain missing values.
    parts = raw.where(valid_format).str.extract(r'^([0-9]{1,3})/([0-9]{1,3})$')
    parts = parts.astype('Int64')
    invalid_order = (parts[0].le(parts[1]) | parts[1].le(0)).fillna(False)
    bad = ~valid_format | invalid_order
    normalized = frame.drop(columns=['blood_pressure']).copy()
    normalized['systolic'] = parts[0]
    normalized['diastolic'] = parts[1]
    quarantine = frame.loc[bad].copy()
    quarantine['error_reason'] = 'invalid_pressure_order'
    quarantine.loc[~valid_format.loc[bad], 'error_reason'] = 'invalid_pressure_format'
    metrics = {
        'input_rows': len(frame), 'clean_rows': int((~bad).sum()),
        'invalid_format': int((~valid_format).sum()),
        'invalid_order': int(invalid_order.sum()), 'quarantine_rows': int(bad.sum()),
    }
    return normalized.loc[~bad].copy(), quarantine, metrics


def validate_batch(batch, gx):
    expectations = {
        'length_of_stay_1_10': gx.expectations.ExpectColumnValuesToBeBetween(
            column='length_of_stay', min_value=1, max_value=10),
        'patient_id_unique': gx.expectations.ExpectColumnValuesToBeUnique(column='patient_id'),
        'readmitted_values': gx.expectations.ExpectColumnValuesToBeInSet(
            column='readmitted_30_days', value_set=['Yes', 'No']),
    }
    results = {}
    for name, expectation in expectations.items():
        result = batch.validate(expectation)
        results[name] = {'success': bool(result.success), 'result': result.result}
    return results
