select
    idempotency_key,
    patient_id::bigint as patient_id,
    age::smallint as age,
    gender,
    split_part(blood_pressure, '/', 1)::smallint as systolic,
    split_part(blood_pressure, '/', 2)::smallint as diastolic,
    cholesterol::smallint as cholesterol,
    bmi::double as bmi,
    {{ yes_no('diabetes') }} as has_diabetes,
    {{ yes_no('hypertension') }} as has_hypertension,
    medication_count::smallint as medication_count,
    length_of_stay::smallint as length_of_stay,
    discharge_destination,
    {{ yes_no('readmitted_30_days') }} as readmitted_30_days
from {{ source('raw', 'encounters') }}
