select
    destination_id,
    discharge_destination,
    gender,
    count(*) as encounter_count,
    count(*) filter (where readmitted_30_days) as readmitted_count,
    round(avg(readmitted_30_days::integer), 6) as readmission_rate,
    round(avg(length_of_stay), 2) as avg_length_of_stay
from {{ ref('fct_encounters') }}
group by destination_id, discharge_destination, gender
