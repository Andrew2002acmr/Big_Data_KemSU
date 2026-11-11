select encounters.*, destination.destination_id
from {{ ref('stg_encounters') }} as encounters
left join {{ ref('dim_destination') }} as destination using (discharge_destination)
