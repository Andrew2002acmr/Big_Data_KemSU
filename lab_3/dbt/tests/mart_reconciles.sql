with fact as (
    select count(*) as n, count(*) filter (where readmitted_30_days) as r
    from {{ ref('fct_encounters') }}
), mart as (
    select sum(encounter_count) as n, sum(readmitted_count) as r
    from {{ ref('mart_readmissions') }}
)
select * from fact cross join mart
where fact.n is distinct from mart.n or fact.r is distinct from mart.r
