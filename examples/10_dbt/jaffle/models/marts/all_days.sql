{{ config(materialized = 'table') }}

with days as (
    {{ dbt.date_spine('day', "make_date(2020, 1, 1)", "make_date(2027, 1, 1)") }}
),

final as (
    select cast(date_day as date) as date_day
    from days
)

select * from final
