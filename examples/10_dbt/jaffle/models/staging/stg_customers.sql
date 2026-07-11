select
    id as customer_id,
    name as customer_name,
    region,
    cast(customer_since as date) as customer_since
from {{ source('jaffle', 'raw_customers') }}
