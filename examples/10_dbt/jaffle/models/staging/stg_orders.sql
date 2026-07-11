select
    id as order_id,
    customer_id,
    amount,
    status as order_status,
    is_food_order,
    order_date
from {{ source('jaffle', 'raw_orders') }}
