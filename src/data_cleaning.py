import pandas as pd
import numpy as np


def load_data():
    orders = pd.read_csv(
        "data/raw/olist_orders_dataset.csv"
    )

    order_items = pd.read_csv(
        "data/raw/olist_order_items_dataset.csv"
    )

    customers = pd.read_csv(
        "data/raw/olist_customers_dataset.csv"
    )

    products = pd.read_csv(
        "data/raw/olist_products_dataset.csv"
    )

    sellers = pd.read_csv(
        "data/raw/olist_sellers_dataset.csv"
    )

    return orders, order_items, customers, products, sellers


def clean_orders(orders):

    date_columns = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]

    for column in date_columns:
        orders[column] = pd.to_datetime(
            orders[column],
            errors='coerce'
        )

    orders['delivery_time_days'] = (
        orders['order_delivered_customer_date']
        - orders['order_purchase_timestamp']
    ).dt.total_seconds() / (24 * 60 * 60)

    orders['delivery_delay_days'] = (
        orders['order_delivered_customer_date']
        - orders['order_estimated_delivery_date']
    ).dt.total_seconds() / (24 * 60 * 60)

    orders.loc[
        orders['delivery_time_days'] < 0,
        'delivery_time_days'
    ] = np.nan

    orders['on_time_delivery'] = np.where(
        orders['delivery_delay_days'] <= 0,
        1,
        0
    )

    return orders


def main():

    orders, order_items, customers, products, sellers = load_data()

    orders = clean_orders(orders)

    order_items['total_item_cost'] = (
        order_items['price']
        + order_items['freight_value']
    )

    merged_data = orders.merge(
        order_items,
        on='order_id',
        how='left'
    )

    merged_data = merged_data.merge(
        customers,
        on='customer_id',
        how='left'
    )

    merged_data = merged_data.merge(
        sellers,
        on='seller_id',
        how='left'
    )

    merged_data = merged_data.merge(
        products,
        on='product_id',
        how='left'
    )

    merged_data.to_csv(
        "data/processed/olist_logistics_cleaned.csv",
        index=False
    )

    print("Data cleaning completed successfully.")


if __name__ == "__main__":
    main()