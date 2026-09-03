from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "olist_logistics_cleaned.csv"
SCREENSHOT_DIR = BASE_DIR / "screenshots"

SCREENSHOT_DIR.mkdir(exist_ok=True)


# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

# Convert date columns
date_columns = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    df[column] = pd.to_datetime(df[column], errors="coerce")


print("Dataset loaded successfully.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ---------------------------------------------------------
# 1. Delivery Time Distribution
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="delivery_time_days",
    bins=30,
    kde=True
)

plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Orders")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "week3_delivery_time_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 2. Delivery Delay Distribution
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="delivery_delay_days",
    bins=30,
    kde=True
)

plt.title("Distribution of Delivery Delays")
plt.xlabel("Delivery Delay (Days)")
plt.ylabel("Number of Orders")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "week3_delivery_delay_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 3. Delivery Time Boxplot
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

sns.boxplot(
    x=df["delivery_time_days"]
)

plt.title("Boxplot of Delivery Time")
plt.xlabel("Delivery Time (Days)")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "week3_delivery_time_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 4. Freight Value Boxplot
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

sns.boxplot(
    x=df["freight_value"]
)

plt.title("Distribution of Freight Value")
plt.xlabel("Freight Value")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "week3_freight_value_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 5. Correlation Heatmap
# ---------------------------------------------------------

correlation_columns = [
    "delivery_time_days",
    "delivery_delay_days",
    "price",
    "freight_value",
    "total_item_cost",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "price_normalized",
    "freight_normalized"
]

correlation_matrix = df[correlation_columns].corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Matrix of Logistics Variables")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "week3_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 6. Price vs Freight Value
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="price",
    y="freight_value",
    alpha=0.5
)

plt.title("Relationship Between Product Price and Freight Value")
plt.xlabel("Product Price")
plt.ylabel("Freight Value")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "week3_price_freight_relationship.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 7. Delivery Time vs Delay
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="delivery_time_days",
    y="delivery_delay_days",
    alpha=0.5
)

plt.title("Relationship Between Delivery Time and Delivery Delay")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Delivery Delay (Days)")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "week3_delivery_time_vs_delay.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 8. Product Weight vs Freight
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="product_weight_g",
    y="freight_value",
    alpha=0.5
)

plt.title("Relationship Between Product Weight and Freight Value")
plt.xlabel("Product Weight (g)")
plt.ylabel("Freight Value")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "week3_weight_vs_freight.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 9. Top Product Categories
# ---------------------------------------------------------

category_counts = (
    df["product_category_name"]
    .value_counts()
    .head(15)
)

plt.figure(figsize=(12, 7))

sns.barplot(
    x=category_counts.values,
    y=category_counts.index
)

plt.title("Top 15 Product Categories by Number of Orders")
plt.xlabel("Number of Orders")
plt.ylabel("Product Category")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "week3_top_product_categories.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 10. Customer State Order Volume
# ---------------------------------------------------------

state_orders = (
    df["customer_state"]
    .value_counts()
    .head(15)
)

plt.figure(figsize=(12, 7))

sns.barplot(
    x=state_orders.values,
    y=state_orders.index
)

plt.title("Top 15 Customer States by Order Volume")
plt.xlabel("Number of Orders")
plt.ylabel("Customer State")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "week3_customer_state_orders.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Week 3 visualizations generated successfully.")
print("Screenshots saved to:", SCREENSHOT_DIR)