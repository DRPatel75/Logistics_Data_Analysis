import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# ============================================================
# WEEK 4: PREDICTIVE MODELING AND OPTIMIZATION
# ============================================================

print("=" * 60)
print("WEEK 4 - PREDICTIVE MODELING AND OPTIMIZATION")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load cleaned dataset
# ------------------------------------------------------------

df = pd.read_csv(
    "data/processed/olist_logistics_cleaned.csv"
)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ------------------------------------------------------------
# 2. Prepare modeling dataset
# ------------------------------------------------------------

df_model = df.dropna(
    subset=["delivery_time_days"]
).copy()

print("Modeling dataset shape:", df_model.shape)


# ------------------------------------------------------------
# 3. Define features and target
# ------------------------------------------------------------

numeric_features = [
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

categorical_features = [
    "customer_state",
    "seller_state",
    "product_category_name"
]

target = "delivery_time_days"


# ------------------------------------------------------------
# 4. Create X and y
# ------------------------------------------------------------

X = df_model[
    numeric_features + categorical_features
].copy()

y = df_model[target].copy()


# ------------------------------------------------------------
# 5. One-hot encode categorical features
# ------------------------------------------------------------

X_encoded = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=True,
    dtype=int
)

print("\nEncoded feature shape:", X_encoded.shape)


# ------------------------------------------------------------
# 6. Train-test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# 7. Linear Regression
# ------------------------------------------------------------

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_pred = linear_model.predict(
    X_test
)


# ------------------------------------------------------------
# 8. Decision Tree Regression
# ------------------------------------------------------------

tree_model = DecisionTreeRegressor(
    max_depth=10,
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)

tree_pred = tree_model.predict(
    X_test
)


# ------------------------------------------------------------
# 9. Random Forest Regression
# ------------------------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(
    X_test
)


# ------------------------------------------------------------
# 10. Gradient Boosting Regression
# ------------------------------------------------------------

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_model.fit(
    X_train,
    y_train
)

gb_pred = gb_model.predict(
    X_test
)


# ------------------------------------------------------------
# 11. Model evaluation function
# ------------------------------------------------------------

def evaluate_model(name, y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ------------------------------------------------------------
# 12. Compare models
# ------------------------------------------------------------

results = [

    evaluate_model(
        "Linear Regression",
        y_test,
        linear_pred
    ),

    evaluate_model(
        "Decision Tree",
        y_test,
        tree_pred
    ),

    evaluate_model(
        "Random Forest",
        y_test,
        rf_pred
    ),

    evaluate_model(
        "Gradient Boosting",
        y_test,
        gb_pred
    )
]

results_df = pd.DataFrame(results)

results_df = results_df.round(4)


print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# ------------------------------------------------------------
# 13. Select best model using RMSE
# ------------------------------------------------------------

best_model = results_df.loc[
    results_df["RMSE"].idxmin()
]

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Model:", best_model["Model"])
print("MAE:", best_model["MAE"])
print("RMSE:", best_model["RMSE"])
print("R2:", best_model["R2"])


# ------------------------------------------------------------
# 14. Feature importance - Random Forest
# ------------------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("=" * 60)
print("TOP 15 FEATURES")
print("=" * 60)

print(
    feature_importance.head(15).to_string(index=False)
)


# ------------------------------------------------------------
# 15. Logistics KPI analysis
# ------------------------------------------------------------

logistics_kpi = (
    df_model.groupby("customer_state")
    .agg(
        average_delivery_time=("delivery_time_days", "mean"),
        average_delivery_delay=("delivery_delay_days", "mean"),
        late_delivery_rate=(
            "on_time_delivery",
            lambda x: (1 - x.mean()) * 100
        ),
        average_freight_cost=("freight_value", "mean"),
        average_order_value=("total_item_cost", "mean"),
        order_count=("order_id", "count")
    )
    .sort_values(
        "late_delivery_rate",
        ascending=False
    )
)

print("\n")
print("=" * 60)
print("TOP 10 STATES BY LATE DELIVERY RATE")
print("=" * 60)

print(
    logistics_kpi.head(10).round(2).to_string()
)


# ------------------------------------------------------------
# 16. Identify priority regions
# ------------------------------------------------------------

late_rate_threshold = (
    logistics_kpi["late_delivery_rate"].median()
)

freight_cost_threshold = (
    logistics_kpi["average_freight_cost"].median()
)

priority_regions = logistics_kpi[
    (logistics_kpi["late_delivery_rate"] > late_rate_threshold)
    &
    (logistics_kpi["average_freight_cost"] > freight_cost_threshold)
].copy()


# ------------------------------------------------------------
# 17. Save results
# ------------------------------------------------------------

results_df.to_csv(
    "reports/week4_model_comparison.csv",
    index=False
)

feature_importance.to_csv(
    "reports/week4_feature_importance.csv",
    index=False
)

logistics_kpi.to_csv(
    "reports/week4_logistics_kpi.csv"
)

priority_regions.to_csv(
    "reports/week4_priority_regions.csv"
)


print("\n")
print("=" * 60)
print("RESULTS SAVED SUCCESSFULLY")
print("=" * 60)

print("Model comparison saved.")
print("Feature importance saved.")
print("Logistics KPI saved.")
print("Priority regions saved.")


print("\nWeek 4 analysis completed successfully!")