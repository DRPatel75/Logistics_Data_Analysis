# E-Commerce Logistics Data Analysis

## Project Overview

This project analyzes e-commerce logistics data using Python
to understand delivery performance, freight costs, customer
patterns, and route optimization opportunities.

## Objectives

- Analyze delivery performance
- Calculate logistics KPIs
- Identify factors affecting delivery delays
- Perform exploratory data analysis
- Apply regression
- Apply clustering
- Explore vehicle route optimization

## Dataset

Brazilian E-Commerce Public Dataset by Olist.

Dataset:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google OR-Tools
- Jupyter Notebook

## Project Structure

```text
data/
notebooks/
src/
reports/
screenshots/



## Week 2 - Data Collection, Cleaning and Preprocessing

This week focuses on preparing the Brazilian E-Commerce Public Dataset
by Olist for logistics analysis.

### Data Collection
The project uses publicly available Brazilian e-commerce logistics data
containing orders, products, customers, sellers, payments and reviews.

### Data Cleaning
The following preprocessing operations were performed:

- Dataset inspection
- Missing value analysis
- Duplicate detection
- Date conversion
- Invalid delivery-time detection
- Delivery-time calculation
- Delivery-delay calculation
- On-time delivery classification
- Product measurement imputation
- Product category handling
- Outlier detection
- Dataset merging

### Processed Dataset

The cleaned dataset is stored in:

data/processed/olist_logistics_cleaned.csv

### Key Logistics Variables

- Delivery time
- Delivery delay
- On-time delivery
- Product price
- Freight value
- Total item cost
- Customer location
- Seller location
- Product category

## Week 4: Predictive Modeling and Optimization

### Objective

Week 4 focuses on applying predictive modeling and optimization
techniques to the logistics dataset to forecast delivery time and
identify opportunities for improving logistics operations.

### Prediction Target

The target variable is:

- `delivery_time_days`

### Features Used

Numerical features:

- `price`
- `freight_value`
- `total_item_cost`
- `product_weight_g`
- `product_length_cm`
- `product_height_cm`
- `product_width_cm`
- `price_normalized`
- `freight_normalized`

Categorical features:

- `customer_state`
- `seller_state`
- `product_category_name`

### Models

Four regression models were evaluated:

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression
4. Gradient Boosting Regression

### Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Best Model

Random Forest Regression achieved the strongest performance.

- MAE: 5.0511 days
- RMSE: 8.0400 days
- R²: 0.2483

### Optimization Analysis

The project analyzes:

- Delivery time
- Delivery delays
- Late-delivery rates
- Freight costs
- Regional logistics performance

Priority regions were identified using above-median late-delivery rates
and above-median freight costs.

### Recommendations

- Focus operational improvements on high late-delivery regions.
- Investigate high freight-cost regions for transportation optimization.
- Consider geographic factors when planning seller allocation.
- Prioritize orders predicted to require longer delivery times.
- Improve resource allocation in regions with operational challenges.