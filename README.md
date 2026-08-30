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