
# DoorDash Delivery Duration Prediction

This project builds a supervised machine learning model to predict delivery duration (in seconds) for DoorDash orders using historical order, marketplace, and store data. The prediction is based on data exploration, feature engineering, and evaluation of multiple regression models.

## Project Description
DoorDash, a leading food delivery platform, estimates delivery time at order placement. Accurately predicting delivery duration is crucial to customer satisfaction and logistics planning. The dataset used in this project contains order-level data from 2015, including timestamps, item details, dasher availability, and pre-estimated time metrics.

The goal is to predict delivery duration from order creation to final delivery using available features. This involves preprocessing timestamps, creating engineered variables, encoding categorical data, selecting features, and modeling with regression algorithms.

## Project Objectives and Modeling Approach

**Primary Objectives:**
- Predict total delivery duration (in seconds)
- Engineer informative features from timestamps and marketplace data
- Compare and evaluate multiple regression models

**Solution Design:**
- Preprocess timestamps to compute `delivery_duration`
- Engineer features like `busy_ratio`, `estimated_total_duration`, and price-based metrics
- Remove high-cardinality columns and multicollinear features using VIF
- Rank features using Random Forest Gini importance and test top predictors
- Compare performance of models across feature sets using RMSE

## Feature Engineering Summary
The following derived features were used to enhance model prediction:
- **busy_ratio**: Total busy dashers / total onshift dashers
- **estimated_total_duration**: System-estimated order duration
- **avg_price_per_item**: Subtotal / total_items
- **item_diversity_ratio**: num_distinct_items / total_items
- **price_range_of_items**: max_item_price - min_item_price

Categorical variables (`market_id`, `order_protocol`, `store_primary_category`) were one-hot encoded. High-cardinality identifiers like `store_id` were dropped.

## Step-by-Step Modeling

### Step 1: Data Preparation
- Converted timestamps to datetime and computed `delivery_duration`
- Dropped rows with missing duration or system estimates
- Cleaned extreme values (inf, NaNs) and standardized types

### Step 2: Multicollinearity Removal
- Used Variance Inflation Factor (VIF)
- Iteratively removed features with VIF > 20
- Final selected feature set: 75+ features

### Step 3: Feature Importance
- Trained Random Forest on filtered data
- Ranked features by Gini importance
- Defined three subsets: top 10, top 20, and all features

### Step 4: Model Comparison
- Models tested: Linear, Ridge, Lasso, PLS, Decision Tree, Random Forest, XGBoost
- Evaluated each model on each feature subset (train/test split = 80/20)
- Metrics: Train RMSE, Test RMSE, R^2

### Step 5: Final Evaluation
- Best model: **Lasso Regression** on **top_10** features
- RMSE: ~0.28 (test)
- Final predictions plotted vs actual values

## Key Business Insights
- **Model Simplicity Wins**: Simpler models (like Lasso) outperformed complex models like XGBoost due to regularization and robustness
- **Pre-estimated Duration Helps**: `estimated_total_duration` was consistently among the top predictors
- **Dasher Load Matters**: Marketplace congestion (`busy_ratio`) influences delivery speed
- **Feature Volume vs Quality**: Top 10 features gave better generalization than using all

## Assumptions
- Timestamps are accurate and recorded in UTC
- Dasher-related metrics are correctly aligned with order time
- Feature importance derived from Random Forest generalizes across models

## Tech Stack
- Python (pandas, numpy, matplotlib, seaborn)
- Scikit-learn
- XGBoost
- Cursor IDE / Jupyter Notebook

## Project Structure
```
doordash-duration-model/
├── data/
│   └── historical_data.csv
├── notebooks/
│   └── doordash_delivery_prediction.ipynb
├── plots/
│   └── feature_importances.png
│   └── model_rmse_comparison.png
├── README.md
```

## How to Run
1. Open the notebook in Jupyter or Cursor IDE
2. Load `historical_data.csv`
3. Follow cells step-by-step for preprocessing, modeling, and evaluation

## Authors
- Satkar Karki  
University of South Dakota, MSBA Spring 2025
