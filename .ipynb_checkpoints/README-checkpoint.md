# Ferry Ticket Redemption Forecasting System

## Overview
This project develops a forecasting system to predict ferry ticket redemptions and sales using a combination of traditional statistical methods and advanced machine learning techniques. The goal is to enable ferry operators to better plan ticket availability and staffing by providing accurate and reliable demand forecasts.


The system analyzes historical ferry ticket data from the Toronto Island Ferry Ticket Counts.csv dataset, capturing seasonal trends, calendar effects, and longer-term demand shifts.

The primary goal is to support the City of Toronto in optimizing ferry operations by predicting daily demand.


## Dataset
Toronto Island Ferry Ticket Counts.csv
Contains daily historical counts of ferry ticket redemptions and sales over multiple years.


**Key attributes include:**
- Date and time
- Number of ticket redemptions
- Number of ticket sales

  
## Assumptions
- The City of Toronto is interested in predicting daily ferry demand to better plan resources.
- Redemptions (rides) and sales (purchases) are treated separately due to their timing.
- The data shows strong weekly and yearly seasonality (e.g., weekends, summer peaks).
- Ticket redemptions and sales were modeled independently for clarity.
- External factors (e.g., weather, local events) were not included but could enhance accuracy.


## Methodology
### Models Implemented

**Base Model (Seasonal Decomposition)**
Decomposes time series data into seasonal components and uses average seasonal patterns for forecasting.

**Base Model Two (Trend + Seasonal Decomposition)**
Extends the Base Model by adding a trend component to capture long-term demand changes.

**Sales Model**
Focuses specifically on forecasting sales counts using trend and seasonal components.

**XGBoost Machine Learning Model**
Utilizes gradient boosting regression with engineered features such as lagged values, rolling averages, day of week, month, and holiday indicators to capture complex temporal patterns.


## Feature Engineering
- **Time-based**: Day of week, month, weekend flag  
- **Lagged Values**: 1-day, 7-day, and 14-day lags for redemptions  
- **Rolling Stats**: 3-day and 7-day rolling mean  
- **Categoricals**: Month, day of week, hour  
- **Derived**: Trend components from seasonal decomposition


## Evaluation Strategy
- Time-series cross-validation with 4 sequential folds, each with a test size of one year (365 days).

**Performance metrics:**

- Mean Absolute Percentage Error (MAPE)

- Root Mean Squared Error (RMSE)

- R-squared (R²)


## Results Summary

| **Model**         | **Avg MAPE** | **Avg RMSE** | **Avg R²** | **Notes**                                   |
|-------------------|--------------|--------------|------------|---------------------------------------------|
| Base Model        | 86.5%        | 3769         | 0.17       | Limited accuracy with seasonality only      |
| Base Model Two    | 48.0%        | 2860         | 0.50       | Improved accuracy with trend addition       |
| Sales Model       | 57.5%        | 2530         | 0.58       | Moderate performance on sales data          |
| XGBoost           | 29.7%        | 1342         | 0.89       | Best performance with engineered features   |


## Visualizations
- Heatmaps to analyze feature correlations

- Residual plots for model diagnostics

- Line graphs comparing predicted vs. actual ticket counts

These visual tools support model validation and interpretation.


## Development Process
All modeling was encapsulated within a reusable, modular class: `RedemptionModel`.

- Handles preprocessing, feature engineering, model training, evaluation, and plotting.
- Enables easy experimentation with multiple model types via flags.
- Facilitates reproducibility and collaborative development.


## Repository Contents
- `Model.py`: Contains the `RedemptionModel` class and all forecasting logic.
- `modeling.ipynb`: Jupyter notebook for experimentation and visualizations.
- `README.md`: This file, with project overview and documentation.


## Environment 
1. Install dependencies:
```bash
pip install pandas statsmodels scikit-learn matplotlib xgboost numpy seaborn
```
2. Run the notebook `modeling.ipynb` or use `Model.py` directly.
3. Customize parameters like model type and fold settings as needed.


## Resources
**Statsmodels**
•	statsmodels.api as sm: Statsmodels API Reference

**Scikit-learn Metrics and Model Selection**
•	mean_absolute_percentage_error (MAPE): MAPE Documentation
•	mean_squared_error: MSE Documentation
•	r2_score: R² Score Documentation
•	TimeSeriesSplit: TimeSeriesSplit Documentation

 **XGBoost**
•	XGBRegressor: XGBRegressor Documentation

## AI Usage Disclosure (Copilot)
This disclosure is made in the interest of transparency regarding the use of AI tools:
- Formatting Support: AI was used to assist with bullet points, tables, and summaries after the initial draft to meet word count and formatting requirements for submission.

- All model development, data analysis, and coding—including model selection, feature engineering, training, and evaluation—were performed independently by me using Python and relevant libraries (e.g., XGBoost, statsmodels, scikit-learn, etc.).



## Contact
 **Celestine Jiwueze**
 Email: jiwuezec@yahoo.co.uk
 Phone No: 2045574601
Applied Scientist - Data Science Applicant  










