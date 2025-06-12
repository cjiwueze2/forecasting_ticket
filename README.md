# 🧾 Ferry Ticket Redemption Forecasting

## 📌 Overview
This project forecasts ferry ticket **redemptions** and **sales** in Toronto using time series analysis and machine learning.

The primary goal is to support the City of Toronto in optimizing ferry operations by predicting daily demand.

Models developed include:
- Seasonal Baseline Models
- Trend + Seasonality Models
- Machine Learning (XGBoost)
- Deep Learning (LSTM)


## 🚀 Key Results

| Model      | Avg MAPE | Avg RMSE | Avg R²  |
|------------|----------|----------|---------|
| XGBoost    | 0.30     | 1342.07  | 0.89    |
| Base       | 0.87     | 3768.80  | 0.17    |

🔍 **XGBoost significantly outperformed the baseline models**, reducing RMSE by over 60% and improving R² from 0.17 to 0.89.


## ⚙️ Assumptions
- The City of Toronto is interested in predicting daily ferry demand to better plan resources.
- Redemptions (rides) and sales (purchases) are treated separately due to their timing.
- The data shows strong weekly and yearly seasonality (e.g., weekends, summer peaks).
- Ticket redemptions and sales were modeled independently for clarity.
- External factors (e.g., weather, local events) were not included but could enhance accuracy.


## 🔧 Development Process
All modeling was encapsulated within a reusable, modular class: `RedemptionModel`.

- Handles preprocessing, feature engineering, model training, evaluation, and plotting.
- Enables easy experimentation with multiple model types via flags.
- Facilitates reproducibility and collaborative development.


## 📊 Data Summary
- **Source**: City of Toronto Open Data Portal  
- **Time Range**: Daily time series ticket data  
- **Key Columns**: `Redemption Count`, `Sales Count`  
- **Preprocessing**:
  - Parsed dates and set as datetime index
  - Resampled to daily frequency
  - Filled missing values post-feature generation


## 🧠 Feature Engineering
- **Time-based**: Day of week, month, weekend flag  
- **Lagged Values**: 1-day, 7-day, and 14-day lags for redemptions  
- **Rolling Stats**: 3-day and 7-day rolling mean  
- **Categoricals**: Month, day of week, hour  
- **Derived**: Trend components from seasonal decomposition


## 📁 Repository Contents
- `Model.py`: Contains the `RedemptionModel` class and all forecasting logic.
- `modeling.ipynb`: Jupyter notebook for experimentation and visualizations.
- `README.md`: This file, with project overview and documentation.


## ✅ How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the notebook `modeling.ipynb` or use `Model.py` directly.
3. Customize parameters like model type and fold settings as needed.

## Resources
📊 Statsmodels
•	statsmodels.api as sm: Statsmodels API Reference
📈 Scikit-learn Metrics and Model Selection
•	mean_absolute_percentage_error (MAPE): MAPE Documentation
•	mean_squared_error: MSE Documentation
•	r2_score: R² Score Documentation
•	TimeSeriesSplit: TimeSeriesSplit Documentation
🌲 XGBoost
•	XGBRegressor: XGBRegressor Documentation
🤖 TensorFlow Keras
•	Sequential model: Sequential Model Documentation
•	LSTM layer: LSTM Layer Documentation
•	Dense layer: Dense Layer Documentation
•	Dropout layer: Dropout Layer Documentation

## 📬 Contact
Project by: **Celestine Jiwueze**  
Applied Scientist - Data Science Applicant  
