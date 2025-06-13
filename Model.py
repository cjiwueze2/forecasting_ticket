import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error as MAPE, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import numpy as np

class RedemptionModel:
    """
    A forecasting class to model and evaluate different techniques
    for predicting ferry ticket redemptions and sales.

    Supports baseline statistical models, and machine learning (XGBoost).
    """

    def __init__(self, X, target_col):
        
        '''
        Args:
        X (pandas.DataFrame): Dataset of predictors, output from load_data()
        target_col (str): column name for target variable
        '''
        self._predictions = {}
        self.X = X
        self.target_col = target_col
        self.results = {} # dict of dicts with model results

    
    def score(self, truth, preds):
        """
        Calculates evaluation metrics for model performance.
        Ignores cases where ground truth is zero to avoid division errors.

        Returns:
        - Dictionary of MAPE, RMSE, R-squared
        """
        mask = truth != 0
        truth = truth[mask]
        preds = preds[mask]
        return {
            'MAPE': MAPE(truth, preds),
            'RMSE': mean_squared_error(truth, preds) ** 0.5,
            'R2': r2_score(truth, preds)
        }


    def run_models(self, n_splits=4, test_size=365, period=365,
                   base_model=1, sales_model=False, ml_model=False):
        """
        Run selected models and store results for cross-validated splits in self.results.
    
        Parameters:
        - n_splits (int): Number of cross-validation splits.
        - test_size (int): Size of each test fold.
        - period (int): Seasonal period (e.g., 365 for daily seasonality).
        - base_model (int): Choose 1 or 2 to select base model variant.
        - sales_model (bool): If True, run the sales-focused model.
        - ml_model (bool): If True, run the XGBoost machine learning model.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        cnt = 0  # Fold counter
    
        for train_idx, test_idx in tscv.split(self.X):
            X_train = self.X.iloc[train_idx]
            X_test = self.X.iloc[test_idx]
    
            # Base Model 1
            if base_model == 1:
                preds = self._base_model(X_train, X_test, period)
                model_name = 'Base'
    
            # Base Model 2
            elif base_model == 2:
                preds = self._base_model_two(X_train, X_test, period)
                model_name = 'BaseModelTwo'
    
            # Sales Model
            elif sales_model:
                preds = self._sales_model(X_train, X_test, period)
                model_name = 'SalesModel'
    
            # Machine Learning Model
            elif ml_model:
                preds = self._ml_model(X_train, X_test)
                model_name = 'XGBoost'
    
            else:
                preds = self._base_model(X_train, X_test, period)
                model_name = 'Base'
    
            # Align prediction and truth to avoid length mismatch errors
            truth = X_test[self.target_col]
            truth, preds = truth.align(preds, join='inner')
    
            # Initialize results dictionary if not already present
            if model_name not in self.results:
                self.results[model_name] = {}
    
            self.results[model_name][cnt] = self.score(truth, preds)
            self.plot(preds, model_name)
            cnt += 1
    
        # Summarize all results after cross-validation
        self.summarize_results()


    def _store_results(self, X_test, preds, model_key, cnt, sales=False):
        """
        Stores evaluation results for a specific model and split.
        Also plots predictions against actuals.

        Parameters:
        - X_test (pd.DataFrame): Test dataset containing true target values
        - preds (pd.Series): Model predictions for the test set
        - model_key (str): Identifier for the model being evaluated
        - cnt (int): Fold number or split index
        - sales (bool): Whether to use 'Sales Count' as target instead of default
        """
        if model_key not in self.results:
            self.results[model_key] = {}

        target = 'Sales Count' if sales else self.target_col
        truth = X_test[target]
        truth, preds = truth.align(preds, join='inner')
        self.results[model_key][cnt] = self.score(truth, preds)
        self.plot(preds, model_key)

    
    # Original Models wth modification   
    def _base_model(self, train, test, period):
        '''
        Our base, too-simple model.
        Your model needs to take the training and test datasets (dataframes)
        and output a prediction based on the test data.

        Please leave this method as-is.

        '''
        res = sm.tsa.seasonal_decompose(train[self.target_col], period=period)
        seasonal = res.seasonal.clip(lower=0)
        seasonal.index = seasonal.index.dayofyear
        daily_avg = seasonal.groupby(seasonal.index).mean()
        return pd.Series(index=test.index, data=[daily_avg.get(x.dayofyear, 0) for x in test.index])
        
    
    # Improved Base Model Two using trend + seasonal decomposition components
    def _base_model_two(self, train, test, period):
        """
        Improved base model using both trend and seasonal components.
        Aggregates by dayofyear and fills missing values in trend.
    
        Parameters:
        - train (pd.DataFrame): Training dataset containing the target variable
        - test (pd.DataFrame): Test dataset to predict
        - period (int): Seasonality period for decomposition (e.g., 365 for yearly)
    
        Returns:
        - pd.Series: Predictions aligned with the test index
        """
        res = sm.tsa.seasonal_decompose(train[self.target_col], period=period)
        trend = res.trend.bfill().ffill()
        seasonal = res.seasonal
        combined = (trend + seasonal).dropna()
        combined.index = combined.index.dayofyear
        daily_avg = combined.groupby(combined.index).mean()
        return pd.Series(index=test.index, data=[daily_avg.get(x.dayofyear, 0) for x in test.index])

        
    # Sales forecasting model using seasonal decomposition (trend + seasonal)
    def _sales_model(self, train, test, period):
        """
        Predicts ticket sales using seasonal decomposition of time series data.
    
        The model extracts the trend and seasonal components from the 'Sales Count'
        column, combines them, averages by day-of-year, and uses these averages
        to forecast the test period.
    
        Parameters:
        - train (pd.DataFrame): Training dataset
        - test (pd.DataFrame): Test dataset
        - period (int): Period for seasonal decomposition (e.g., 365 for yearly)
    
        Returns:
        - pd.Series: Predicted sales counts for the test set
        """
        res = sm.tsa.seasonal_decompose(train['Sales Count'], period=period)
        trend = res.trend.bfill().ffill()
        seasonal = res.seasonal
        combined = (trend + seasonal).dropna()
        combined.index = combined.index.dayofyear
        daily_avg = combined.groupby(combined.index).mean()
        return pd.Series(index=test.index, data=[daily_avg.get(x.dayofyear, 0) for x in test.index])

    
    # Machine learning model using XGBoost for time series prediction
    def _ml_model(self, train, test):
        """
        Trains an XGBoost regressor on engineered time series features
        and predicts target values on the test set.
    
        Steps:
        - Creates lag and calendar-based features using `create_features`.
        - Trains an XGBoost model on the training data.
        - Returns predictions as a pandas Series indexed by test data timestamps.
    
        Parameters:
        - train (pd.DataFrame): Training set
        - test (pd.DataFrame): Test set
    
        Returns:
        - pd.Series: Predicted values for the test set
    
        """
        train = self.create_features(train)
        test = self.create_features(test)
        features = ['lag_1', 'lag_7', 'lag_14', 'rolling_mean_3', 'rolling_mean_7',
                    'dayofweek', 'hour', 'month', 'is_weekend']

        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
        model.fit(train[features], train[self.target_col])
        preds = model.predict(test[features])
        return pd.Series(index=test.index, data=preds)
        

    def create_features(self, df):
        """
        Generates time-based and lag-based features for time series modeling.
    
        Features created:
        - Lag values (1, 7, 14 days)
        - Rolling averages (3-day, 7-day)
        - Calendar-based features: day of week, hour, month, weekend indicator
    
        Parameters:
        - df (pd.DataFrame): Input DataFrame with a datetime index and target column.
    
        Returns:
        - pd.DataFrame: DataFrame with engineered features and no missing values.
        """
        df = df.copy()
        df['lag_1'] = df[self.target_col].shift(1)
        df['lag_7'] = df[self.target_col].shift(7)
        df['lag_14'] = df[self.target_col].shift(14)
        df['rolling_mean_3'] = df[self.target_col].rolling(3).mean()
        df['rolling_mean_7'] = df[self.target_col].rolling(7).mean()
        df['dayofweek'] = df.index.dayofweek
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        return df.dropna() 


    def plot(self, preds, label):
        """
        Plots model predictions against actual observed values for visual comparison.
    
        Parameters:
        - preds (pd.Series): Predicted values indexed by date.
        - label (str): Label for the prediction line in the plot legend.
        """
        plt.figure(figsize=(15, 5))
        plt.scatter(self.X.index, self.X[self.target_col], s=0.4, color='grey', label='Observed')
        plt.plot(preds.index, preds.values, label=label, color='red')
        plt.title(f"{label} Prediction vs Actual")
        plt.xlabel("Date")
        plt.ylabel(self.target_col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
   

    def summarize_results(self):
        """
        Prints a detailed summary of evaluation metrics (MAPE, RMSE, R²) 
        for each model across all time-series cross-validation folds.
    
        For each model:
        - Displays metrics per fold
        - Computes and displays the average performance
        """
        print(f"\n Summary for '{self.target_col}' Forecasting:\n")
        for model_name, splits in self.results.items():
            print(f"Model: {model_name}")
            mape_list, rmse_list, r2_list = [], [], []
            for i, metrics in splits.items():
                print(f"  Fold {i}: MAPE={metrics['MAPE']:.4f}, RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.4f}")
                mape_list.append(metrics['MAPE'])
                rmse_list.append(metrics['RMSE'])
                r2_list.append(metrics['R2'])
            print(f"  ➤ Avg MAPE: {np.mean(mape_list):.4f}")
            print(f"  ➤ Avg RMSE: {np.mean(rmse_list):.2f}")
            print(f"  ➤ Avg R²: {np.mean(r2_list):.4f}\n")


   

