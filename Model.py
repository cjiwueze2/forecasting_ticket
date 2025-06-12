import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error as MAPE, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

class RedemptionModel:
    """
    A forecasting class to model and evaluate different techniques
    for predicting ferry ticket redemptions and sales.

    Supports baseline statistical models, machine learning (XGBoost), 
    and deep learning (LSTM) approaches.
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
        self.results = {}

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
            'RMSE': mean_squared_error(truth, preds, squared=False),
            'R2': r2_score(truth, preds)
        }

    def _store_results(self, X_test, preds, model_key, cnt, sales=False):
        """
        Stores evaluation results for a given model and split.
        Also plots predictions against actuals.
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
        


    def run_models(self, n_splits=4, test_size=365, period=365,
                   base_model=1, sales_model=False, ml_model=False, dl_model=False):
        """
        Trains and evaluates different models using time-series cross-validation.
        Function modified to accomodate running different models based on params passed

        Parameters:
        - n_splits: number of splits for time-series cross-validation
        - test_size: size of the test set in each split
        - period: seasonality period for decomposition
        - base_model: 1 or 2 or 'all' to select base statistical models
        - sales_model, ml_model, dl_model: Boolean flags to activate respective models
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        for fold, (train_idx, test_idx) in enumerate(tscv.split(self.X)):
            X_train = self.X.iloc[train_idx]
            X_test = self.X.iloc[test_idx]

            # original base model with modification
            if base_model == 1:
                preds = self._base_model(X_train, X_test, period)
                self._store_results(X_test, preds, 'Base', fold)
                
            # model built from the base model (Seasonal decomposition using trend + seasonal components)
            elif base_model == 2:
                preds = self._base_model_two(X_train, X_test, period)
                self._store_results(X_test, preds, 'BaseModelTwo', fold)
                
            # New Sales Model targeting sales count prediction
            elif sales_model:
                preds = self._sales_model(X_train, X_test, period)
                self._store_results(X_test, preds, 'SalesModel', fold, sales=True)
                
            # this ML models predict recemption count as the label example of params base_model=0, ml_model=True
            elif ml_model:
                preds = self._ml_model(X_train, X_test)
                self._store_results(X_test, preds, 'XGBoost', fold)

             # this Deep learning models predict recemption count as the label example of params base_model=0, ml_model=False, dl_model=True
            elif dl_model:
                preds = self._dl_model(X_train, X_test)
                self._store_results(X_test, preds, 'LSTM', fold)
            else:
                preds = self._base_model(X_train, X_test, period)
                self._store_results(X_test, preds, 'Base', fold)

        self.summarize_results()
    
    #
    def _base_model_two(self, train, test, period):
        """
        Seasonal decomposition using trend + seasonal components.
        """
        res = sm.tsa.seasonal_decompose(train[self.target_col], period=period)
        trend = res.trend.bfill().ffill()
        seasonal = res.seasonal
        combined = (trend + seasonal).dropna()
        combined.index = combined.index.dayofyear
        daily_avg = combined.groupby(combined.index).mean()
        return pd.Series(index=test.index, data=[daily_avg.get(x.dayofyear, 0) for x in test.index])

        
    # new sales model to predict sales count
    def _sales_model(self, train, test, period):
        """
        Forecasts ticket sales based on trend + seasonal decomposition.
        """
        res = sm.tsa.seasonal_decompose(train['Sales Count'], period=period)
        trend = res.trend.bfill().ffill()
        seasonal = res.seasonal
        combined = (trend + seasonal).dropna()
        combined.index = combined.index.dayofyear
        daily_avg = combined.groupby(combined.index).mean()
        return pd.Series(index=test.index, data=[daily_avg.get(x.dayofyear, 0) for x in test.index])

    # Trying out a Machine learning model using XGBoost regressor
    def _ml_model(self, train, test):
        """
        Machine learning model using XGBoost regressor with engineered features.
        """
        train = self.create_features(train)
        test = self.create_features(test)
        features = ['lag_1', 'lag_7', 'lag_14', 'rolling_mean_3', 'rolling_mean_7',
                    'dayofweek', 'hour', 'month', 'is_weekend']

        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
        model.fit(train[features], train[self.target_col])
        preds = model.predict(test[features])
        return pd.Series(index=test.index, data=preds)
        
    # Trring out a Deep learning model using stacked LSTM layers:
    def _dl_model(self, train, test, sequence_length=14):
        """
        Deep learning model using stacked LSTM layers: 256 → 128 → 64 → 32.
        
        Args:
            train (pd.DataFrame): Training data.
            test (pd.DataFrame): Testing data.
            sequence_length (int): Number of past timesteps to use as input.
    
        Returns:
            pd.Series: Predicted values aligned with test index.
        """
        from tensorflow.keras.layers import Dropout
    
        # Merge train and test for continuous feature generation
        df = pd.concat([train, test])
        df = self.create_features(df)
    
        features = ['lag_1', 'lag_7', 'lag_14', 'rolling_mean_3', 'rolling_mean_7']
        df = df.dropna()
    
        # Prepare training sequences
        X, y = [], []
        for i in range(sequence_length, len(train)):
            X.append(df[features].iloc[i-sequence_length:i].values)
            y.append(df[self.target_col].iloc[i])
    
        X = np.array(X)  # Shape: (samples, sequence_length, features)
        y = np.array(y)
    
        # Define stacked LSTM model
        model = Sequential([
            LSTM(256, activation='relu', return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(128, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=20, verbose=0)
    
        # Prepare test sequences
        X_test = []
        test_start = len(train)
        for i in range(test_start, len(df)):
            if i - sequence_length < 0:
                continue
            X_test.append(df[features].iloc[i-sequence_length:i].values)
    
        X_test = np.array(X_test)
    
        # Predict
        preds = model.predict(X_test, verbose=0)
    
        # Align predictions with test index
        aligned_index = df.iloc[-len(preds):].index
        return pd.Series(index=aligned_index, data=preds.flatten())



    def create_features(self, df):
        """
        Create lag, rolling, and datetime features for time series forecasting.
        Assumes 'date' is either in the index or as a column.
        """
        df = df.copy()
    
        # If 'date' column is missing, create it from index
        if 'date' not in df.columns:
            df['date'] = df.index
    
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df.set_index('date', inplace=True)
    
        # Lag features
        df['lag_1'] = df[self.target_col].shift(1)
        df['lag_7'] = df[self.target_col].shift(7)
        df['lag_14'] = df[self.target_col].shift(14)
    
        # Rolling averages
        df['rolling_mean_3'] = df[self.target_col].shift(1).rolling(window=3).mean()
        df['rolling_mean_7'] = df[self.target_col].shift(1).rolling(window=7).mean()
    
        # Datetime-based features
        df['dayofweek'] = df.index.dayofweek
        df['hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
        df['month'] = df.index.month
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
        df.dropna(inplace=True)
        return df


    




    def plot(self, preds, label):
        """
        Visualizes predictions against observed data.
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
        Prints a summary of the performance metrics for each model and fold.
        """
        print(f"\n📊 Summary for '{self.target_col}' Forecasting:\n")
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


    # supplimentary run a single model based on the model name params
    def run_single_model(self, model_name, n_splits, test_size, period):
        """
        Helper function to run a specific model by name.
    
        Valid options: Base, BaseModelTwo, Sales, ML, DL
        """
        if model_name == 'ML':
            from xgboost import XGBRegressor
            from sklearn.model_selection import TimeSeriesSplit
    
            # Assume self.df is your dataset and has already been preprocessed
            features, target = self.prepare_features(self.df)  # ← define this method or replace appropriately
    
            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
            for train_index, test_index in tscv.split(features):
                X_train, X_test = features.iloc[train_index], features.iloc[test_index]
                y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    
            # Train the model
            model = XGBRegressor(n_estimators=100, learning_rate=0.1)
            model.fit(X_train, y_train)
    
            # Save trained model to the object
            self.model = model
    
            # Evaluate Mmdel
            preds = model.predict(X_test)
            print("ML Model trained and stored in self.model")
    
        else:
            # Fall back to your flag-based system
            model_flags = {
                'base_model': False,
                'sales_model': False,
                'ml_model': model_name == 'ML',
                'dl_model': model_name == 'DL',
            }
    
            self.run_models(**model_flags, test_size=test_size, period=period, n_splits=n_splits)


