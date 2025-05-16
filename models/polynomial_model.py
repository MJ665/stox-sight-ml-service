# models/polynomial_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline # Pipeline simplifies an already simple process
from datetime import timedelta
from typing import Tuple # For type hinting

def train_polynomial_model(
    train_df: pd.DataFrame, 
    degree: int = 3
) -> Tuple[make_pipeline, MinMaxScaler, pd.DataFrame, np.ndarray]: # model, y_scaler, X_train_time, y_train_scaled
    
    train_df_copy = train_df.copy() # Work on a copy

    # Create a time feature (days since the start OF THE TRAINING SET for this model)
    # This 'Time' feature is local to the training phase of this model.
    # For prediction/evaluation, 'Time' must be calculated relative to the *original dataset's start*.
    original_train_min_date = train_df_copy['Date'].min()
    train_df_copy['Time'] = (train_df_copy['Date'] - original_train_min_date).dt.days
    
    X_train_time = train_df_copy[['Time']] # Features (Time)
    y_train_actual = train_df_copy['Close'].values.reshape(-1, 1) # Target (Close Price)

    # Scale the target variable (Close Price)
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_actual)

    # Create and train the polynomial regression model pipeline
    # The pipeline will first create polynomial features from 'Time', then fit LinearRegression
    poly_model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
    poly_model.fit(X_train_time, y_train_scaled) # Fit on 'Time' and scaled 'Close'
    
    return poly_model, y_scaler, X_train_time, y_train_scaled

def evaluate_polynomial_model(
    model: make_pipeline, 
    test_df: pd.DataFrame, 
    y_scaler: MinMaxScaler, 
    original_min_date_for_time_feature: pd.Timestamp, # Min date of the *entire original* dataset
    degree: int # Degree is passed for context, though not directly used if polyfeatures is in pipeline
) -> Tuple[np.ndarray, np.ndarray]:

    test_df_copy = test_df.copy()

    # Create 'Time' feature for test set, relative to the *original dataset's start date*
    test_df_copy['Time'] = (test_df_copy['Date'] - original_min_date_for_time_feature).dt.days
    
    X_test_time = test_df_copy[['Time']]
    y_test_actual = test_df_copy['Close'].values.reshape(-1,1) # Original scale for comparison

    # Predict using the pipeline (applies PolynomialFeatures then LinearRegression)
    y_pred_test_scaled = model.predict(X_test_time)
    
    # Inverse transform predictions to original price scale
    y_pred_test_rescaled = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1,1))
    
    return y_test_actual, y_pred_test_rescaled

def predict_polynomial_future(
    model: make_pipeline, 
    full_df_for_last_date: pd.DataFrame,
    original_min_date_for_time_feature: pd.Timestamp,
    y_scaler: MinMaxScaler, 
    degree: int,
    n_future_days: int
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    
    last_known_date = full_df_for_last_date['Date'].iloc[-1]

    future_time_values: list[int] = []
    future_dates_list: list[pd.Timestamp] = [] # Changed from pd.DatetimeIndex to list for appending

    for i in range(1, n_future_days + 1):
        future_d = last_known_date + timedelta(days=i)
        future_dates_list.append(future_d)
        
        # Calculate the difference, which is a Timedelta object
        time_difference = future_d - original_min_date_for_time_feature
        # Access the .days attribute directly from the Timedelta object
        time_val = time_difference.days # <--- CORRECTED LINE
        future_time_values.append(time_val)
        
    X_future_time = pd.DataFrame(future_time_values, columns=['Time'])

    future_preds_scaled = model.predict(X_future_time)
    future_preds_rescaled = y_scaler.inverse_transform(future_preds_scaled.reshape(-1,1))
    
    return future_preds_rescaled, pd.to_datetime(future_dates_list) # Convert list of Timestamps to DatetimeIndex at the end


























# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import make_pipeline
# from datetime import timedelta

# def train_polynomial_model(train_df, degree=3):
#     train_df = train_df.copy()
#     # Create a time feature (e.g., days since the start)
#     train_df['Time'] = (train_df['Date'] - train_df['Date'].min()).dt.days
    
#     X_train = train_df[['Time']]
#     y_train = train_df['Close'].values.reshape(-1, 1)

#     # Scale target variable for stability if needed, or scale input 'Time'
#     # Here, we'll scale y for consistency with LSTM approach and inverse transform predictions
#     y_scaler = MinMaxScaler()
#     y_train_scaled = y_scaler.fit_transform(y_train)

#     # Create polynomial features and fit the model
#     # The pipeline handles feature transformation and regression
#     poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
#     poly_model.fit(X_train, y_train_scaled)
    
#     return poly_model, y_scaler, X_train, y_train_scaled # Return X_train for plotting if needed


# # def evaluate_polynomial_model(model, test_df, y_scaler, degree=3):
# # models/polynomial_model.py
# def evaluate_polynomial_model(model, test_df, y_scaler, original_min_date_for_time_feature, degree=3):
#     # ...

#     test_df = test_df.copy()
#     test_df['Time'] = (test_df['Date'] - test_df['Date'].min()).dt.days # Relative to test start for consistency with how it might be used
#                                                                       # OR better: use Time from start of entire dataset
#                                                                       # For now, using test_df's min date for Time feature in test
#                                                                       # This assumes `Time` is number of days from *its own dataset start*
#                                                                       # This needs careful handling if train/test 'Time' need to be aligned

#     # A more robust way for 'Time' feature for test_df:
#     # It should be relative to the *original* training data's start date.
#     # This requires passing the original min_date or the transformed time feature.
#     # For simplicity in this example, let's assume Time can be calculated for test_df like this:
#     # (This is a common point of error if not handled carefully in time series with poly reg)

#     # To make it more robust, let's assume the Time feature for testing should continue from training
#     # This requires access to the last time step of training or min_date of train.
#     # Let's assume test_df['Date'].min() is the first day AFTER train_df['Date'].max() for 'Time' continuity
#     # If `train_df` was passed, we could get train_df['Date'].min()
#     # For now, this simple relative time for test is used.
    
#     X_test = test_df[['Time']]
#     y_test_actual = test_df['Close'].values.reshape(-1,1) # Original scale

#     y_pred_test_scaled = model.predict(X_test)
#     y_pred_test_rescaled = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1,1))
    
#     return y_test_actual, y_pred_test_rescaled


# # def predict_polynomial_future(model, full_df, y_scaler, degree, n_future_days):

# def predict_polynomial_future(model, full_df_for_last_date, original_min_date_for_time_feature, y_scaler, degree, n_future_days): # degree is positional here
#     # ...
#     # Create future 'Time' features
#     last_date_in_full_df = full_df['Date'].iloc[-1]
#     original_min_date = full_df['Date'].min() # Min date from the entire dataset for 'Time' reference

#     future_time_features = []
#     future_dates = []
#     for i in range(1, n_future_days + 1):
#         future_d = last_date_in_full_df + timedelta(days=i)
#         future_dates.append(future_d)
#         time_val = (future_d - original_min_date).days
#         future_time_features.append([time_val])
        
#     X_future = pd.DataFrame(future_time_features, columns=['Time'])

#     future_preds_scaled = model.predict(X_future)
#     future_preds_rescaled = y_scaler.inverse_transform(future_preds_scaled.reshape(-1,1))
    
#     return future_preds_rescaled, pd.to_datetime(future_dates)