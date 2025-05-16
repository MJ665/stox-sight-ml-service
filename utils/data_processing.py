# utils/data_processing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np # Add numpy import

def preprocess_data(df, sequence_length, train_split_ratio=0.8):
    # ... (same as before)
    train_size = int(len(df) * train_split_ratio)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy().reset_index(drop=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[['Close']])
    return train_df, test_df, scaler

def create_sequences_for_tuner(data_scaled_1d_array, sequence_length):
    """
    Creates sequences for Keras models, where y is 1D.
    data_scaled_1d_array should be the scaled 'Close' price as a 1D numpy array.
    """
    x, y = [], []
    # Ensure data_scaled_1d_array is 1D
    if data_scaled_1d_array.ndim > 1:
        data_scaled_1d_array = data_scaled_1d_array.flatten()

    for i in range(sequence_length, len(data_scaled_1d_array)):
        x.append(data_scaled_1d_array[i-sequence_length:i]) # Input sequence
        y.append(data_scaled_1d_array[i])                  # Single next value
    return np.array(x), np.array(y)





















# # utils/data_processing.py
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# def preprocess_data(df, sequence_length, train_split_ratio=0.8):
#     # df already has 'Date' as datetime and sorted, 'Close' is numeric
    
#     # Splitting data
#     train_size = int(len(df) * train_split_ratio)
#     train_df = df.iloc[:train_size].copy()
#     test_df = df.iloc[train_size:].copy().reset_index(drop=True)

#     # Scaler for LSTM (and potentially for Poly if we scale target there)
#     # Fit scaler ONLY on training data's 'Close' price
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaler.fit(train_df[['Close']]) # Fit on the 'Close' column of training data
    
#     return train_df, test_df, scaler