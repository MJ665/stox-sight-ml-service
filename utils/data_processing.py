import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df, sequence_length, train_split_ratio=0.8):
    # df already has 'Date' as datetime and sorted, 'Close' is numeric
    
    # Splitting data
    train_size = int(len(df) * train_split_ratio)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy().reset_index(drop=True)

    # Scaler for LSTM (and potentially for Poly if we scale target there)
    # Fit scaler ONLY on training data's 'Close' price
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[['Close']]) # Fit on the 'Close' column of training data
    
    return train_df, test_df, scaler