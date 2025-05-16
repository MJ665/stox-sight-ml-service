# models/gru_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam # For custom learning rate
from datetime import timedelta
from typing import Tuple

# Helper from lstm_model.py (can be moved to a common utils if used by multiple models)
def create_sequences(data_scaled, sequence_length):
    x, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        x.append(data_scaled[i-sequence_length:i, 0])
        y.append(data_scaled[i, 0])
    return np.array(x), np.array(y)



# models/gru_model.py
# ... (imports and create_sequences)
from tensorflow.keras.layers import GRU, Dropout, Dense, Input # Add Input

def build_gru_model_for_search(input_shape, units=64, dropout_rate=0.2, learning_rate=0.001, activation='tanh'): # No 'recurrent_dropout_rate'
    """
    Builds a GRU model. This function is designed to be compatible with
    KerasRegressor for RandomizedSearchCV.
    'dropout_rate' will be used for GRU's recurrent_dropout and for Dropout layers.
    """
    model = Sequential()
    model.add(Input(shape=input_shape)) # Explicit Input layer

    model.add(GRU(
        units=units, 
        return_sequences=True, 
        activation=activation, 
        recurrent_dropout=dropout_rate # Using dropout_rate here
    ))
    model.add(Dropout(dropout_rate)) # Standard dropout layer

    model.add(GRU(
        units=int(units * 0.75), 
        return_sequences=False, # Last GRU before Dense
        activation=activation, 
        recurrent_dropout=dropout_rate # Using dropout_rate here
    ))
    model.add(Dropout(dropout_rate)) # Standard dropout layer
    
    model.add(Dense(units=1))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def train_gru_model_with_params(
    train_df: pd.DataFrame, 
    scaler: MinMaxScaler, 
    sequence_length: int, 
    model_params: dict, # Best params from RandomizedSearch
    epochs: int = 50, # Default epochs for final training
    batch_size: int = 32
) -> Tuple[Sequential, any]: # Model and history

    scaled_train_close = scaler.transform(train_df[['Close']])
    x_train, y_train = create_sequences(scaled_train_close, sequence_length)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build model with the best parameters found
    gru_model = build_gru_model_for_search(
        input_shape=(x_train.shape[1], 1),
        units=model_params.get('units', 128),
        dropout_rate=model_params.get('dropout_rate', 0.2),
        learning_rate=model_params.get('learning_rate', 0.001),
        activation=model_params.get('activation', 'relu')
    )
    
    print(f"Training final GRU model with params: {model_params}, epochs: {epochs}")
    history = gru_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return gru_model, history


def evaluate_gru_model(
    model: Sequential, 
    test_df: pd.DataFrame, 
    train_df_for_sequence: pd.DataFrame, # Last part of training data for initial sequence
    scaler: MinMaxScaler, 
    sequence_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: # y_actual, y_pred, x_test

    last_sequence_from_train_scaled = scaler.transform(train_df_for_sequence[['Close']])[-sequence_length:]
    test_close_scaled = scaler.transform(test_df[['Close']])
    
    combined_input_scaled = np.concatenate((last_sequence_from_train_scaled, test_close_scaled), axis=0)
    
    x_test, y_test_actual_scaled = create_sequences(combined_input_scaled, sequence_length)
    x_test_reshaped = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    y_pred_test_scaled = model.predict(x_test_reshaped)
    
    y_pred_test_rescaled = scaler.inverse_transform(y_pred_test_scaled)
    y_test_actual_rescaled = scaler.inverse_transform(y_test_actual_scaled.reshape(-1,1))
    
    return y_test_actual_rescaled, y_pred_test_rescaled, x_test_reshaped


def predict_gru_future(
    model: Sequential, 
    full_df: pd.DataFrame, 
    scaler: MinMaxScaler, 
    sequence_length: int, 
    n_future_days: int
) -> Tuple[np.ndarray, pd.DatetimeIndex]:

    last_sequence_scaled = scaler.transform(full_df[['Close']])[-sequence_length:]
    current_sequence_for_model = last_sequence_scaled.reshape(1, sequence_length, 1)

    future_predictions_scaled = []
    for _ in range(n_future_days):
        next_pred_scaled = model.predict(current_sequence_for_model, verbose=0) # verbose=0 for less log spam
        future_predictions_scaled.append(next_pred_scaled[0, 0])
        
        new_sequence_part = np.append(current_sequence_for_model[0, 1:, 0], next_pred_scaled[0,0])
        current_sequence_for_model = new_sequence_part.reshape(1, sequence_length, 1)

    future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    
    last_date = full_df['Date'].iloc[-1]
    future_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, n_future_days + 1)])
    
    return future_predictions_rescaled, future_dates