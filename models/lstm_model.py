import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from datetime import timedelta

def create_lstm_sequences(data_scaled, sequence_length):
    x, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        x.append(data_scaled[i-sequence_length:i, 0])
        y.append(data_scaled[i, 0])
    return np.array(x), np.array(y)

def train_lstm_model(train_df, scaler, sequence_length, epochs=3, batch_size=32):
    # Scaler should already be fitted on train_df['Close'] in data_processing
    scaled_train_close = scaler.transform(train_df[['Close']])

    x_train, y_train = create_lstm_sequences(scaled_train_close, sequence_length)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1), activation='relu'),
        Dropout(0.2),
        LSTM(units=60, return_sequences=True, activation='relu'),
        Dropout(0.3),
        LSTM(units=80, return_sequences=True, activation='relu'),
        Dropout(0.4),
        LSTM(units=120, activation='relu'),
        Dropout(0.5),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model, history

def evaluate_lstm_model(model, test_df, train_df_for_sequence, scaler, sequence_length):
    # Prepare test data
    last_sequence_from_train_scaled = scaler.transform(train_df_for_sequence[['Close']])[-sequence_length:]
    test_close_scaled = scaler.transform(test_df[['Close']])
    
    combined_input_scaled = np.concatenate((last_sequence_from_train_scaled, test_close_scaled), axis=0)
    
    x_test, y_test_actual_scaled = create_lstm_sequences(combined_input_scaled, sequence_length)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    y_pred_test_scaled = model.predict(x_test)
    
    y_pred_test_rescaled = scaler.inverse_transform(y_pred_test_scaled)
    y_test_actual_rescaled = scaler.inverse_transform(y_test_actual_scaled.reshape(-1,1))
    
    return y_test_actual_rescaled, y_pred_test_rescaled, x_test # Return x_test if needed for other things

def predict_lstm_future(model, full_df, scaler, sequence_length, n_future_days):
    # Use the last 'sequence_length' days from the entire dataset (scaled)
    last_sequence_scaled = scaler.transform(full_df[['Close']])[-sequence_length:]
    current_sequence_for_model = last_sequence_scaled.reshape(1, sequence_length, 1)

    future_predictions_scaled = []
    for _ in range(n_future_days):
        next_pred_scaled = model.predict(current_sequence_for_model)
        future_predictions_scaled.append(next_pred_scaled[0, 0])
        # Update sequence: append new prediction, drop oldest
        new_sequence_part = np.append(current_sequence_for_model[0, 1:, 0], next_pred_scaled[0,0])
        current_sequence_for_model = new_sequence_part.reshape(1, sequence_length, 1)

    future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    
    last_date = full_df['Date'].iloc[-1]
    future_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, n_future_days + 1)])
    
    return future_predictions_rescaled, future_dates