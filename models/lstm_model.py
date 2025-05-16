# # models/lstm_model.py
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
# from tensorflow.keras.callbacks import History
# from datetime import timedelta
# from typing import Tuple, List # For type hinting

# def create_lstm_sequences(data_scaled: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
#     x, y = [], []
#     for i in range(sequence_length, len(data_scaled)):
#         x.append(data_scaled[i-sequence_length:i, 0]) # Assuming data_scaled is [samples, 1 feature]
#         y.append(data_scaled[i, 0])
#     return np.array(x), np.array(y)

# def train_lstm_model(
#     train_df: pd.DataFrame, 
#     scaler: MinMaxScaler, 
#     sequence_length: int, 
#     epochs: int = 3, 
#     batch_size: int = 32
# ) -> Tuple[Sequential, History]:
    
#     scaled_train_close = scaler.transform(train_df[['Close']]) # Ensure 2D for transform

#     x_train, y_train = create_lstm_sequences(scaled_train_close, sequence_length)
    
#     # Reshape x_train for LSTM: [samples, time_steps, features]
#     x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#     model = Sequential()
#     # Using Input layer for explicit input shape definition (good practice)
#     model.add(Input(shape=(x_train.shape[1], 1))) 
#     model.add(LSTM(units=50, activation='relu', return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=60, activation='relu', return_sequences=True))
#     model.add(Dropout(0.3))
#     model.add(LSTM(units=80, activation='relu', return_sequences=True))
#     model.add(Dropout(0.4))
#     model.add(LSTM(units=120, activation='relu')) # Last LSTM layer, return_sequences=False (default)
#     model.add(Dropout(0.5))
#     model.add(Dense(units=1))
    
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     print(f"Training LSTM with x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
#     history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False) # shuffle=False for time series
#     return model, history

# def evaluate_lstm_model(
#     model: Sequential, 
#     test_df: pd.DataFrame, 
#     train_df_for_sequence: pd.DataFrame, # Last part of training data
#     scaler: MinMaxScaler, 
#     sequence_length: int
# )
# # -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
#     # Get last 'sequence_length' days from SCALED training data
#     # Ensure original train_df_for_sequence is used for transform to get the right scaling context
#     last_sequence_from_train_scaled = scaler.transform(train_df_for_sequence[['Close']].iloc[-sequence_length:])
    
#     # Scale the test data
#     test_close_scaled = scaler.transform(test_df[['Close']])
    
#     # Concatenate for creating test sequences
#     combined_input_scaled = np.concatenate((last_sequence_from_train_scaled, test_close_scaled), axis=0)
    
#     x_test, y_test_actual_scaled = create_lstm_sequences(combined_input_scaled, sequence_length)
#     x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#     if x_test.shape[0] == 0: # Handle cases with insufficient test data for sequences
#         print("Warning: Not enough data to create any test sequences for LSTM evaluation.")
#         return np.array([]), np.array([]), np.array([])

#     y_pred_test_scaled = model.predict(x_test)
    
#     y_pred_test_rescaled = scaler.inverse_transform(y_pred_test_scaled)
#     y_test_actual_rescaled = scaler.inverse_transform(y_test_actual_scaled.reshape(-1,1))
    
#     return y_test_actual_rescaled, y_pred_test_rescaled, x_test

# def predict_lstm_future(
#     model: Sequential, 
#     full_df: pd.DataFrame, # Entire available dataframe
#     scaler: MinMaxScaler, 
#     sequence_length: int, 
#     n_future_days: int
# ) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    
#     # Use the last 'sequence_length' days from the entire dataset (scaled)
#     # Make sure to use the 'Close' column and scale it
#     last_actual_sequence_scaled = scaler.transform(full_df[['Close']].iloc[-sequence_length:])
    
#     current_sequence_for_model = last_actual_sequence_scaled.reshape(1, sequence_length, 1)

#     future_predictions_scaled_list: List[float] = [] # Use a list for appending

#     for _ in range(n_future_days):
#         next_pred_scaled_array = model.predict(current_sequence_for_model) # Output shape (1, 1)
#         next_pred_scalar = next_pred_scaled_array[0, 0]
#         future_predictions_scaled_list.append(next_pred_scalar)
        
#         # Update sequence: append new prediction (scalar), drop oldest value
#         # current_sequence_for_model is (1, sequence_length, 1)
#         # new_day_data is (1,1,1)
#         new_day_data = np.array([[next_pred_scalar]]).reshape(1,1,1) # Reshape scalar to (1,1,1)
        
#         # Take all but the first time step, and append the new prediction
#         current_sequence_for_model = np.concatenate((current_sequence_for_model[:, 1:, :], new_day_data), axis=1)

#     future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions_scaled_list).reshape(-1, 1))
    
#     last_date = full_df['Date'].iloc[-1]
#     future_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, n_future_days + 1)])
    
#     return future_predictions_rescaled, future_dates




























# models/lstm_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input # Import Input
from tensorflow.keras.optimizers import Adam
from datetime import timedelta
from typing import Tuple # For type hinting


def create_lstm_sequences(data_scaled, sequence_length): # Renamed to avoid conflict if you have another create_sequences
    x, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        x.append(data_scaled[i-sequence_length:i, 0])
        y.append(data_scaled[i, 0])
    return np.array(x), np.array(y)

def build_lstm_model_for_search(input_shape, units=64, dropout_rate=0.2, learning_rate=0.001, activation='tanh'): # Changed default activation
    """
    Builds an LSTM model, compatible with KerasRegressor for hyperparameter search.
    input_shape should be (sequence_length, num_features=1)
    """
    model = Sequential()
    
    # Using Input layer explicitly as recommended by Keras warnings
    model.add(Input(shape=input_shape)) 

    # Layer 1
    # If this is the only LSTM layer OR if there's another LSTM after it, return_sequences=True
    # If it's the LAST LSTM before Dense, return_sequences=False
    model.add(LSTM(units=units, return_sequences=True, activation=activation)) # If more LSTMs follow
    model.add(Dropout(dropout_rate))

    # Layer 2 (Example)
    model.add(LSTM(units=int(units * 0.75), return_sequences=True, activation=activation)) # If more LSTMs follow
    model.add(Dropout(dropout_rate))

    # Layer 3 (Last LSTM layer before Dense)
    model.add(LSTM(units=int(units * 0.5), return_sequences=False, activation=activation)) # Last LSTM, so False
    model.add(Dropout(dropout_rate))
    
    # Output Layer
    model.add(Dense(units=1)) # Activation is linear by default for regression
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error') # or 'huber_loss'
    return model

def train_lstm_model_with_params(
    train_df: pd.DataFrame, 
    scaler: MinMaxScaler, 
    sequence_length: int, 
    model_params: dict, # Best params from RandomizedSearch
    epochs: int = 50, 
    batch_size: int = 32
) -> Tuple[Sequential, any]:

    scaled_train_close = scaler.transform(train_df[['Close']])
    x_train, y_train = create_lstm_sequences(scaled_train_close, sequence_length)
    x_train_shaped = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Ensure all parameters expected by build_lstm_model_for_search are provided
    # Default them if not present in model_params from RandomizedSearch
    current_units = model_params.get('units', 64) # Default units
    current_dropout = model_params.get('dropout_rate', 0.2)
    current_lr = model_params.get('learning_rate', 0.001)
    current_activation = model_params.get('activation', 'tanh')


    lstm_model = build_lstm_model_for_search(
        input_shape=(x_train_shaped.shape[1], 1), # sequence_length, num_features
        units=current_units,
        dropout_rate=current_dropout,
        learning_rate=current_lr,
        activation=current_activation
    )
    
    print(f"Training final LSTM model with params: units={current_units}, dropout={current_dropout}, lr={current_lr}, activation={current_activation}, epochs={epochs}")
    history = lstm_model.fit(x_train_shaped, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return lstm_model, history

# evaluate_lstm_model and predict_lstm_future remain the same
def evaluate_lstm_model(model, test_df, train_df_for_sequence, scaler, sequence_length):
    last_sequence_from_train_scaled = scaler.transform(train_df_for_sequence[['Close']])[-sequence_length:]
    test_close_scaled = scaler.transform(test_df[['Close']])
    combined_input_scaled = np.concatenate((last_sequence_from_train_scaled, test_close_scaled), axis=0)
    x_test, y_test_actual_scaled = create_lstm_sequences(combined_input_scaled, sequence_length)
    x_test_reshaped = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_pred_test_scaled = model.predict(x_test_reshaped)
    y_pred_test_rescaled = scaler.inverse_transform(y_pred_test_scaled)
    y_test_actual_rescaled = scaler.inverse_transform(y_test_actual_scaled.reshape(-1,1))
    return y_test_actual_rescaled, y_pred_test_rescaled, x_test_reshaped

def predict_lstm_future(model, full_df, scaler, sequence_length, n_future_days):
    last_sequence_scaled = scaler.transform(full_df[['Close']])[-sequence_length:]
    current_sequence_for_model = last_sequence_scaled.reshape(1, sequence_length, 1)
    future_predictions_scaled = []
    for _ in range(n_future_days):
        next_pred_scaled = model.predict(current_sequence_for_model, verbose=0)
        future_predictions_scaled.append(next_pred_scaled[0, 0])
        new_sequence_part = np.append(current_sequence_for_model[0, 1:, 0], next_pred_scaled[0,0])
        current_sequence_for_model = new_sequence_part.reshape(1, sequence_length, 1)
    future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    last_date = full_df['Date'].iloc[-1]
    future_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, n_future_days + 1)])
    return future_predictions_rescaled, future_dates





















































# # single layer LSTM





# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dropout, Dense
# from tensorflow.keras.optimizers import Adam # Import Adam


# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dropout, Dense
# from datetime import timedelta

# def create_lstm_sequences(data_scaled, sequence_length):
#     x, y = [], []
#     for i in range(sequence_length, len(data_scaled)):
#         x.append(data_scaled[i-sequence_length:i, 0])
#         y.append(data_scaled[i, 0])
#     return np.array(x), np.array(y)





# # models/lstm_model.py
# # ...
# def build_lstm_model_for_search(input_shape, units=50, dropout_rate=0.2, learning_rate=0.001, activation='relu'):
#     model = Sequential([
#         LSTM(units=units, return_sequences=True, input_shape=input_shape, activation=activation),
#         Dropout(dropout_rate),
#         LSTM(units=int(units*1.2), return_sequences=True, activation=activation), # Output: 3D
#         Dropout(dropout_rate + 0.1 if dropout_rate + 0.1 <= 0.5 else 0.5),
#         LSTM(units=int(units*1.5), return_sequences=True, activation=activation), # Output: 3D
#         Dropout(dropout_rate + 0.2 if dropout_rate + 0.2 <= 0.5 else 0.5),
#         # This LSTM below is the problem if the one above it has return_sequences=True
#         LSTM(units=int(units*2) if int(units*2) <=240 else 240 , activation=activation), # Implicitly return_sequences=False. Output: 2D
#         Dropout(dropout_rate + 0.2 if dropout_rate + 0.2 <= 0.5 else 0.5),
#         Dense(units=1) # Expects 2D input (batch_size, features)
#     ])
#     optimizer = Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer, loss='mean_squared_error')
#     return model



# def train_lstm_model_with_params(
#     train_df: pd.DataFrame, 
#     scaler: MinMaxScaler, 
#     sequence_length: int, 
#     model_params: dict, # Best params from RandomizedSearch
#     epochs: int = 50, 
#     batch_size: int = 32
# ) -> tuple[Sequential, any]:

#     scaled_train_close = scaler.transform(train_df[['Close']])
#     x_train, y_train = create_lstm_sequences(scaled_train_close, sequence_length)
#     x_train_shaped = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#     lstm_model = build_lstm_model_for_search(
#         input_shape=(x_train_shaped.shape[1], 1),
#         units=model_params.get('units', 50),
#         dropout_rate=model_params.get('dropout_rate', 0.2),
#         learning_rate=model_params.get('learning_rate', 0.001),
#         activation=model_params.get('activation','relu') # if you add activation to LSTM search space
#     )
    
#     print(f"Training final LSTM model with params: {model_params}, epochs: {epochs}")
#     history = lstm_model.fit(x_train_shaped, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
#     return lstm_model, history

# # ... (evaluate_lstm_model, predict_lstm_future remain mostly the same, ensure they take the trained model)





# # def train_lstm_model(train_df, scaler, sequence_length, epochs=3, batch_size=32):
# #     # Scaler should already be fitted on train_df['Close'] in data_processing
# #     scaled_train_close = scaler.transform(train_df[['Close']])

# #     x_train, y_train = create_lstm_sequences(scaled_train_close, sequence_length)
# #     x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# #     model = Sequential([
# #         LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1), activation='relu'),
# #         Dropout(0.2),
# #         LSTM(units=60, return_sequences=True, activation='relu'),
# #         Dropout(0.3),
# #         LSTM(units=80, return_sequences=True, activation='relu'),
# #         Dropout(0.4),
# #         LSTM(units=120, activation='relu'),
# #         Dropout(0.5),
# #         Dense(units=1)
# #     ])
# #     model.compile(optimizer='adam', loss='mean_squared_error')
# #     history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
# #     return model, history

# def evaluate_lstm_model(model, test_df, train_df_for_sequence, scaler, sequence_length):
#     # Prepare test data
#     last_sequence_from_train_scaled = scaler.transform(train_df_for_sequence[['Close']])[-sequence_length:]
#     test_close_scaled = scaler.transform(test_df[['Close']])
    
#     combined_input_scaled = np.concatenate((last_sequence_from_train_scaled, test_close_scaled), axis=0)
    
#     x_test, y_test_actual_scaled = create_lstm_sequences(combined_input_scaled, sequence_length)
#     x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#     y_pred_test_scaled = model.predict(x_test)
    
#     y_pred_test_rescaled = scaler.inverse_transform(y_pred_test_scaled)
#     y_test_actual_rescaled = scaler.inverse_transform(y_test_actual_scaled.reshape(-1,1))
    
#     return y_test_actual_rescaled, y_pred_test_rescaled, x_test # Return x_test if needed for other things

# def predict_lstm_future(model, full_df, scaler, sequence_length, n_future_days):
#     # Use the last 'sequence_length' days from the entire dataset (scaled)
#     last_sequence_scaled = scaler.transform(full_df[['Close']])[-sequence_length:]
#     current_sequence_for_model = last_sequence_scaled.reshape(1, sequence_length, 1)

#     future_predictions_scaled = []
#     for _ in range(n_future_days):
#         next_pred_scaled = model.predict(current_sequence_for_model)
#         future_predictions_scaled.append(next_pred_scaled[0, 0])
#         # Update sequence: append new prediction, drop oldest
#         new_sequence_part = np.append(current_sequence_for_model[0, 1:, 0], next_pred_scaled[0,0])
#         current_sequence_for_model = new_sequence_part.reshape(1, sequence_length, 1)

#     future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    
#     last_date = full_df['Date'].iloc[-1]
#     future_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, n_future_days + 1)])
    
#     return future_predictions_rescaled, future_dates