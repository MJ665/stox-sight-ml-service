# models/transformer_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Add # Embedding, Reshape, Flatten (not used in this revision)
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from datetime import timedelta
from typing import Tuple

def create_sequences(data_scaled, sequence_length): # Assuming this is correctly defined elsewhere or here
    x, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        x.append(data_scaled[i-sequence_length:i, 0])
        y.append(data_scaled[i, 0])
    return np.array(x), np.array(y)

def transformer_encoder_block(inputs, num_heads, ff_dim, dropout_rate=0.1):
    # Inputs to this block should now have shape (batch_size, sequence_length, model_dim)
    model_dim = inputs.shape[-1]
    
    # Ensure key_dim is not zero
    if model_dim < num_heads:
        # This can happen if model_dim is small and num_heads is chosen from a list like [2,4,8]
        # Option: adjust num_heads or skip this combination in hyperparameter search
        # For now, let's force num_heads to be 1 if model_dim is too small, or raise error.
        # Or, ensure model_dim is always >= num_heads (e.g. by choosing ff_dim_projection wisely)
        # A simpler fix for now is to ensure ff_dim_projection in build_transformer_model_for_search
        # is large enough.
        # Here, let's ensure key_dim is at least 1.
        key_dim_calc = model_dim // num_heads
        if key_dim_calc == 0:
            # This would still cause issues if num_heads > model_dim.
            # The main fix is the projection layer before the encoder.
            # This check is a safeguard if the projection_dim itself is very small.
            print(f"Warning: model_dim ({model_dim}) < num_heads ({num_heads}). MultiHeadAttention might behave unexpectedly or error.")
            # A possible adjustment, though not ideal:
            # num_heads = model_dim # or 1, but this changes the hyperparameter.
            # key_dim_calc = 1
            # For now, rely on the projection_dim to be sufficient.
            # If this error persists, it means the projection_dim itself (from param search) is too small for num_heads.
            pass # Let Keras handle it, it might error if key_dim becomes 0 after division

    attention_output = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=model_dim // num_heads # key_dim is per head; model_dim must be divisible by num_heads
    )(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    ffn_output = Dense(ff_dim, activation="relu")(out1)
    ffn_output = Dense(model_dim)(ffn_output) # Project back to model_dim
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return out2

def build_transformer_model_for_search(
    input_shape, # (sequence_length, num_features=1)
    projection_dim=64, # NEW: Dimension to project input features into
    num_transformer_blocks=2,
    num_heads=4,
    ff_dim=32,       # ff_dim for the transformer block's feed-forward network
    dense_units=32,  # dense_units for the final dense layer after pooling
    dropout_rate=0.1,
    learning_rate=0.001
):
    if projection_dim % num_heads != 0:
        # Adjust projection_dim to be divisible by num_heads or adjust num_heads
        # For simplicity in hyperparam search, let's ensure projection_dim is a multiple of num_heads
        # A quick fix: find nearest valid projection_dim, or adjust num_heads.
        # Here, we'll rely on careful param_space definition, but a check is good.
        # This can be a common source of error if projection_dim is small.
        print(f"Warning: projection_dim ({projection_dim}) is not divisible by num_heads ({num_heads}). Adjusting num_heads for this build.")
        # Find largest num_heads that divides projection_dim from common choices [1,2,4,8]
        possible_heads = [h for h in [8,4,2,1] if projection_dim % h == 0]
        if not possible_heads: # Should not happen if projection_dim is e.g. >=8
            num_heads = 1 
        else:
            num_heads = possible_heads[0]
        print(f"Adjusted num_heads to {num_heads} for projection_dim {projection_dim}")


    inputs = Input(shape=input_shape) # e.g., (60, 1)
    
    # Project the single input feature to a higher dimension (projection_dim)
    # This projection_dim will be the "model_dim" for the transformer blocks
    x = Dense(projection_dim, activation='relu')(inputs) # Output: (batch, seq_len, projection_dim)
    # No positional encoding added here for simplicity, but recommended for real applications

    for _ in range(num_transformer_blocks):
        # Pass the current 'x' which has model_dim = projection_dim
        x = transformer_encoder_block(x, num_heads, ff_dim, dropout_rate)

    x = GlobalAveragePooling1D(data_format="channels_last")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model

# train_transformer_model_with_params, evaluate_transformer_model, predict_transformer_future
# need to be updated to pass 'projection_dim' if it's tuned.

def train_transformer_model_with_params(
    train_df: pd.DataFrame,
    scaler: MinMaxScaler,
    sequence_length: int,
    model_params: dict, # Best params from RandomizedSearch
    epochs: int = 50,
    batch_size: int = 32
) -> Tuple[Model, any]:

    scaled_train_close = scaler.transform(train_df[['Close']])
    x_train, y_train = create_sequences(scaled_train_close, sequence_length)
    x_train_shaped = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    transformer_model = build_transformer_model_for_search(
        input_shape=(x_train_shaped.shape[1], 1),
        projection_dim=model_params.get('projection_dim', 64), # NEW
        num_transformer_blocks=model_params.get('num_transformer_blocks', 2),
        num_heads=model_params.get('num_heads', 4),
        ff_dim=model_params.get('ff_dim', 32),
        dense_units=model_params.get('dense_units', 32),
        dropout_rate=model_params.get('dropout_rate', 0.1),
        learning_rate=model_params.get('learning_rate', 0.001)
    )

    print(f"Training final Transformer model with params: {model_params}, epochs: {epochs}")
    history = transformer_model.fit(x_train_shaped, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return transformer_model, history

# evaluate and predict functions for transformer remain largely the same,
# as they just use the trained model.
def evaluate_transformer_model(
    model: Model, test_df: pd.DataFrame, train_df_for_sequence: pd.DataFrame,
    scaler: MinMaxScaler, sequence_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ... (same as before)
    last_sequence_from_train_scaled = scaler.transform(train_df_for_sequence[['Close']])[-sequence_length:]
    test_close_scaled = scaler.transform(test_df[['Close']])
    combined_input_scaled = np.concatenate((last_sequence_from_train_scaled, test_close_scaled), axis=0)
    x_test, y_test_actual_scaled = create_sequences(combined_input_scaled, sequence_length)
    x_test_shaped = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_pred_test_scaled = model.predict(x_test_shaped)
    y_pred_test_rescaled = scaler.inverse_transform(y_pred_test_scaled)
    y_test_actual_rescaled = scaler.inverse_transform(y_test_actual_scaled.reshape(-1,1))
    return y_test_actual_rescaled, y_pred_test_rescaled, x_test_shaped




# models/transformer_model.py
# ... (imports and other functions) ...

def predict_transformer_future(
    model: Model,
    full_df: pd.DataFrame,
    scaler: MinMaxScaler,
    sequence_length: int,
    n_future_days: int
) -> Tuple[np.ndarray, pd.DatetimeIndex]:

    last_sequence_scaled = scaler.transform(full_df[['Close']])[-sequence_length:]
    # current_sequence_for_model should be (1, sequence_length, num_features)
    current_sequence_for_model = last_sequence_scaled.reshape(1, sequence_length, 1)

    future_predictions_scaled = []

    for _ in range(n_future_days):
        next_pred_scaled_array = model.predict(current_sequence_for_model, verbose=0) # Output shape (1, 1)
        next_pred_scaled = next_pred_scaled_array[0, 0] # Get the scalar prediction
        future_predictions_scaled.append(next_pred_scaled)

        # Prepare the new value to be appended. It needs to be (1, 1) for concatenation
        # with a slice that will be (sequence_length - 1, 1).
        new_value_to_append_2d = np.array([[next_pred_scaled]]) # Shape (1, 1)

        # Get all but the first timestep from the current sequence, maintaining 2D shape (seq_len-1, features)
        # current_sequence_for_model[0] has shape (sequence_length, 1)
        # current_sequence_for_model[0, 1:, :] also has shape (sequence_length - 1, 1)
        sequence_without_oldest = current_sequence_for_model[0, 1:, :] # Shape (sequence_length - 1, 1)

        # Concatenate along the timesteps axis (axis=0)
        updated_sequence_2d = np.concatenate((sequence_without_oldest, new_value_to_append_2d), axis=0) # Shape (sequence_length, 1)
        
        # Reshape back to (1, sequence_length, 1) for the next model prediction
        current_sequence_for_model = updated_sequence_2d.reshape(1, sequence_length, 1)

    future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    
    last_date = full_df['Date'].iloc[-1]
    future_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, n_future_days + 1)])
    
    return future_predictions_rescaled, future_dates





































































































































# # models/transformer_model.py

# # models/transformer_model.py
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from tensorflow.keras.layers import (
#     Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
#     GlobalAveragePooling1D, Add, Embedding, Reshape, Flatten
# )
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from datetime import timedelta
# from typing import Tuple

# # Helper from lstm_model.py (can be moved to a common utils if used by multiple models)
# def create_sequences(data_scaled, sequence_length):
#     x, y = [], []
#     for i in range(sequence_length, len(data_scaled)):
#         x.append(data_scaled[i-sequence_length:i, 0])
#         y.append(data_scaled[i, 0])
#     return np.array(x), np.array(y)

# # --- Transformer Components ---
# def transformer_encoder_block(inputs, num_heads, ff_dim, dropout_rate=0.1):
#     # Attention and Normalization
#     attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1] // num_heads)(inputs, inputs)
#     attention_output = Dropout(dropout_rate)(attention_output)
#     out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output) # Add & Norm

#     # Feed Forward Part
#     ffn_output = Dense(ff_dim, activation="relu")(out1)
#     ffn_output = Dense(inputs.shape[-1])(ffn_output) # Project back to input dimension
#     ffn_output = Dropout(dropout_rate)(ffn_output)
#     out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output) # Add & Norm
#     return out2

# def build_transformer_model_for_search(
#     input_shape, # (sequence_length, num_features=1)
#     num_transformer_blocks=2,
#     num_heads=4,
#     ff_dim=32,
#     dense_units=32,
#     dropout_rate=0.1,
#     learning_rate=0.001
# ):
#     inputs = Input(shape=input_shape)
#     x = inputs

#     # Positional Encoding (Simple version: learnable embedding)
#     # More complex sine/cosine positional encoding can be added if needed
#     # For now, let's assume the sequence order itself carries enough info for short sequences
#     # or that the model can learn positions through dense layers if needed.
#     # For a more robust solution, explicit positional encoding is better.
#     # Here, we'll just proceed, but this is an area for improvement.
#     # A simple approach if features are 1: treat timesteps as categories for embedding
#     # if input_shape[1] == 1: # if num_features is 1
#     #     positions = tf.range(start=0, limit=input_shape[0], delta=1)
#     #     position_embeddings = Embedding(input_dim=input_shape[0], output_dim=input_shape[1])(positions)
#     #     x = x + Reshape((input_shape[0], input_shape[1]))(position_embeddings) # requires input_shape[1] == output_dim of embedding

#     for _ in range(num_transformer_blocks):
#         x = transformer_encoder_block(x, num_heads, ff_dim, dropout_rate)

#     # After Transformer blocks, the output is still a sequence (batch, seq_len, features)
#     # We need to reduce it to a single vector per sequence for the final Dense layer
#     x = GlobalAveragePooling1D(data_format="channels_last")(x) # Or Flatten()
#     x = Dropout(dropout_rate)(x)
#     x = Dense(dense_units, activation="relu")(x)
#     x = Dropout(dropout_rate)(x)
#     outputs = Dense(1)(x) # Regression output

#     model = Model(inputs=inputs, outputs=outputs)
#     optimizer = Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer, loss="mean_squared_error") # or 'huber_loss'
#     return model

# def train_transformer_model_with_params(
#     train_df: pd.DataFrame,
#     scaler: MinMaxScaler,
#     sequence_length: int,
#     model_params: dict, # Best params from RandomizedSearch
#     epochs: int = 50,
#     batch_size: int = 32
# ) -> Tuple[Model, any]: # Keras Model and history

#     scaled_train_close = scaler.transform(train_df[['Close']])
#     x_train, y_train = create_sequences(scaled_train_close, sequence_length)
#     # Transformer expects input shape: (batch_size, sequence_length, num_features)
#     x_train_shaped = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # num_features = 1

#     transformer_model = build_transformer_model_for_search(
#         input_shape=(x_train_shaped.shape[1], 1),
#         num_transformer_blocks=model_params.get('num_transformer_blocks', 2),
#         num_heads=model_params.get('num_heads', 4),
#         ff_dim=model_params.get('ff_dim', 32),
#         dense_units=model_params.get('dense_units', 32),
#         dropout_rate=model_params.get('dropout_rate', 0.1),
#         learning_rate=model_params.get('learning_rate', 0.001)
#     )

#     print(f"Training final Transformer model with params: {model_params}, epochs: {epochs}")
#     history = transformer_model.fit(x_train_shaped, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
#     return transformer_model, history

# def evaluate_transformer_model(
#     model: Model,
#     test_df: pd.DataFrame,
#     train_df_for_sequence: pd.DataFrame,
#     scaler: MinMaxScaler,
#     sequence_length: int
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

#     last_sequence_from_train_scaled = scaler.transform(train_df_for_sequence[['Close']])[-sequence_length:]
#     test_close_scaled = scaler.transform(test_df[['Close']])
#     combined_input_scaled = np.concatenate((last_sequence_from_train_scaled, test_close_scaled), axis=0)

#     x_test, y_test_actual_scaled = create_sequences(combined_input_scaled, sequence_length)
#     x_test_shaped = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#     y_pred_test_scaled = model.predict(x_test_shaped)

#     y_pred_test_rescaled = scaler.inverse_transform(y_pred_test_scaled)
#     y_test_actual_rescaled = scaler.inverse_transform(y_test_actual_scaled.reshape(-1,1))

#     return y_test_actual_rescaled, y_pred_test_rescaled, x_test_shaped

# def predict_transformer_future(
#     model: Model,
#     full_df: pd.DataFrame,
#     scaler: MinMaxScaler,
#     sequence_length: int,
#     n_future_days: int
# ) -> Tuple[np.ndarray, pd.DatetimeIndex]:

#     # For Transformer, prediction is often done one step at a time if it's autoregressive,
#     # or if it predicts the whole sequence, then the input needs to be prepared for that.
#     # This implementation assumes a similar autoregressive prediction as LSTM/GRU for simplicity.
    
#     last_sequence_scaled = scaler.transform(full_df[['Close']])[-sequence_length:]
#     # The input for the Transformer model for prediction needs to be (1, sequence_length, num_features)
#     current_sequence_for_model = last_sequence_scaled.reshape(1, sequence_length, 1)

#     future_predictions_scaled = []
    
#     # This loop structure is more typical for RNNs. Transformers can sometimes predict
#     # all future steps at once if trained for that, or use a different generation strategy.
#     # For simplicity, we'll adapt the RNN-style iterative prediction.
#     for _ in range(n_future_days):
#         next_pred_scaled_array = model.predict(current_sequence_for_model, verbose=0) # Output shape (1, 1)
#         next_pred_scaled = next_pred_scaled_array[0, 0] # Get the scalar prediction
#         future_predictions_scaled.append(next_pred_scaled)
        
#         # Update sequence: append new prediction (as a 2D array for concatenate), drop oldest
#         new_val_scaled = np.array([[next_pred_scaled]]) # Shape (1,1)
        
#         # Remove the oldest timestep from the sequence part and append the new prediction
#         # current_sequence_for_model is (1, seq_len, 1)
#         # current_sequence_for_model[0, 1:, 0] gives (seq_len-1,)
#         # We need to make it (seq_len-1, 1) then append new_val_scaled (1,1) -> (seq_len,1)
#         # then reshape to (1, seq_len, 1)
        
#         updated_sequence_scaled = np.append(current_sequence_for_model[0, 1:, :], [[new_val_scaled]], axis=0)
#         current_sequence_for_model = updated_sequence_scaled.reshape(1, sequence_length, 1)

#     future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    
#     last_date = full_df['Date'].iloc[-1]
#     future_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, n_future_days + 1)])
    
#     return future_predictions_rescaled, future_dates