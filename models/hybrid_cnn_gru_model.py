# models/hybrid_cnn_gru_model.py

# models/hybrid_cnn_gru_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv1D, MaxPooling1D, GRU, Bidirectional,
    Attention, Layer # For custom Attention if needed, or use tf.keras.layers.Attention
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from datetime import timedelta
from typing import Tuple

# Ensure create_sequences is available (either here or from a common util)
def create_sequences(data_scaled, sequence_length):
    x, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        x.append(data_scaled[i-sequence_length:i, 0])
        y.append(data_scaled[i, 0])
    return np.array(x), np.array(y)


# Note: The 'Attention' layer in tf.keras.layers.Attention is a specific type (Luong-style).
# The example might imply a more generic attention or self-attention.
# For simplicity, we'll use tf.keras.layers.Attention which performs self-attention if query=value=key.
# A custom attention layer might be needed for more specific behaviors.

def build_hybrid_cnn_gru_attention_for_search(
    input_shape, # (sequence_length, num_features=1)
    cnn_filters=64,
    cnn_kernel_size=3,
    gru_units=128,
    # attention_units, # Not directly a param for tf.keras.layers.Attention
    dropout_rate=0.2,
    learning_rate=0.001
):
    inputs = Input(shape=input_shape)

    # CNN for local pattern extraction
    x = Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, activation='relu', padding='same')(inputs)
    # No MaxPooling in the suggestion, but often useful. Let's stick to suggestion first.
    # x = MaxPooling1D(pool_size=2)(x) # Optional

    # Bidirectional GRU for temporal relationships
    # The output of Conv1D is (batch, new_seq_len_after_cnn_if_no_padding, cnn_filters)
    # GRU expects (batch, timesteps, features)
    x = Bidirectional(GRU(units=gru_units, return_sequences=True))(x) # return_sequences=True for Attention
    x = Dropout(dropout_rate)(x)

    # Attention mechanism
    # tf.keras.layers.Attention computes attention scores between query and value, using key for value.
    # For self-attention on the GRU output: query=x, value=x, key=x (implicitly if only one input)
    # The output shape of Attention is (batch_size, query_sequence_length, value_embedding_dimension)
    # which will be (batch_size, sequence_length_after_cnn, gru_units * 2 for Bidirectional)
    attention_output = Attention()([x, x]) # Self-attention on GRU output
    # The suggestion implies a reduction after attention before Dense, often by GlobalAveragePooling1D or Flatten
    # Or directly feeding to Dense if the attention layer itself reduces dimensionality (not default for tf.keras.layers.Attention)
    # Let's add pooling to reduce sequence to a vector
    x = tf.keras.layers.GlobalAveragePooling1D()(attention_output)


    # Regression output
    x = Dense(int(gru_units/2), activation='relu')(x) # An intermediate dense layer
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x) # Final regression output

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    # Huber loss is good for time series as it's less sensitive to outliers than MSE
    model.compile(optimizer=optimizer, loss='huber_loss')
    return model

# ... (Add train_hybrid_model_with_params, evaluate_hybrid_model, predict_hybrid_future functions)
# These will be very similar in structure to the LSTM/GRU ones:
# - train_hybrid_model_with_params: takes model_params, calls build_hybrid_..., fits model.
# - evaluate_hybrid_model: takes trained model, test data, scaler, seq_len, returns actuals & preds.
# - predict_hybrid_future: takes trained model, full_df, scaler, seq_len, n_days, returns future_preds & dates.
# The input preparation (create_sequences, reshaping) will be the same.



# models/hybrid_cnn_gru_model.py
# ... (imports, build_hybrid_cnn_gru_attention_for_search, create_sequences) ...

def train_hybrid_model_with_params(
    train_df: pd.DataFrame,
    scaler: MinMaxScaler,
    sequence_length: int,
    model_params: dict,
    epochs: int = 50,
    batch_size: int = 32
) -> Tuple[Model, any]:
    scaled_train_close = scaler.transform(train_df[['Close']])
    x_train, y_train = create_sequences(scaled_train_close, sequence_length)
    x_train_shaped = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    hybrid_model = build_hybrid_cnn_gru_attention_for_search(
        input_shape=(x_train_shaped.shape[1], 1),
        cnn_filters=model_params.get('cnn_filters', 64),
        cnn_kernel_size=model_params.get('cnn_kernel_size', 3),
        gru_units=model_params.get('gru_units', 128),
        dropout_rate=model_params.get('dropout_rate', 0.2),
        learning_rate=model_params.get('learning_rate', 0.001)
    )
    print(f"Training final Hybrid CNN-GRU-Attention model with params: {model_params}, epochs: {epochs}")
    history = hybrid_model.fit(x_train_shaped, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return hybrid_model, history

# ... implement evaluate_hybrid_model and predict_hybrid_future ...
# (These will be almost identical to the evaluate_... and predict_..._future for LSTM/GRU,
# just ensure they use the correct model variable and input/output shapes if they differ)