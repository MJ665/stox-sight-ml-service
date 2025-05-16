# utils/hyperparameter_tuner.py
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasRegressor
import numpy as np
from scipy.stats import randint as sp_randint
from scipy.stats import uniform, loguniform

from models.lstm_model import build_lstm_model_for_search
from models.gru_model import build_gru_model_for_search # Assumes this takes 'dropout_rate' for GRU's recurrent_dropout
from models.transformer_model import build_transformer_model_for_search

# Parameter spaces
LSTM_PARAM_SPACE = {
    'model__units': sp_randint(32, 128),
    'model__dropout_rate': uniform(0.1, 0.4), # This will be passed as dropout_rate
    'optimizer__learning_rate': loguniform(1e-4, 1e-2),
    'model__activation': ['tanh', 'relu'],
}
GRU_PARAM_SPACE = {
    'model__units': sp_randint(32, 128),
    'model__dropout_rate': uniform(0.1, 0.4), # This will be passed as dropout_rate
                                              # and build_gru_model_for_search uses it for GRU's recurrent_dropout
    'optimizer__learning_rate': loguniform(1e-4, 1e-2),
    'model__activation': ['tanh', 'relu']
}
TRANSFORMER_PARAM_SPACE = {
    'model__projection_dim': [32, 64], # NEW - Ensure this is divisible by num_heads choices
    'model__num_transformer_blocks': sp_randint(1, 3), # 1 or 2 blocks
    'model__num_heads': [2, 4], # Must be a divisor of projection_dim
    'model__ff_dim': sp_randint(32, 65), # Feed-forward internal dimension (e.g. 32, 64)
    'model__dense_units': sp_randint(16, 33), # Units in dense layer after pooling
    'model__dropout_rate': uniform(0.1, 0.2), # range 0.1 to 0.3
    'optimizer__learning_rate': loguniform(1e-4, 1e-2),
}

MODEL_CONFIGS = {
    "lstm": {
        "build_fn": build_lstm_model_for_search,
        "param_space": LSTM_PARAM_SPACE,
        "default_model_params": {
            "model__units": 64,
            "model__dropout_rate": 0.2,
            "model__activation": "tanh",
        },
        "fit_params": {'epochs': 10, 'batch_size': 32, 'verbose': 0}
    },
    "gru": {
        "build_fn": build_gru_model_for_search,
        "param_space": GRU_PARAM_SPACE,
        "default_model_params": { # Defaults specific to GRU
            "model__units": 64,
            "model__dropout_rate": 0.2, # This will be used as 'dropout_rate' in build_gru_model_for_search
            "model__activation": "tanh",
        },
        "fit_params": {'epochs': 10, 'batch_size': 32, 'verbose': 0}
    },
      "transformer": {
        "build_fn": build_transformer_model_for_search,
        "param_space": TRANSFORMER_PARAM_SPACE,
        "default_model_params": { # Defaults specific to Transformer
            "model__projection_dim": 64, # NEW
            "model__num_transformer_blocks": 1, # Keep low for speed
            "model__num_heads": 4,
            "model__ff_dim": 32,
            "model__dense_units": 16,
            "model__dropout_rate": 0.1,
        },
        "fit_params": {'epochs': 15, 'batch_size': 32, 'verbose': 0}
    },
}

def tune_model_hyperparameters(
    model_type: str,
    x_train_shaped: np.ndarray,
    y_train_scaled: np.ndarray,
    n_iter: int = 10,
    cv: int = 3
) -> dict:
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model type for tuning: {model_type}")

    config = MODEL_CONFIGS[model_type]

    base_regressor_params = {
        "model": config["build_fn"],
        "input_shape": (x_train_shaped.shape[1], x_train_shaped.shape[2]),
        "loss": "mean_squared_error",
        "optimizer": "adam",
        "optimizer__learning_rate": 0.001,
        "verbose": 0
    }
    
    # These defaults in current_regressor_params are passed to KerasRegressor
    # KerasRegressor passes them to the build_fn (model)
    # The important part is that param_space keys (e.g. 'model__projection_dim') correctly
    # map to parameters your build_fn expects (e.g. 'projection_dim')
    current_regressor_params = {
        **base_regressor_params, 
        **(config.get("default_model_params", {}))
    }
    # Remove defaults that are not part of *this specific model's* build_fn signature
    # to prevent TypeErrors when KerasRegressor calls the build_fn.
    # SciKeras is generally good at filtering, but being explicit can help.
    # However, for hyperparameter search, KerasRegressor should only receive parameters
    # that are either fixed (like input_shape) or part of the current param_distributions.
    # The main change is ensuring the param_space keys are correct and the build_fn matches.
    # The default_model_params in MODEL_CONFIGS are used to provide base values if
    # a parameter is NOT in the RandomizedSearchCV space for that model.
    
    regressor = KerasRegressor(**current_regressor_params)
    # ... (rest of the function)


    random_search = RandomizedSearchCV(
        estimator=regressor,
        param_distributions=config["param_space"],
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1,
        random_state=42,
        error_score='raise'
    )

    print(f"Starting RandomizedSearchCV for {model_type} with scikeras...")
    random_search.fit(x_train_shaped, y_train_scaled, **config["fit_params"])

    print(f"Best parameters found for {model_type}: {random_search.best_params_}")
    print(f"Best score for {model_type}: {random_search.best_score_}")

    best_model_arch_params = {}
    for key, value in random_search.best_params_.items():
        if key.startswith("model__"):
            best_model_arch_params[key[len("model__"):]] = value
        elif key.startswith("optimizer__"):
            best_model_arch_params[key[len("optimizer__"):]] = value
    
    return best_model_arch_params







# # utils/hyperparameter_tuner.py
# from sklearn.model_selection import RandomizedSearchCV
# from scikeras.wrappers import KerasRegressor
# import numpy as np
# from scipy.stats import randint as sp_randint
# from scipy.stats import uniform, loguniform

# from models.lstm_model import build_lstm_model_for_search
# from models.gru_model import build_gru_model_for_search
# from models.transformer_model import build_transformer_model_for_search
# # from models.hybrid_cnn_gru_model import build_hybrid_cnn_gru_attention_for_search # Assuming not yet fully implemented

# # Parameter spaces
# LSTM_PARAM_SPACE = {
#     'model__units': sp_randint(32, 128),
#     'model__dropout_rate': uniform(0.1, 0.4),
#     'optimizer__learning_rate': loguniform(1e-4, 1e-2),
#     'model__activation': ['tanh', 'relu'],
# }


# GRU_PARAM_SPACE = {
#     'model__units': sp_randint(32, 128),
#     # 'model__dropout_rate': uniform(0.1, 0.4), # This is for the tf.keras.layers.Dropout
#     'model__recurrent_dropout_for_gru_layer': uniform(0.0, 0.4), # NEW specific name if you want to tune it separately
#                                                                # OR, if you want a single dropout for both:
#     'model__dropout_rate': uniform(0.1, 0.4), # This will be passed as 'dropout_rate' to build_gru_model_for_search
#     'optimizer__learning_rate': loguniform(1e-4, 1e-2),
#     'model__activation': ['tanh', 'relu']
# }


# # GRU_PARAM_SPACE = {
# #     'model__units': sp_randint(32, 128),
# #     'model__dropout_rate': uniform(0.1, 0.4),
# #     'model__recurrent_dropout_rate': uniform(0.0, 0.3),
# #     'optimizer__learning_rate': loguniform(1e-4, 1e-2),
# #     'model__activation': ['tanh', 'relu']
# # }
# TRANSFORMER_PARAM_SPACE = {
#     'model__num_transformer_blocks': sp_randint(1, 3),
#     'model__num_heads': [2, 4], # Keep small for speed
#     'model__ff_dim': sp_randint(32, 65), # Smaller range
#     'model__dense_units': sp_randint(16, 33), # Smaller range
#     'model__dropout_rate': uniform(0.1, 0.3),
#     'optimizer__learning_rate': loguniform(1e-4, 1e-2),
# }
# # HYBRID_CNN_GRU_PARAM_SPACE = { ... } # Define when ready

# MODEL_CONFIGS = {
#     "lstm": {
#         "build_fn": build_lstm_model_for_search,
#         "param_space": LSTM_PARAM_SPACE,
#         "default_model_params": { # Defaults specific to LSTM
#             "model__units": 64,
#             "model__dropout_rate": 0.2,
#             "model__activation": "tanh",
#         },
#         "fit_params": {'epochs': 10, 'batch_size': 32, 'verbose': 0}
#     },
#     # "gru": {
#     #     "build_fn": build_gru_model_for_search,
#     #     "param_space": GRU_PARAM_SPACE,
#     #     "default_model_params": { # Defaults specific to GRU
#     #         "model__units": 64,
#     #         "model__dropout_rate": 0.2,
#     #         "model__recurrent_dropout_rate": 0.1,
#     #         "model__activation": "tanh",
#     #     },
#     #     "fit_params": {'epochs': 10, 'batch_size': 32, 'verbose': 0}
#     # },
    
#      "gru": {
#         "build_fn": build_gru_model_for_search,
#         "param_space": GRU_PARAM_SPACE,
#         "default_model_params": {
#             "model__units": 64,
#             "model__dropout_rate": 0.2, # This will be used for GRU's recurrent_dropout AND Dropout layer
#             # Remove "model__recurrent_dropout_rate": 0.1, if you go with single dropout_rate
#             "model__activation": "tanh",
#         },
#         "fit_params": {'epochs': 10, 'batch_size': 32, 'verbose': 0}
#     },
#     "transformer": {
#         "build_fn": build_transformer_model_for_search,
#         "param_space": TRANSFORMER_PARAM_SPACE,
#         "default_model_params": { # Defaults specific to Transformer
#             "model__num_transformer_blocks": 2,
#             "model__num_heads": 4,
#             "model__ff_dim": 32,
#             "model__dense_units": 32,
#             "model__dropout_rate": 0.1,
#         },
#         "fit_params": {'epochs': 15, 'batch_size': 32, 'verbose': 0}
#     },
#     # "hybrid_cnn_gru": { ... } # Define when ready
# }

# def tune_model_hyperparameters(
#     model_type: str,
#     x_train_shaped: np.ndarray,
#     y_train_scaled: np.ndarray,
#     n_iter: int = 10,
#     cv: int = 3
# ) -> dict:
#     if model_type not in MODEL_CONFIGS:
#         raise ValueError(f"Unsupported model type for tuning: {model_type}")

#     config = MODEL_CONFIGS[model_type]

#     # Common KerasRegressor parameters
#     base_regressor_params = {
#         "model": config["build_fn"],
#         "input_shape": (x_train_shaped.shape[1], x_train_shaped.shape[2]), # Passed to build_fn
#         "loss": "mean_squared_error", # Default loss
#         "optimizer": "adam",          # Default optimizer
#         "optimizer__learning_rate": 0.001, # Default optimizer lr, will be overridden by search if in param_space
#         "verbose": 0 # Keras verbosity during RandomizedSearchCV's internal fits
#     }

#     # Combine base params with model-specific defaults
#     # The model-specific defaults will override base_regressor_params if there's overlap
#     # on keys like optimizer__learning_rate if it's also in model_params
#     # However, it's better to keep optimizer params separate or consistent.
#     # For now, we assume model_params in MODEL_CONFIGS are for model architecture.
    
#     current_regressor_params = {
#         **base_regressor_params,
#         **(config.get("default_model_params", {}))
#         # NO model__recurrent_dropout_rate HERE unless your build_fn explicitly takes it
#     }
    
#     regressor = KerasRegressor(**current_regressor_params)

#     random_search = RandomizedSearchCV(
#         estimator=regressor,
#         param_distributions=config["param_space"],
#         n_iter=n_iter,
#         cv=cv,
#         scoring='neg_mean_squared_error',
#         verbose=1,
#         n_jobs=-1,
#         random_state=42,
#         error_score='raise' # Important for debugging individual fit failures
#     )

#     print(f"Starting RandomizedSearchCV for {model_type} with scikeras...")
#     random_search.fit(x_train_shaped, y_train_scaled, **config["fit_params"])

#     print(f"Best parameters found for {model_type}: {random_search.best_params_}")
#     print(f"Best score for {model_type}: {random_search.best_score_}")

#     best_model_arch_params = {}
#     for key, value in random_search.best_params_.items():
#         if key.startswith("model__"):
#             best_model_arch_params[key[len("model__"):]] = value
#         elif key.startswith("optimizer__"):
#             best_model_arch_params[key[len("optimizer__"):]] = value
    
#     return best_model_arch_params












# # utils/hyperparameter_tuner.py
# from sklearn.model_selection import RandomizedSearchCV
# # from tensorflow.keras.wrappers.scikit_learn import KerasRegressor # OLD AND DEPRECATED
# from scikeras.wrappers import KerasRegressor # NEW - USE THIS
# import numpy as np
# from scipy.stats import randint as sp_randint
# from scipy.stats import uniform, loguniform

# # Import your model building functions (these should remain the same)
# from models.lstm_model import build_lstm_model_for_search
# from models.gru_model import build_gru_model_for_search



# from models.transformer_model import build_transformer_model_for_search # NEW

# TRANSFORMER_PARAM_SPACE = {
#     'model__num_transformer_blocks': sp_randint(1, 4), # 1 to 3 blocks
#     'model__num_heads': [2, 4, 8], # Must be a divisor of model dimension (which depends on ff_dim or embedding)
#     'model__ff_dim': sp_randint(32, 128), # Feed-forward internal dimension
#     'model__dense_units': sp_randint(16, 64), # Units in dense layer after pooling
#     'model__dropout_rate': uniform(0.1, 0.3),
#     'optimizer__learning_rate': loguniform(1e-4, 1e-2),
# }



# # Define parameter spaces (these remain the same)
# LSTM_PARAM_SPACE = {
#     'model__units': sp_randint(32, 128), # Prefix with 'model__' for scikeras
#     'model__dropout_rate': uniform(0.1, 0.4),
#     'optimizer__learning_rate': loguniform(1e-4, 1e-2), # If learning_rate is for optimizer
#     # For scikeras, epochs and batch_size are fit_params, not part of model construction usually
# }

# GRU_PARAM_SPACE = {
#     'model__units': sp_randint(32, 128),
#     'model__dropout_rate': uniform(0.1, 0.4), # scikeras calls this dropout_rate if your build_fn uses it
#     'optimizer__learning_rate': loguniform(1e-4, 1e-2),
#     'model__activation': ['relu', 'tanh'] # Prefix with 'model__'
# }

# MODEL_CONFIGS = {
#     "lstm": {
#         "build_fn": build_lstm_model_for_search, # This is the function that returns a Keras model
#         "param_space": LSTM_PARAM_SPACE,
#         "fit_params": {'epochs': 10, 'batch_size': 32, 'verbose': 0}
#     },
#     "gru": {
#         "build_fn": build_gru_model_for_search,
#         "param_space": GRU_PARAM_SPACE,
#         "fit_params": {'epochs': 10, 'batch_size': 32, 'verbose': 0}
#     },
#      "transformer": { # NEW
#         "build_fn": build_transformer_model_for_search,
#         "param_space": TRANSFORMER_PARAM_SPACE,
#         "fit_params": {'epochs': 15, 'batch_size': 32, 'verbose': 0} # Transformers might need a bit more epochs
#     }
# }

# def tune_model_hyperparameters(
#     model_type: str,
#     x_train_shaped: np.ndarray, # Shape: (samples, sequence_length, features=1)
#     y_train_scaled: np.ndarray, # Shape: (samples,)
#     n_iter: int = 10,
#     cv: int = 3
# ) -> dict:

#     if model_type not in MODEL_CONFIGS:
#         raise ValueError(f"Unsupported model type for tuning: {model_type}")

#     config = MODEL_CONFIGS[model_type]

#     # Create KerasRegressor wrapper using scikeras
#     # Parameters for your build_fn (like units, dropout_rate) are passed directly
#     # to KerasRegressor, and it will pass them to your model building function.
#     # The 'model' argument takes your function that returns a compiled Keras model.
#     # 'input_shape' will be passed to your build_fn if it accepts it.
    
    

#     regressor = KerasRegressor(
#         model=config["build_fn"],
#         input_shape=(x_train_shaped.shape[1], x_train_shaped.shape[2]),
#         loss="mean_squared_error",
#         optimizer="adam",
#         optimizer__learning_rate=0.001, # Default for optimizer
#         # Defaults for model params (scikeras will pass these to build_fn)
#         # These should cover all params your build_fn might expect
#         # that are also in your param_space, to avoid errors if a
#         # search iteration doesn't pick a value for one of them.
#         model__units=64,             # For LSTM/GRU
#         model__dropout_rate=0.2,     # For LSTM/GRU/Transformer
#         model__activation='relu',    # For GRU
#         model__num_transformer_blocks=2, # For Transformer
#         model__num_heads=4,              # For Transformer
#         model__ff_dim=32,                # For Transformer
#         model__dense_units=32,           # For Transformer
#         verbose=0
#     )

#     # Parameters in param_distributions for RandomizedSearchCV need to be prefixed
#     # with 'model__' if they are arguments to your model building function,
#     # // or 'optimizer__' if they are arguments to the optimizer, etc.
#     # Example: if your build_lstm_model_for_search takes 'units', then in param_space use 'model__units'.
#     # If your build_fn configures an Adam optimizer and takes 'learning_rate' for it,
#     # then use 'optimizer__learning_rate' in KerasRegressor and param_space.

#     random_search = RandomizedSearchCV(
#         estimator=regressor,
#         param_distributions=config["param_space"],
#         n_iter=n_iter,
#         cv=cv,
#         scoring='neg_mean_squared_error',
#         verbose=1,
#         n_jobs=-1, # Use all cores; set to 1 if issues arise
#         random_state=42
#     )

#     print(f"Starting RandomizedSearchCV for {model_type}...")
#     # Pass fit_params (epochs, batch_size) to the fit method of RandomizedSearchCV
#     random_search.fit(x_train_shaped, y_train_scaled, **config["fit_params"])

#     print(f"Best parameters found for {model_type}: {random_search.best_params_}")
#     print(f"Best score for {model_type}: {random_search.best_score_}")

#     # Extract model-specific parameters, removing the 'model__' or 'optimizer__' prefix
#     best_model_arch_params = {}
#     for key, value in random_search.best_params_.items():
#         if key.startswith("model__"):
#             best_model_arch_params[key[len("model__"):]] = value
#         elif key.startswith("optimizer__"):
#             # You might want to store optimizer params separately or include them
#             # For now, let's assume they are part of model_params for simplicity if your build_fn handles it
#             best_model_arch_params[key[len("optimizer__"):]] = value # e.g. learning_rate
#         # else: # Parameters like epochs, batch_size if they were in search space
#             # best_model_arch_params[key] = value


#     return best_model_arch_params








































































# # utils/hyperparameter_tuner.py
# from sklearn.model_selection import RandomizedSearchCV
# # from tensorflow.keras.wrappers.scikit_learn import KerasRegressor # Deprecated in TF 2.10+, use scikeras.wrappers.KerasRegressor
# # If using TF 2.10+ and scikeras is installed:
# from scikeras.wrappers import KerasRegressor 
# import numpy as np
# from scipy.stats import randint as sp_randint
# from scipy.stats import uniform, loguniform

# # Import your model building functions
# from models.lstm_model import build_lstm_model_for_search # Assuming you create this
# from models.gru_model import build_gru_model_for_search

# # Define parameter spaces
# LSTM_PARAM_SPACE = {
#     'units': sp_randint(32, 128), # Integer range
#     'dropout_rate': uniform(0.1, 0.4), # Continuous uniform distribution (0.1 to 0.1+0.4=0.5)
#     'learning_rate': loguniform(1e-4, 1e-2),
#     # 'batch_size': [16, 32, 64], # KerasRegressor can take batch_size and epochs
#     # 'epochs': [10, 20]       # These are fit_params for RandomizedSearchCV
# }

# GRU_PARAM_SPACE = {
#     'units': sp_randint(32, 128),
#     'dropout_rate': uniform(0.1, 0.4), # for GRU, this might be recurrent_dropout or just dropout
#     'learning_rate': loguniform(1e-4, 1e-2),
#     'activation': ['relu', 'tanh']
#     # 'batch_size': [16, 32, 64],
#     # 'epochs': [10, 20]
# }

# # Add more for other models if needed (Transformer, Hybrid)

# MODEL_CONFIGS = {
#     "lstm": {
#         "build_fn": build_lstm_model_for_search,
#         "param_space": LSTM_PARAM_SPACE,
#         "fit_params": {'epochs': 10, 'batch_size': 32, 'verbose': 0} # epochs for RandomizedSearch CV folds
#     },
#     "gru": {
#         "build_fn": build_gru_model_for_search,
#         "param_space": GRU_PARAM_SPACE,
#         "fit_params": {'epochs': 10, 'batch_size': 32, 'verbose': 0}
#     }
#     # Add "transformer", "hybrid_cnn_gru" here later
# }

# def tune_model_hyperparameters(
#     model_type: str, 
#     x_train_shaped: np.ndarray, # Shape: (samples, sequence_length, features=1)
#     y_train_scaled: np.ndarray, # Shape: (samples,)
#     n_iter: int = 10, # Number of parameter settings that are sampled
#     cv: int = 3       # Number of cross-validation folds
# ) -> dict: # Returns best_params

#     if model_type not in MODEL_CONFIGS:
#         raise ValueError(f"Unsupported model type for tuning: {model_type}")

#     config = MODEL_CONFIGS[model_type]
    
#     # Create KerasRegressor wrapper
#     # Note: input_shape needs to be passed to build_fn.
#     # KerasRegressor doesn't directly pass dynamic args like input_shape from fit() to build_fn.
#     # We pass it via a fixed parameter in the KerasRegressor constructor.
#     # Or, ensure build_fn can infer it or doesn't strictly need it if using input_dim in first layer.
#     # For sequence models, input_shape is critical.
    
#     # Solution: KerasRegressor takes build_fn args as kwargs
#     regressor = KerasRegressor(
#         build_fn=config["build_fn"], 
#         input_shape=(x_train_shaped.shape[1], x_train_shaped.shape[2]) # Pass input_shape here
#     )

#     random_search = RandomizedSearchCV(
#         estimator=regressor,
#         param_distributions=config["param_space"],
#         n_iter=n_iter,
#         cv=cv,
#         scoring='neg_mean_squared_error', # Lower MSE is better, so neg_mse (higher is better)
#         verbose=1, # Set to 0 or 1 for less/more output
#         n_jobs=-1, # Use all available cores, careful on shared systems
#         random_state=42 # For reproducibility
#     )

#     print(f"Starting RandomizedSearchCV for {model_type}...")
#     random_search.fit(x_train_shaped, y_train_scaled, **config["fit_params"])
    
#     print(f"Best parameters found for {model_type}: {random_search.best_params_}")
#     print(f"Best score for {model_type}: {random_search.best_score_}")
    
#     # The best_params_ might include batch_size/epochs if they were in param_space.
#     # We are primarily interested in the model architecture params (units, dropout, lr).
#     best_model_params = {k: v for k, v in random_search.best_params_.items() 
#                          if k not in ['batch_size', 'epochs']} # Filter out fit params if they were in search space
    
#     return best_model_params