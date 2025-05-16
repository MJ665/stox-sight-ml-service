# ./main.py
import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks , Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np # For reshaping for tuner
import uuid
from datetime import datetime, timedelta
import io
import json
from dotenv import load_dotenv

load_dotenv()

from utils.email_sender import send_email_with_attachment # NEW IMPORT

from sklearn.metrics import mean_absolute_error, r2_score

# Data processing
from utils.data_processing import preprocess_data, create_sequences_for_tuner # Add create_sequences_for_tuner
from utils.plotting import plot_test_predictions, plot_future_predictions, plot_polynomial_regression
from utils.pdf_generator import generate_prediction_report, PDF_DIR
from utils.gemini_analyzer import generate_analysis_prompt, get_gemini_analysis
from utils.hyperparameter_tuner import tune_model_hyperparameters # NEW IMPORT

# Model functions
from models.lstm_model import (
    train_lstm_model_with_params, # UPDATED
    evaluate_lstm_model,
    predict_lstm_future
)
from models.polynomial_model import (
    train_polynomial_model,
    predict_polynomial_future,
    evaluate_polynomial_model
)
from models.gru_model import ( # NEW IMPORTS
    train_gru_model_with_params,
    evaluate_gru_model,
    predict_gru_future
)


# Configuration
SEQUENCE_LENGTH =int(os.getenv("SEQUENCE_LENGTH")) # Adjusted
N_FUTURE_DAYS_PREDICTION =int(os.getenv("N_FUTURE_DAYS_PREDICTION"))
# FINAL_TRAIN_EPOCHS =int (os.getenv("FINAL_TRAIN_EPOCHS")) # Epochs for final model training AFTER tuning
FINAL_TRAIN_EPOCHS =int(os.getenv("FINAL_TRAIN_EPOCHS")) # Epochs for final model training AFTER tuning
POLYNOMIAL_DEGREE =int(os.getenv("POLYNOMIAL_DEGREE"))
BUY_SELL_THRESHOLD_PERCENT =float(os.getenv("BUY_SELL_THRESHOLD_PERCENT"))
RANDOM_SEARCH_N_ITER =int(os.getenv("RANDOM_SEARCH_N_ITER")) # Number of iterations for RandomizedSearchCV (keep low for hackathon speed)
RANDOM_SEARCH_CV =int(os.getenv("RANDOM_SEARCH_CV"))     # Number of CV folds (keep low for speed)


# It's better to configure the API key once
API_KEY = os.getenv("GOOGLE_API_KEY_ANALYZER")







# --- PATH CONFIGURATION FOR GENERATED FILES ---
IS_ON_HUGGINGFACE = os.path.exists("data") and os.access("data", os.W_OK)

if IS_ON_HUGGINGFACE:
    RUNTIME_GENERATED_BASE_DIR = "data/stox_sight_outputs" # Use a simpler name directly under /data
    print(f"INFO (Startup): Running on Hugging Face Spaces. Using base: {RUNTIME_GENERATED_BASE_DIR}")
else:
    RUNTIME_GENERATED_BASE_DIR = os.path.join(os.getcwd(), "data")
    print(f"INFO (Startup): Not on Hugging Face Spaces (or /data not writable). Using local fallback: {RUNTIME_GENERATED_BASE_DIR}")

# These paths are now dynamically determined
PLOTS_DIR = os.path.join(RUNTIME_GENERATED_BASE_DIR, "plots")
PDF_GENERATION_DIR = os.path.join(RUNTIME_GENERATED_BASE_DIR, "pdfs")

# URL base path for serving these generated files via FastAPI
SERVED_GENERATED_CONTENT_URL_PREFIX = "/outputs" # Make it short and distinct

app = FastAPI(title="Advanced Stock Prediction ML Service")

# Ensure the BASE directory exists at startup for the mount.
# Subdirectories (plots, pdfs) will be created per-request.
try:
    os.makedirs(RUNTIME_GENERATED_BASE_DIR, exist_ok=True)
    app.mount(SERVED_GENERATED_CONTENT_URL_PREFIX, StaticFiles(directory=RUNTIME_GENERATED_BASE_DIR), name="generated_outputs")
    print(f"INFO (Startup): Mounted '{RUNTIME_GENERATED_BASE_DIR}' at URL prefix '{SERVED_GENERATED_CONTENT_URL_PREFIX}'")
except OSError as e:
    print(f"WARNING (Startup): Could not create or mount base directory '{RUNTIME_GENERATED_BASE_DIR}'. Error: {e}. Static file serving might fail.")





# STATIC_DIR = "static"
# PLOTS_DIR = os.path.join(STATIC_DIR, "plots")
# os.makedirs(PLOTS_DIR, exist_ok=True)
# os.makedirs(PDF_DIR, exist_ok=True)


# Mount the base directory for generated content
# This allows files under RUNTIME_GENERATED_BASE_DIR to be served via SERVED_GENERATED_CONTENT_URL_PREFIX
app.mount(SERVED_GENERATED_CONTENT_URL_PREFIX, StaticFiles(directory=RUNTIME_GENERATED_BASE_DIR), name="generated_outputs")


def cleanup_old_files(directory, max_age_minutes=60):
    # ... (same as before) ...
    now = datetime.now()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if (now - file_mod_time) > timedelta(minutes=max_age_minutes):
                    os.remove(file_path)
        except Exception:
            pass # Ignore cleanup errors

class ModelResultDetail(BaseModel):
    best_params: dict = {}
    test_metrics: dict = {}
    future_predictions: dict = {}
    training_loss: list = [] # For deep learning models

class TrainResponse(BaseModel):
    message: str
    run_id: str
    stock_symbol: str # Added for clarity in response
    lstm_results: ModelResultDetail
    gru_results: ModelResultDetail # NEW
    transformer_results: ModelResultDetail # NEW
    polynomial_results: ModelResultDetail # Keep ModelResultDetail structure for consistency
    trading_suggestion_tomorrow: dict # Based on one model, e.g., LSTM or an ensemble
    ai_qualitative_analysis: dict | str
    plot_urls: dict
    pdf_report_url: str
    email_sent_status: str # NEW

from models.transformer_model import ( # NEW
    train_transformer_model_with_params,
    evaluate_transformer_model,
    predict_transformer_future
)





# ... (cleanup_old_files, Pydantic models) ...
# ... (import model functions) ...

@app.post("/train-predict/", response_model=TrainResponse)
async def train_and_predict_models(
    background_tasks: BackgroundTasks,
    csv_file: UploadFile = File(...),
    user_email_to_send_to: str = Form(...),
):
    run_id = str(uuid.uuid4())
    # ... (results_payload initialization) ...

    # --- Ensure specific output subdirectories exist AT REQUEST TIME ---
    # This happens AFTER RUNTIME_GENERATED_BASE_DIR is confirmed/created at startup.
    try:
        # These are now based on the globally defined PLOTS_DIR and PDF_GENERATION_DIR
        os.makedirs(PLOTS_DIR, exist_ok=True)
        os.makedirs(PDF_GENERATION_DIR, exist_ok=True)
        print(f"[{run_id}] Ensured runtime directories exist: Plots='{PLOTS_DIR}', PDFs='{PDF_GENERATION_DIR}'")
    except OSError as e:
        print(f"[{run_id}] CRITICAL ERROR: Could not create runtime output subdirectories. Error: {e}")
        # This error indicates a problem with the RUNTIME_GENERATED_BASE_DIR itself being unwritable
        raise HTTPException(status_code=500, detail=f"Server configuration error: Cannot create output directories. Base: {RUNTIME_GENERATED_BASE_DIR}. Error: {str(e)}")


    stock_symbol_from_csv = "UNKNOWN" # Default

    results_payload = {
        "message": "Processing started.",
        "run_id": run_id,
        "stock_symbol": stock_symbol_from_csv,
        "csv_filename": csv_file.filename,
        "lstm_results": {"best_params": {}, "test_metrics": {}, "future_predictions": {}, "training_loss": []},
        "gru_results": {"best_params": {}, "test_metrics": {}, "future_predictions": {}, "training_loss": []}, # NEW
         "transformer_results": {"best_params": {}, "test_metrics": {}, "future_predictions": {}, "training_loss": []}, # NEW
        "polynomial_results": {"test_metrics": {}, "future_predictions": {}}, # Poly doesn't have 'best_params' from RandomizedSearch in this setup
        "trading_suggestion_tomorrow": {},
        "ai_qualitative_analysis": {},
        "plot_urls": {},
        "pdf_report_url": "",
         "email_sent_status": "Not attempted" # Initialize
    }

    try:
        contents = await csv_file.read()
        df_full = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        if 'Date' not in df_full.columns or 'Close' not in df_full.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'Date' and 'Close' columns.")
        if 'Symbol' in df_full.columns and not df_full['Symbol'].empty:
            stock_symbol_from_csv = df_full['Symbol'].iloc[0]
            results_payload["stock_symbol"] = stock_symbol_from_csv


        df_full['Date'] = pd.to_datetime(df_full['Date'])
        df_full.sort_values(by='Date', inplace=True)
        df_full.reset_index(drop=True, inplace=True)
        df_full['Close'] = pd.to_numeric(df_full['Close'], errors='coerce')
        df_full.dropna(subset=['Close'], inplace=True)

        if len(df_full) < SEQUENCE_LENGTH + N_FUTURE_DAYS_PREDICTION + 20: # Min data check
             raise HTTPException(status_code=400, detail=f"Not enough data. Need at least {SEQUENCE_LENGTH + N_FUTURE_DAYS_PREDICTION + 20} rows.")

        original_min_date = df_full['Date'].min()
        last_actual_close_price = df_full['Close'].iloc[-1]
        last_actual_date = df_full['Date'].iloc[-1]

        train_df, test_df, price_scaler = preprocess_data(df_full.copy(), sequence_length=SEQUENCE_LENGTH)
        
        # Data for hyperparameter tuner (needs to be reshaped for KerasRegressor)
        # This data should be SCALED.
        x_train_for_tuner_scaled, y_train_for_tuner_scaled = create_sequences_for_tuner(
            price_scaler.transform(train_df[['Close']]), SEQUENCE_LENGTH
        )
        x_train_for_tuner_reshaped = np.reshape(
            x_train_for_tuner_scaled, 
            (x_train_for_tuner_scaled.shape[0], x_train_for_tuner_scaled.shape[1], 1)
        )



        # --- LSTM Model with Hyperparameter Tuning ---
        print(f"[{run_id}] Tuning LSTM hyperparameters...")
        best_lstm_params = tune_model_hyperparameters(
            "lstm", x_train_for_tuner_reshaped, y_train_for_tuner_scaled,
            n_iter=RANDOM_SEARCH_N_ITER, cv=RANDOM_SEARCH_CV
        )
        results_payload["lstm_results"]["best_params"] = best_lstm_params
        
        print(f"[{run_id}] Training final LSTM model with best params: {best_lstm_params}...")
        lstm_model, lstm_history = train_lstm_model_with_params(
            train_df, price_scaler, SEQUENCE_LENGTH, best_lstm_params, epochs=FINAL_TRAIN_EPOCHS
        )
        results_payload["lstm_results"]["training_loss"] = lstm_history.history.get('loss', [0.0]) # Ensure it's a list

        y_test_actual_lstm, y_pred_test_lstm, _ = evaluate_lstm_model(lstm_model, test_df, train_df, price_scaler, SEQUENCE_LENGTH)
        results_payload["lstm_results"]["test_metrics"] = {"mae": float(mean_absolute_error(y_test_actual_lstm, y_pred_test_lstm)), "r2_score": float(r2_score(y_test_actual_lstm, y_pred_test_lstm))}
        
        future_preds_lstm, future_dates_lstm = predict_lstm_future(lstm_model, df_full, price_scaler, SEQUENCE_LENGTH, N_FUTURE_DAYS_PREDICTION)
        results_payload["lstm_results"]["future_predictions"] = {dt.strftime('%Y-%m-%d'): float(pred) for dt, pred in zip(future_dates_lstm, future_preds_lstm.flatten())}

        # LSTM Plotting (same as before, ensure plot_urls is populated in results_payload)
        # ...
        
                # LSTM Plotting
        # lstm_plot_test_filename = f"{run_id}_lstm_test_predictions.png"
        # lstm_plot_test_path = os.path.join(PLOTS_DIR, lstm_plot_test_filename)
        # plot_test_predictions(test_df['Date'], y_test_actual_lstm, y_pred_test_lstm, "LSTM: Test Set Predictions", lstm_plot_test_path)
        # results_payload["plot_urls"]["lstm_test_plot"] = f"/static/plots/{lstm_plot_test_filename}"

        # lstm_plot_future_filename = f"{run_id}_lstm_future_predictions.png"
        # lstm_plot_future_path = os.path.join(PLOTS_DIR, lstm_plot_future_filename)
        # plot_future_predictions(df_full['Date'], df_full['Close'], test_df['Date'], y_pred_test_lstm, future_dates_lstm, future_preds_lstm, "LSTM: Historical, Test & Future Predictions", lstm_plot_future_path)
        # results_payload["plot_urls"]["lstm_future_plot"] = f"/static/plots/{lstm_plot_future_filename}"
        lstm_plot_test_filename = f"{run_id}_lstm_test_predictions.png"
        lstm_plot_test_disk_path = os.path.join(PLOTS_DIR, lstm_plot_test_filename) # DISK PATH
        plot_test_predictions(test_df['Date'], y_test_actual_lstm, y_pred_test_lstm, "LSTM: Test Set Predictions", lstm_plot_test_disk_path)
        results_payload["plot_urls"]["lstm_test_plot_url"] = f"{SERVED_GENERATED_CONTENT_URL_PREFIX}/plots/{lstm_plot_test_filename}" # URL
        results_payload["plot_urls"]["lstm_test_plot_path"] = lstm_plot_test_disk_path # Store disk path for PDF

        lstm_plot_future_filename = f"{run_id}_lstm_future_predictions.png"
        lstm_plot_future_disk_path = os.path.join(PLOTS_DIR, lstm_plot_future_filename)
        plot_future_predictions(df_full['Date'], df_full['Close'], test_df['Date'], y_pred_test_lstm, future_dates_lstm, future_preds_lstm, "LSTM: Historical, Test & Future Predictions", lstm_plot_future_disk_path)
        results_payload["plot_urls"]["lstm_future_plot_url"] = f"{SERVED_GENERATED_CONTENT_URL_PREFIX}/plots/{lstm_plot_future_filename}"
        results_payload["plot_urls"]["lstm_future_plot_path"] = lstm_plot_future_disk_path




        # --- GRU Model with Hyperparameter Tuning ---
        print(f"[{run_id}] Tuning GRU hyperparameters...")
        best_gru_params = tune_model_hyperparameters(
            "gru", x_train_for_tuner_reshaped, y_train_for_tuner_scaled,
            n_iter=RANDOM_SEARCH_N_ITER, cv=RANDOM_SEARCH_CV
        )
        results_payload["gru_results"]["best_params"] = best_gru_params

        print(f"[{run_id}] Training final GRU model with best params: {best_gru_params}...")
        gru_model, gru_history = train_gru_model_with_params(
            train_df, price_scaler, SEQUENCE_LENGTH, best_gru_params, epochs=FINAL_TRAIN_EPOCHS
        )
        results_payload["gru_results"]["training_loss"] = gru_history.history.get('loss', [0.0])

        y_test_actual_gru, y_pred_test_gru, _ = evaluate_gru_model(gru_model, test_df, train_df, price_scaler, SEQUENCE_LENGTH)
        results_payload["gru_results"]["test_metrics"] = {"mae": float(mean_absolute_error(y_test_actual_gru, y_pred_test_gru)), "r2_score": float(r2_score(y_test_actual_gru, y_pred_test_gru))}

        future_preds_gru, future_dates_gru = predict_gru_future(gru_model, df_full, price_scaler, SEQUENCE_LENGTH, N_FUTURE_DAYS_PREDICTION)
        results_payload["gru_results"]["future_predictions"] = {dt.strftime('%Y-%m-%d'): float(pred) for dt, pred in zip(future_dates_gru, future_preds_gru.flatten())}

        # GRU Plotting (create new plot functions or adapt existing ones for GRU)
        # gru_plot_test_filename = f"{run_id}_gru_test_predictions.png"
        # gru_plot_test_path = os.path.join(PLOTS_DIR, gru_plot_test_filename)
        # plot_test_predictions(test_df['Date'], y_test_actual_gru, y_pred_test_gru, "GRU: Test Set Predictions", gru_plot_test_path)
        # results_payload["plot_urls"]["gru_test_plot"] = f"/static/plots/{gru_plot_test_filename}"

        # gru_plot_future_filename = f"{run_id}_gru_future_predictions.png"
        # gru_plot_future_path = os.path.join(PLOTS_DIR, gru_plot_future_filename)
        # plot_future_predictions(df_full['Date'], df_full['Close'], test_df['Date'], y_pred_test_gru, future_dates_gru, future_preds_gru, "GRU: Historical, Test & Future Predictions", gru_plot_future_path)
        # results_payload["plot_urls"]["gru_future_plot"] = f"/static/plots/{gru_plot_future_filename}"


        gru_plot_test_filename = f"{run_id}_gru_test_predictions.png"
        gru_plot_test_disk_path = os.path.join(PLOTS_DIR, gru_plot_test_filename)
        plot_test_predictions(test_df['Date'], y_test_actual_gru, y_pred_test_gru, "GRU: Test Set Predictions", gru_plot_test_disk_path)
        results_payload["plot_urls"]["gru_test_plot_url"] = f"{SERVED_GENERATED_CONTENT_URL_PREFIX}/plots/{gru_plot_test_filename}"
        results_payload["plot_urls"]["gru_test_plot_path"] = gru_plot_test_disk_path
        
        gru_plot_future_filename = f"{run_id}_gru_future_predictions.png"
        gru_plot_future_disk_path = os.path.join(PLOTS_DIR, gru_plot_future_filename)
        plot_future_predictions(df_full['Date'], df_full['Close'], test_df['Date'], y_pred_test_gru, future_dates_gru, future_preds_gru, "GRU: Historical, Test & Future Predictions", gru_plot_future_disk_path)
        results_payload["plot_urls"]["gru_future_plot_url"] = f"{SERVED_GENERATED_CONTENT_URL_PREFIX}/plots/{gru_plot_future_filename}"
        results_payload["plot_urls"]["gru_future_plot_path"] = gru_plot_future_disk_path





        # --- Transformer Model with Hyperparameter Tuning ---
        print(f"[{run_id}] Tuning Transformer hyperparameters...")
        best_transformer_params = tune_model_hyperparameters(
            "transformer", x_train_for_tuner_reshaped, y_train_for_tuner_scaled,
            n_iter=RANDOM_SEARCH_N_ITER, cv=RANDOM_SEARCH_CV
        )
        results_payload["transformer_results"]["best_params"] = best_transformer_params

        print(f"[{run_id}] Training final Transformer model with best params: {best_transformer_params}...")
        transformer_model, transformer_history = train_transformer_model_with_params(
            train_df, price_scaler, SEQUENCE_LENGTH, best_transformer_params, epochs=FINAL_TRAIN_EPOCHS # Adjust epochs if needed
        )
        results_payload["transformer_results"]["training_loss"] = transformer_history.history.get('loss', [0.0])

        y_test_actual_transformer, y_pred_test_transformer, _ = evaluate_transformer_model(transformer_model, test_df, train_df, price_scaler, SEQUENCE_LENGTH)
        results_payload["transformer_results"]["test_metrics"] = {"mae": float(mean_absolute_error(y_test_actual_transformer, y_pred_test_transformer)), "r2_score": float(r2_score(y_test_actual_transformer, y_pred_test_transformer))}

        future_preds_transformer, future_dates_transformer = predict_transformer_future(transformer_model, df_full, price_scaler, SEQUENCE_LENGTH, N_FUTURE_DAYS_PREDICTION)
        results_payload["transformer_results"]["future_predictions"] = {dt.strftime('%Y-%m-%d'): float(pred) for dt, pred in zip(future_dates_transformer, future_preds_transformer.flatten())}

        # Transformer Plotting
        # transformer_plot_test_filename = f"{run_id}_transformer_test_predictions.png"
        # transformer_plot_test_path = os.path.join(PLOTS_DIR, transformer_plot_test_filename)
        # plot_test_predictions(test_df['Date'], y_test_actual_transformer, y_pred_test_transformer, "Transformer: Test Set Predictions", transformer_plot_test_path)
        # results_payload["plot_urls"]["transformer_test_plot"] = f"/static/plots/{transformer_plot_test_filename}"

        # transformer_plot_future_filename = f"{run_id}_transformer_future_predictions.png"
        # transformer_plot_future_path = os.path.join(PLOTS_DIR, transformer_plot_future_filename)
        # plot_future_predictions(df_full['Date'], df_full['Close'], test_df['Date'], y_pred_test_transformer, future_dates_transformer, future_preds_transformer, "Transformer: Historical, Test & Future Predictions", transformer_plot_future_path)
        # results_payload["plot_urls"]["transformer_future_plot"] = f"/static/plots/{transformer_plot_future_filename}"


        transformer_plot_test_filename = f"{run_id}_transformer_test_predictions.png"
        transformer_plot_test_disk_path = os.path.join(PLOTS_DIR, transformer_plot_test_filename)
        plot_test_predictions(test_df['Date'], y_test_actual_transformer, y_pred_test_transformer, "Transformer: Test Set Predictions", transformer_plot_test_disk_path)
        results_payload["plot_urls"]["transformer_test_plot_url"] = f"{SERVED_GENERATED_CONTENT_URL_PREFIX}/plots/{transformer_plot_test_filename}"
        results_payload["plot_urls"]["transformer_test_plot_path"] = transformer_plot_test_disk_path

        transformer_plot_future_filename = f"{run_id}_transformer_future_predictions.png"
        transformer_plot_future_disk_path = os.path.join(PLOTS_DIR, transformer_plot_future_filename)
        plot_future_predictions(df_full['Date'], df_full['Close'], test_df['Date'], y_pred_test_transformer, future_dates_transformer, future_preds_transformer, "Transformer: Historical, Test & Future Predictions", transformer_plot_future_disk_path)
        results_payload["plot_urls"]["transformer_future_plot_url"] = f"{SERVED_GENERATED_CONTENT_URL_PREFIX}/plots/{transformer_plot_future_filename}"
        results_payload["plot_urls"]["transformer_future_plot_path"] = transformer_plot_future_disk_path
        



        # --- Polynomial Regression Model (No hyperparameter tuning in this example, but could be added) ---
        print(f"[{run_id}] Training Polynomial Regression model...")
        poly_model, poly_y_scaler, _, _ = train_polynomial_model(train_df.copy(), degree=POLYNOMIAL_DEGREE)
        
        y_test_actual_poly, y_pred_test_poly = evaluate_polynomial_model(poly_model, test_df.copy(), poly_y_scaler, original_min_date, degree=POLYNOMIAL_DEGREE)
        results_payload["polynomial_results"]["test_metrics"] = {"mae": float(mean_absolute_error(y_test_actual_poly, y_pred_test_poly)), "r2_score": float(r2_score(y_test_actual_poly, y_pred_test_poly))}
        
        future_preds_poly, future_dates_poly = predict_polynomial_future(poly_model, df_full, original_min_date, poly_y_scaler, POLYNOMIAL_DEGREE, N_FUTURE_DAYS_PREDICTION)
        results_payload["polynomial_results"]["future_predictions"] = {dt.strftime('%Y-%m-%d'): float(pred) for dt, pred in zip(future_dates_poly, future_preds_poly.flatten())}



        poly_plot_filename = f"{run_id}_polynomial_regression.png"
        poly_plot_path = os.path.join(PLOTS_DIR, poly_plot_filename)
        plot_polynomial_regression(
            df_full['Date'],      # Historical dates
            df_full['Close'],     # Historical close prices
            test_df['Date'],      # Test dates for overlay
            y_pred_test_poly,     # Polynomial predictions on test set
            future_dates_poly,    # Future dates
            future_preds_poly,    # Polynomial future predictions
            f"Polynomial Regression (Degree {POLYNOMIAL_DEGREE})",
            poly_plot_path
        )
        results_payload["plot_urls"]["polynomial_plot"] = f"/static/plots/{poly_plot_filename}"
        # Polynomial Plotting (same as before)
        # ...
        
        poly_plot_filename = f"{run_id}_polynomial_regression.png"
        poly_plot_disk_path = os.path.join(PLOTS_DIR, poly_plot_filename)
        plot_polynomial_regression(
            df_full['Date'], df_full['Close'], test_df['Date'], y_pred_test_poly,
            future_dates_poly, future_preds_poly,
            f"Polynomial Regression (Degree {POLYNOMIAL_DEGREE})", poly_plot_disk_path
        )
        results_payload["plot_urls"]["polynomial_plot_url"] = f"{SERVED_GENERATED_CONTENT_URL_PREFIX}/plots/{poly_plot_filename}"
        results_payload["plot_urls"]["polynomial_plot_path"] = poly_plot_disk_path
 
#         # --- Polynomial Regression Model ---
#         print(f"[{run_id}] Training Polynomial Regression model...")
#         poly_model, poly_scaler, _, _ = train_polynomial_model(train_df.copy(), degree=POLYNOMIAL_DEGREE)
#         print(f"[{run_id}] Polynomial Regression model training complete.")


#         y_test_actual_poly, y_pred_test_poly = evaluate_polynomial_model(
#             poly_model,          # 1. model
#             test_df.copy(),      # 2. test_df
#             poly_scaler,         # 3. y_scaler
#             original_min_date,   # 4. original_min_date_for_time_feature
#             degree=POLYNOMIAL_DEGREE # 5. degree (as keyword)
#         )
        
        
# #         # Polynomial Test Set Evaluation
# #         y_test_actual_poly, y_pred_test_poly = evaluate_polynomial_model(
# #             poly_model,
# #             test_df.copy(), # Use a copy
# #             poly_scaler,
# #             degree=POLYNOMIAL_DEGREE
# #         )
#         results_payload["polynomial_results"]["test_metrics"] = {
#             "mae": float(mean_absolute_error(y_test_actual_poly, y_pred_test_poly)),
#             "r2_score": float(r2_score(y_test_actual_poly, y_pred_test_poly))
#         }

#         future_preds_poly, future_dates_poly = predict_polynomial_future(
#             poly_model, df, original_min_date, poly_scaler, POLYNOMIAL_DEGREE, N_FUTURE_DAYS_PREDICTION # Pass original_min_date
#         )
        
#         # main.py

# # #         # Polynomial Future Predictions
# #         future_preds_poly, future_dates_poly = predict_polynomial_future(
# #             poly_model,
# #             df, # Full dataframe for last date
# #             poly_scaler,
# #             degree=POLYNOMIAL_DEGREE,
# #             n_future_days=N_FUTURE_DAYS_PREDICTION
# #         )
        
        
        
#         results_payload["polynomial_results"]["future_predictions"] = {
#             dt.strftime('%Y-%m-%d'): float(pred) for dt, pred in zip(future_dates_poly, future_preds_poly.flatten())
#         }

#         poly_plot_filename = f"{run_id}_polynomial_regression.png"
#         poly_plot_path = os.path.join(PLOTS_DIR, poly_plot_filename)
#         plot_polynomial_regression(df['Date'], df['Close'], test_df['Date'], y_pred_test_poly, future_dates_poly, future_preds_poly, f"Polynomial Regression (Degree {POLYNOMIAL_DEGREE})", poly_plot_path)
#         results_payload["plot_urls"]["polynomial_plot"] = f"/static/plots/{poly_plot_filename}"


        # --- Trading Suggestion (Decide which model to base this on, or an ensemble) ---
        # For now, let's keep it based on LSTM as an example
        if len(future_preds_lstm) > 0:
            # ... (your existing trading suggestion logic based on LSTM) ...
            predicted_tomorrow_price = future_preds_lstm.flatten()[0] # Using LSTM
            price_diff_percent = ((predicted_tomorrow_price - last_actual_close_price) / last_actual_close_price) * 100
            signal = "HOLD/NEUTRAL"
            # ... (rest of your logic)
            results_payload["trading_suggestion_tomorrow"] = {
                "signal": signal, "predicted_price_lstm": float(predicted_tomorrow_price), # Clarify source
                "last_actual_price": float(last_actual_close_price), "percentage_change": float(price_diff_percent),
                # ...
            }
        else:
            results_payload["trading_suggestion_tomorrow"] = {"signal": "N/A", "reason": "Not enough future predictions from primary model."}


        # --- Generate AI Analysis with Gemini (Pass new model results) ---
        print(f"[{run_id}] Generating AI analysis with Gemini...")
        gemini_prompt = generate_analysis_prompt(
            stock_symbol=results_payload["stock_symbol"],
            historical_data_df=df_full.tail(SEQUENCE_LENGTH + 10),
            lstm_test_actual=y_test_actual_lstm, lstm_test_pred=y_pred_test_lstm, lstm_test_dates=test_df['Date'],
            lstm_future_pred=future_preds_lstm, lstm_future_dates=future_dates_lstm,
            gru_test_actual=y_test_actual_gru, gru_test_pred=y_pred_test_gru, gru_test_dates=test_df['Date'],
            gru_future_pred=future_preds_gru, gru_future_dates=future_dates_gru,
            
            transformer_test_actual=y_test_actual_transformer, # NEW
            transformer_test_pred=y_pred_test_transformer,     # NEW
            transformer_test_dates=test_df['Date'],            # NEW
            transformer_future_pred=future_preds_transformer,  # NEW
            transformer_future_dates=future_dates_transformer, # NEW
            transformer_best_params=best_transformer_params,    # NEW
            poly_test_actual=y_test_actual_poly, poly_test_pred=y_pred_test_poly, poly_test_dates=test_df['Date'],
            poly_future_pred=future_preds_poly, poly_future_dates=future_dates_poly,
            trading_suggestion=results_payload["trading_suggestion_tomorrow"],
            lstm_best_params=best_lstm_params,
            gru_best_params=best_gru_params
        )
        
        # get_gemini_analysis now returns a Python dict (parsed JSON or error dict)
        ai_analysis_result_dict = await get_gemini_analysis(gemini_prompt) 
        
        # Check if it's an error dictionary or actual analysis
        if "error" in ai_analysis_result_dict:
            print(f"[{run_id}] AI analysis from Gemini reported an error: {ai_analysis_result_dict.get('error')}")
            print(f"[{run_id}] Raw response causing error (if available): {ai_analysis_result_dict.get('raw_response', ai_analysis_result_dict.get('extracted_string', 'N/A'))}")
            results_payload["ai_qualitative_analysis"] = ai_analysis_result_dict # Store the error dict
        else:
            # It's the parsed JSON dictionary, perform your data formatting
            print(f"[{run_id}] AI analysis successfully received and parsed.")
            
            # Your existing formatting logic:
            # Ensure ai_analysis_result_dict is indeed a dict before accessing keys
            if isinstance(ai_analysis_result_dict, dict):
                if ai_analysis_result_dict.get("analysisDate") == "YYYY-MM-DD (Today's Date)" or not ai_analysis_result_dict.get("analysisDate"):
                    ai_analysis_result_dict["analysisDate"] = datetime.now().strftime('%Y-%m-%d')
                
                data_summary = ai_analysis_result_dict.get("dataSummary")
                if isinstance(data_summary, dict): # Check if dataSummary is a dict
                    data_summary["lastActualClose"] = f"{last_actual_close_price:.2f}"
                    data_summary["lastActualDate"] = last_actual_date.strftime('%Y-%m-%d')
                else: # If dataSummary is not a dict or missing, initialize it
                    ai_analysis_result_dict["dataSummary"] = {
                        "lastActualClose": f"{last_actual_close_price:.2f}",
                        "lastActualDate": last_actual_date.strftime('%Y-%m-%d')
                    }
                results_payload["ai_qualitative_analysis"] = ai_analysis_result_dict
            else:
                # This case should ideally not be reached if get_gemini_analysis type hints are correct
                # and it always returns a dict. But as a fallback:
                print(f"[{run_id}] Warning: ai_analysis_result_dict was not a dictionary as expected after Gemini call.")
                results_payload["ai_qualitative_analysis"] = {"error": "Unexpected format from AI analysis step.", "received_data": str(ai_analysis_result_dict)}
        
        print(f"[{run_id}] AI analysis processing in main.py complete.")
        
        
        # --- Generate PDF Report (Update to include GRU results) ---
        print(f"[{run_id}] Generating PDF report...")
        
        disk_plot_paths_for_pdf = {
            "lstm_test_plot_path": results_payload["plot_urls"].get("lstm_test_plot_path"),
            "lstm_future_plot_path": results_payload["plot_urls"].get("lstm_future_plot_path"),
            "gru_test_plot_path": results_payload["plot_urls"].get("gru_test_plot_path"),
            "gru_future_plot_path": results_payload["plot_urls"].get("gru_future_plot_path"),
            "transformer_test_plot_path": results_payload["plot_urls"].get("transformer_test_plot_path"),
            "transformer_future_plot_path": results_payload["plot_urls"].get("transformer_future_plot_path"),
            "polynomial_plot_path": results_payload["plot_urls"].get("polynomial_plot_path"),
        }
        
        # Filter out any None paths
        disk_plot_paths_for_pdf = {k: v for k, v in disk_plot_paths_for_pdf.items() if v}

        # pdf_file_disk_path = generate_prediction_report(
        #     run_id, 
        #     results_payload, 
        #     disk_plot_paths_for_pdf, # Pass disk paths
        #     PDF_GENERATION_DIR # Pass the directory where PDF should be saved
        # )
        # pdf_filename_only = os.path.basename(pdf_file_disk_path)
        # results_payload["pdf_report_url"] = f"{SERVED_GENERATED_CONTENT_URL_PREFIX}/pdfs/{pdf_filename_only}" # URL for client
        # print(f"[{run_id}] PDF report generated: {pdf_file_disk_path}")

        pdf_file_disk_path = generate_prediction_report(
            run_id,
            results_payload,
            disk_plot_paths_for_pdf,
            PDF_GENERATION_DIR # Pass the correct directory for PDFs
        )
        pdf_filename_only = os.path.basename(pdf_file_disk_path)
        # URL for the PDF uses the SERVED_GENERATED_CONTENT_URL_PREFIX and the 'pdfs' subfolder
        results_payload["pdf_report_url"] = f"{SERVED_GENERATED_CONTENT_URL_PREFIX}/pdfs/{pdf_filename_only}"
        print(f"[{run_id}] PDF report generated: {pdf_file_disk_path}")


        results_payload["message"] = "Processing complete."
        background_tasks.add_task(cleanup_old_files, PLOTS_DIR)
        background_tasks.add_task(cleanup_old_files, PDF_DIR)
        
        
        
        # --- Send Email with PDF Attachment ---
        if user_email_to_send_to: # Check if an email address is provided
            print(f"[{run_id}] Preparing to send PDF report to {user_email_to_send_to}...")
            stock_symbol_for_email = results_payload.get("stock_symbol", "Requested_Stock") # Fallback

            email_subject = f"Stox Sight - Prediction Report for {stock_symbol_for_email} (Run ID: {run_id})" # CORRECTED
            email_body_html = f"""
            <html>
            <body>
                <p>Dear User,</p>
                <p>Please find attached the Stox Sight prediction report for <strong>{stock_symbol_for_email}</strong> (Run ID: {run_id}).</p> 
                <p>This report includes analysis from LSTM, GRU, Transformer, and Polynomial Regression models, along with AI-driven qualitative insights.</p>
                <p><strong>Key Sentiment:</strong> {results_payload.get("ai_qualitative_analysis", {}).get("overallSentiment", "N/A")}</p>
                <p><strong>Trading Suggestion for Tomorrow:</strong> {results_payload.get("trading_suggestion_tomorrow", {}).get("signal", "N/A")}</p>
                 <p>You can also download the report <a href="YOUR_FASTAPI_BASE_URL{results_payload['pdf_report_url']}">here</a> if your service is publicly accessible and configured to serve it.</p> 
                <br>
                <p><em>Disclaimer: This is an automated report for informational purposes only and does not constitute financial advice.</em></p>
                <p>Thank you,<br>Stox Sight Team</p>
            </body>
            </html>
            """
            # Replace YOUR_FASTAPI_BASE_URL with your actual deployed base URL if making download link work
            
            email_sent_successfully, email_message = await send_email_with_attachment(
                to_email=user_email_to_send_to,
                subject=email_subject,
                body_html=email_body_html,
                attachment_path=pdf_file_disk_path, # Use the disk path of the PDF
                attachment_filename=pdf_filename_only
            )
            if email_sent_successfully:
                results_payload["email_sent_status"] = "Success"
                print(f"[{run_id}] Email successfully queued/sent to {user_email_to_send_to}.")
            else:
                results_payload["email_sent_status"] = f"Failed: {email_message}"
                print(f"[{run_id}] Failed to send email: {email_message}")
        else:
            results_payload["email_sent_status"] = "Skipped: No recipient email provided."
            print(f"[{run_id}] Email sending skipped: No recipient email provided.")


        results_payload["message"] = "Processing complete."
        final_plot_urls = {k: v for k, v in results_payload["plot_urls"].items() if k.endswith("_url")}
        results_payload["plot_urls"] = final_plot_urls
        background_tasks.add_task(cleanup_old_files, PLOTS_DIR)
        background_tasks.add_task(cleanup_old_files, PDF_DIR)
        return results_payload

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[{run_id}] Error during processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# ... (StaticFiles mount, /reports/{pdf_filename} GET endpoint, and root GET endpoint remain the same) ...



# PDF serving endpoint needs to use PDF_GENERATION_DIR
@app.get(f"{SERVED_GENERATED_CONTENT_URL_PREFIX}/pdfs/{{pdf_filename}}")
async def get_pdf_report(pdf_filename: str):
    file_path = os.path.join(PDF_GENERATION_DIR, pdf_filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, media_type='application/pdf', filename=pdf_filename)
    else:
        print(f"PDF Get Error: File not found at {file_path}")
        raise HTTPException(status_code=404, detail="PDF report not found.")


from fastapi.staticfiles import StaticFiles
# app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static") # For plots

@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Prediction ML Service. POST to /train-predict/ with a CSV file."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)





















































































































































# # ./main.py



# # Data processing
# from utils.data_processing import preprocess_data, create_sequences_for_tuner # Add create_sequences_for_tuner

# from utils.hyperparameter_tuner import tune_model_hyperparameters # NEW IMPORT
# import os
# import shutil
# from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
# # Ensure FileResponse is imported if not already for serving PDF
# from fastapi.responses import JSONResponse, FileResponse
# from pydantic import BaseModel
# import pandas as pd
# import uuid
# from datetime import datetime, timedelta
# import io
# import pandas as pd
# import os
# # ... other imports ...
# import json # For handling potential JSON string from Gemini before parsing

# from utils.gemini_analyzer import generate_analysis_prompt, get_gemini_analysis
# from dotenv import load_dotenv # For .env file

# load_dotenv() # Load environment variables at the start of the app

# from sklearn.metrics import mean_absolute_error, r2_score

# from models.lstm_model import (
#     train_lstm_model,
#     predict_lstm_future,
#     evaluate_lstm_model
# )
# from models.polynomial_model import (
#     train_polynomial_model,
#     predict_polynomial_future,
#     evaluate_polynomial_model
# )
# from utils.data_processing import preprocess_data
# from utils.plotting import (
#     plot_test_predictions,
#     plot_future_predictions,
#     plot_polynomial_regression
# )


# from models.gru_model import ( # NEW IMPORTS
#     train_gru_model_with_params,
#     evaluate_gru_model,
#     predict_gru_future
# )

# # --- NEW IMPORT ---
# from utils.pdf_generator import generate_prediction_report, PDF_DIR # Import PDF_DIR

# # Configuration
# SEQUENCE_LENGTH = 70
# N_FUTURE_DAYS_PREDICTION =30
# LSTM_EPOCHS = 50 # You had 10, changed back to 3 as per original request
# POLYNOMIAL_DEGREE = 3 # You had 5, changed back to 3
# BUY_SELL_THRESHOLD_PERCENT = 0.5 # e.g., 0.5% change for buy/sell signal






# STATIC_DIR = "static"
# PLOTS_DIR = os.path.join(STATIC_DIR, "plots")
# # PDF_DIR is now imported from pdf_generator
# os.makedirs(PLOTS_DIR, exist_ok=True)
# os.makedirs(PDF_DIR, exist_ok=True) # Ensure PDF_DIR from pdf_generator also exists

# app = FastAPI(title="Stock Prediction ML Service")

# def cleanup_old_files(directory, max_age_minutes=60): # Generic cleanup
#     now = datetime.now()
#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)
#         try:
#             if os.path.isfile(file_path): # Ensure it's a file
#                 file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
#                 if (now - file_mod_time) > timedelta(minutes=max_age_minutes):
#                     os.remove(file_path)
#                     print(f"Cleaned up old file: {filename} from {directory}")
#         except Exception as e:
#             print(f"Error cleaning up file {filename} from {directory}: {e}")



# class ModelResultDetail(BaseModel):
#     best_params: dict = {}
#     test_metrics: dict = {}
#     future_predictions: dict = {}
#     training_loss: list = [] # For deep learning models



# # main.py
# class TrainResponse(BaseModel):
#     message: str
#     run_id: str
#     lstm_results: dict
#     polynomial_results: dict
#     trading_suggestion_tomorrow: dict
#     ai_qualitative_analysis: dict | str # Can be dict or error string from Gemini
#     plot_urls: dict
#     pdf_report_url: str

# @app.post("/train-predict/", response_model=TrainResponse)
# async def train_and_predict_models(
#     background_tasks: BackgroundTasks,
#     csv_file: UploadFile = File(...)
# ):
#     run_id = str(uuid.uuid4())
#     # Initialize with new fields
#     results_payload = { # This will be used to build the JSON response and pass to PDF generator
#         "message": "Processing started.",
#         "run_id": run_id,
#         "csv_filename": csv_file.filename, # Store original filename
#         "lstm_results": {},
#         "polynomial_results": {},
#         "trading_suggestion_tomorrow": {},
#         "plot_urls": {},
#         "pdf_report_url": ""
#     }

#     try:
#         contents = await csv_file.read()
#         df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

#         if 'Date' not in df.columns or 'Close' not in df.columns:
#             raise HTTPException(status_code=400, detail="CSV must contain 'Date' and 'Close' columns.")

#         df['Date'] = pd.to_datetime(df['Date'])
#         df.sort_values(by='Date', inplace=True)
#         df.reset_index(drop=True, inplace=True)
#         df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
#         df.dropna(subset=['Close'], inplace=True)

#         if len(df) < SEQUENCE_LENGTH + N_FUTURE_DAYS_PREDICTION + 10: # Adjusted minimum data
#              raise HTTPException(status_code=400, detail=f"Not enough data. Need at least {SEQUENCE_LENGTH + N_FUTURE_DAYS_PREDICTION + 10} rows after cleaning.")

#         original_min_date = df['Date'].min() # For consistent Polynomial 'Time' feature
#         last_actual_close_price = df['Close'].iloc[-1]
#         last_actual_date = df['Date'].iloc[-1]

#         train_df, test_df, scaler = preprocess_data(df.copy(), sequence_length=SEQUENCE_LENGTH)

#         # --- LSTM Model ---
#         print(f"[{run_id}] Training LSTM model...")
#         lstm_model, lstm_history = train_lstm_model(train_df, scaler, SEQUENCE_LENGTH, LSTM_EPOCHS)
#         results_payload["lstm_results"]["training_loss"] = lstm_history.history.get('loss', []) # Store loss
#         print(f"[{run_id}] LSTM model training complete.")

#         y_test_actual_lstm, y_pred_test_lstm, _ = evaluate_lstm_model(
#             lstm_model, test_df, train_df, scaler, SEQUENCE_LENGTH
#         )
#         results_payload["lstm_results"]["test_metrics"] = {
#             "mae": float(mean_absolute_error(y_test_actual_lstm, y_pred_test_lstm)),
#             "r2_score": float(r2_score(y_test_actual_lstm, y_pred_test_lstm))
#         }

#         future_preds_lstm, future_dates_lstm = predict_lstm_future(
#             lstm_model, df, scaler, SEQUENCE_LENGTH, N_FUTURE_DAYS_PREDICTION
#         )
#         results_payload["lstm_results"]["future_predictions"] = {
#             dt.strftime('%Y-%m-%d'): float(pred) for dt, pred in zip(future_dates_lstm, future_preds_lstm.flatten())
#         }
#         # --- Trading Suggestion (based on LSTM's next day prediction) ---
#         if len(future_preds_lstm) > 0:
#             predicted_tomorrow_price_lstm = future_preds_lstm.flatten()[0]
#             price_diff_percent = ((predicted_tomorrow_price_lstm - last_actual_close_price) / last_actual_close_price) * 100
#             signal = "HOLD/NEUTRAL"
#             reason = f"Predicted LSTM price for tomorrow: {predicted_tomorrow_price_lstm:.2f}. Last close: {last_actual_close_price:.2f} on {last_actual_date.strftime('%Y-%m-%d')}."
            
#             if price_diff_percent > BUY_SELL_THRESHOLD_PERCENT:
#                 signal = "BUY"
#                 reason += f" Change: +{price_diff_percent:.2f}% (>{BUY_SELL_THRESHOLD_PERCENT}%)"
#             elif price_diff_percent < -BUY_SELL_THRESHOLD_PERCENT:
#                 signal = "SELL"
#                 reason += f" Change: {price_diff_percent:.2f}% (<{-BUY_SELL_THRESHOLD_PERCENT}%)"
#             else:
#                  reason += f" Change: {price_diff_percent:.2f}% (within +/-{BUY_SELL_THRESHOLD_PERCENT}%)"

#             results_payload["trading_suggestion_tomorrow"] = {
#                 "signal": signal,
#                 "predicted_price_lstm": float(predicted_tomorrow_price_lstm),
#                 "last_actual_price": float(last_actual_close_price),
#                 "percentage_change": float(price_diff_percent),
#                 "reason": reason
#             }
#         else:
#             results_payload["trading_suggestion_tomorrow"] = {"signal": "N/A", "reason": "Not enough future predictions from LSTM."}


#         # LSTM Plotting
#         lstm_plot_test_filename = f"{run_id}_lstm_test_predictions.png"
#         lstm_plot_test_path = os.path.join(PLOTS_DIR, lstm_plot_test_filename)
#         plot_test_predictions(test_df['Date'], y_test_actual_lstm, y_pred_test_lstm, "LSTM: Test Set Predictions", lstm_plot_test_path)
#         results_payload["plot_urls"]["lstm_test_plot"] = f"/static/plots/{lstm_plot_test_filename}"

#         lstm_plot_future_filename = f"{run_id}_lstm_future_predictions.png"
#         lstm_plot_future_path = os.path.join(PLOTS_DIR, lstm_plot_future_filename)
#         plot_future_predictions(df['Date'], df['Close'], test_df['Date'], y_pred_test_lstm, future_dates_lstm, future_preds_lstm, "LSTM: Historical, Test & Future Predictions", lstm_plot_future_path)
#         results_payload["plot_urls"]["lstm_future_plot"] = f"/static/plots/{lstm_plot_future_filename}"


#         # --- Polynomial Regression Model ---
#         print(f"[{run_id}] Training Polynomial Regression model...")
#         poly_model, poly_scaler, _, _ = train_polynomial_model(train_df.copy(), degree=POLYNOMIAL_DEGREE)
#         print(f"[{run_id}] Polynomial Regression model training complete.")


#         y_test_actual_poly, y_pred_test_poly = evaluate_polynomial_model(
#             poly_model,          # 1. model
#             test_df.copy(),      # 2. test_df
#             poly_scaler,         # 3. y_scaler
#             original_min_date,   # 4. original_min_date_for_time_feature
#             degree=POLYNOMIAL_DEGREE # 5. degree (as keyword)
#         )
        
        
# #         # Polynomial Test Set Evaluation
# #         y_test_actual_poly, y_pred_test_poly = evaluate_polynomial_model(
# #             poly_model,
# #             test_df.copy(), # Use a copy
# #             poly_scaler,
# #             degree=POLYNOMIAL_DEGREE
# #         )
#         results_payload["polynomial_results"]["test_metrics"] = {
#             "mae": float(mean_absolute_error(y_test_actual_poly, y_pred_test_poly)),
#             "r2_score": float(r2_score(y_test_actual_poly, y_pred_test_poly))
#         }

#         future_preds_poly, future_dates_poly = predict_polynomial_future(
#             poly_model, df, original_min_date, poly_scaler, POLYNOMIAL_DEGREE, N_FUTURE_DAYS_PREDICTION # Pass original_min_date
#         )
        
#         # main.py

# # #         # Polynomial Future Predictions
# #         future_preds_poly, future_dates_poly = predict_polynomial_future(
# #             poly_model,
# #             df, # Full dataframe for last date
# #             poly_scaler,
# #             degree=POLYNOMIAL_DEGREE,
# #             n_future_days=N_FUTURE_DAYS_PREDICTION
# #         )
        
        
        
#         results_payload["polynomial_results"]["future_predictions"] = {
#             dt.strftime('%Y-%m-%d'): float(pred) for dt, pred in zip(future_dates_poly, future_preds_poly.flatten())
#         }

#         poly_plot_filename = f"{run_id}_polynomial_regression.png"
#         poly_plot_path = os.path.join(PLOTS_DIR, poly_plot_filename)
#         plot_polynomial_regression(df['Date'], df['Close'], test_df['Date'], y_pred_test_poly, future_dates_poly, future_preds_poly, f"Polynomial Regression (Degree {POLYNOMIAL_DEGREE})", poly_plot_path)
#         results_payload["plot_urls"]["polynomial_plot"] = f"/static/plots/{poly_plot_filename}"

#         # --- Generate PDF Report ---
#         print(f"[{run_id}] Generating PDF report...")
#         pdf_file_path = generate_prediction_report(run_id, results_payload, results_payload["plot_urls"])
#         pdf_filename_only = os.path.basename(pdf_file_path)
#         results_payload["pdf_report_url"] = f"/reports/{pdf_filename_only}" # URL to download PDF
#         print(f"[{run_id}] PDF report generated: {pdf_file_path}")

#         results_payload["message"] = "Processing complete."
#         background_tasks.add_task(cleanup_old_files, PLOTS_DIR)
#         background_tasks.add_task(cleanup_old_files, PDF_DIR) # Cleanup PDFs too
        
        

#         # --- Generate AI Analysis with Gemini (THIS SECTION IS CORRECTLY PLACED) ---
#         print(f"[{run_id}] Generating AI analysis with Gemini...")
#         gemini_prompt = generate_analysis_prompt(
#             stock_symbol=df.get('Symbol', ['UNKNOWN_SYMBOL'])[0] if 'Symbol' in df.columns else results_payload["csv_filename"].split('_')[0],
#             historical_data_df=df.tail(SEQUENCE_LENGTH + 10),
#             lstm_test_actual=y_test_actual_lstm,
#             lstm_test_pred=y_pred_test_lstm,
#             lstm_test_dates=test_df['Date'],
#             lstm_future_pred=future_preds_lstm,
#             lstm_future_dates=future_dates_lstm,
#             poly_test_actual=y_test_actual_poly,
#             poly_test_pred=y_pred_test_poly,
#             poly_test_dates=test_df['Date'],
#             poly_future_pred=future_preds_poly,
#             poly_future_dates=future_dates_poly,
#             trading_suggestion=results_payload["trading_suggestion_tomorrow"]
#         )
        
#         ai_analysis_json = await get_gemini_analysis(gemini_prompt)
        
#         if isinstance(ai_analysis_json, dict) and ai_analysis_json.get("analysisDate") == "YYYY-MM-DD (Today's Date)":
#             ai_analysis_json["analysisDate"] = datetime.now().strftime('%Y-%m-%d')
#         if isinstance(ai_analysis_json, dict) and "dataSummary" in ai_analysis_json:
#              ai_analysis_json["dataSummary"]["lastActualClose"] = f"{last_actual_close_price:.2f}"
#              ai_analysis_json["dataSummary"]["lastActualDate"] = last_actual_date.strftime('%Y-%m-%d')

#         results_payload["ai_qualitative_analysis"] = ai_analysis_json
#         print(f"[{run_id}] AI analysis generated.")

#         # --- Generate PDF Report (NOW THIS IS THE ONLY CALL, AND IT'S CORRECTLY PLACED) ---
#         print(f"[{run_id}] Generating PDF report...")
#         pdf_file_path = generate_prediction_report(run_id, results_payload, results_payload["plot_urls"])
#         pdf_filename_only = os.path.basename(pdf_file_path)
#         results_payload["pdf_report_url"] = f"/reports/{pdf_filename_only}"
#         print(f"[{run_id}] PDF report generated: {pdf_file_path}")

#         # --- Finalize results and return ---
#         results_payload["message"] = "Processing complete."
#         background_tasks.add_task(cleanup_old_files, PLOTS_DIR)
#         background_tasks.add_task(cleanup_old_files, PDF_DIR)
        
#         return results_payload # NOW RETURN AFTER ALL STEPS

#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         print(f"[{run_id}] Error during processing: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# # --- Endpoint to serve generated PDFs ---
# @app.get("/reports/{pdf_filename}")
# async def get_pdf_report(pdf_filename: str):
#     file_path = os.path.join(PDF_DIR, pdf_filename)
#     if os.path.exists(file_path):
#         return FileResponse(path=file_path, media_type='application/pdf', filename=pdf_filename)
#     else:
#         raise HTTPException(status_code=404, detail="PDF report not found.")

# from fastapi.staticfiles import StaticFiles
# app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static") # For plots

# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Stock Prediction ML Service. POST to /train-predict/ with a CSV file."}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
























































































































































# # import os
# # import shutil
# # from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
# # from fastapi.responses import JSONResponse, FileResponse
# # from pydantic import BaseModel
# # import pandas as pd
# # import uuid # For unique filenames/run IDs
# # from datetime import datetime, timedelta
# # import io

# # # --- ADD THIS IMPORT ---
# # from sklearn.metrics import mean_absolute_error, r2_score
# # # -----------------------

# # # Import your model and utility functions
# # from models.lstm_model import (
# #     train_lstm_model,
# #     predict_lstm_future,
# #     evaluate_lstm_model
# # )
# # from models.polynomial_model import (
# #     train_polynomial_model,
# #     predict_polynomial_future,
# #     evaluate_polynomial_model
# # )
# # from utils.data_processing import preprocess_data
# # from utils.plotting import (
# #     plot_test_predictions,
# #     plot_future_predictions,
# #     plot_polynomial_regression
# # )

# # # Configuration
# # SEQUENCE_LENGTH = 100 # For LSTM
# # N_FUTURE_DAYS_PREDICTION = 30
# # LSTM_EPOCHS = 10 # As requested
# # POLYNOMIAL_DEGREE = 5 # Example, can be tuned or passed as param

# # # Ensure static directories exist
# # STATIC_DIR = "static"
# # PLOTS_DIR = os.path.join(STATIC_DIR, "plots")
# # os.makedirs(PLOTS_DIR, exist_ok=True)

# # app = FastAPI(title="Stock Prediction ML Service")

# # # --- Helper Function to clean up old plots ---
# # def cleanup_old_plots(max_age_minutes=60):
# #     now = datetime.now()
# #     for filename in os.listdir(PLOTS_DIR):
# #         file_path = os.path.join(PLOTS_DIR, filename)
# #         try:
# #             file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
# #             if (now - file_mod_time) > timedelta(minutes=max_age_minutes):
# #                 os.remove(file_path)
# #                 print(f"Cleaned up old plot: {filename}")
# #         except Exception as e:
# #             print(f"Error cleaning up plot {filename}: {e}")


# # class TrainResponse(BaseModel):
# #     message: str
# #     run_id: str
# #     lstm_results: dict
# #     polynomial_results: dict
# #     plot_urls: dict

# # @app.post("/train-predict/", response_model=TrainResponse)
# # async def train_and_predict_models(
# #     background_tasks: BackgroundTasks,
# #     csv_file: UploadFile = File(...)
# # ):
# #     run_id = str(uuid.uuid4())
# #     results = {
# #         "message": "Processing started.",
# #         "run_id": run_id,
# #         "lstm_results": {},
# #         "polynomial_results": {},
# #         "plot_urls": {}
# #     }

# #     try:
# #         # 1. Read and Preprocess CSV
# #         contents = await csv_file.read()
# #         df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

# #         # Ensure 'Date' is datetime and not index, 'Close' is numeric
# #         if 'Date' not in df.columns or 'Close' not in df.columns:
# #             raise HTTPException(status_code=400, detail="CSV must contain 'Date' and 'Close' columns.")

# #         df['Date'] = pd.to_datetime(df['Date'])
# #         df.sort_values(by='Date', inplace=True)
# #         df.reset_index(drop=True, inplace=True) # Ensure Date is not index
# #         df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
# #         df.dropna(subset=['Close'], inplace=True)

# #         if len(df) < SEQUENCE_LENGTH + 10: # Minimum data for train/test/future
# #              raise HTTPException(status_code=400, detail=f"Not enough data. Need at least {SEQUENCE_LENGTH + 10} rows after cleaning.")

# #         train_df, test_df, scaler = preprocess_data(df.copy(), sequence_length=SEQUENCE_LENGTH) # Use a copy

# #         # --- LSTM Model ---
# #         print(f"[{run_id}] Training LSTM model...")
# #         lstm_model, lstm_history = train_lstm_model(
# #             train_df,
# #             scaler,
# #             sequence_length=SEQUENCE_LENGTH,
# #             epochs=LSTM_EPOCHS
# #         )
# #         print(f"[{run_id}] LSTM model training complete.")

# #         # LSTM Test Set Evaluation
# #         y_test_actual_lstm, y_pred_test_lstm, x_test_lstm = evaluate_lstm_model(
# #             lstm_model,
# #             test_df,
# #             train_df, # Pass train_df for last sequence
# #             scaler,
# #             sequence_length=SEQUENCE_LENGTH
# #         )
# #         results["lstm_results"]["test_metrics"] = {
# #             "mae": float(mean_absolute_error(y_test_actual_lstm, y_pred_test_lstm)),
# #             "r2_score": float(r2_score(y_test_actual_lstm, y_pred_test_lstm))
# #         }

# #         # LSTM Future Predictions
# #         future_preds_lstm, future_dates_lstm = predict_lstm_future(
# #             lstm_model,
# #             df, # Full dataframe for last sequence
# #             scaler,
# #             sequence_length=SEQUENCE_LENGTH,
# #             n_future_days=N_FUTURE_DAYS_PREDICTION
# #         )
# #         results["lstm_results"]["future_predictions"] = {
# #             str(date.date()): float(pred) for date, pred in zip(future_dates_lstm, future_preds_lstm.flatten())
# #         }

# #         # LSTM Plotting
# #         lstm_plot_test_path = os.path.join(PLOTS_DIR, f"{run_id}_lstm_test_predictions.png")
# #         plot_test_predictions(
# #             test_df['Date'],
# #             y_test_actual_lstm,
# #             y_pred_test_lstm,
# #             title="LSTM: Test Set Predictions",
# #             save_path=lstm_plot_test_path
# #         )
# #         results["plot_urls"]["lstm_test_plot"] = f"/static/plots/{run_id}_lstm_test_predictions.png"

# #         lstm_plot_future_path = os.path.join(PLOTS_DIR, f"{run_id}_lstm_future_predictions.png")
# #         plot_future_predictions(
# #             df['Date'],
# #             df['Close'],
# #             test_df['Date'], # Dates for test predictions
# #             y_pred_test_lstm, # Test predictions
# #             future_dates_lstm,
# #             future_preds_lstm,
# #             title="LSTM: Historical, Test & Future Predictions",
# #             save_path=lstm_plot_future_path
# #         )
# #         results["plot_urls"]["lstm_future_plot"] = f"/static/plots/{run_id}_lstm_future_predictions.png"


# #         # --- Polynomial Regression Model ---
# #         # print(f"[{run_id}] Training Polynomial Regression model...")
# #         # poly_model, poly_scaler, time_feature_train, X_poly_train, y_poly_train = train_polynomial_model(
# #         #     train_df.copy(), # Use a copy
# #         #     degree=POLYNOMIAL_DEGREE
# #         # )
        
        
# #         # NEW corrected line in main.py:
# #         poly_model, poly_scaler, X_train_poly_time_feature, y_train_poly_scaled = train_polynomial_model(
# #             train_df.copy(),
# #             degree=POLYNOMIAL_DEGREE
# #         )
# #         # You can then use X_train_poly_time_feature if you need the 'Time' feature DataFrame (which is X_train from the function)
# #         # and y_train_poly_scaled is the scaled target variable from training.
# #         # For the current logic, these returned training features/targets (X_train_poly_time_feature, y_train_poly_scaled)
# #         # are not directly used later in the main.py flow, but it's good to unpack correctly.


# #         print(f"[{run_id}] Polynomial Regression model training complete.")

# #         # Polynomial Test Set Evaluation
# #         y_test_actual_poly, y_pred_test_poly = evaluate_polynomial_model(
# #             poly_model,
# #             test_df.copy(), # Use a copy
# #             poly_scaler,
# #             degree=POLYNOMIAL_DEGREE
# #         )
# #         results["polynomial_results"]["test_metrics"] = {
# #             "mae": float(mean_absolute_error(y_test_actual_poly, y_pred_test_poly)),
# #             "r2_score": float(r2_score(y_test_actual_poly, y_pred_test_poly))
# #         }

# #         # Polynomial Future Predictions
# #         future_preds_poly, future_dates_poly = predict_polynomial_future(
# #             poly_model,
# #             df, # Full dataframe for last date
# #             poly_scaler,
# #             degree=POLYNOMIAL_DEGREE,
# #             n_future_days=N_FUTURE_DAYS_PREDICTION
# #         )
# #         results["polynomial_results"]["future_predictions"] = {
# #             str(date.date()): float(pred) for date, pred in zip(future_dates_poly, future_preds_poly.flatten())
# #         }

# #         # Polynomial Plotting
# #         poly_plot_path = os.path.join(PLOTS_DIR, f"{run_id}_polynomial_regression.png")
# #         plot_polynomial_regression(
# #             df['Date'], df['Close'], # All historical data
# #             test_df['Date'], y_pred_test_poly, # Test predictions
# #             future_dates_poly, future_preds_poly,
# #             title=f"Polynomial Regression (Degree {POLYNOMIAL_DEGREE})",
# #             save_path=poly_plot_path
# #         )
# #         results["plot_urls"]["polynomial_plot"] = f"/static/plots/{run_id}_polynomial_regression.png"


# #         results["message"] = "Processing complete."
# #         background_tasks.add_task(cleanup_old_plots) # Schedule cleanup
# #         return results

# #     except HTTPException as e:
# #         raise e # Re-raise HTTP exceptions
# #     except Exception as e:
# #         print(f"[{run_id}] Error during processing: {e}")
# #         import traceback
# #         traceback.print_exc()
# #         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# # # --- Mount static directory to serve plots ---
# # from fastapi.staticfiles import StaticFiles
# # app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# # @app.get("/")
# # def read_root():
# #     return {"message": "Welcome to the Stock Prediction ML Service. POST to /train-predict/ with a CSV file."}

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8000)