import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
# Ensure FileResponse is imported if not already for serving PDF
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import pandas as pd
import uuid
from datetime import datetime, timedelta
import io

from sklearn.metrics import mean_absolute_error, r2_score

from models.lstm_model import (
    train_lstm_model,
    predict_lstm_future,
    evaluate_lstm_model
)
from models.polynomial_model import (
    train_polynomial_model,
    predict_polynomial_future,
    evaluate_polynomial_model
)
from utils.data_processing import preprocess_data
from utils.plotting import (
    plot_test_predictions,
    plot_future_predictions,
    plot_polynomial_regression
)
# --- NEW IMPORT ---
from utils.pdf_generator import generate_prediction_report, PDF_DIR # Import PDF_DIR

# Configuration
SEQUENCE_LENGTH = 100
N_FUTURE_DAYS_PREDICTION = 30
LSTM_EPOCHS = 3 # You had 10, changed back to 3 as per original request
POLYNOMIAL_DEGREE = 3 # You had 5, changed back to 3
BUY_SELL_THRESHOLD_PERCENT = 0.5 # e.g., 0.5% change for buy/sell signal

STATIC_DIR = "static"
PLOTS_DIR = os.path.join(STATIC_DIR, "plots")
# PDF_DIR is now imported from pdf_generator
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True) # Ensure PDF_DIR from pdf_generator also exists

app = FastAPI(title="Stock Prediction ML Service")

def cleanup_old_files(directory, max_age_minutes=60): # Generic cleanup
    now = datetime.now()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path): # Ensure it's a file
                file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if (now - file_mod_time) > timedelta(minutes=max_age_minutes):
                    os.remove(file_path)
                    print(f"Cleaned up old file: {filename} from {directory}")
        except Exception as e:
            print(f"Error cleaning up file {filename} from {directory}: {e}")

class TrainResponse(BaseModel):
    message: str
    run_id: str
    lstm_results: dict
    polynomial_results: dict
    trading_suggestion_tomorrow: dict # Added
    plot_urls: dict
    pdf_report_url: str # Added

@app.post("/train-predict/", response_model=TrainResponse)
async def train_and_predict_models(
    background_tasks: BackgroundTasks,
    csv_file: UploadFile = File(...)
):
    run_id = str(uuid.uuid4())
    # Initialize with new fields
    results_payload = { # This will be used to build the JSON response and pass to PDF generator
        "message": "Processing started.",
        "run_id": run_id,
        "csv_filename": csv_file.filename, # Store original filename
        "lstm_results": {},
        "polynomial_results": {},
        "trading_suggestion_tomorrow": {},
        "plot_urls": {},
        "pdf_report_url": ""
    }

    try:
        contents = await csv_file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        if 'Date' not in df.columns or 'Close' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'Date' and 'Close' columns.")

        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values(by='Date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)

        if len(df) < SEQUENCE_LENGTH + N_FUTURE_DAYS_PREDICTION + 10: # Adjusted minimum data
             raise HTTPException(status_code=400, detail=f"Not enough data. Need at least {SEQUENCE_LENGTH + N_FUTURE_DAYS_PREDICTION + 10} rows after cleaning.")

        original_min_date = df['Date'].min() # For consistent Polynomial 'Time' feature
        last_actual_close_price = df['Close'].iloc[-1]
        last_actual_date = df['Date'].iloc[-1]

        train_df, test_df, scaler = preprocess_data(df.copy(), sequence_length=SEQUENCE_LENGTH)

        # --- LSTM Model ---
        print(f"[{run_id}] Training LSTM model...")
        lstm_model, lstm_history = train_lstm_model(train_df, scaler, SEQUENCE_LENGTH, LSTM_EPOCHS)
        results_payload["lstm_results"]["training_loss"] = lstm_history.history.get('loss', []) # Store loss
        print(f"[{run_id}] LSTM model training complete.")

        y_test_actual_lstm, y_pred_test_lstm, _ = evaluate_lstm_model(
            lstm_model, test_df, train_df, scaler, SEQUENCE_LENGTH
        )
        results_payload["lstm_results"]["test_metrics"] = {
            "mae": float(mean_absolute_error(y_test_actual_lstm, y_pred_test_lstm)),
            "r2_score": float(r2_score(y_test_actual_lstm, y_pred_test_lstm))
        }

        future_preds_lstm, future_dates_lstm = predict_lstm_future(
            lstm_model, df, scaler, SEQUENCE_LENGTH, N_FUTURE_DAYS_PREDICTION
        )
        results_payload["lstm_results"]["future_predictions"] = {
            dt.strftime('%Y-%m-%d'): float(pred) for dt, pred in zip(future_dates_lstm, future_preds_lstm.flatten())
        }
        # --- Trading Suggestion (based on LSTM's next day prediction) ---
        if len(future_preds_lstm) > 0:
            predicted_tomorrow_price_lstm = future_preds_lstm.flatten()[0]
            price_diff_percent = ((predicted_tomorrow_price_lstm - last_actual_close_price) / last_actual_close_price) * 100
            signal = "HOLD/NEUTRAL"
            reason = f"Predicted LSTM price for tomorrow: {predicted_tomorrow_price_lstm:.2f}. Last close: {last_actual_close_price:.2f} on {last_actual_date.strftime('%Y-%m-%d')}."
            
            if price_diff_percent > BUY_SELL_THRESHOLD_PERCENT:
                signal = "BUY"
                reason += f" Change: +{price_diff_percent:.2f}% (>{BUY_SELL_THRESHOLD_PERCENT}%)"
            elif price_diff_percent < -BUY_SELL_THRESHOLD_PERCENT:
                signal = "SELL"
                reason += f" Change: {price_diff_percent:.2f}% (<{-BUY_SELL_THRESHOLD_PERCENT}%)"
            else:
                 reason += f" Change: {price_diff_percent:.2f}% (within +/-{BUY_SELL_THRESHOLD_PERCENT}%)"

            results_payload["trading_suggestion_tomorrow"] = {
                "signal": signal,
                "predicted_price_lstm": float(predicted_tomorrow_price_lstm),
                "last_actual_price": float(last_actual_close_price),
                "percentage_change": float(price_diff_percent),
                "reason": reason
            }
        else:
            results_payload["trading_suggestion_tomorrow"] = {"signal": "N/A", "reason": "Not enough future predictions from LSTM."}


        # LSTM Plotting
        lstm_plot_test_filename = f"{run_id}_lstm_test_predictions.png"
        lstm_plot_test_path = os.path.join(PLOTS_DIR, lstm_plot_test_filename)
        plot_test_predictions(test_df['Date'], y_test_actual_lstm, y_pred_test_lstm, "LSTM: Test Set Predictions", lstm_plot_test_path)
        results_payload["plot_urls"]["lstm_test_plot"] = f"/static/plots/{lstm_plot_test_filename}"

        lstm_plot_future_filename = f"{run_id}_lstm_future_predictions.png"
        lstm_plot_future_path = os.path.join(PLOTS_DIR, lstm_plot_future_filename)
        plot_future_predictions(df['Date'], df['Close'], test_df['Date'], y_pred_test_lstm, future_dates_lstm, future_preds_lstm, "LSTM: Historical, Test & Future Predictions", lstm_plot_future_path)
        results_payload["plot_urls"]["lstm_future_plot"] = f"/static/plots/{lstm_plot_future_filename}"


        # --- Polynomial Regression Model ---
        print(f"[{run_id}] Training Polynomial Regression model...")
        poly_model, poly_scaler, _, _ = train_polynomial_model(train_df.copy(), degree=POLYNOMIAL_DEGREE)
        print(f"[{run_id}] Polynomial Regression model training complete.")


        y_test_actual_poly, y_pred_test_poly = evaluate_polynomial_model(
            poly_model,          # 1. model
            test_df.copy(),      # 2. test_df
            poly_scaler,         # 3. y_scaler
            # original_min_date,   # 4. original_min_date_for_time_feature
            degree=POLYNOMIAL_DEGREE # 5. degree (as keyword)
        )
        
        
#         # Polynomial Test Set Evaluation
#         y_test_actual_poly, y_pred_test_poly = evaluate_polynomial_model(
#             poly_model,
#             test_df.copy(), # Use a copy
#             poly_scaler,
#             degree=POLYNOMIAL_DEGREE
#         )
        results_payload["polynomial_results"]["test_metrics"] = {
            "mae": float(mean_absolute_error(y_test_actual_poly, y_pred_test_poly)),
            "r2_score": float(r2_score(y_test_actual_poly, y_pred_test_poly))
        }

        # future_preds_poly, future_dates_poly = predict_polynomial_future(
        #     poly_model, df, original_min_date, poly_scaler, POLYNOMIAL_DEGREE, N_FUTURE_DAYS_PREDICTION # Pass original_min_date
        # )
        
        # main.py

#         # Polynomial Future Predictions
        future_preds_poly, future_dates_poly = predict_polynomial_future(
            poly_model,
            df, # Full dataframe for last date
            poly_scaler,
            degree=POLYNOMIAL_DEGREE,
            n_future_days=N_FUTURE_DAYS_PREDICTION
        )
        
        
        
        results_payload["polynomial_results"]["future_predictions"] = {
            dt.strftime('%Y-%m-%d'): float(pred) for dt, pred in zip(future_dates_poly, future_preds_poly.flatten())
        }

        poly_plot_filename = f"{run_id}_polynomial_regression.png"
        poly_plot_path = os.path.join(PLOTS_DIR, poly_plot_filename)
        plot_polynomial_regression(df['Date'], df['Close'], test_df['Date'], y_pred_test_poly, future_dates_poly, future_preds_poly, f"Polynomial Regression (Degree {POLYNOMIAL_DEGREE})", poly_plot_path)
        results_payload["plot_urls"]["polynomial_plot"] = f"/static/plots/{poly_plot_filename}"

        # --- Generate PDF Report ---
        print(f"[{run_id}] Generating PDF report...")
        pdf_file_path = generate_prediction_report(run_id, results_payload, results_payload["plot_urls"])
        pdf_filename_only = os.path.basename(pdf_file_path)
        results_payload["pdf_report_url"] = f"/reports/{pdf_filename_only}" # URL to download PDF
        print(f"[{run_id}] PDF report generated: {pdf_file_path}")

        results_payload["message"] = "Processing complete."
        background_tasks.add_task(cleanup_old_files, PLOTS_DIR)
        background_tasks.add_task(cleanup_old_files, PDF_DIR) # Cleanup PDFs too
        
        return results_payload # Return the full payload

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[{run_id}] Error during processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# --- Endpoint to serve generated PDFs ---
@app.get("/reports/{pdf_filename}")
async def get_pdf_report(pdf_filename: str):
    file_path = os.path.join(PDF_DIR, pdf_filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, media_type='application/pdf', filename=pdf_filename)
    else:
        raise HTTPException(status_code=404, detail="PDF report not found.")

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static") # For plots

@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Prediction ML Service. POST to /train-predict/ with a CSV file."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


















































































































































# import os
# import shutil
# from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
# from fastapi.responses import JSONResponse, FileResponse
# from pydantic import BaseModel
# import pandas as pd
# import uuid # For unique filenames/run IDs
# from datetime import datetime, timedelta
# import io

# # --- ADD THIS IMPORT ---
# from sklearn.metrics import mean_absolute_error, r2_score
# # -----------------------

# # Import your model and utility functions
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

# # Configuration
# SEQUENCE_LENGTH = 100 # For LSTM
# N_FUTURE_DAYS_PREDICTION = 30
# LSTM_EPOCHS = 10 # As requested
# POLYNOMIAL_DEGREE = 5 # Example, can be tuned or passed as param

# # Ensure static directories exist
# STATIC_DIR = "static"
# PLOTS_DIR = os.path.join(STATIC_DIR, "plots")
# os.makedirs(PLOTS_DIR, exist_ok=True)

# app = FastAPI(title="Stock Prediction ML Service")

# # --- Helper Function to clean up old plots ---
# def cleanup_old_plots(max_age_minutes=60):
#     now = datetime.now()
#     for filename in os.listdir(PLOTS_DIR):
#         file_path = os.path.join(PLOTS_DIR, filename)
#         try:
#             file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
#             if (now - file_mod_time) > timedelta(minutes=max_age_minutes):
#                 os.remove(file_path)
#                 print(f"Cleaned up old plot: {filename}")
#         except Exception as e:
#             print(f"Error cleaning up plot {filename}: {e}")


# class TrainResponse(BaseModel):
#     message: str
#     run_id: str
#     lstm_results: dict
#     polynomial_results: dict
#     plot_urls: dict

# @app.post("/train-predict/", response_model=TrainResponse)
# async def train_and_predict_models(
#     background_tasks: BackgroundTasks,
#     csv_file: UploadFile = File(...)
# ):
#     run_id = str(uuid.uuid4())
#     results = {
#         "message": "Processing started.",
#         "run_id": run_id,
#         "lstm_results": {},
#         "polynomial_results": {},
#         "plot_urls": {}
#     }

#     try:
#         # 1. Read and Preprocess CSV
#         contents = await csv_file.read()
#         df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

#         # Ensure 'Date' is datetime and not index, 'Close' is numeric
#         if 'Date' not in df.columns or 'Close' not in df.columns:
#             raise HTTPException(status_code=400, detail="CSV must contain 'Date' and 'Close' columns.")

#         df['Date'] = pd.to_datetime(df['Date'])
#         df.sort_values(by='Date', inplace=True)
#         df.reset_index(drop=True, inplace=True) # Ensure Date is not index
#         df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
#         df.dropna(subset=['Close'], inplace=True)

#         if len(df) < SEQUENCE_LENGTH + 10: # Minimum data for train/test/future
#              raise HTTPException(status_code=400, detail=f"Not enough data. Need at least {SEQUENCE_LENGTH + 10} rows after cleaning.")

#         train_df, test_df, scaler = preprocess_data(df.copy(), sequence_length=SEQUENCE_LENGTH) # Use a copy

#         # --- LSTM Model ---
#         print(f"[{run_id}] Training LSTM model...")
#         lstm_model, lstm_history = train_lstm_model(
#             train_df,
#             scaler,
#             sequence_length=SEQUENCE_LENGTH,
#             epochs=LSTM_EPOCHS
#         )
#         print(f"[{run_id}] LSTM model training complete.")

#         # LSTM Test Set Evaluation
#         y_test_actual_lstm, y_pred_test_lstm, x_test_lstm = evaluate_lstm_model(
#             lstm_model,
#             test_df,
#             train_df, # Pass train_df for last sequence
#             scaler,
#             sequence_length=SEQUENCE_LENGTH
#         )
#         results["lstm_results"]["test_metrics"] = {
#             "mae": float(mean_absolute_error(y_test_actual_lstm, y_pred_test_lstm)),
#             "r2_score": float(r2_score(y_test_actual_lstm, y_pred_test_lstm))
#         }

#         # LSTM Future Predictions
#         future_preds_lstm, future_dates_lstm = predict_lstm_future(
#             lstm_model,
#             df, # Full dataframe for last sequence
#             scaler,
#             sequence_length=SEQUENCE_LENGTH,
#             n_future_days=N_FUTURE_DAYS_PREDICTION
#         )
#         results["lstm_results"]["future_predictions"] = {
#             str(date.date()): float(pred) for date, pred in zip(future_dates_lstm, future_preds_lstm.flatten())
#         }

#         # LSTM Plotting
#         lstm_plot_test_path = os.path.join(PLOTS_DIR, f"{run_id}_lstm_test_predictions.png")
#         plot_test_predictions(
#             test_df['Date'],
#             y_test_actual_lstm,
#             y_pred_test_lstm,
#             title="LSTM: Test Set Predictions",
#             save_path=lstm_plot_test_path
#         )
#         results["plot_urls"]["lstm_test_plot"] = f"/static/plots/{run_id}_lstm_test_predictions.png"

#         lstm_plot_future_path = os.path.join(PLOTS_DIR, f"{run_id}_lstm_future_predictions.png")
#         plot_future_predictions(
#             df['Date'],
#             df['Close'],
#             test_df['Date'], # Dates for test predictions
#             y_pred_test_lstm, # Test predictions
#             future_dates_lstm,
#             future_preds_lstm,
#             title="LSTM: Historical, Test & Future Predictions",
#             save_path=lstm_plot_future_path
#         )
#         results["plot_urls"]["lstm_future_plot"] = f"/static/plots/{run_id}_lstm_future_predictions.png"


#         # --- Polynomial Regression Model ---
#         # print(f"[{run_id}] Training Polynomial Regression model...")
#         # poly_model, poly_scaler, time_feature_train, X_poly_train, y_poly_train = train_polynomial_model(
#         #     train_df.copy(), # Use a copy
#         #     degree=POLYNOMIAL_DEGREE
#         # )
        
        
#         # NEW corrected line in main.py:
#         poly_model, poly_scaler, X_train_poly_time_feature, y_train_poly_scaled = train_polynomial_model(
#             train_df.copy(),
#             degree=POLYNOMIAL_DEGREE
#         )
#         # You can then use X_train_poly_time_feature if you need the 'Time' feature DataFrame (which is X_train from the function)
#         # and y_train_poly_scaled is the scaled target variable from training.
#         # For the current logic, these returned training features/targets (X_train_poly_time_feature, y_train_poly_scaled)
#         # are not directly used later in the main.py flow, but it's good to unpack correctly.


#         print(f"[{run_id}] Polynomial Regression model training complete.")

#         # Polynomial Test Set Evaluation
#         y_test_actual_poly, y_pred_test_poly = evaluate_polynomial_model(
#             poly_model,
#             test_df.copy(), # Use a copy
#             poly_scaler,
#             degree=POLYNOMIAL_DEGREE
#         )
#         results["polynomial_results"]["test_metrics"] = {
#             "mae": float(mean_absolute_error(y_test_actual_poly, y_pred_test_poly)),
#             "r2_score": float(r2_score(y_test_actual_poly, y_pred_test_poly))
#         }

#         # Polynomial Future Predictions
#         future_preds_poly, future_dates_poly = predict_polynomial_future(
#             poly_model,
#             df, # Full dataframe for last date
#             poly_scaler,
#             degree=POLYNOMIAL_DEGREE,
#             n_future_days=N_FUTURE_DAYS_PREDICTION
#         )
#         results["polynomial_results"]["future_predictions"] = {
#             str(date.date()): float(pred) for date, pred in zip(future_dates_poly, future_preds_poly.flatten())
#         }

#         # Polynomial Plotting
#         poly_plot_path = os.path.join(PLOTS_DIR, f"{run_id}_polynomial_regression.png")
#         plot_polynomial_regression(
#             df['Date'], df['Close'], # All historical data
#             test_df['Date'], y_pred_test_poly, # Test predictions
#             future_dates_poly, future_preds_poly,
#             title=f"Polynomial Regression (Degree {POLYNOMIAL_DEGREE})",
#             save_path=poly_plot_path
#         )
#         results["plot_urls"]["polynomial_plot"] = f"/static/plots/{run_id}_polynomial_regression.png"


#         results["message"] = "Processing complete."
#         background_tasks.add_task(cleanup_old_plots) # Schedule cleanup
#         return results

#     except HTTPException as e:
#         raise e # Re-raise HTTP exceptions
#     except Exception as e:
#         print(f"[{run_id}] Error during processing: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# # --- Mount static directory to serve plots ---
# from fastapi.staticfiles import StaticFiles
# app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Stock Prediction ML Service. POST to /train-predict/ with a CSV file."}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)