import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for server-side plotting
import pandas as pd

def plot_test_predictions(dates_actual, y_actual, y_pred, title, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(dates_actual, y_actual, 'b-', label='Actual Price')
    plt.plot(dates_actual, y_pred, 'r-', label='Predicted Price')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.savefig(save_path)
    plt.close() # Close the figure to free memory

def plot_future_predictions(hist_dates, hist_close, 
                            test_pred_dates, test_pred_close,
                            future_dates, future_close, title, save_path):
    plt.figure(figsize=(15, 7))
    plt.plot(hist_dates, hist_close, 'k-', label='Historical Actual Price', alpha=0.7)
    if test_pred_dates is not None and test_pred_close is not None:
         plt.plot(test_pred_dates, test_pred_close, color='red', linestyle='-', label='Test Predicted Price')
    plt.plot(future_dates, future_close, color='green', linestyle='--', marker='o', markersize=4, label='Future Predicted Price')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.savefig(save_path)
    plt.close()

def plot_polynomial_regression(hist_dates, hist_close, 
                               test_pred_dates, test_pred_values,
                               future_dates, future_pred_values, 
                               title, save_path):
    plt.figure(figsize=(15, 7))
    plt.plot(hist_dates, hist_close, 'k.', label='Historical Actual Price', alpha=0.5, markersize=3)
    
    # Combine test and future dates/preds for a continuous line if desired, or plot separately
    # For simplicity, plotting test and future separately for polynomial
    if test_pred_dates is not None and test_pred_values is not None:
        # Sort test predictions by date if they aren't already for smooth line
        sorted_test_indices = test_pred_dates.argsort()
        plt.plot(test_pred_dates[sorted_test_indices], test_pred_values[sorted_test_indices], 'r-', label='Polynomial Fit (Test)', linewidth=2)

    if future_dates is not None and future_pred_values is not None:
        plt.plot(future_dates, future_pred_values, 'g--', label='Polynomial Future Prediction', linewidth=2)
        
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.savefig(save_path)
    plt.close()