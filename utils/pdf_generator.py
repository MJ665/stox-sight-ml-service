# utils/pdf_generator.py
from fpdf import FPDF
from datetime import datetime
import os

PDF_DIR = "static/pdfs" # Define where PDFs will be saved
os.makedirs(PDF_DIR, exist_ok=True)

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Stock Prediction Report', 0, 1, 'C')
        self.set_font('Arial', '', 8)
        self.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, content):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, content)
        self.ln()

    def add_table(self, headers, data, col_widths=None):
        self.set_font('Arial', 'B', 9)
        if col_widths is None:
            col_width = self.w / (len(headers) + 1) # distribute width
            col_widths = [col_width] * len(headers)
        
        # Header
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, 1, 0, 'C')
        self.ln()
        
        # Data
        self.set_font('Arial', '', 8)
        for row in data:
            for i, item in enumerate(row):
                self.cell(col_widths[i], 6, str(item), 1, 0, 'L')
            self.ln()
        self.ln(5)

    def add_metric(self, model_name, metric_name, value):
        self.set_font('Arial', '', 10)
        self.cell(0, 6, f"{model_name} - {metric_name}: {value}", 0, 1)
        self.ln(1)

    def add_suggestion(self, suggestion):
        self.set_font('Arial', 'B', 11)
        self.cell(0, 10, f"Trading Suggestion for Tomorrow: {suggestion}", 0, 1, 'L')
        self.ln(5)

    def add_plot_image(self, image_path, title, width=180):
        if os.path.exists(image_path):
            self.chapter_title(title)
            self.image(image_path, x=None, y=None, w=width)
            self.ln(5)
        else:
            self.chapter_body(f"Plot not found: {image_path}")


def generate_prediction_report(run_id: str, report_data: dict, plot_paths: dict):
    pdf_filename = f"{run_id}_report.pdf"
    pdf_filepath = os.path.join(PDF_DIR, pdf_filename)

    pdf = PDFReport()
    pdf.alias_nb_pages() # For total page numbers
    pdf.add_page()

    # --- General Information ---
    pdf.chapter_title("Run Information")
    pdf.chapter_body(f"Report ID: {run_id}")
    pdf.chapter_body(f"CSV Processed: {report_data.get('csv_filename', 'N/A')}") # Assuming you add this
    pdf.ln(5)

    # --- Trading Suggestion ---
    if report_data.get('trading_suggestion_tomorrow'):
        pdf.add_suggestion(report_data['trading_suggestion_tomorrow']['signal'])
        pdf.chapter_body(f"Basis: {report_data['trading_suggestion_tomorrow']['reason']}")

    # --- LSTM Model Results ---
    if "lstm_results" in report_data:
        pdf.chapter_title("LSTM Model Analysis")
        lstm_metrics = report_data["lstm_results"].get("test_metrics", {})
        pdf.add_metric("LSTM", "MAE", f"{lstm_metrics.get('mae', 'N/A'):.4f}")
        pdf.add_metric("LSTM", "R2 Score", f"{lstm_metrics.get('r2_score', 'N/A'):.4f}")
        
        future_preds_lstm = report_data["lstm_results"].get("future_predictions", {})
        if future_preds_lstm:
            pdf.ln(3)
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 6, "LSTM Future Predictions:", 0, 1)
            headers = ["Date", "Predicted Price"]
            table_data = [[date, f"{price:.2f}"] for date, price in future_preds_lstm.items()]
            pdf.add_table(headers, table_data, col_widths=[40, 40])

    # --- Polynomial Regression Results ---
    if "polynomial_results" in report_data:
        pdf.chapter_title("Polynomial Regression Analysis")
        poly_metrics = report_data["polynomial_results"].get("test_metrics", {})
        pdf.add_metric("Polynomial", "MAE", f"{poly_metrics.get('mae', 'N/A'):.4f}")
        pdf.add_metric("Polynomial", "R2 Score", f"{poly_metrics.get('r2_score', 'N/A'):.4f}")

        future_preds_poly = report_data["polynomial_results"].get("future_predictions", {})
        if future_preds_poly:
            pdf.ln(3)
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 6, "Polynomial Future Predictions:", 0, 1)
            headers = ["Date", "Predicted Price"]
            table_data = [[date, f"{price:.2f}"] for date, price in future_preds_poly.items()]
            pdf.add_table(headers, table_data, col_widths=[40, 40])
    
    pdf.add_page() # New page for plots

    # --- Add Plots ---
    if plot_paths.get("lstm_test_plot"):
        pdf.add_plot_image(plot_paths["lstm_test_plot"].replace("/static/", "static/"), "LSTM: Test Set Predictions")
    if plot_paths.get("lstm_future_plot"):
        pdf.add_plot_image(plot_paths["lstm_future_plot"].replace("/static/", "static/"), "LSTM: Historical, Test & Future Predictions")
    if plot_paths.get("polynomial_plot"):
         pdf.add_plot_image(plot_paths["polynomial_plot"].replace("/static/", "static/"), "Polynomial Regression Predictions")

    pdf.output(pdf_filepath, 'F')
    print(f"Generated PDF report: {pdf_filepath}")
    return pdf_filepath # Return the path to the generated PDF