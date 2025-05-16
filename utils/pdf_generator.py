

# utils/pdf_generator.py
from fpdf import FPDF, fpdf
from datetime import datetime
import os
import pandas as pd
import textwrap
import json

PDF_DIR = "static/pdfs"
os.makedirs(PDF_DIR, exist_ok=True)
from dotenv import load_dotenv

load_dotenv()

POLYNOMIAL_DEGREE =os.getenv("POLYNOMIAL_DEGREE")



class PDFReport(FPDF):
    def __init__(self, orientation='P', unit='mm', format='A4'):
        super().__init__(orientation, unit, format)
        self.set_auto_page_break(auto=True, margin=15) # Standard bottom margin for page break
        self.table_headers = [] # To store headers for re-drawing
        self.table_col_widths = [] # To store col_widths

    # ... (header, footer, chapter_title, _render_wrapped_text_lines, chapter_body, add_observation, add_suggestion, add_metric, add_plot_image as in the previous successful version) ...
    # I'll re-paste them here for completeness, assuming they were working.
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

    def _render_wrapped_text_lines(self, lines: list, line_height: float, initial_x_offset: float = 0):
        is_first_line_of_block = True
        for line_content in lines:
            current_x_pos = self.get_x() 
            if not is_first_line_of_block and initial_x_offset > 0:
                self.set_x(self.l_margin + initial_x_offset)
            
            try:
                self.multi_cell(0, line_height, line_content, border=0, align='L', new_x="LMARGIN", new_y="NEXT")
            except fpdf.FPDFException as e:
                print(f"FPDFException rendering line '{line_content[:50]}...': {e}. Skipping line.")
            
            if not is_first_line_of_block and initial_x_offset > 0:
                 self.set_x(current_x_pos) 
            is_first_line_of_block = False

    def chapter_body(self, content, char_wrap_limit=110, line_height=5): # Adjusted limit from prev
        original_font_family, original_font_style, original_font_size = self.font_family, self.font_style, self.font_size
        self.set_font('Arial', '', 10)
        
        content_str = str(content)
        wrapped_lines = textwrap.wrap(
            content_str, width=char_wrap_limit, break_long_words=True, 
            replace_whitespace=False, drop_whitespace=False, break_on_hyphens=True,
            fix_sentence_endings=True
        )
        self._render_wrapped_text_lines(wrapped_lines, line_height)
        
        self.set_font(original_font_family, original_font_style, original_font_size)
        self.ln(1)

    def add_observation(self, observation_text, char_wrap_limit=75, line_height=5):
        original_font_family, original_font_style, original_font_size = self.font_family, self.font_style, self.font_size
        self.set_font('Arial', '', 9)
        
        prefix = "- "
        observation_str = str(observation_text)
        text_part_lines = textwrap.wrap(
            observation_str, width=char_wrap_limit, break_long_words=True,
            replace_whitespace=False, drop_whitespace=False, break_on_hyphens=True,
            fix_sentence_endings=True
        )
        
        if not text_part_lines:
            try:
                self.multi_cell(0, line_height, prefix, border=0, align='L', new_x="LMARGIN", new_y="NEXT")
            except fpdf.FPDFException as e: print(f"FPDFException rendering observation prefix: {e}")
            self.set_font(original_font_family, original_font_style, original_font_size)
            return

        try:
            self.multi_cell(0, line_height, f"{prefix}{text_part_lines[0]}", border=0, align='L', new_x="LMARGIN", new_y="NEXT")
        except fpdf.FPDFException as e: print(f"FPDFException rendering observation line '{prefix}{text_part_lines[0][:50]}...': {e}")

        if len(text_part_lines) > 1:
            prefix_width = self.get_string_width(prefix)
            self._render_wrapped_text_lines(text_part_lines[1:], line_height, initial_x_offset=prefix_width + 0.5)
            
        self.set_font(original_font_family, original_font_style, original_font_size)

    def add_suggestion(self, suggestion_signal, suggestion_basis, char_wrap_limit=80, line_height=5):
        self.set_font('Arial', 'B', 11)
        self.cell(0, 10, f"Trading Suggestion for Tomorrow: {suggestion_signal}", 0, 1, 'L')
        self.ln(1)
        
        original_font_family, original_font_style, original_font_size = self.font_family, self.font_style, self.font_size
        self.set_font('Arial', '', 10)
        
        wrapped_basis = textwrap.wrap(
            str(suggestion_basis), width=char_wrap_limit, break_long_words=True,
            replace_whitespace=False, drop_whitespace=False, break_on_hyphens=True,
            fix_sentence_endings=True
        )
        self._render_wrapped_text_lines(wrapped_basis, line_height)
        
        self.set_font(original_font_family, original_font_style, original_font_size)
        self.ln(3)

    def add_metric(self, model_name, metric_name, value):
        self.set_font('Arial', '', 10)
        self.cell(0, 6, f"{model_name} - {metric_name}: {value}", 0, 1)
        self.ln(1)

    # def add_plot_image(self, image_path, title, width=170): 
    #     if os.path.exists(image_path):
    #         approx_image_height = 60 # Rough estimate for plot height + title
    #         if self.get_y() + approx_image_height > self.page_break_trigger:
    #             self.add_page()
    #         self.chapter_title(title)
    #         try:
    #             self.image(image_path, x=None, y=None, w=width) 
    #             self.ln(5)
    #         except Exception as e:
    #             print(f"Error adding image {image_path} to PDF: {e}")
    #             self.chapter_body(f"Error rendering plot: {os.path.basename(image_path)}")
    def add_plot_image(self, image_disk_path: str, title: str, width=170): # Parameter is now disk_path
        print(f"PDF Generator: Attempting to add image from disk path: {image_disk_path}")
        if os.path.exists(image_disk_path):
            approx_image_height = 80 # Increased estimate for plot height + title
            if self.get_y() + approx_image_height > self.page_break_trigger:
                self.add_page()
                # If you have table headers that need to be redrawn on new page, call that here too
                # self._draw_table_header() # Example if you had a table before plots
            self.chapter_title(title)
            try:
                self.image(image_disk_path, x=None, y=None, w=width)
                self.ln(5)
            except Exception as e:
                print(f"Error adding image {image_disk_path} to PDF: {e}")
                self.chapter_body(f"Error rendering plot: {os.path.basename(image_disk_path)}")
        else:
            print(f"PDF Generator: Plot not found at disk path: {image_disk_path}")
            self.chapter_body(f"Plot not found: {os.path.basename(image_disk_path)}")

    def add_json_dump(self, title: str, data_dict: dict):
        # ... (same as before) ...
        self.chapter_title(title)
        self.set_font("Courier", size=8)
        try:
            json_string = json.dumps(data_dict, indent=2)
        except Exception as e:
            json_string = f"Error serializing data to JSON: {e}\nRaw data: {str(data_dict)}"
            self.set_font("Arial", size=8)

        wrapped_json_lines = []
        for line in json_string.split('\n'):
            wrapped_json_lines.extend(textwrap.wrap(
                line, width=90, break_long_words=False, replace_whitespace=False,
                drop_whitespace=False, subsequent_indent="  "
            ))
        for line in wrapped_json_lines:
            try:
                self.cell(0, 4, line, 0, 1, 'L') 
            except fpdf.FPDFException as e: print(f"FPDFException rendering JSON line '{line[:50]}...': {e}. Skipping.")
        self.ln(5)
        self.set_font("Arial", size=10)
    # --- End of helper methods from previous step ---

    def _draw_table_header(self):
        if not self.table_headers or not self.table_col_widths:
            return
        self.set_font('Arial', 'B', 9)
        current_x = self.get_x() # Save current X, might be LMARGIN
        for i, header_text in enumerate(self.table_headers):
            self.cell(self.table_col_widths[i], 7, str(header_text), 1, 0, 'C')
        self.ln()
        self.set_x(current_x) # Reset X in case cells shifted it and ln() didn't reset fully

    def add_table(self, headers, data, col_widths=None):
        page_width = self.w - self.l_margin - self.r_margin
        
        # Store headers and calculate/store column widths for potential re-drawing
        self.table_headers = headers
        if col_widths:
            self.table_col_widths = col_widths
        else:
            num_cols = len(headers)
            if num_cols > 0:
                col_width_val = page_width / num_cols
                self.table_col_widths = [col_width_val] * num_cols
            else:
                self.table_col_widths = []
                return # No headers, no table
        
        # Draw initial header
        self._draw_table_header()
        
        self.set_font('Arial', '', 8)
        row_height = 6 # Assuming fixed row height for simplicity with `cell`

        for row_data in data:
            # Check if the current row will fit on the page
            # Add header height if we are about to break page and need to redraw it
            header_height_if_new_page = 7 if self.get_y() + row_height > self.page_break_trigger else 0
            
            if self.get_y() + row_height + header_height_if_new_page > self.page_break_trigger:
                self.add_page()
                self._draw_table_header() # Redraw header on new page
                self.set_font('Arial', '', 8) # Reset font for data rows

            current_x_start_of_row = self.l_margin # Ensure row starts at left margin
            self.set_x(current_x_start_of_row)

            for i, item_text in enumerate(row_data):
                # self.set_xy(current_x_start_of_row + sum(self.table_col_widths[:i]), self.get_y()) # Not needed if using flow
                self.cell(self.table_col_widths[i], row_height, str(item_text), 1, 0, 'L')
            self.ln(row_height) # Move to next line, respecting the row_height
        self.ln(5) # Space after table


def generate_prediction_report(run_id: str, report_data: dict, disk_plot_paths: dict, output_directory: str):
    # ... (The rest of this function, calling the PDFReport methods, remains the same)
    # The key is that the add_table method within the PDFReport class is now improved.
    # I'll re-paste the call structure for clarity, assuming the methods it calls are updated.

    pdf_filename = f"{run_id}_report.pdf"
    # pdf_filepath = os.path.join(PDF_DIR, pdf_filename)
    pdf_filepath = os.path.join(output_directory, pdf_filename)

    pdf = PDFReport() # Uses the updated class
    pdf.alias_nb_pages()
    pdf.add_page()

    # 1. Run Information (with disclaimer)
    pdf.chapter_title("Disclaimer")
    pdf.chapter_body("This AI-generated analysis is for informational and educational purposes (e.g., Hackathon, College Project) ONLY and is NOT financial advice. Predictions are speculative. Consult a qualified financial advisor before making any investment decisions. Do NOT trust this report for actual trading.", char_wrap_limit=100) # Example disclaimer
    pdf.ln(3)
    pdf.chapter_title("Run Information")
    pdf.chapter_body(f"Stock Symbol: {report_data.get('stock_symbol', 'N/A')}") # Added stock symbol
    pdf.chapter_body(f"Report ID: {run_id}")
    pdf.chapter_body(f"CSV Processed: {report_data.get('csv_filename', 'N/A')}")
    pdf.chapter_body(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    pdf.ln(5)
    pdf.ln(5)


    # 2. Trading Suggestion
    trading_suggestion_data = report_data.get('trading_suggestion_tomorrow', {})
    if trading_suggestion_data and trading_suggestion_data.get('signal') != "N/A":
        pdf.add_suggestion(
            trading_suggestion_data.get('signal', 'N/A'),
            trading_suggestion_data.get('reason', 'N/A')
        )
    else:
        pdf.chapter_title("Trading Suggestion")
        pdf.chapter_body("No definitive trading suggestion generated for tomorrow.")
        pdf.ln(5)

    # 3. AI Qualitative Analysis
    ai_analysis = report_data.get("ai_qualitative_analysis")
    if isinstance(ai_analysis, dict):
        pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
        # ... (Keep your existing detailed rendering of ai_analysis sections) ...
        pdf.chapter_body(f"Stock Symbol: {ai_analysis.get('stockSymbol', 'N/A')}")
        pdf.chapter_body(f"Analysis Date: {ai_analysis.get('analysisDate', 'N/A')}")
        pdf.ln(1)
        pdf.set_font('Arial', 'B', 10); pdf.cell(0, 6, f"Overall Sentiment: {ai_analysis.get('overallSentiment', 'N/A')}", 0, 1)
        pdf.chapter_body(f"Rationale: {ai_analysis.get('sentimentRationale', 'N/A')}")

        
        pdf.ln(3)

        lstm_analysis = ai_analysis.get("lstmModelAnalysis", {})
        pdf.set_font('Arial', 'BU', 10); pdf.cell(0, 6, "LSTM Model Insights:", 0, 1)
        pdf.chapter_body(f"Test Performance: {lstm_analysis.get('performanceOnTest', 'N/A')}")
        pdf.chapter_body(f"Future Outlook: {lstm_analysis.get('futureOutlook', 'N/A')}")
        pdf.chapter_body(f"Confidence: {lstm_analysis.get('confidenceInOutlook', 'N/A')}")
        pdf.ln(2)
        
        

        # GRU AI Insights
        gru_ai = ai_analysis.get("gruModelAnalysis", {})
        if gru_ai:
            pdf.set_font('Arial', 'BU', 10); pdf.cell(0, 6, "GRU Model Insights (AI):", 0, 1)
            if gru_ai.get("bestParamsFound"): pdf.chapter_body(f"Best Tuned Params: {json.dumps(gru_ai.get('bestParamsFound'))}")
            pdf.chapter_body(f"Test Performance: {gru_ai.get('performanceOnTest', 'N/A')}")
            pdf.chapter_body(f"Future Outlook: {gru_ai.get('futureOutlook', 'N/A')}")
            pdf.chapter_body(f"Confidence: {gru_ai.get('confidenceInOutlook', 'N/A')}")
            pdf.ln(2)

        # TRANSFORMER AI Insights (NEW)
        transformer_ai = ai_analysis.get("transformerModelAnalysis", {})
        if transformer_ai:
            pdf.set_font('Arial', 'BU', 10); pdf.cell(0, 6, "Transformer Model Insights (AI):", 0, 1)
            if transformer_ai.get("bestParamsFound"): pdf.chapter_body(f"Best Tuned Params: {json.dumps(transformer_ai.get('bestParamsFound'))}")
            pdf.chapter_body(f"Test Performance: {transformer_ai.get('performanceOnTest', 'N/A')}")
            pdf.chapter_body(f"Future Outlook: {transformer_ai.get('futureOutlook', 'N/A')}")
            pdf.chapter_body(f"Confidence: {transformer_ai.get('confidenceInOutlook', 'N/A')}")
            pdf.ln(2)

        poly_analysis = ai_analysis.get("polynomialRegressionAnalysis", {})
        pdf.set_font('Arial', 'BU', 10); pdf.cell(0, 6, "Polynomial Regression Insights:", 0, 1)
        pdf.chapter_body(f"Test Performance: {poly_analysis.get('performanceOnTest', 'N/A')}")
        pdf.chapter_body(f"Future Outlook: {poly_analysis.get('futureOutlook', 'N/A')}")
        pdf.chapter_body(f"Confidence: {poly_analysis.get('confidenceInOutlook', 'N/A')}")
        pdf.ln(2)
        
        combined_outlook = ai_analysis.get("combinedOutlook", {})
        pdf.set_font('Arial', 'BU', 10); pdf.cell(0, 6, "Combined Outlook & Observations:", 0, 1)
        pdf.chapter_body(f"Synopsis: {combined_outlook.get('shortTermForecastSynopsis', 'N/A')}")
        key_observations = combined_outlook.get("keyObservations", [])
        if key_observations:
            pdf.set_font('Arial', 'B', 9); pdf.cell(0, 5, "Key Observations:", 0, 1)
            for obs in key_observations: pdf.add_observation(obs) 
        pdf.ln(2)

        risk_factors = ai_analysis.get("riskFactors", [])
        if risk_factors:
            pdf.set_font('Arial', 'BU', 10); pdf.cell(0, 6, "Identified Risk Factors:", 0, 1)
            for risk in risk_factors: pdf.add_observation(risk) 
        pdf.ln(2)
        
        pdf.chapter_body(ai_analysis.get('disclaimer', "Standard AI analysis disclaimer applies."), char_wrap_limit=100)
        pdf.ln(5)
        
        
    elif isinstance(ai_analysis, str): # Error from Gemini
        pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
        pdf.chapter_body(f"Error or non-JSON response during AI Analysis: {ai_analysis}")
        pdf.ln(5)
    else:
        pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
        pdf.chapter_body("AI analysis was not performed or data is unavailable in expected format.")
        pdf.ln(5)


    # 3. Numerical Results (Model by Model)
    pdf.chapter_title("Model Numerical Summaries")

    # 4. LSTM Numerical Results
    if "lstm_results" in report_data:
        pdf.chapter_title("LSTM Model - Numerical Summary")
        lstm_metrics = report_data["lstm_results"].get("test_metrics", {})
        pdf.add_metric("LSTM", "Test MAE", f"{lstm_metrics.get('mae', 'N/A'):.4f}")
        pdf.add_metric("LSTM", "Test R2 Score", f"{lstm_metrics.get('r2_score', 'N/A'):.4f}")
        pdf.ln(2) # Add some space before the table
        
        future_preds_lstm = report_data["lstm_results"].get("future_predictions", {})
        if future_preds_lstm:
            pdf.set_font('Arial', 'B', 10); pdf.cell(0, 6, "LSTM Future Predictions (Table):", 0, 1, 'L')
            headers = ["Date", "Predicted Price (LSTM)"]
            table_data = [[date, f"{price:.2f}"] for date, price in future_preds_lstm.items()]
            # Ensure col_widths are appropriate for the content
            pdf.add_table(headers, table_data, col_widths=[pdf.w * 0.25, pdf.w * 0.25]) # Example: 25% of page width each
        pdf.ln(3)

    # GRU Numerical Results (NEW)
    if "gru_results" in report_data:
        pdf.chapter_title("GRU Model - Numerical Summary")
        gru_data = report_data["gru_results"]
        if gru_data.get("best_params"):
             pdf.chapter_body(f"Best Hyperparameters Found: {json.dumps(gru_data['best_params'])}")
        
        gru_metrics = gru_data.get("test_metrics", {})
        pdf.add_metric("GRU", "Test MAE", f"{gru_metrics.get('mae', 'N/A'):.4f}" if isinstance(gru_metrics.get('mae'), (int,float)) else 'N/A')
        pdf.add_metric("GRU", "Test R2 Score", f"{gru_metrics.get('r2_score', 'N/A'):.4f}" if isinstance(gru_metrics.get('r2_score'), (int,float)) else 'N/A')
        pdf.ln(2)
        
        future_preds_gru = gru_data.get("future_predictions", {})
        if future_preds_gru:
            pdf.set_font('Arial', 'B', 10); pdf.cell(0, 6, "GRU Future Predictions (Table):", 0, 1, 'L')
            headers = ["Date", "Predicted Price (GRU)"]
            table_data = [[date, f"{price:.2f}" if isinstance(price, (int,float)) else 'N/A'] for date, price in future_preds_gru.items()]
            pdf.add_table(headers, table_data, col_widths=[pdf.w * 0.25, pdf.w * 0.25])
        pdf.ln(3)




    # TRANSFORMER Numerical Results (NEW)
    if "transformer_results" in report_data and isinstance(report_data["transformer_results"], dict):
        pdf.set_font('Arial', 'B', 11); pdf.cell(0, 8, "Transformer Model:", 0, 1, 'L')
        transformer_data = report_data["transformer_results"]
        if transformer_data.get("best_params"): pdf.chapter_body(f"Best Hyperparameters: {json.dumps(transformer_data['best_params'])}")
        transformer_metrics = transformer_data.get("test_metrics", {})
        pdf.add_metric("Transformer", "Test MAE", f"{transformer_metrics.get('mae', 'N/A'):.4f}" if isinstance(transformer_metrics.get('mae'), (int,float)) else 'N/A')
        pdf.add_metric("Transformer", "Test R2 Score", f"{transformer_metrics.get('r2_score', 'N/A'):.4f}" if isinstance(transformer_metrics.get('r2_score'), (int,float)) else 'N/A')
        pdf.ln(1)
        future_preds_transformer = transformer_data.get("future_predictions", {})
        if future_preds_transformer:
            pdf.set_font('Arial', 'B', 9); pdf.cell(0, 6, "Future Predictions (Transformer):", 0, 1, 'L')
            headers = ["Date", "Predicted Price"]
            table_data = [[date, f"{price:.2f}" if isinstance(price, (int,float)) else 'N/A'] for date, price in list(future_preds_transformer.items())[:7]]
            pdf.add_table(headers, table_data, col_widths=[pdf.w * 0.20, pdf.w * 0.20])
        pdf.ln(3)
        
        
    # 5. Polynomial Regression Numerical Results
    if "polynomial_results" in report_data:
        pdf.chapter_title("Polynomial Regression - Numerical Summary")
        poly_metrics = report_data["polynomial_results"].get("test_metrics", {})
        pdf.add_metric("Polynomial", "Test MAE", f"{poly_metrics.get('mae', 'N/A'):.4f}")
        pdf.add_metric("Polynomial", "Test R2 Score", f"{poly_metrics.get('r2_score', 'N/A'):.4f}")
        pdf.ln(2)

        future_preds_poly = report_data["polynomial_results"].get("future_predictions", {})
        if future_preds_poly:
            pdf.set_font('Arial', 'B', 10); pdf.cell(0, 6, "Polynomial Future Predictions (Table):", 0, 1, 'L')
            headers = ["Date", "Predicted Price (Poly)"]
            table_data = [[date, f"{price:.2f}"] for date, price in future_preds_poly.items()]
            pdf.add_table(headers, table_data, col_widths=[pdf.w * 0.25, pdf.w * 0.25])
        pdf.ln(3)

    # 6. Raw JSON Data Summary
    # ... (same as before, calls pdf.add_json_dump)
    summary_data_for_json_dump = {
        "run_id": report_data.get("run_id"),
        "csv_filename": report_data.get("csv_filename"),
        "trading_suggestion_tomorrow": report_data.get("trading_suggestion_tomorrow"),
        "lstm_metrics": report_data.get("lstm_results", {}).get("test_metrics"),
        "lstm_future_predictions_sample": dict(list(report_data.get("lstm_results", {}).get("future_predictions", {}).items())[:5]),
        "gru_best_params": report_data.get("gru_results", {}).get("best_params"),
        "gru_metrics": report_data.get("gru_results", {}).get("test_metrics"),
        "gru_future_predictions_sample": dict(list(report_data.get("gru_results", {}).get("future_predictions", {}).items())[:5]),
        "polynomial_metrics": report_data.get("polynomial_results", {}).get("test_metrics"),
        "polynomial_future_predictions_sample": dict(list(report_data.get("polynomial_results", {}).get("future_predictions", {}).items())[:5])
    }
    if pdf.get_y() + 60 > pdf.page_break_trigger : pdf.add_page() 
    pdf.add_json_dump("Raw Data Summary (JSON Snippet)", summary_data_for_json_dump)


    # 7. Visualizations
    
    
    
    
    if disk_plot_paths:
        if pdf.get_y() + 70 > pdf.page_break_trigger : pdf.add_page()
        pdf.chapter_title("Visualizations")
        
        # Use the disk paths directly
        if disk_plot_paths.get("lstm_test_plot_path"):
            pdf.add_plot_image(disk_plot_paths["lstm_test_plot_path"], "LSTM: Test Set Predictions")
        if disk_plot_paths.get("lstm_future_plot_path"):
            pdf.add_plot_image(disk_plot_paths["lstm_future_plot_path"], "LSTM: All Predictions")
        
        if disk_plot_paths.get("gru_test_plot_path"):
            pdf.add_plot_image(disk_plot_paths["gru_test_plot_path"], "GRU: Test Set Predictions")
        if disk_plot_paths.get("gru_future_plot_path"):
            pdf.add_plot_image(disk_plot_paths["gru_future_plot_path"], "GRU: All Predictions")

        if disk_plot_paths.get("transformer_test_plot_path"):
            pdf.add_plot_image(disk_plot_paths["transformer_test_plot_path"], "Transformer: Test Set Predictions")
        if disk_plot_paths.get("transformer_future_plot_path"):
            pdf.add_plot_image(disk_plot_paths["transformer_future_plot_path"], "Transformer: All Predictions")

        if disk_plot_paths.get("polynomial_plot_path"):
            # Get POLYNOMIAL_DEGREE from report_data if possible, or use a global/default
            # Assuming POLYNOMIAL_DEGREE from main.py's scope is available or passed if needed for title
            # For simplicity, let's assume main.py's POLYNOMIAL_DEGREE is what was used.
            # You might want to pass it in report_data if it can vary.
            from main import POLYNOMIAL_DEGREE as APP_POLYNOMIAL_DEGREE # Quick way to get it for title
            pdf.add_plot_image(disk_plot_paths["polynomial_plot_path"], f"Polynomial Regression (Degree {APP_POLYNOMIAL_DEGREE})")
            
            


    try:
        pdf.output(pdf_filepath, 'F')
        print(f"Generated PDF report: {pdf_filepath}")
    except Exception as e:
        print(f"Error saving PDF {pdf_filepath}: {e}")
        raise
    return pdf_filepath











# # utils/pdf_generator.py



# from fpdf import FPDF, fpdf
# from datetime import datetime
# import os
# import pandas as pd
# import textwrap
# import json # For pretty-printing JSON

# PDF_DIR = "static/pdfs"
# os.makedirs(PDF_DIR, exist_ok=True)

# class PDFReport(FPDF):
#     # ... (header, footer, chapter_title, chapter_body, add_observation, add_table, add_metric, add_suggestion, add_plot_image)
#     # These methods seem to be working well with the textwrap and error handling,
#     # so we'll keep them as they were in the previous successful iteration unless a specific part needs adjustment.
#     # For brevity, I'll assume these helper methods are correct from the previous version that fixed the FPDFException.
#     # Ensure these methods are copied from the version that worked without FPDFException.
    
#     # --- Start of re-pasted helper methods (ensure these are from your working version) ---
#     def header(self):
#         self.set_font('Arial', 'B', 12)
#         self.cell(0, 10, 'Stock Prediction Report', 0, 1, 'C')
#         self.set_font('Arial', '', 8)
#         self.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
#         self.ln(5)

#     def footer(self):
#         self.set_y(-15)
#         self.set_font('Arial', 'I', 8)
#         self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

#     def chapter_title(self, title):
#         self.set_font('Arial', 'B', 12)
#         self.cell(0, 10, title, 0, 1, 'L')
#         self.ln(2)

#     def _render_wrapped_text_lines(self, lines: list, line_height: float, initial_x_offset: float = 0):
#         is_first_line_of_block = True
#         for line_content in lines:
#             current_x_pos = self.get_x() # Get current X before potential modification
#             if not is_first_line_of_block and initial_x_offset > 0:
#                 self.set_x(self.l_margin + initial_x_offset)
            
#             try:
#                 self.multi_cell(0, line_height, line_content, border=0, align='L', new_x="LMARGIN", new_y="NEXT")
#             except fpdf.FPDFException as e:
#                 print(f"FPDFException rendering line '{line_content[:50]}...': {e}. Skipping line.")
            
#             if not is_first_line_of_block and initial_x_offset > 0:
#                  self.set_x(current_x_pos) # Reset X if it was an indented line, for safety, though new_x=LMARGIN should handle
#             is_first_line_of_block = False

#     def chapter_body(self, content, char_wrap_limit=110, line_height=5):
#         original_font_family, original_font_style, original_font_size = self.font_family, self.font_style, self.font_size
#         self.set_font('Arial', '', 10)
        
#         content_str = str(content)
#         wrapped_lines = textwrap.wrap(
#             content_str, width=char_wrap_limit, break_long_words=True, 
#             replace_whitespace=False, drop_whitespace=False, break_on_hyphens=True,
#             fix_sentence_endings=True
#         )
#         self._render_wrapped_text_lines(wrapped_lines, line_height)
        
#         self.set_font(original_font_family, original_font_style, original_font_size)
#         self.ln(1)

#     def add_observation(self, observation_text, char_wrap_limit=75, line_height=5):
#         original_font_family, original_font_style, original_font_size = self.font_family, self.font_style, self.font_size
#         self.set_font('Arial', '', 9)
        
#         prefix = "- "
#         observation_str = str(observation_text)
#         text_part_lines = textwrap.wrap(
#             observation_str, width=char_wrap_limit, break_long_words=True,
#             replace_whitespace=False, drop_whitespace=False, break_on_hyphens=True,
#             fix_sentence_endings=True
#         )
        
#         if not text_part_lines:
#             try:
#                 self.multi_cell(0, line_height, prefix, border=0, align='L', new_x="LMARGIN", new_y="NEXT")
#             except fpdf.FPDFException as e: print(f"FPDFException rendering observation prefix: {e}")
#             self.set_font(original_font_family, original_font_style, original_font_size)
#             return

#         try:
#             self.multi_cell(0, line_height, f"{prefix}{text_part_lines[0]}", border=0, align='L', new_x="LMARGIN", new_y="NEXT")
#         except fpdf.FPDFException as e: print(f"FPDFException rendering observation line '{prefix}{text_part_lines[0][:50]}...': {e}")

#         if len(text_part_lines) > 1:
#             prefix_width = self.get_string_width(prefix)
#             self._render_wrapped_text_lines(text_part_lines[1:], line_height, initial_x_offset=prefix_width + 0.5) # Reduced indent slightly
            
#         self.set_font(original_font_family, original_font_style, original_font_size)

#     def add_suggestion(self, suggestion_signal, suggestion_basis, char_wrap_limit=80, line_height=5):
#         self.set_font('Arial', 'B', 11)
#         self.cell(0, 10, f"Trading Suggestion for Tomorrow: {suggestion_signal}", 0, 1, 'L')
#         self.ln(1)
        
#         original_font_family, original_font_style, original_font_size = self.font_family, self.font_style, self.font_size
#         self.set_font('Arial', '', 10)
        
#         wrapped_basis = textwrap.wrap(
#             str(suggestion_basis), width=char_wrap_limit, break_long_words=True,
#             replace_whitespace=False, drop_whitespace=False, break_on_hyphens=True,
#             fix_sentence_endings=True
#         )
#         self._render_wrapped_text_lines(wrapped_basis, line_height)
        
#         self.set_font(original_font_family, original_font_style, original_font_size)
#         self.ln(3)

#     def add_table(self, headers, data, col_widths=None):
#         self.set_font('Arial', 'B', 9)
#         page_width = self.w - self.l_margin - self.r_margin
#         if col_widths is None:
#             num_cols = len(headers)
#             if num_cols > 0: col_width_val = page_width / num_cols; col_widths = [col_width_val] * num_cols
#             else: return 
#         for i, header in enumerate(headers): self.cell(col_widths[i], 7, str(header), 1, 0, 'C')
#         self.ln()
#         self.set_font('Arial', '', 8)
#         for row_data in data:
#             # Simple fixed height for table rows. For dynamic height based on content,
#             # each cell would need to be a multi_cell and max height calculated.
#             max_h = 6 
#             current_y_for_row = self.get_y()
#             for i, item_text in enumerate(row_data):
#                 self.set_xy(self.l_margin + sum(col_widths[:i]), current_y_for_row)
#                 # Using multi_cell in tables is complex for height alignment. Cell truncates.
#                 # If text wrapping in table cells is essential, a more complex table method is needed.
#                 self.cell(col_widths[i], max_h, str(item_text), 1, 0, 'L') 
#             self.ln(max_h) # Move to next row based on fixed height
#         self.ln(5)

#     def add_metric(self, model_name, metric_name, value):
#         self.set_font('Arial', '', 10)
#         self.cell(0, 6, f"{model_name} - {metric_name}: {value}", 0, 1)
#         self.ln(1)

#     def add_plot_image(self, image_path, title, width=170): 
#         if os.path.exists(image_path):
#             # Check if enough space on the current page for title + image (approximate)
#             # Image height is proportional to width if not specified, assume roughly width/2 for landscape
#             approx_image_height = (width / self.w) * self.h * 0.3 if width < self.w else 50 # very rough estimate
#             title_height = 12 
#             if self.get_y() + title_height + approx_image_height > self.page_break_trigger:
#                 self.add_page()
            
#             self.chapter_title(title)
#             try:
#                 self.image(image_path, x=None, y=None, w=width) 
#                 self.ln(5)
#             except Exception as e:
#                 print(f"Error adding image {image_path} to PDF: {e}")
#                 self.chapter_body(f"Error rendering plot: {os.path.basename(image_path)}")
#         else:
#             self.chapter_body(f"Plot not found: {os.path.basename(image_path)}")
#     # --- End of re-pasted helper methods ---

#     def add_json_dump(self, title: str, data_dict: dict):
#         """Adds a pretty-printed JSON string to the PDF."""
#         self.chapter_title(title)
#         self.set_font("Courier", size=8) # Monospaced font for JSON
        
#         try:
#             json_string = json.dumps(data_dict, indent=2)
#         except Exception as e:
#             json_string = f"Error serializing data to JSON: {e}\nRaw data: {str(data_dict)}"
#             self.set_font("Arial", size=8) # Switch back if error

#         # textwrap for the JSON string, as it can be very long
#         # For JSON, we want to preserve spaces for indentation
#         wrapped_json_lines = []
#         for line in json_string.split('\n'):
#             wrapped_json_lines.extend(textwrap.wrap(
#                 line, 
#                 width=90, # Adjust based on Courier font and page width
#                 break_long_words=False, # Try not to break in middle of keys/values
#                 replace_whitespace=False,
#                 drop_whitespace=False,
#                 subsequent_indent="  " # For readability of wrapped JSON lines
#             ))
        
#         for line in wrapped_json_lines:
#             try:
#                 # Using cell for pre-formatted text to avoid multi_cell's complex breaking
#                 # This might truncate very long individual lines if they still don't fit.
#                 self.cell(0, 4, line, 0, 1, 'L') 
#             except fpdf.FPDFException as e:
#                 print(f"FPDFException rendering JSON line '{line[:50]}...': {e}. Skipping.")
#         self.ln(5)
#         self.set_font("Arial", size=10) # Reset to default body font


# def generate_prediction_report(run_id: str, report_data: dict, plot_paths: dict):
#     pdf_filename = f"{run_id}_report.pdf"
#     pdf_filepath = os.path.join(PDF_DIR, pdf_filename)

#     pdf = PDFReport()
#     pdf.alias_nb_pages()
#     pdf.add_page()

#     # --- Section Order Recommendation: ---
#     # 1. Run Information
#     # 2. Trading Suggestion (Key takeaway upfront)
#     # 3. AI Qualitative Analysis (Gemini)
#     # 4. LSTM Numerical Results (Metrics & Table)
#     # 5. Polynomial Numerical Results (Metrics & Table)
#     # 6. Raw JSON Data Summary (Optional, but requested)
#     # 7. Visualizations (Plots on new pages)

#     # 1. Run Information
#     pdf.chapter_title("Disclaimer: DO NOT TRUST THE ML REPORT and GEMINI REPORT")
#     pdf.chapter_title("This is for Hackathon Purpose and College Project Purpose")
#     pdf.chapter_title("This is AI Generated analysis is for informational purposes only and not financial advice. ")
#     pdf.chapter_title("Predictions are speculative. Consult a qualified financial advisor before making investment decisions.")
#     pdf.chapter_title("Run Information")
#     pdf.chapter_body(f"Report ID: {run_id}")
#     pdf.chapter_body(f"CSV Processed: {report_data.get('csv_filename', 'N/A')}")
#     pdf.chapter_body(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     pdf.ln(5)

#     # 2. Trading Suggestion
#     trading_suggestion_data = report_data.get('trading_suggestion_tomorrow', {})
#     if trading_suggestion_data and trading_suggestion_data.get('signal') != "N/A":
#         pdf.add_suggestion(
#             trading_suggestion_data.get('signal', 'N/A'),
#             trading_suggestion_data.get('reason', 'N/A')
#         )
#     else:
#         pdf.chapter_title("Trading Suggestion")
#         pdf.chapter_body("No definitive trading suggestion generated for tomorrow.")
#         pdf.ln(5)

#     # 3. AI Qualitative Analysis
#     ai_analysis = report_data.get("ai_qualitative_analysis")
#     if isinstance(ai_analysis, dict):
#         pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
#         # ... (Keep your existing detailed rendering of ai_analysis sections) ...
#         pdf.chapter_body(f"Stock Symbol: {ai_analysis.get('stockSymbol', 'N/A')}")
#         pdf.chapter_body(f"Analysis Date: {ai_analysis.get('analysisDate', 'N/A')}")
#         pdf.ln(1)
#         pdf.set_font('Arial', 'B', 10); pdf.cell(0, 6, f"Overall Sentiment: {ai_analysis.get('overallSentiment', 'N/A')}", 0, 1)
#         pdf.chapter_body(f"Rationale: {ai_analysis.get('sentimentRationale', 'N/A')}")

        
#         pdf.ln(3)

#         lstm_analysis = ai_analysis.get("lstmModelAnalysis", {})
#         pdf.set_font('Arial', 'BU', 10); pdf.cell(0, 6, "LSTM Model Insights:", 0, 1)
#         pdf.chapter_body(f"Test Performance: {lstm_analysis.get('performanceOnTest', 'N/A')}")
#         pdf.chapter_body(f"Future Outlook: {lstm_analysis.get('futureOutlook', 'N/A')}")
#         pdf.chapter_body(f"Confidence: {lstm_analysis.get('confidenceInOutlook', 'N/A')}")
#         pdf.ln(2)

#         poly_analysis = ai_analysis.get("polynomialRegressionAnalysis", {})
#         pdf.set_font('Arial', 'BU', 10); pdf.cell(0, 6, "Polynomial Regression Insights:", 0, 1)
#         pdf.chapter_body(f"Test Performance: {poly_analysis.get('performanceOnTest', 'N/A')}")
#         pdf.chapter_body(f"Future Outlook: {poly_analysis.get('futureOutlook', 'N/A')}")
#         pdf.chapter_body(f"Confidence: {poly_analysis.get('confidenceInOutlook', 'N/A')}")
#         pdf.ln(2)
        
#         combined_outlook = ai_analysis.get("combinedOutlook", {})
#         pdf.set_font('Arial', 'BU', 10); pdf.cell(0, 6, "Combined Outlook & Observations:", 0, 1)
#         pdf.chapter_body(f"Synopsis: {combined_outlook.get('shortTermForecastSynopsis', 'N/A')}")
#         key_observations = combined_outlook.get("keyObservations", [])
#         if key_observations:
#             pdf.set_font('Arial', 'B', 9); pdf.cell(0, 5, "Key Observations:", 0, 1)
#             for obs in key_observations: pdf.add_observation(obs) 
#         pdf.ln(2)

#         risk_factors = ai_analysis.get("riskFactors", [])
#         if risk_factors:
#             pdf.set_font('Arial', 'BU', 10); pdf.cell(0, 6, "Identified Risk Factors:", 0, 1)
#             for risk in risk_factors: pdf.add_observation(risk) 
#         pdf.ln(2)
        
#         pdf.chapter_body(ai_analysis.get('disclaimer', "Standard AI analysis disclaimer applies."), char_wrap_limit=100)
#         pdf.ln(5)
#     elif isinstance(ai_analysis, str): # Error from Gemini
#         pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
#         pdf.chapter_body(f"Error or non-JSON response during AI Analysis: {ai_analysis}")
#         pdf.ln(5)
#     else:
#         pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
#         pdf.chapter_body("AI analysis was not performed or data is unavailable in expected format.")
#         pdf.ln(5)

#     # 4. LSTM Numerical Results
#     if "lstm_results" in report_data:
#         pdf.chapter_title("LSTM Model - Numerical Summary")
#         lstm_metrics = report_data["lstm_results"].get("test_metrics", {})
#         pdf.add_metric("LSTM", "Test MAE", f"{lstm_metrics.get('mae', 'N/A'):.4f}")
#         pdf.add_metric("LSTM", "Test R2 Score", f"{lstm_metrics.get('r2_score', 'N/A'):.4f}")
        
#         future_preds_lstm = report_data["lstm_results"].get("future_predictions", {})
#         if future_preds_lstm:
#             pdf.ln(3); pdf.set_font('Arial', 'B', 10); pdf.cell(0, 6, "LSTM Future Predictions (Table):", 0, 1)
#             headers = ["Date", "Predicted Price (LSTM)"]
#             table_data = [[date, f"{price:.2f}"] for date, price in future_preds_lstm.items()]
#             pdf.add_table(headers, table_data, col_widths=[40, 50]) # Adjusted col width
#         pdf.ln(3)

#     # 5. Polynomial Regression Numerical Results
#     if "polynomial_results" in report_data:
#         pdf.chapter_title("Polynomial Regression - Numerical Summary")
#         poly_metrics = report_data["polynomial_results"].get("test_metrics", {})
#         pdf.add_metric("Polynomial", "Test MAE", f"{poly_metrics.get('mae', 'N/A'):.4f}")
#         pdf.add_metric("Polynomial", "Test R2 Score", f"{poly_metrics.get('r2_score', 'N/A'):.4f}")

#         future_preds_poly = report_data["polynomial_results"].get("future_predictions", {})
#         if future_preds_poly:
#             pdf.ln(3); pdf.set_font('Arial', 'B', 10); pdf.cell(0, 6, "Polynomial Future Predictions (Table):", 0, 1)
#             headers = ["Date", "Predicted Price (Poly)"]
#             table_data = [[date, f"{price:.2f}"] for date, price in future_preds_poly.items()]
#             pdf.add_table(headers, table_data, col_widths=[40, 50]) # Adjusted col width
#         pdf.ln(3)

#     # 6. Raw JSON Data Summary (Selected parts of results_payload)
#     # We will pretty-print relevant parts of the report_data, excluding verbose items like full plot_urls or AI analysis again.
#     summary_data_for_json_dump = {
#         "run_id": report_data.get("run_id"),
#         "csv_filename": report_data.get("csv_filename"),
#         "trading_suggestion_tomorrow": report_data.get("trading_suggestion_tomorrow"),
#         "lstm_metrics": report_data.get("lstm_results", {}).get("test_metrics"),
#         "lstm_future_predictions_sample": dict(list(report_data.get("lstm_results", {}).get("future_predictions", {}).items())[:5]), # Sample
#         "polynomial_metrics": report_data.get("polynomial_results", {}).get("test_metrics"),
#         "polynomial_future_predictions_sample": dict(list(report_data.get("polynomial_results", {}).get("future_predictions", {}).items())[:5]) # Sample
#     }
#     if pdf.get_y() + 60 > pdf.page_break_trigger : pdf.add_page() # Check space before large JSON dump
#     pdf.add_json_dump("Raw Data Summary (JSON Snippet)", summary_data_for_json_dump)


#     # 7. Visualizations
#     if plot_paths:
#       pdf.add_page() # Always start plots on a new page for better layout
#       pdf.chapter_title("Visualizations")
#       if plot_paths.get("lstm_test_plot"):
#           pdf.add_plot_image(plot_paths["lstm_test_plot"].replace("/static/", "static/"), "LSTM: Test Set Predictions")
#       if plot_paths.get("lstm_future_plot"):
#           pdf.add_plot_image(plot_paths["lstm_future_plot"].replace("/static/", "static/"), "LSTM: Historical, Test & Future Predictions")
#       if plot_paths.get("polynomial_plot"):
#           pdf.add_plot_image(plot_paths["polynomial_plot"].replace("/static/", "static/"), "Polynomial Regression Predictions")

#     try:
#         pdf.output(pdf_filepath, 'F')
#         print(f"Generated PDF report: {pdf_filepath}")
#     except Exception as e:
#         print(f"Error saving PDF {pdf_filepath}: {e}")
#         # Consider logging the full report_data here for debugging PDF generation
#         # print("Data passed to PDF generator:", json.dumps(report_data, indent=2, default=str))
#         raise
#     return pdf_filepath












































































































# # utils/pdf_generator.py
# from fpdf import FPDF, fpdf # Import fpdf module itself for FPDFException
# from datetime import datetime
# import os
# import pandas as pd
# import textwrap

# PDF_DIR = "static/pdfs"
# os.makedirs(PDF_DIR, exist_ok=True)

# class PDFReport(FPDF):
#     def header(self):
#         self.set_font('Arial', 'B', 12)
#         self.cell(0, 10, 'Stock Prediction Report', 0, 1, 'C')
#         self.set_font('Arial', '', 8)
#         self.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
#         self.ln(5)

#     def footer(self):
#         self.set_y(-15)
#         self.set_font('Arial', 'I', 8)
#         self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

#     def chapter_title(self, title):
#         self.set_font('Arial', 'B', 12)
#         self.cell(0, 10, title, 0, 1, 'L')
#         self.ln(2)

#     def safe_multi_cell(self, w, h, txt, border=0, align='L', fill=False, max_char_per_word_segment=30):
#         """
#         A more robust multi_cell that attempts to break very long words
#         if FPDF's internal breaking fails.
#         """
#         try:
#             self.multi_cell(w, h, txt, border, align, fill)
#         except fpdf.FPDFException as e: # Catch the specific FPDFException
#             if "Not enough horizontal space" in str(e) and len(txt) > max_char_per_word_segment :
#                 print(f"FPDFException caught for text: '{txt[:100]}...'. Attempting to segment.")
#                 # Try to break the problematic text into smaller chunks
#                 # This is a simple split, can be made smarter
#                 segmented_text = ""
#                 current_segment = ""
#                 for char in txt:
#                     current_segment += char
#                     if len(current_segment) >= max_char_per_word_segment:
#                         # Try to find a natural break near here if possible, otherwise force it
#                         # This example forces break, can be improved
#                         segmented_text += current_segment + "\n" # Add a newline as a hint
#                         current_segment = ""
#                 if current_segment: # Add any remaining part
#                     segmented_text += current_segment
                
#                 # Re-attempt with the segmented text
#                 # This might still fail if a segment itself contains an unbreakable part for FPDF
#                 # but it's an attempt to mitigate.
#                 print(f"Retrying with segmented text: '{segmented_text[:100]}...'")
#                 try:
#                     self.multi_cell(w, h, segmented_text, border, align, fill)
#                 except fpdf.FPDFException as e2:
#                     print(f"FPDFException on retry with segmented text: {e2}. Writing placeholder.")
#                     self.multi_cell(w, h, "[Text too long to render, see JSON output]", border, align, fill)
#             else:
#                 # If it's another FPDFException or text is already short, re-raise or handle
#                 print(f"Unhandled FPDFException or text too short to segment: {e}. Writing placeholder.")
#                 self.multi_cell(w, h, "[Error rendering text, see JSON output]", border, align, fill)
#         except Exception as e_other: # Catch any other unexpected error
#             print(f"Unexpected error in safe_multi_cell for text '{txt[:50]}...': {e_other}")
#             self.multi_cell(w, h, "[Unexpected error rendering text]", border, align, fill)


#     def chapter_body(self, content, char_wrap_limit=80):
#         # ... (font saving/restoring logic) ...
#         self.set_font('Arial', '', 10)
#         content_str = str(content)
#         wrapped_lines = textwrap.wrap(
#             content_str, width=char_wrap_limit, break_long_words=True, 
#             replace_whitespace=False, drop_whitespace=False, break_on_hyphens=True
#         )
#         for line in wrapped_lines:
#             self.safe_multi_cell(0, 5, line, align='L') # Use safe_multi_cell
#         self.ln(1)
#         # ... (font restoring) ...

#     def add_observation(self, observation_text, char_wrap_limit=75):
#         # ... (font saving/restoring logic) ...
#         self.set_font('Arial', '', 9)
#         prefix = "- "
#         observation_str = str(observation_text)
#         wrapped_lines = textwrap.wrap(
#             observation_str, width=char_wrap_limit, break_long_words=True,
#             replace_whitespace=False, drop_whitespace=False, break_on_hyphens=True
#         )
#         for i, line in enumerate(wrapped_lines):
#             text_to_render = f"{prefix}{line}" if i == 0 else line
#             x_offset = 0 if i == 0 else (self.l_margin + self.get_string_width(prefix) + 1)
            
#             if x_offset > 0 : self.set_x(x_offset)
#             self.safe_multi_cell(0, 5, text_to_render, align='L') # Use safe_multi_cell
#             if x_offset > 0 : self.set_x(self.l_margin) # Reset x for next line if it was indented
#         # ... (font restoring) ...

#     def add_suggestion(self, suggestion_signal, suggestion_basis, char_wrap_limit=80):
#         self.set_font('Arial', 'B', 11)
#         self.cell(0, 10, f"Trading Suggestion for Tomorrow: {suggestion_signal}", 0, 1, 'L')
#         self.ln(1)
        
#         # ... (font saving/restoring logic) ...
#         self.set_font('Arial', '', 10)
#         wrapped_basis = textwrap.wrap(
#             str(suggestion_basis), width=char_wrap_limit, break_long_words=True,
#             replace_whitespace=False, drop_whitespace=False, break_on_hyphens=True
#         )
#         for line in wrapped_basis:
#             self.safe_multi_cell(0, 5, line, align='L') # Use safe_multi_cell
#         self.ln(3)
#         # ... (font restoring) ...

#     # ... (add_table, add_metric, add_plot_image remain the same) ...


# def generate_prediction_report(run_id: str, report_data: dict, plot_paths: dict):
#     # ... (setup is the same) ...
#     # The calls to pdf.chapter_body, pdf.add_observation, pdf.add_suggestion
#     # will now use the self.safe_multi_cell internally if they call it.
#     # Ensure the parts of generate_prediction_report that directly call multi_cell
#     # are also updated or use chapter_body/add_observation.

#     # Example: if ai_analysis is a string (error message)
#     # elif isinstance(ai_analysis, str):
#     #     pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
#     #     pdf.set_font('Arial', 'B', 10)
#     #     # pdf.multi_cell(0, 5, f"Error during AI Analysis: {ai_analysis}") # OLD
#     #     pdf.chapter_body(f"Error during AI Analysis: {ai_analysis}") # NEW (uses safe_multi_cell via chapter_body)
#     #     pdf.ln(5)

#     # The rest of generate_prediction_report structure can remain largely the same,
#     # as the methods it calls (`chapter_body`, `add_observation`, `add_suggestion`)
#     # have been internally updated to be more robust or use `safe_multi_cell`.

#     # ... (Your existing structure for generate_prediction_report) ...
#     # Make sure to call the updated methods:
#     # pdf.chapter_body(...) for general text
#     # pdf.add_observation(...) for list items
#     # pdf.add_suggestion(...) for the trading suggestion basis

#     # The structure provided in the previous answer for generate_prediction_report
#     # should largely work if the class methods (chapter_body, add_observation, add_suggestion)
#     # are correctly implemented with robust text handling as shown above.
#     # I will re-paste the relevant parts of generate_prediction_report with these calls in mind.

#     pdf_filename = f"{run_id}_report.pdf"
#     pdf_filepath = os.path.join(PDF_DIR, pdf_filename)

#     pdf = PDFReport()
#     pdf.alias_nb_pages()
#     pdf.add_page()

#     pdf.chapter_title("Run Information")
#     pdf.chapter_body(f"Report ID: {run_id}")
#     pdf.chapter_body(f"CSV Processed: {report_data.get('csv_filename', 'N/A')}")
#     pdf.ln(5)
    
#     ai_analysis = report_data.get("ai_qualitative_analysis")
#     if isinstance(ai_analysis, dict):
#         pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
#         pdf.chapter_body(f"Stock Symbol: {ai_analysis.get('stockSymbol', 'N/A')}")
#         pdf.chapter_body(f"Analysis Date: {ai_analysis.get('analysisDate', 'N/A')}")
#         pdf.ln(1)
#         pdf.set_font('Arial', 'B', 10)
#         pdf.cell(0, 6, f"Overall Sentiment: {ai_analysis.get('overallSentiment', 'N/A')}", 0, 1)
#         pdf.chapter_body(f"Rationale: {ai_analysis.get('sentimentRationale', 'N/A')}") # Uses safe_multi_cell
#         pdf.ln(3)

#         lstm_analysis = ai_analysis.get("lstmModelAnalysis", {})
#         # ... (similar for other sections, using chapter_body or add_observation) ...
#         pdf.set_font('Arial', 'BU', 10)
#         pdf.cell(0, 6, "LSTM Model Insights:", 0, 1)
#         pdf.chapter_body(f"Test Performance: {lstm_analysis.get('performanceOnTest', 'N/A')}")
#         pdf.chapter_body(f"Future Outlook: {lstm_analysis.get('futureOutlook', 'N/A')}")
#         pdf.chapter_body(f"Confidence: {lstm_analysis.get('confidenceInOutlook', 'N/A')}")
#         pdf.ln(2)

#         poly_analysis = ai_analysis.get("polynomialRegressionAnalysis", {})
#         pdf.set_font('Arial', 'BU', 10)
#         pdf.cell(0, 6, "Polynomial Regression Insights:", 0, 1)
#         pdf.chapter_body(f"Test Performance: {poly_analysis.get('performanceOnTest', 'N/A')}")
#         pdf.chapter_body(f"Future Outlook: {poly_analysis.get('futureOutlook', 'N/A')}")
#         pdf.chapter_body(f"Confidence: {poly_analysis.get('confidenceInOutlook', 'N/A')}")
#         pdf.ln(2)
        
#         combined_outlook = ai_analysis.get("combinedOutlook", {})
#         pdf.set_font('Arial', 'BU', 10)
#         pdf.cell(0, 6, "Combined Outlook & Observations:", 0, 1)
#         pdf.chapter_body(f"Synopsis: {combined_outlook.get('shortTermForecastSynopsis', 'N/A')}")
#         key_observations = combined_outlook.get("keyObservations", [])
#         if key_observations:
#             pdf.set_font('Arial', 'B', 9) # Set font before calling add_observation if it doesn't set it
#             pdf.cell(0, 5, "Key Observations:", 0, 1)
#             for obs in key_observations:
#                 pdf.add_observation(obs) 
#         pdf.ln(2)

#         risk_factors = ai_analysis.get("riskFactors", [])
#         if risk_factors:
#             pdf.set_font('Arial', 'BU', 10)
#             pdf.cell(0, 6, "Identified Risk Factors:", 0, 1)
#             for risk in risk_factors:
#                 pdf.add_observation(risk) 
#         pdf.ln(2)
        
#         pdf.chapter_body(ai_analysis.get('disclaimer', "Standard AI analysis disclaimer applies."), char_wrap_limit=100)
#         pdf.ln(5)

#     elif isinstance(ai_analysis, str): # Error message from Gemini
#         pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
#         pdf.chapter_body(f"Error during AI Analysis: {ai_analysis}") # Will use safe_multi_cell
#         pdf.ln(5)
#     else:
#         pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
#         pdf.chapter_body("AI analysis was not performed or data is unavailable.")
#         pdf.ln(5)

#     trading_suggestion_data = report_data.get('trading_suggestion_tomorrow', {})
#     if trading_suggestion_data and trading_suggestion_data.get('signal') != "N/A":
#         pdf.add_suggestion( # This method now has enhanced wrapping
#             trading_suggestion_data.get('signal', 'N/A'),
#             trading_suggestion_data.get('reason', 'N/A')
#         )

#     # ... (Numerical results and plots sections - these use add_metric, add_table, add_plot_image which are generally fine)
#     if "lstm_results" in report_data:
#         pdf.chapter_title("LSTM Model Numerical Results")
#         lstm_metrics = report_data["lstm_results"].get("test_metrics", {})
#         pdf.add_metric("LSTM", "MAE", f"{lstm_metrics.get('mae', 'N/A'):.4f}")
#         pdf.add_metric("LSTM", "R2 Score", f"{lstm_metrics.get('r2_score', 'N/A'):.4f}")
        
#         future_preds_lstm = report_data["lstm_results"].get("future_predictions", {})
#         if future_preds_lstm:
#             pdf.ln(3); pdf.set_font('Arial', 'B', 10); pdf.cell(0, 6, "LSTM Future Predictions (Table):", 0, 1)
#             pdf.add_table(["Date", "Predicted Price"], [[date, f"{price:.2f}"] for date, price in future_preds_lstm.items()], col_widths=[40, 40])

#     if "polynomial_results" in report_data:
#         pdf.chapter_title("Polynomial Regression Numerical Results")
#         poly_metrics = report_data["polynomial_results"].get("test_metrics", {})
#         pdf.add_metric("Polynomial", "MAE", f"{poly_metrics.get('mae', 'N/A'):.4f}")
#         pdf.add_metric("Polynomial", "R2 Score", f"{poly_metrics.get('r2_score', 'N/A'):.4f}")

#         future_preds_poly = report_data["polynomial_results"].get("future_predictions", {})
#         if future_preds_poly:
#             pdf.ln(3); pdf.set_font('Arial', 'B', 10); pdf.cell(0, 6, "Polynomial Future Predictions (Table):", 0, 1)
#             pdf.add_table(["Date", "Predicted Price"], [[date, f"{price:.2f}"] for date, price in future_preds_poly.items()], col_widths=[40, 40])
    
#     if plot_paths:
#       pdf.add_page()
#       pdf.chapter_title("Visualizations")
#       if plot_paths.get("lstm_test_plot"):
#           pdf.add_plot_image(plot_paths["lstm_test_plot"].replace("/static/", "static/"), "LSTM: Test Set Predictions")
#       if plot_paths.get("lstm_future_plot"):
#           pdf.add_plot_image(plot_paths["lstm_future_plot"].replace("/static/", "static/"), "LSTM: Historical, Test & Future Predictions")
#       if plot_paths.get("polynomial_plot"):
#           pdf.add_plot_image(plot_paths["polynomial_plot"].replace("/static/", "static/"), "Polynomial Regression Predictions")

#     try:
#         pdf.output(pdf_filepath, 'F')
#         print(f"Generated PDF report: {pdf_filepath}")
#     except Exception as e:
#         print(f"Error saving PDF {pdf_filepath}: {e}")
#         raise
#     return pdf_filepath










# # utils/pdf_generator.py
# from fpdf import FPDF
# from datetime import datetime
# import os
# import pandas as pd
# import textwrap

# PDF_DIR = "static/pdfs"
# os.makedirs(PDF_DIR, exist_ok=True)

# class PDFReport(FPDF):
#     def header(self):
#         self.set_font('Arial', 'B', 12)
#         self.cell(0, 10, 'Stock Prediction Report', 0, 1, 'C')
#         self.set_font('Arial', '', 8)
#         self.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
#         self.ln(5)

#     def footer(self):
#         self.set_y(-15)
#         self.set_font('Arial', 'I', 8)
#         self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

#     def chapter_title(self, title):
#         self.set_font('Arial', 'B', 12)
#         self.cell(0, 10, title, 0, 1, 'L')
#         self.ln(2)

#     def chapter_body(self, content, char_wrap_limit=80): # Reduced char_wrap_limit
#         current_font_family = self.font_family
#         current_font_style = self.font_style
#         current_font_size = self.font_size
#         self.set_font('Arial', '', 10) # Ensure consistent font for wrapping

#         content_str = str(content)
#         # Using a slightly smaller width for textwrap to give FPDF more leeway
#         wrapped_lines = textwrap.wrap(
#             content_str, 
#             width=char_wrap_limit, 
#             break_long_words=True, # This is key
#             replace_whitespace=False, 
#             drop_whitespace=False,
#             break_on_hyphens=True
#         )
#         for line in wrapped_lines:
#             self.multi_cell(0, 5, line, border=0, align='L') # Use 'L' align for multi_cell
#         self.ln(1) # Reduced ln after body

#         # Restore font if it was different
#         self.set_font(current_font_family, current_font_style, current_font_size)


#     def add_observation(self, observation_text, char_wrap_limit=75): # Reduced further
#         current_font_family = self.font_family
#         current_font_style = self.font_style
#         current_font_size = self.font_size
#         self.set_font('Arial', '', 9) # Consistent font for this method

#         prefix = "- "
#         observation_str = str(observation_text)
        
#         # Calculate available width for the text part (after "- ")
#         available_width = self.w - self.l_margin - self.r_margin - self.get_string_width(prefix) - 2 # -2 for a small buffer

#         # Estimate char_wrap_limit based on available width (very rough)
#         # This is tricky as char width varies. A fixed smaller char_wrap_limit is safer.
#         # For font size 9, a char is roughly 9 * 0.2 = 1.8 points wide on average.
#         # effective_char_limit = int(available_width / (self.font_size * 0.25)) if self.font_size > 0 else char_wrap_limit
#         # Using the passed char_wrap_limit is usually more reliable if tuned.

#         wrapped_lines = textwrap.wrap(
#             observation_str, 
#             width=char_wrap_limit, 
#             break_long_words=True, 
#             replace_whitespace=False, 
#             drop_whitespace=False,
#             break_on_hyphens=True
#         )
        
#         for i, line in enumerate(wrapped_lines):
#             if i == 0:
#                 self.multi_cell(0, 5, f"{prefix}{line}", border=0, align='L')
#             else:
#                 # Indent subsequent lines
#                 self.set_x(self.l_margin + self.get_string_width(prefix) + 1) # Indent
#                 self.multi_cell(0, 5, line, border=0, align='L')
#                 self.set_x(self.l_margin) # Reset X for next potential observation
#         # No self.ln() needed as multi_cell adds it.

#         self.set_font(current_font_family, current_font_style, current_font_size)


#     def add_table(self, headers, data, col_widths=None):
#         # ... (no changes here, seems fine) ...
#         self.set_font('Arial', 'B', 9)
#         page_width = self.w - self.l_margin - self.r_margin
#         if col_widths is None:
#             num_cols = len(headers)
#             if num_cols > 0:
#                 col_width_val = page_width / num_cols
#                 col_widths = [col_width_val] * num_cols
#             else:
#                 return 
#         for i, header in enumerate(headers):
#             self.cell(col_widths[i], 7, str(header), 1, 0, 'C')
#         self.ln()
#         self.set_font('Arial', '', 8)
#         for row in data:
#             for i, item in enumerate(row):
#                 self.cell(col_widths[i], 6, str(item), 1, 0, 'L')
#             self.ln()
#         self.ln(5)

#     def add_metric(self, model_name, metric_name, value):
#         # ... (no changes here) ...
#         self.set_font('Arial', '', 10)
#         self.cell(0, 6, f"{model_name} - {metric_name}: {value}", 0, 1)
#         self.ln(1)

#     def add_suggestion(self, suggestion_signal, suggestion_basis, char_wrap_limit=80): # Reduced char_wrap_limit
#         self.set_font('Arial', 'B', 11)
#         self.cell(0, 10, f"Trading Suggestion for Tomorrow: {suggestion_signal}", 0, 1, 'L')
#         self.ln(1)
        
#         current_font_family = self.font_family
#         current_font_style = self.font_style
#         current_font_size = self.font_size
#         self.set_font('Arial', '', 10) # Ensure consistent font for basis

#         wrapped_basis = textwrap.wrap(
#             str(suggestion_basis), 
#             width=char_wrap_limit, 
#             break_long_words=True, 
#             replace_whitespace=False, 
#             drop_whitespace=False,
#             break_on_hyphens=True
#         )
#         for line in wrapped_basis:
#             self.multi_cell(0, 5, line, border=0, align='L') # Use 'L' align
#         self.ln(3) # Reduced ln

#         self.set_font(current_font_family, current_font_style, current_font_size)


#     def add_plot_image(self, image_path, title, width=170):
#         # ... (no changes here) ...
#         if os.path.exists(image_path):
#             self.chapter_title(title)
#             try:
#                 self.image(image_path, x=None, y=None, w=width)
#                 self.ln(5)
#             except Exception as e:
#                 print(f"Error adding image {image_path} to PDF: {e}")
#                 self.chapter_body(f"Error rendering plot: {os.path.basename(image_path)}")
#         else:
#             self.chapter_body(f"Plot not found: {os.path.basename(image_path)}")


# def generate_prediction_report(run_id: str, report_data: dict, plot_paths: dict):
#     # ... (same setup) ...
#     pdf_filename = f"{run_id}_report.pdf"
#     pdf_filepath = os.path.join(PDF_DIR, pdf_filename)

#     pdf = PDFReport()
#     pdf.alias_nb_pages()
#     pdf.add_page()

#     # ... (Run Information) ...
#     pdf.chapter_title("Run Information")
#     pdf.chapter_body(f"Report ID: {run_id}")
#     pdf.chapter_body(f"CSV Processed: {report_data.get('csv_filename', 'N/A')}") # Assuming you add this
#     pdf.ln(5)

#     # --- AI Qualitative Analysis (Order changed: AI Analysis before Trading Suggestion) ---
#     ai_analysis = report_data.get("ai_qualitative_analysis")
#     if isinstance(ai_analysis, dict):
#         pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
#         pdf.chapter_body(f"Stock Symbol: {ai_analysis.get('stockSymbol', 'N/A')}")
#         pdf.chapter_body(f"Analysis Date: {ai_analysis.get('analysisDate', 'N/A')}")
#         pdf.ln(1)
#         pdf.set_font('Arial', 'B', 10)
#         pdf.cell(0, 6, f"Overall Sentiment: {ai_analysis.get('overallSentiment', 'N/A')}", 0, 1)
#         pdf.set_font('Arial', '', 10) # Ensure font for rationale
#         pdf.chapter_body(f"Rationale: {ai_analysis.get('sentimentRationale', 'N/A')}")
#         pdf.ln(3)

#         # LSTM Insights
#         lstm_analysis = ai_analysis.get("lstmModelAnalysis", {})
#         pdf.set_font('Arial', 'BU', 10)
#         pdf.cell(0, 6, "LSTM Model Insights:", 0, 1)
#         pdf.set_font('Arial', '', 10)
#         pdf.chapter_body(f"Test Performance: {lstm_analysis.get('performanceOnTest', 'N/A')}")
#         pdf.chapter_body(f"Future Outlook: {lstm_analysis.get('futureOutlook', 'N/A')}")
#         pdf.chapter_body(f"Confidence: {lstm_analysis.get('confidenceInOutlook', 'N/A')}")
#         pdf.ln(2)

#         # Polynomial Insights
#         poly_analysis = ai_analysis.get("polynomialRegressionAnalysis", {})
#         pdf.set_font('Arial', 'BU', 10)
#         pdf.cell(0, 6, "Polynomial Regression Insights:", 0, 1)
#         pdf.set_font('Arial', '', 10)
#         pdf.chapter_body(f"Test Performance: {poly_analysis.get('performanceOnTest', 'N/A')}")
#         pdf.chapter_body(f"Future Outlook: {poly_analysis.get('futureOutlook', 'N/A')}")
#         pdf.chapter_body(f"Confidence: {poly_analysis.get('confidenceInOutlook', 'N/A')}")
#         pdf.ln(2)
        
#         # Combined Outlook
#         combined_outlook = ai_analysis.get("combinedOutlook", {})
#         pdf.set_font('Arial', 'BU', 10)
#         pdf.cell(0, 6, "Combined Outlook & Observations:", 0, 1)
#         pdf.set_font('Arial', '', 10)
#         pdf.chapter_body(f"Synopsis: {combined_outlook.get('shortTermForecastSynopsis', 'N/A')}")
#         key_observations = combined_outlook.get("keyObservations", [])
#         if key_observations:
#             pdf.set_font('Arial', 'B', 9)
#             pdf.cell(0, 5, "Key Observations:", 0, 1)
#             for obs in key_observations:
#                 pdf.add_observation(obs) 
#         pdf.ln(2)

#         # Risk Factors
#         risk_factors = ai_analysis.get("riskFactors", [])
#         if risk_factors:
#             pdf.set_font('Arial', 'BU', 10)
#             pdf.cell(0, 6, "Identified Risk Factors:", 0, 1)
#             for risk in risk_factors:
#                 pdf.add_observation(risk) 
#         pdf.ln(2)
        
#         pdf.set_font('Arial', 'I', 8)
#         pdf.chapter_body(ai_analysis.get('disclaimer', "Standard AI analysis disclaimer applies."), char_wrap_limit=100)
#         pdf.ln(5)

#     elif isinstance(ai_analysis, str):
#         pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
#         pdf.set_font('Arial', 'B', 10)
#         pdf.chapter_body(f"Error during AI Analysis: {ai_analysis}")
#         pdf.ln(5)
#     else:
#         pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
#         pdf.chapter_body("AI analysis was not performed or data is unavailable.")
#         pdf.ln(5)

#     # --- Trading Suggestion (Now after AI Analysis in PDF order) ---
#     trading_suggestion_data = report_data.get('trading_suggestion_tomorrow', {})
#     if trading_suggestion_data and trading_suggestion_data.get('signal') != "N/A": # Only add if signal is not N/A
#         pdf.add_suggestion(
#             trading_suggestion_data.get('signal', 'N/A'),
#             trading_suggestion_data.get('reason', 'N/A')
#         )
#     # ... (rest of numerical results and plots) ...
#     # (The order of sections for LSTM and Polynomial numerical results, then plots, seems fine)

#     if "lstm_results" in report_data:
#         pdf.chapter_title("LSTM Model Numerical Results")
#         # ... (rest of LSTM table)
#         lstm_metrics = report_data["lstm_results"].get("test_metrics", {})
#         pdf.add_metric("LSTM", "MAE", f"{lstm_metrics.get('mae', 'N/A'):.4f}")
#         pdf.add_metric("LSTM", "R2 Score", f"{lstm_metrics.get('r2_score', 'N/A'):.4f}")
        
#         future_preds_lstm = report_data["lstm_results"].get("future_predictions", {})
#         if future_preds_lstm:
#             pdf.ln(3)
#             pdf.set_font('Arial', 'B', 10)
#             pdf.cell(0, 6, "LSTM Future Predictions (Table):", 0, 1)
#             headers = ["Date", "Predicted Price"]
#             table_data = [[date, f"{price:.2f}"] for date, price in future_preds_lstm.items()]
#             pdf.add_table(headers, table_data, col_widths=[40, 40])


#     if "polynomial_results" in report_data:
#         pdf.chapter_title("Polynomial Regression Numerical Results")
#         # ... (rest of Poly table)
#         poly_metrics = report_data["polynomial_results"].get("test_metrics", {})
#         pdf.add_metric("Polynomial", "MAE", f"{poly_metrics.get('mae', 'N/A'):.4f}")
#         pdf.add_metric("Polynomial", "R2 Score", f"{poly_metrics.get('r2_score', 'N/A'):.4f}")

#         future_preds_poly = report_data["polynomial_results"].get("future_predictions", {})
#         if future_preds_poly:
#             pdf.ln(3)
#             pdf.set_font('Arial', 'B', 10)
#             pdf.cell(0, 6, "Polynomial Future Predictions (Table):", 0, 1)
#             headers = ["Date", "Predicted Price"]
#             table_data = [[date, f"{price:.2f}"] for date, price in future_preds_poly.items()]
#             pdf.add_table(headers, table_data, col_widths=[40, 40])

    
#     if plot_paths: # Check if plot_paths dictionary is not empty
#       pdf.add_page()
#       pdf.chapter_title("Visualizations") # General title for plots page
#       if plot_paths.get("lstm_test_plot"):
#           pdf.add_plot_image(plot_paths["lstm_test_plot"].replace("/static/", "static/"), "LSTM: Test Set Predictions")
#       if plot_paths.get("lstm_future_plot"):
#           pdf.add_plot_image(plot_paths["lstm_future_plot"].replace("/static/", "static/"), "LSTM: Historical, Test & Future Predictions")
#       if plot_paths.get("polynomial_plot"):
#           pdf.add_plot_image(plot_paths["polynomial_plot"].replace("/static/", "static/"), "Polynomial Regression Predictions")

#     try:
#         pdf.output(pdf_filepath, 'F')
#         print(f"Generated PDF report: {pdf_filepath}")
#     except Exception as e:
#         print(f"Error saving PDF {pdf_filepath}: {e}")
#         raise
#     return pdf_filepath



















































































# # utils/pdf_generator.py
# from fpdf import FPDF
# from datetime import datetime
# import os
# import pandas as pd # Make sure this is imported if you use pd types, though not directly in this snippet
# import textwrap # Ensure this is imported

# PDF_DIR = "static/pdfs"
# os.makedirs(PDF_DIR, exist_ok=True)

# class PDFReport(FPDF):
#     def header(self):
#         self.set_font('Arial', 'B', 12)
#         self.cell(0, 10, 'Stock Prediction Report', 0, 1, 'C')
#         self.set_font('Arial', '', 8)
#         self.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
#         self.ln(5)

#     def footer(self):
#         self.set_y(-15)
#         self.set_font('Arial', 'I', 8)
#         self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

#     def chapter_title(self, title):
#         self.set_font('Arial', 'B', 12)
#         self.cell(0, 10, title, 0, 1, 'L')
#         self.ln(2)

#     def chapter_body(self, content, char_wrap_limit=90):
#         self.set_font('Arial', '', 10)
#         # Ensure content is a string
#         content_str = str(content)
#         wrapped_lines = textwrap.wrap(content_str, width=char_wrap_limit, break_long_words=True, replace_whitespace=False, drop_whitespace=False)
#         for line in wrapped_lines:
#             self.multi_cell(0, 5, line)
#         self.ln() # Add a small space after the multi_cell block

#     def add_observation(self, observation_text, char_wrap_limit=85):
#         self.set_font('Arial', '', 9)
#         prefix = "- "
#         # Ensure observation_text is a string
#         observation_str = str(observation_text)
#         wrapped_lines = textwrap.wrap(observation_str, width=char_wrap_limit, break_long_words=True, replace_whitespace=False, drop_whitespace=False)
        
#         first_line = True
#         for line_idx, line in enumerate(wrapped_lines):
#             current_x = self.get_x()
#             current_y = self.get_y()
#             if first_line:
#                 self.multi_cell(0, 5, f"{prefix}{line}")
#                 first_line = False
#             else:
#                 # For subsequent lines, set X manually before multi_cell
#                 self.set_xy(current_x + self.get_string_width(prefix) + 1, current_y) # More precise indent
#                 self.multi_cell(0, 5, line)
#                 self.set_y(self.get_y()) # Ensure Y position is updated correctly after multi_cell
#         # No self.ln() here, as multi_cell handles its own line breaks.

#     def add_table(self, headers, data, col_widths=None):
#         self.set_font('Arial', 'B', 9)
#         # Calculate effective page width for table
#         page_width = self.w - self.l_margin - self.r_margin
#         if col_widths is None:
#             num_cols = len(headers)
#             if num_cols > 0:
#                 col_width_val = page_width / num_cols
#                 col_widths = [col_width_val] * num_cols
#             else:
#                 return # No headers, no table

#         # Header
#         for i, header in enumerate(headers):
#             self.cell(col_widths[i], 7, str(header), 1, 0, 'C')
#         self.ln()
        
#         # Data
#         self.set_font('Arial', '', 8)
#         for row in data:
#             for i, item in enumerate(row):
#                 self.cell(col_widths[i], 6, str(item), 1, 0, 'L')
#             self.ln()
#         self.ln(5)

#     def add_metric(self, model_name, metric_name, value):
#         self.set_font('Arial', '', 10)
#         self.cell(0, 6, f"{model_name} - {metric_name}: {value}", 0, 1)
#         self.ln(1)

#     def add_suggestion(self, suggestion_signal, suggestion_basis, char_wrap_limit=90):
#         self.set_font('Arial', 'B', 11)
#         self.cell(0, 10, f"Trading Suggestion for Tomorrow: {suggestion_signal}", 0, 1, 'L')
#         self.ln(1) # Smaller ln
#         self.set_font('Arial', '', 10)
#         # Wrap the basis text
#         wrapped_basis = textwrap.wrap(str(suggestion_basis), width=char_wrap_limit, break_long_words=True, replace_whitespace=False, drop_whitespace=False)
#         for line in wrapped_basis:
#             self.multi_cell(0, 50, line)
#         self.ln(5)


#     def add_plot_image(self, image_path, title, width=170): # Reduced width slightly
#         if os.path.exists(image_path):
#             self.chapter_title(title)
#             try:
#                 self.image(image_path, x=None, y=None, w=width)
#                 self.ln(5)
#             except Exception as e:
#                 print(f"Error adding image {image_path} to PDF: {e}")
#                 self.chapter_body(f"Error rendering plot: {os.path.basename(image_path)}")
#         else:
#             self.chapter_body(f"Plot not found: {os.path.basename(image_path)}")


# def generate_prediction_report(run_id: str, report_data: dict, plot_paths: dict):
#     pdf_filename = f"{run_id}_report.pdf"
#     pdf_filepath = os.path.join(PDF_DIR, pdf_filename)

#     pdf = PDFReport()
#     pdf.alias_nb_pages()
#     pdf.add_page()

#     pdf.chapter_title("Run Information")
#     pdf.chapter_body(f"Report ID: {run_id}")
#     pdf.chapter_body(f"CSV Processed: {report_data.get('csv_filename', 'N/A')}")
#     pdf.ln(5)
    
#     ai_analysis = report_data.get("ai_qualitative_analysis")
#     if isinstance(ai_analysis, dict):
#         pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
#         pdf.chapter_body(f"Stock Symbol: {ai_analysis.get('stockSymbol', 'N/A')}")
#         pdf.chapter_body(f"Analysis Date: {ai_analysis.get('analysisDate', 'N/A')}")
#         pdf.ln(1) # smaller ln
#         pdf.set_font('Arial', 'B', 10)
#         pdf.cell(0, 6, f"Overall Sentiment: {ai_analysis.get('overallSentiment', 'N/A')}", 0, 1)
#         pdf.set_font('Arial', '', 10)
#         pdf.chapter_body(f"Rationale: {ai_analysis.get('sentimentRationale', 'N/A')}")
#         pdf.ln(3)

#         # LSTM Insights
#         lstm_analysis = ai_analysis.get("lstmModelAnalysis", {})
#         pdf.set_font('Arial', 'BU', 10)
#         pdf.cell(0, 6, "LSTM Model Insights:", 0, 1)
#         pdf.set_font('Arial', '', 10)
#         pdf.chapter_body(f"Test Performance: {lstm_analysis.get('performanceOnTest', 'N/A')}")
#         pdf.chapter_body(f"Future Outlook: {lstm_analysis.get('futureOutlook', 'N/A')}")
#         pdf.chapter_body(f"Confidence: {lstm_analysis.get('confidenceInOutlook', 'N/A')}")
#         pdf.ln(2)

#         # Polynomial Insights
#         poly_analysis = ai_analysis.get("polynomialRegressionAnalysis", {})
#         pdf.set_font('Arial', 'BU', 10)
#         pdf.cell(0, 6, "Polynomial Regression Insights:", 0, 1)
#         pdf.set_font('Arial', '', 10)
#         pdf.chapter_body(f"Test Performance: {poly_analysis.get('performanceOnTest', 'N/A')}")
#         pdf.chapter_body(f"Future Outlook: {poly_analysis.get('futureOutlook', 'N/A')}")
#         pdf.chapter_body(f"Confidence: {poly_analysis.get('confidenceInOutlook', 'N/A')}")
#         pdf.ln(2)
        
#         # Combined Outlook
#         combined_outlook = ai_analysis.get("combinedOutlook", {})
#         pdf.set_font('Arial', 'BU', 10)
#         pdf.cell(0, 6, "Combined Outlook & Observations:", 0, 1)
#         pdf.set_font('Arial', '', 10)
#         pdf.chapter_body(f"Synopsis: {combined_outlook.get('shortTermForecastSynopsis', 'N/A')}")
#         key_observations = combined_outlook.get("keyObservations", [])
#         if key_observations:
#             pdf.set_font('Arial', 'B', 9)
#             pdf.cell(0, 5, "Key Observations:", 0, 1)
#             for obs in key_observations:
#                 pdf.add_observation(obs) # Using the new method
#         pdf.ln(2)

#         # Risk Factors
#         risk_factors = ai_analysis.get("riskFactors", [])
#         if risk_factors:
#             pdf.set_font('Arial', 'BU', 10)
#             pdf.cell(0, 6, "Identified Risk Factors:", 0, 1)
#             for risk in risk_factors:
#                 pdf.add_observation(risk) # Using the new method
#         pdf.ln(2)
        
#         pdf.set_font('Arial', 'I', 8)
#         pdf.chapter_body(ai_analysis.get('disclaimer', "Standard AI analysis disclaimer applies."), char_wrap_limit=100)
#         pdf.ln(5)

#     elif isinstance(ai_analysis, str):
#         pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
#         pdf.set_font('Arial', 'B', 10)
#         pdf.chapter_body(f"Error during AI Analysis: {ai_analysis}") # Use chapter_body for wrapping
#         pdf.ln(5)
#     else:
#         pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
#         pdf.chapter_body("AI analysis was not performed or data is unavailable.")
#         pdf.ln(5)

#     trading_suggestion_data = report_data.get('trading_suggestion_tomorrow', {})
#     if trading_suggestion_data:
#         pdf.add_suggestion(
#             trading_suggestion_data.get('signal', 'N/A'),
#             trading_suggestion_data.get('reason', 'N/A')
#         )

#     if "lstm_results" in report_data:
#         pdf.chapter_title("LSTM Model Numerical Results")
#         # ... (rest of LSTM table)
#         lstm_metrics = report_data["lstm_results"].get("test_metrics", {})
#         pdf.add_metric("LSTM", "MAE", f"{lstm_metrics.get('mae', 'N/A'):.4f}")
#         pdf.add_metric("LSTM", "R2 Score", f"{lstm_metrics.get('r2_score', 'N/A'):.4f}")
        
#         future_preds_lstm = report_data["lstm_results"].get("future_predictions", {})
#         if future_preds_lstm:
#             pdf.ln(3)
#             pdf.set_font('Arial', 'B', 10)
#             pdf.cell(0, 6, "LSTM Future Predictions (Table):", 0, 1)
#             headers = ["Date", "Predicted Price"]
#             table_data = [[date, f"{price:.2f}"] for date, price in future_preds_lstm.items()]
#             pdf.add_table(headers, table_data, col_widths=[40, 40])


#     if "polynomial_results" in report_data:
#         pdf.chapter_title("Polynomial Regression Numerical Results")
#         # ... (rest of Poly table)
#         poly_metrics = report_data["polynomial_results"].get("test_metrics", {})
#         pdf.add_metric("Polynomial", "MAE", f"{poly_metrics.get('mae', 'N/A'):.4f}")
#         pdf.add_metric("Polynomial", "R2 Score", f"{poly_metrics.get('r2_score', 'N/A'):.4f}")

#         future_preds_poly = report_data["polynomial_results"].get("future_predictions", {})
#         if future_preds_poly:
#             pdf.ln(3)
#             pdf.set_font('Arial', 'B', 10)
#             pdf.cell(0, 6, "Polynomial Future Predictions (Table):", 0, 1)
#             headers = ["Date", "Predicted Price"]
#             table_data = [[date, f"{price:.2f}"] for date, price in future_preds_poly.items()]
#             pdf.add_table(headers, table_data, col_widths=[40, 40])

    
#     if plot_paths: # Check if plot_paths dictionary is not empty
#       pdf.add_page()
#       pdf.chapter_title("Visualizations") # General title for plots page
#       if plot_paths.get("lstm_test_plot"):
#           pdf.add_plot_image(plot_paths["lstm_test_plot"].replace("/static/", "static/"), "LSTM: Test Set Predictions")
#       if plot_paths.get("lstm_future_plot"):
#           pdf.add_plot_image(plot_paths["lstm_future_plot"].replace("/static/", "static/"), "LSTM: Historical, Test & Future Predictions")
#       if plot_paths.get("polynomial_plot"):
#           pdf.add_plot_image(plot_paths["polynomial_plot"].replace("/static/", "static/"), "Polynomial Regression Predictions")

#     try:
#         pdf.output(pdf_filepath, 'F')
#         print(f"Generated PDF report: {pdf_filepath}")
#     except Exception as e:
#         print(f"Error saving PDF {pdf_filepath}: {e}")
#         # Potentially re-raise or handle more gracefully
#         raise
#     return pdf_filepath
















# # utils/pdf_generator.py
# from fpdf import FPDF
# from datetime import datetime
# import os
# import pandas as pd
# import textwrap # <-- ADD THIS IMPORT


# PDF_DIR = "static/pdfs" # Define where PDFs will be saved
# os.makedirs(PDF_DIR, exist_ok=True)

# class PDFReport(FPDF):
#     def header(self):
#         self.set_font('Arial', 'B', 12)
#         self.cell(0, 10, 'Stock Prediction Report', 0, 1, 'C')
#         self.set_font('Arial', '', 8)
#         self.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
#         self.ln(5)

#     def footer(self):
#         self.set_y(-15)
#         self.set_font('Arial', 'I', 8)
#         self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

#     def chapter_title(self, title):
#         self.set_font('Arial', 'B', 12)
#         self.cell(0, 10, title, 0, 1, 'L')
#         self.ln(2)

#     def chapter_body(self, content, char_wrap_limit=90): # Added char_wrap_limit
#         self.set_font('Arial', '', 10)
#         # Wrap long lines that might not have spaces
#         wrapped_lines = textwrap.wrap(str(content), width=char_wrap_limit, break_long_words=True, replace_whitespace=False)
#         for line in wrapped_lines:
#             self.multi_cell(0, 5, line) # Process each wrapped line
#         self.ln()

#     def add_observation(self, observation_text, char_wrap_limit=85): # Specific for observations
#         self.set_font('Arial', '', 9)
#         # Add the bullet point manually
#         prefix = "- "
#         remaining_width = self.w - self.l_margin - self.r_margin - self.get_string_width(prefix) # Approx remaining width
        
#         # Estimate wrap width based on remaining width and font size (rough)
#         # char_width_approx = self.font_size * 0.3 # Very rough estimate
#         # effective_wrap_limit = int(remaining_width / char_width_approx) if char_width_approx > 0 else char_wrap_limit

#         wrapped_lines = textwrap.wrap(str(observation_text), width=char_wrap_limit, break_long_words=True, replace_whitespace=False)
        
#         first_line = True
#         for line in wrapped_lines:
#             if first_line:
#                 self.multi_cell(0, 5, f"{prefix}{line}")
#                 first_line = False
#             else:
#                 # Indent subsequent lines of the same observation
#                 self.set_x(self.l_margin + self.get_string_width(prefix) + 1) # Indent by width of "- " + a bit
#                 self.multi_cell(0, 5, line)
#         # self.ln() # multi_cell adds its own line break, so an extra ln() might not be needed unless you want more space

#     def add_table(self, headers, data, col_widths=None):
#         self.set_font('Arial', 'B', 9)
#         if col_widths is None:
#             col_width = self.w / (len(headers) + 1) # distribute width
#             col_widths = [col_width] * len(headers)
        
#         # Header
#         for i, header in enumerate(headers):
#             self.cell(col_widths[i], 7, header, 1, 0, 'C')
#         self.ln()
        
#         # Data
#         self.set_font('Arial', '', 8)
#         for row in data:
#             for i, item in enumerate(row):
#                 self.cell(col_widths[i], 6, str(item), 1, 0, 'L')
#             self.ln()
#         self.ln(5)

#     def add_metric(self, model_name, metric_name, value):
#         self.set_font('Arial', '', 10)
#         self.cell(0, 6, f"{model_name} - {metric_name}: {value}", 0, 1)
#         self.ln(1)

#     def add_suggestion(self, suggestion):
#         self.set_font('Arial', 'B', 11)
#         self.cell(0, 10, f"Trading Suggestion for Tomorrow: {suggestion}", 0, 1, 'L')
#         self.ln(5)

#     def add_plot_image(self, image_path, title, width=180):
#         if os.path.exists(image_path):
#             self.chapter_title(title)
#             self.image(image_path, x=None, y=None, w=width)
#             self.ln(5)
#         else:
#             self.chapter_body(f"Plot not found: {image_path}")


# def generate_prediction_report(run_id: str, report_data: dict, plot_paths: dict):
#     pdf_filename = f"{run_id}_report.pdf"
#     pdf_filepath = os.path.join(PDF_DIR, pdf_filename)

#     pdf = PDFReport()
#     pdf.alias_nb_pages() # For total page numbers
#     pdf.add_page()

#     # --- General Information ---
#     pdf.chapter_title("Run Information")
#     pdf.chapter_body(f"Report ID: {run_id}")
#     pdf.chapter_body(f"CSV Processed: {report_data.get('csv_filename', 'N/A')}") # Assuming you add this
#     pdf.ln(5)
    
#     # --- AI Qualitative Analysis ---
#     ai_analysis = report_data.get("ai_qualitative_analysis")
#     if isinstance(ai_analysis, dict): # Check if it's a successfully parsed dict
#         pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
#         pdf.chapter_body(f"Stock Symbol: {ai_analysis.get('stockSymbol', 'N/A')}")
#         pdf.chapter_body(f"Analysis Date: {ai_analysis.get('analysisDate', 'N/A')}")
#         pdf.chapter_body(f"Rationale: {ai_analysis.get('sentimentRationale', 'N/A')}") # Uses updated chapter_body
#         pdf.ln(2)
#         pdf.set_font('Arial', 'B', 10)
#         pdf.cell(0, 6, f"Overall Sentiment: {ai_analysis.get('overallSentiment', 'N/A')}", 0, 1)
#         pdf.set_font('Arial', '', 10)
#         pdf.chapter_body(f"Rationale: {ai_analysis.get('sentimentRationale', 'N/A')}")
#         pdf.ln(3)

#         if "lstmModelAnalysis" in ai_analysis:
#             pdf.set_font('Arial', 'BU', 10) # Underlined Bold
#             pdf.cell(0, 6, "LSTM Model Insights:", 0, 1)
#             pdf.set_font('Arial', '', 10)
#             pdf.chapter_body(f"Test Performance: {ai_analysis['lstmModelAnalysis'].get('performanceOnTest', 'N/A')}")
#             pdf.chapter_body(f"Future Outlook: {ai_analysis['lstmModelAnalysis'].get('futureOutlook', 'N/A')}")
#             pdf.chapter_body(f"Confidence: {ai_analysis['lstmModelAnalysis'].get('confidenceInOutlook', 'N/A')}")
#             pdf.ln(2)

#         if "polynomialRegressionAnalysis" in ai_analysis:
#             pdf.set_font('Arial', 'BU', 10)
#             pdf.cell(0, 6, "Polynomial Regression Insights:", 0, 1)
#             pdf.set_font('Arial', '', 10)
#             pdf.chapter_body(f"Test Performance: {ai_analysis['polynomialRegressionAnalysis'].get('performanceOnTest', 'N/A')}")
#             pdf.chapter_body(f"Future Outlook: {ai_analysis['polynomialRegressionAnalysis'].get('futureOutlook', 'N/A')}")
#             pdf.chapter_body(f"Confidence: {ai_analysis['polynomialRegressionAnalysis'].get('confidenceInOutlook', 'N/A')}")
#             pdf.ln(2)
        
#         if "combinedOutlook" in ai_analysis:
#             pdf.set_font('Arial', 'BU', 10)
#             pdf.cell(0, 6, "Combined Outlook & Observations:", 0, 1)
#             pdf.set_font('Arial', '', 10)
#             pdf.chapter_body(f"Synopsis: {ai_analysis['combinedOutlook'].get('shortTermForecastSynopsis', 'N/A')}")
#             if "keyObservations" in ai_analysis['combinedOutlook']:
#                 pdf.set_font('Arial', 'B', 9)
#                 pdf.cell(0, 5, "Key Observations:", 0, 1)
#                 # pdf.set_font('Arial', '', 9) # Font is set in add_observation
#                 for obs in ai_analysis['combinedOutlook']['keyObservations']:
#                     # pdf.multi_cell(0, 5, f"- {obs}")
#                      # pdf.multi_cell(0, 5, f"- {obs}") # OLD WAY
#                     pdf.add_observation(obs) # NEW WAY using textwrap
#             pdf.ln(2)
#             pdf.ln(2)

#         if "riskFactors" in ai_analysis:
#             pdf.set_font('Arial', 'BU', 10)
#             pdf.cell(0, 6, "Identified Risk Factors:", 0, 1)
#             # pdf.set_font('Arial', '', 9)
#             for risk in ai_analysis['riskFactors']:
#                 pdf.multi_cell(0, 5, f"- {risk}")
#             pdf.ln(2)
        
#         pdf.set_font('Arial', 'I', 8)
#         # pdf.multi_cell(0,5, ai_analysis.get('disclaimer', "Standard AI analysis disclaimer applies.")) # OLD WAY
#         pdf.chapter_body(ai_analysis.get('disclaimer', "Standard AI analysis disclaimer applies."), char_wrap_limit=100) # NEW WAY

#     elif isinstance(ai_analysis, str): # Error message from Gemini
#         pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
#         pdf.set_font('Arial', 'B', 10)
#         pdf.multi_cell(0, 5, f"Error during AI Analysis: {ai_analysis}")
#         pdf.ln(5)
#     else: # No AI analysis data
#         pdf.chapter_title("AI Qualitative Analysis (via Gemini)")
#         pdf.chapter_body("AI analysis was not performed or data is unavailable.")
#         pdf.ln(5)

#     # --- Trading Suggestion ---
#     if report_data.get('trading_suggestion_tomorrow'):
#         pdf.add_suggestion(report_data['trading_suggestion_tomorrow']['signal'])
#         pdf.chapter_body(f"Basis: {report_data['trading_suggestion_tomorrow']['reason']}")

#     # --- LSTM Model Results ---
#     if "lstm_results" in report_data:
#         pdf.chapter_title("LSTM Model Analysis")
#         lstm_metrics = report_data["lstm_results"].get("test_metrics", {})
#         pdf.add_metric("LSTM", "MAE", f"{lstm_metrics.get('mae', 'N/A'):.4f}")
#         pdf.add_metric("LSTM", "R2 Score", f"{lstm_metrics.get('r2_score', 'N/A'):.4f}")
        
#         future_preds_lstm = report_data["lstm_results"].get("future_predictions", {})
#         if future_preds_lstm:
#             pdf.ln(3)
#             pdf.set_font('Arial', 'B', 10)
#             pdf.cell(0, 6, "LSTM Future Predictions:", 0, 1)
#             headers = ["Date", "Predicted Price"]
#             table_data = [[date, f"{price:.2f}"] for date, price in future_preds_lstm.items()]
#             pdf.add_table(headers, table_data, col_widths=[40, 40])

#     # --- Polynomial Regression Results ---
#     if "polynomial_results" in report_data:
#         pdf.chapter_title("Polynomial Regression Analysis")
#         poly_metrics = report_data["polynomial_results"].get("test_metrics", {})
#         pdf.add_metric("Polynomial", "MAE", f"{poly_metrics.get('mae', 'N/A'):.4f}")
#         pdf.add_metric("Polynomial", "R2 Score", f"{poly_metrics.get('r2_score', 'N/A'):.4f}")

#         future_preds_poly = report_data["polynomial_results"].get("future_predictions", {})
#         if future_preds_poly:
#             pdf.ln(3)
#             pdf.set_font('Arial', 'B', 10)
#             pdf.cell(0, 6, "Polynomial Future Predictions:", 0, 1)
#             headers = ["Date", "Predicted Price"]
#             table_data = [[date, f"{price:.2f}"] for date, price in future_preds_poly.items()]
#             pdf.add_table(headers, table_data, col_widths=[40, 40])
    
#     pdf.add_page() # New page for plots

#     # --- Add Plots ---
#     if plot_paths.get("lstm_test_plot"):
#         pdf.add_plot_image(plot_paths["lstm_test_plot"].replace("/static/", "static/"), "LSTM: Test Set Predictions")
#     if plot_paths.get("lstm_future_plot"):
#         pdf.add_plot_image(plot_paths["lstm_future_plot"].replace("/static/", "static/"), "LSTM: Historical, Test & Future Predictions")
#     if plot_paths.get("polynomial_plot"):
#          pdf.add_plot_image(plot_paths["polynomial_plot"].replace("/static/", "static/"), "Polynomial Regression Predictions")

#     pdf.output(pdf_filepath, 'F')
#     print(f"Generated PDF report: {pdf_filepath}")
#     return pdf_filepath # Return the path to the generated PDF