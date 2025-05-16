# utils/gemini_analyzer.py
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
import pandas as pd

load_dotenv() # Load variables from .env file

# It's better to configure the API key once
API_KEY = os.getenv("GOOGLE_API_KEY_ANALYZER")
if not API_KEY:
    print("Warning: GOOGLE_API_KEY_ANALYZER not found in .env file.")
    # Potentially raise an error or use a default dummy key for offline testing
else:
    genai.configure(api_key=API_KEY)


def format_predictions_for_prompt(title: str, dates, predictions, actuals=None, limit=10):
    header = f"\n--- {title} ---\nDate       | Predicted   | Actual (if avail)\n"
    lines = [header]
    for i in range(min(len(dates), limit)):
        date_str = dates[i].strftime('%Y-%m-%d') if hasattr(dates[i], 'strftime') else str(dates[i])
        pred_str = f"{predictions.flatten()[i]:.2f}"
        actual_str = f"{actuals.flatten()[i]:.2f}" if actuals is not None and i < len(actuals) else "N/A"
        lines.append(f"{date_str:<10} | {pred_str:>10} | {actual_str:>17}")
    if len(dates) > limit:
        lines.append(f"... and {len(dates) - limit} more days.")
    return "\n".join(lines)

# utils/gemini_analyzer.py
# ... (imports) ...

def generate_analysis_prompt(
    stock_symbol: str,
    historical_data_df: pd.DataFrame,
    lstm_test_actual, lstm_test_pred, lstm_test_dates,
    lstm_future_pred, lstm_future_dates,
    gru_test_actual, gru_test_pred, gru_test_dates, # NEW
    gru_future_pred, gru_future_dates,             # NEW
    
    # Transformer
    transformer_test_actual, transformer_test_pred, transformer_test_dates, # NEW
    transformer_future_pred, transformer_future_dates,                     # NEW
    transformer_best_params: dict,                                          # NEW
    poly_test_actual, poly_test_pred, poly_test_dates,
    poly_future_pred, poly_future_dates,
    trading_suggestion: dict,
    lstm_best_params: dict, # NEW
    gru_best_params: dict   # NEW
):
    # ... (historical_snippet and formatting functions same as before) ...
    historical_snippet = "Recent Historical Closing Prices (last few days):\nDate       | Close\n"
    for idx, row in historical_data_df.tail(min(len(historical_data_df), 5)).iterrows():
        historical_snippet += f"{row['Date'].strftime('%Y-%m-%d'):<10} | {row['Close']:.2f}\n"

    lstm_test_snippet = format_predictions_for_prompt("LSTM Test Set (sample)", lstm_test_dates, lstm_test_pred, lstm_test_actual)
    lstm_future_snippet = format_predictions_for_prompt("LSTM Future Predictions (sample)", lstm_future_dates, lstm_future_pred)
    
    gru_test_snippet = format_predictions_for_prompt("GRU Test Set (sample)", gru_test_dates, gru_test_pred, gru_test_actual) # NEW
    gru_future_snippet = format_predictions_for_prompt("GRU Future Predictions (sample)", gru_future_dates, gru_future_pred) # NEW

    poly_test_snippet = format_predictions_for_prompt("Poly Reg Test Set (sample)", poly_test_dates, poly_test_pred, poly_test_actual)
    poly_future_snippet = format_predictions_for_prompt("Poly Reg Future Predictions (sample)", poly_future_dates, poly_future_pred)
    
    transformer_test_snippet = format_predictions_for_prompt("Transformer Test Set (sample)", transformer_test_dates, transformer_test_pred, transformer_test_actual) # NEW
    transformer_future_snippet = format_predictions_for_prompt("Transformer Future Predictions (sample)", transformer_future_dates, transformer_future_pred) # NEW






    prompt = f"""
**SYSTEM PROMPT: AI Stock Prediction Analysis & Sentiment Report Generator (JSON Output)**

**Your Role:**
You are an AI financial analyst assistant. Based *exclusively* on the provided stock data summary, LSTM model predictions, Polynomial Regression model predictions, and a basic trading suggestion, your task is to generate a concise, insightful analysis.
The output **MUST be a valid JSON object** adhering to the specified structure.



**Input Data Summary for Stock: {stock_symbol}**
{historical_snippet}

**LSTM Model (Best Params: {json.dumps(lstm_best_params)})**
{lstm_test_snippet}
{lstm_future_snippet}

**GRU Model (Best Params: {json.dumps(gru_best_params)})**
{gru_test_snippet}
{gru_future_snippet}

**Transformer Model (Best Params: {json.dumps(transformer_best_params)})** 
{transformer_test_snippet}
{transformer_future_snippet}


**Polynomial Regression Model**
{poly_test_snippet}
{poly_future_snippet}

Trading Suggestion for Tomorrow (from LSTM prediction vs. last close):
Signal: {trading_suggestion.get('signal', 'N/A')}
Reason: {trading_suggestion.get('reason', 'N/A')}


**Your Task: Generate a JSON Object with Analysis and Sentiment**
(Your existing JSON structure, but now ensure it has sections for GRU model analysis)
Provide qualitative insights based on the data. 

**JSON Output Structure:**
```json
{{
  "stockSymbol": "{stock_symbol}",
  "analysisDate": "YYYY-MM-DD (Today's Date)",
  "overallSentiment": "Neutral",
  "sentimentRationale": "Brief explanation for the overall sentiment, considering both models' outlook and the trading suggestion.",
  "dataSummary": {{
    "lastActualClose": "Price from historical_data_df",
    "lastActualDate": "Date of lastActualClose"
  }},
  "lstmModelAnalysis": {{
    "performanceOnTest": "Qualitative assessment (e.g., 'Appears to follow the trend moderately well', 'Shows some divergence from actuals'). Consider MAE/R2 if they were provided and reflect on them.",
    "futureOutlook": "Qualitative assessment of LSTM's future predictions (e.g., 'Predicts a slight upward trend', 'Indicates potential consolidation', 'Shows a significant drop').",
    "confidenceInOutlook": "Low/Medium/High - based on test performance and stability of future predictions. Be conservative."
  }},
    "gruModelAnalysis": {{  // NEW SECTION
    "bestParamsFound": {json.dumps(gru_best_params)},
    "performanceOnTest": "Qualitative assessment for GRU.",
    "futureOutlook": "Qualitative assessment of GRU's future predictions.",
    "confidenceInOutlook": "Low/Medium/High for GRU."
  }},
  
  "transformerModelAnalysis": {{  // NEW SECTION
    "bestParamsFound": {json.dumps(transformer_best_params)},
    "performanceOnTest": "Qualitative assessment for Transformer.",
    "futureOutlook": "Qualitative assessment of Transformer's future predictions.",
    "confidenceInOutlook": "Low/Medium/High for Transformer."
  }},
  "polynomialRegressionAnalysis": {{
    "performanceOnTest": "Qualitative assessment (e.g., 'Provides a smooth trendline that captures the general direction', 'Struggles with sharp turns').",
    "futureOutlook": "Qualitative assessment of Polynomial's future predictions (e.g., 'Suggests continued growth along the established trend', 'Indicates a potential peak based on the curve').",
    "confidenceInOutlook": "Low/Medium/High - based on how well the polynomial fit represents the data and the nature of polynomial extrapolation."
  }},
  combinedOutlook": {{
    "shortTermForecastSynopsis": "Synthesize insights from LSTM, GRU, Transformer and Polynomial models... Synthesize insights from both models for the short-term (next few days to a week). Note any agreements or disagreements between models.",
    "keyObservations": [
      "Observation 1: e.g., all models suggest a short-term price increase.",
      "Observation 2: e.g., LSTM is more volatile in its short-term predictions compared to the smoother polynomial trend.",
      "Observation 3: e.g., The trading suggestion to '{trading_suggestion.get('signal', 'N/A')}' is primarily driven by the LSTM's next-day forecast."
      // Add 2-4 key, distinct observations based on the provided data.
    ]
  }},
  "riskFactors": [
    "Model-based predictions are inherently uncertain and based on past data, which may not reflect future market conditions.",
    "Polynomial regression is prone to poor extrapolation beyond the observed data range, especially with higher degrees.",
    "LSTM models can be sensitive to input data and may not capture sudden market shocks or events not present in training data.",
    "The 'Buy/Sell' suggestion is a simple heuristic and not comprehensive financial advice."
  ],
  "disclaimer": "This AI-generated analysis is for informational purposes only and not financial advice. Predictions are speculative. Consult a qualified financial advisor before making investment decisions."
}}
"""
    return prompt.strip()

# ... (get_gemini_analysis function remains the same, ensure it returns a string that main.py then parses)


# def generate_analysis_prompt(
#     stock_symbol: str,
#     historical_data_df: pd.DataFrame, # Last ~100 days
#     lstm_test_actual, lstm_test_pred, lstm_test_dates,
#     lstm_future_pred, lstm_future_dates,
#     poly_test_actual, poly_test_pred, poly_test_dates,
#     poly_future_pred, poly_future_dates,
#     trading_suggestion: dict
# ):
#     # Prepare data snippets
#     recent_history_limit = 100 # Days of actual historical data to show
#     prediction_display_limit = 7 # Days of future predictions to show in prompt

#     historical_snippet = "Recent Historical Closing Prices (last few days):\nDate       | Close\n"
#     for idx, row in historical_data_df.tail(min(len(historical_data_df), 5)).iterrows(): # Show last 5 actuals
#         historical_snippet += f"{row['Date'].strftime('%Y-%m-%d'):<10} | {row['Close']:.2f}\n"

#     lstm_test_snippet = format_predictions_for_prompt(
#         "LSTM Test Set Performance (sample)",
#         lstm_test_dates, lstm_test_pred, lstm_test_actual, limit=prediction_display_limit
#     )
#     lstm_future_snippet = format_predictions_for_prompt(
#         "LSTM Future Predictions (sample)",
#         lstm_future_dates, lstm_future_pred, limit=prediction_display_limit
#     )
#     poly_test_snippet = format_predictions_for_prompt(
#         "Polynomial Regression Test Set Performance (sample)",
#         poly_test_dates, poly_test_pred, poly_test_actual, limit=prediction_display_limit
#     )
#     poly_future_snippet = format_predictions_for_prompt(
#         "Polynomial Regression Future Predictions (sample)",
#         poly_future_dates, poly_future_pred, limit=prediction_display_limit
#     )

#     prompt = f"""
# **SYSTEM PROMPT: AI Stock Prediction Analysis & Sentiment Report Generator (JSON Output)**

# **Your Role:**
# You are an AI financial analyst assistant. Based *exclusively* on the provided stock data summary, LSTM model predictions, Polynomial Regression model predictions, and a basic trading suggestion, your task is to generate a concise, insightful analysis.
# The output **MUST be a valid JSON object** adhering to the specified structure.

# **Input Data Summary for Stock: {stock_symbol}**

# {historical_snippet}
# "this is the LSTM MODEL I have made" 
# {lstm_test_snippet}
# this is the LSTM MODEL Future prediction 
# {lstm_future_snippet}
# this is the polynomial test snippet

# {poly_test_snippet}
# thi is the polynomial future predictions
# {poly_future_snippet}

# Trading Suggestion for Tomorrow (from LSTM prediction vs. last close):
# Signal: {trading_suggestion.get('signal', 'N/A')}
# Reason: {trading_suggestion.get('reason', 'N/A')}

# **Your Task: Generate a JSON Object with Analysis and Sentiment**

# Provide qualitative insights based on the data. Be cautious and emphasize that this is not financial advice.

# **JSON Output Structure:**
# ```json
# {{
#   "stockSymbol": "{stock_symbol}",
#   "analysisDate": "YYYY-MM-DD (Today's Date)",
#   "overallSentiment": "Neutral",
#   "sentimentRationale": "Brief explanation for the overall sentiment, considering both models' outlook and the trading suggestion.",
#   "dataSummary": {{
#     "lastActualClose": "Price from historical_data_df",
#     "lastActualDate": "Date of lastActualClose"
#   }},
#   "lstmModelAnalysis": {{
#     "performanceOnTest": "Qualitative assessment (e.g., 'Appears to follow the trend moderately well', 'Shows some divergence from actuals'). Consider MAE/R2 if they were provided and reflect on them.",
#     "futureOutlook": "Qualitative assessment of LSTM's future predictions (e.g., 'Predicts a slight upward trend', 'Indicates potential consolidation', 'Shows a significant drop').",
#     "confidenceInOutlook": "Low/Medium/High - based on test performance and stability of future predictions. Be conservative."
#   }},
#   "polynomialRegressionAnalysis": {{
#     "performanceOnTest": "Qualitative assessment (e.g., 'Provides a smooth trendline that captures the general direction', 'Struggles with sharp turns').",
#     "futureOutlook": "Qualitative assessment of Polynomial's future predictions (e.g., 'Suggests continued growth along the established trend', 'Indicates a potential peak based on the curve').",
#     "confidenceInOutlook": "Low/Medium/High - based on how well the polynomial fit represents the data and the nature of polynomial extrapolation."
#   }},
#   "combinedOutlook": {{
#     "shortTermForecastSynopsis": "Synthesize insights from both models for the short-term (next few days to a week). Note any agreements or disagreements between models.",
#     "keyObservations": [
#       "Observation 1: e.g., Both models suggest a short-term price increase.",
#       "Observation 2: e.g., LSTM is more volatile in its short-term predictions compared to the smoother polynomial trend.",
#       "Observation 3: e.g., The trading suggestion to '{trading_suggestion.get('signal', 'N/A')}' is primarily driven by the LSTM's next-day forecast."
#       // Add 2-4 key, distinct observations based on the provided data.
#     ]
#   }},
#   "riskFactors": [
#     "Model-based predictions are inherently uncertain and based on past data, which may not reflect future market conditions.",
#     "Polynomial regression is prone to poor extrapolation beyond the observed data range, especially with higher degrees.",
#     "LSTM models can be sensitive to input data and may not capture sudden market shocks or events not present in training data.",
#     "The 'Buy/Sell' suggestion is a simple heuristic and not comprehensive financial advice."
#   ],
#   "disclaimer": "This AI-generated analysis is for informational purposes only and not financial advice. Predictions are speculative. Consult a qualified financial advisor before making investment decisions."
# }}
# """
#     return prompt.strip()


async def get_gemini_analysis(prompt_text: str) -> dict: # Ensure return type is dict
    if not API_KEY:
        print("Error: Gemini API key not configured for get_gemini_analysis.")
        return {"error": "Gemini API key not configured."}
    try:
        model = genai.GenerativeModel('gemini-1.5-flash') # Changed to 1.5 flash as per your model files
                                                         # Consider gemini-1.5-pro-latest for better JSON
                                                         # and set generation_config={"response_mime_type":"application/json"}
                                                         # if using a model that robustly supports it.

        print("\n--- Sending Prompt to Gemini for Analysis ---")
        print(f"Prompt length: {len(prompt_text)} characters")
        # print(prompt_text[:1000] + "..." if len(prompt_text) > 1000 else prompt_text) # Log snippet

        # Forcing JSON output is best if the model supports it well.
        # For gemini-1.5-flash, we primarily rely on prompt engineering.
        # If using gemini-1.5-pro, you can add:
        # generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        # response = await model.generate_content_async(prompt_text, generation_config=generation_config)
        
        response = await model.generate_content_async(prompt_text)

        print("\n--- Received Response from Gemini (Raw Text) ---")
        # print(response.text[:1000] + "..." if len(response.text) > 1000 else response.text)

        # Log candidate and prompt feedback for detailed debugging
        if hasattr(response, 'candidates') and response.candidates:
            for i, candidate in enumerate(response.candidates):
                print(f"--- Candidate {i+1} ---")
                candidate_text = ""
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part_idx, part in enumerate(candidate.content.parts):
                        if hasattr(part, 'text'):
                            print(f"Part {part_idx} Text (first 500 chars): {part.text[:500]}...")
                            candidate_text += part.text # Concatenate parts if any
                if not candidate_text: # Fallback if parts structure is different or empty
                    if hasattr(response, 'text'): # Check if response object has text directly
                        candidate_text = response.text
                    else: # Last resort if no text found in common places
                        print("Warning: Could not extract text from candidate parts or response.text")
                        candidate_text = ""

                if hasattr(candidate, 'finish_reason'): print("Finish Reason:", candidate.finish_reason)
                if hasattr(candidate, 'safety_ratings'): print("Safety Ratings:", candidate.safety_ratings)
        elif hasattr(response, 'text'): # If no candidates but direct text
             candidate_text = response.text
             print(f"Direct response text (first 500 chars): {candidate_text[:500]}...")
        else:
            print("Error: No candidates or text found in Gemini response.")
            return {"error": "No text content found in Gemini response.", "raw_response_object": str(response)}


        if hasattr(response, 'prompt_feedback'):
             print("Prompt Feedback:", response.prompt_feedback)
        
        # Robust JSON extraction
        json_string_to_parse = None
        if "```json" in candidate_text:
            # Find the start of the JSON block
            start_index = candidate_text.find("```json") + len("```json")
            # Find the end of the JSON block
            end_index = candidate_text.rfind("```")
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_string_to_parse = candidate_text[start_index:end_index].strip()
            else: # Fallback if ```json is there but closing ``` is missing or before start
                json_string_to_parse = candidate_text[start_index:].strip() # Try to take everything after ```json
        elif candidate_text.strip().startswith("{") and candidate_text.strip().endswith("}"):
             json_string_to_parse = candidate_text.strip()
        
        if not json_string_to_parse:
            print("Error: Could not identify a JSON block in Gemini's response.")
            print(f"Full Gemini Response was: {candidate_text}")
            return {"error": "Could not identify JSON block in Gemini response.", "raw_response": candidate_text}

        try:
            print(f"\n--- Attempting to parse extracted JSON string (first 500 chars): ---\n{json_string_to_parse[:500]}...\n---")
            json_analysis = json.loads(json_string_to_parse)
            print("--- Successfully parsed JSON from Gemini ---")
            return json_analysis
        except json.JSONDecodeError as e:
            print(f"Error decoding extracted Gemini JSON string: {e}")
            print(f"Extracted string that failed parsing: {json_string_to_parse}")
            return {"error": "Failed to parse extracted JSON from Gemini.", "extracted_string": json_string_to_parse, "original_raw": candidate_text}

    except Exception as e:
        print(f"Error communicating with Gemini API or processing response: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Gemini API communication/processing error: {str(e)}"}