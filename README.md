---
title: Stox Sight ML Service
emoji:  चार्ट📈
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_file: main.py # Tells Spaces your main app is in main.py
app_port: 8000    # The port your Uvicorn server listens on inside the Docker container
---

# Stox Sight - FastAPI ML Prediction Service

This service provides stock price prediction using various ML models (LSTM, GRU, Transformer, Polynomial Regression)
and generates an analytical report with plots.

### example curl request

curl -X POST "http://127.0.0.1:8000/train-predict/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "csv_file=@/Users/meet/Desktop/stox-sight/csvGenerated/test_example.com/TCS_INDIA_YY2025MM05DD14_HH14MM23SS26.csv" \
     -F "user_email_to_send_to=email@email.com"


curl -X POST "http://127.0.0.1:8000/train-predict/" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "csv_file=@/Users/meet/Desktop/stox-sight/csvGenerated/test_example.com/ITC_INDIA_20250514_141615.csv" \
    -F "user_email_to_send_to=email@email.com"