from flask import Flask, jsonify, render_template, request
import yfinance as yf
from flask_cors import CORS
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import os
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__, template_folder='.')
CORS(app)

@app.route('/')
def home():
    # Serves the frontend HTML page
    return render_template('index.html')

@app.route('/api/analyze')
def analyze_market():
    # URL parameters with sensible defaults for quantitative analysis
    ticker = request.args.get('ticker', default='TSLA', type=str).upper()
    threshold_sigma = request.args.get('sigma', default=2.5, type=float)
    
    try:
        # Fetching historical asset closing data
        df = yf.download(ticker, start="2024-01-01", end="2026-05-01", progress=False)
        if df.empty:
            return jsonify({"error": f"No data found for ticker {ticker}"}), 400
        
        closing_prices = df['Close'].dropna()
        dates = closing_prices.index.strftime('%Y-%m-%d').tolist()
        prices_list = closing_prices.values.flatten().tolist()
        
        # Fit deterministic statistical ARIMA(1,1,1) engine
        model = ARIMA(prices_list, order=(1, 1, 1))
        fitted_model = model.fit()
        
        # Extract mathematical residuals (Actual Price - Predicted Price)
        residuals = np.array(prices_list) - fitted_model.fittedvalues
        std_dev = np.std(residuals)
        
        # Map indices where the statistical variance breaks our threshold bounds
        anomaly_indices = np.where(np.abs(residuals) > (threshold_sigma * std_dev))[0]
        
        # Format payloads cleanly for frontend consumption
        anomalies_data = []
        for idx in anomaly_indices:
            # Warm-up period: Ignore any anomalies mathematically flagged in the first 30 days
            if int(idx) < 30:
                continue
                
            anomalies_data.append({
                "date": dates[int(idx)],
                "price": round(prices_list[int(idx)], 2),
                "deviation": round(float(residuals[int(idx)]), 2)
            })
            
        return jsonify({
            "ticker": ticker,
            "dates": dates,
            "prices": prices_list,
            "anomalies": anomalies_data,
            "threshold": round(threshold_sigma * std_dev, 2)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("[SYSTEM] High-performance analytics server ignition initialized...")
    app.run(debug=True, port=5000)