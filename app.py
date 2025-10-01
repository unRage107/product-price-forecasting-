from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, origins=["http://localhost:5000", "http://127.0.0.1:5000"])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = jsonify()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        df = pd.read_csv(file)
        price_col = None
        for col in ["Price", "Close", "Close Price", "Adj Close"]:
            if col in df.columns:
                price_col = col
                break
        if not price_col:
            return jsonify({'error': 'CSV must contain a Price, Close, Close Price, or Adj Close column'}), 400
        dates = None
        if "Date" in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                dates = df['Date'].values
            except:
                pass
        prices = df[price_col].astype(float).values
        if len(prices) < 10:
            return jsonify({'error': 'Not enough data points. Need at least 10 values for accurate prediction'}), 400
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices
        split_idx = int(len(prices) * 0.9)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        degree = 2
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train, y_train)
        y_pred = model.predict(X)
        test_pred = model.predict(X_test)
        mse = np.mean((y_test - test_pred) ** 2)
        # Avoid division by zero or nan
        if np.mean(y_test) != 0 and not np.isnan(mse):
            accuracy = max(0, 100 - (mse / np.mean(y_test)) * 100)
        else:
            accuracy = 0.0
        future_X = np.arange(len(prices), len(prices) + 5).reshape(-1, 1)
        predicted_prices = model.predict(future_X)
        predicted_price = predicted_prices[0]
        if predicted_price > prices[-1]:
            prediction = "UP 📈"
        else:
            prediction = "DOWN 📉"
        x_dates = np.arange(len(prices))
        future_dates = np.arange(len(prices), len(prices) + 5)
        plt.figure(figsize=(12, 6))
        if dates is not None:
            plt.plot(dates, prices, 'o-', color='royalblue', label="Historical Prices", linewidth=2, markersize=4)
            plt.fill_between(dates, prices, color='royalblue', alpha=0.08)
            last_date = df['Date'].iloc[-1]
            future_date_list = []
            current_date = last_date
            for i in range(5):
                current_date += timedelta(days=1)
                while current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                future_date_list.append(current_date)
            plt.plot(future_date_list, predicted_prices, 'o--', color='orange', label="Predicted Prices", linewidth=2, markersize=7)
            plt.scatter(future_date_list, predicted_prices, color='orange', s=60, edgecolor='black', zorder=5)
            plt.xticks(rotation=30)
        else:
            plt.plot(x_dates, prices, 'o-', color='royalblue', label="Historical Prices", linewidth=2, markersize=4)
            plt.fill_between(x_dates, prices, color='royalblue', alpha=0.08)
            plt.plot(future_dates, predicted_prices, 'o--', color='orange', label="Predicted Prices", linewidth=2, markersize=7)
            plt.scatter(future_dates, predicted_prices, color='orange', s=60, edgecolor='black', zorder=5)
        plt.plot(x_dates if dates is None else dates, y_pred, '-', color='crimson', label="Regression Line", linewidth=2, alpha=0.7)
        plt.xlabel('Date' if dates is not None else 'Trading Days')
        plt.ylabel('Price (₹)')
        plt.title('Product Price Forecast')
        plt.legend(frameon=True, loc='best')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plot_url = "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()
        response = jsonify({
            'prediction': prediction,
            'predicted_price': float(predicted_price),
            'predicted_prices': [float(price) for price in predicted_prices],
            'mse': float(mse),
            'accuracy': float(accuracy) if not np.isnan(accuracy) and not np.isinf(accuracy) else 0.0,
            'plot_url': plot_url,
            'current_price': float(prices[-1])
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        error_response = jsonify({'error': f'Error processing file: {str(e)}'})
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response, 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')