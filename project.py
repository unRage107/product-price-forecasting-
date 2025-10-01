from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    predicted_price = None
    predicted_prices = None
    plot_url = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            error = "Please upload a CSV file."
            return render_template("index.html", error=error)

        try:
            df = pd.read_csv(file)

            # Flexible column check
            if "Price" in df.columns:
                prices = df["Price"].astype(float).values[::-1]
            elif "Close" in df.columns:
                prices = df["Close"].astype(float).values[::-1]
            elif "Close Price" in df.columns:
                prices = df["Close Price"].astype(float).values[::-1]
            elif "Adj Close" in df.columns:
                prices = df["Adj Close"].astype(float).values[::-1]
            else:
                error = "CSV must contain 'Price', 'Close', 'Close Price', or 'Adj Close' column."
                return render_template("index.html", error=error)

            # Simple moving average prediction
            window = 5
            df["SMA"] = df.iloc[:, 0].rolling(window=window).mean()

            predicted_price = df["SMA"].iloc[-1]
            predicted_prices = list(df["SMA"].dropna().tail(5))

            if predicted_price > df.iloc[:, 0].iloc[-1]:
                prediction = "UP 📈"
            else:
                prediction = "DOWN 📉"

            # Plot
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, df.iloc[:, 0], label="Actual Price", color="blue")
            plt.plot(df.index, df["SMA"], label="Predicted (SMA)", color="red")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.title("Stock Price Prediction")
            plt.legend()
            plt.grid(True)

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plot_url = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            plt.close()

        except Exception as e:
            error = f"Error processing file: {str(e)}"

    return render_template("index.html",
                           prediction=prediction,
                           predicted_price=predicted_price,
                           predicted_prices=predicted_prices,
                           plot_url=plot_url,
                           error=error)

if __name__ == "__main__":
    app.run(debug=True)
