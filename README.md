# AI-Driven Stock Market Prediction using LSTM and Sentiment Analysis

A production-ready, end-to-end machine learning project that combines **LSTM deep learning** with **VADER sentiment analysis** to predict stock closing prices and generate Buy / Hold / Sell signals.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📥 **Data Collection** | Live OHLCV data via `yfinance`; CSV or built-in stub news headlines |
| 🧹 **Preprocessing** | MinMaxScaler normalisation, 60-step LSTM windows, text cleaning, stopword removal |
| 😊 **Sentiment Analysis** | VADER (default) **or** TF-IDF + Logistic Regression |
| 🏗️ **Feature Engineering** | Price + sentiment features merged on date; sequential (X, y) dataset |
| 🧠 **LSTM Model** | 2 LSTM layers + Dropout + Dense layers (TensorFlow/Keras) |
| ⏱️ **Training** | Early stopping, ReduceLROnPlateau, validation split |
| 📐 **Evaluation** | MAE & RMSE reported in original price scale |
| 📊 **Visualization** | Interactive Plotly charts (historical vs. predicted, sentiment trend, loss curve) |
| 🌐 **Streamlit App** | Ticker input → next-day price + Buy/Hold/Sell signal |
| 💾 **Model Persistence** | Saved as `.keras`; downloadable from the app |
| 🔁 **Multiple Stocks** | `get_multiple_stocks()` helper for batch processing |

---

## 📁 Project Structure

```
Stock-Trend-Prediction/
├── data_loader.py      # yfinance data fetching & news loading
├── preprocessing.py    # normalisation, LSTM sequences, sentiment pipeline
├── model.py            # LSTM build / train / evaluate / save / load
├── app.py              # Streamlit web application
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/pankaj1281/Stock-Trend-Prediction.git
cd Stock-Trend-Prediction
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥️ Using the App

1. Enter a **stock ticker** (e.g. `AAPL`, `TSLA`, `GOOG`) in the sidebar.
2. Choose a **historical data period** (1y – 10y).
3. Optionally upload a **news headlines CSV** with `date` and `headline` columns (a built-in stub is used otherwise).
4. Click **Run Prediction**.
5. The app will train an LSTM model (or load a previously saved one) and display:
   - Predicted next-day closing price
   - Buy / Hold / Sell signal
   - Interactive charts (predictions, sentiment trend, training loss)
   - Download button for the trained model (`.keras`)

---

## 🛠️ Using the Modules Programmatically

```python
from data_loader import fetch_stock_data, load_news_headlines
from preprocessing import build_lstm_dataset
from model import build_lstm_model, train_model, evaluate_model, predict_next_close

# 1. Fetch data
stock_df = fetch_stock_data("AAPL", period="5y")
news_df  = load_news_headlines()            # uses stub if no CSV path given

# 2. Preprocess → LSTM dataset
dataset = build_lstm_dataset(stock_df, news_df=news_df, window_size=60)

# 3. Build & train
model = build_lstm_model(
    input_shape=(60, len(dataset["feature_cols"]))
)
history = train_model(model, dataset["X_train"], dataset["y_train"])

# 4. Evaluate
results = evaluate_model(
    model,
    dataset["X_test"],
    dataset["y_test"],
    dataset["scaler"],
    dataset["feature_cols"],
)
print(f"MAE: {results['mae']:.4f}  RMSE: {results['rmse']:.4f}")

# 5. Predict next closing price
from preprocessing import normalize_stock_data
scaled_df, _ = normalize_stock_data(
    dataset["merged_df"], feature_cols=dataset["feature_cols"]
)
last_window = scaled_df.values[-60:]
price = predict_next_close(model, last_window, dataset["scaler"], dataset["feature_cols"])
print(f"Predicted next close: ${price:.2f}")
```

---

## 📰 News Headlines CSV Format

If you supply your own headlines file it must contain **at least** these two columns:

| Column | Type | Example |
|---|---|---|
| `date` | YYYY-MM-DD | `2024-03-15` |
| `headline` | string | `Apple beats Q1 earnings estimates` |

---

## ⚙️ Configuration

Key constants you can tweak:

| Location | Variable | Default | Description |
|---|---|---|---|
| `app.py` | `WINDOW_SIZE` | `60` | LSTM look-back window |
| `model.py` | `RANDOM_SEED` | `42` | Global random seed |
| `model.build_lstm_model` | `lstm_units` | `50` | Units per LSTM layer |
| `model.build_lstm_model` | `dropout_rate` | `0.2` | Dropout fraction |
| `model.train_model` | `epochs` | `50` | Max training epochs |
| `model.train_model` | `patience` | `10` | Early-stopping patience |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Historical stock price data |
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | MinMaxScaler, TF-IDF, Logistic Regression |
| `tensorflow` | LSTM model (Keras API) |
| `nltk` | VADER sentiment, tokenisation, stopwords |
| `streamlit` | Web application |
| `plotly` | Interactive charts |
| `matplotlib` | Static plots (optional) |

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.  
Predictions made by this model should **not** be used as financial advice.  
Always consult a qualified financial advisor before making investment decisions.

---

## 📄 License

[MIT](LICENSE)
