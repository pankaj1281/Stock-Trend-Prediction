# AI-Powered Intelligent Stock Trading System

A production-ready, end-to-end machine learning project that combines
**BiLSTM + Attention deep learning**, **technical indicator feature engineering**,
**VADER / FinBERT sentiment analysis**, and a **backtesting engine** to predict
stock closing prices and generate Buy / Hold / Sell signals with confidence scores.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📥 **Data Collection** | Live OHLCV data via `yfinance`; CSV or built-in stub news headlines |
| 🧹 **Preprocessing** | MinMaxScaler normalisation, 60-step LSTM windows, text cleaning, stopword removal |
| 📐 **Technical Indicators** | RSI, MACD, SMA (20/50/200), EMA (12/26), Bollinger Bands, ATR, OBV, Volatility |
| 😊 **Sentiment Analysis** | VADER (baseline) **or** FinBERT (transformer-based, financial domain) with recency weighting |
| 🧠 **BiLSTM + Attention** | Bidirectional LSTM with Multi-Head Attention, Dropout, LayerNorm (production model) |
| 🏗️ **LSTM Baseline** | 2-layer unidirectional LSTM for comparison |
| 🌲 **Random Forest** | Scikit-learn benchmark with flattened sequences |
| ⏱️ **Training** | Early stopping, ReduceLROnPlateau, `ModelCheckpoint`, validation split |
| 📐 **Evaluation** | MAE, RMSE, R² reported in original price scale |
| 🔁 **Backtesting** | Portfolio simulation with Sharpe ratio, max drawdown, P&L, win rate |
| 🔬 **Explainable AI** | SHAP (if installed) or permutation-based feature importance |
| 📊 **Visualization** | Interactive Plotly charts (predictions, technicals, sentiment, backtest) |
| 🌐 **Streamlit App** | Full dashboard with all features, multi-stock comparison, watchlist, alerts |
| 💾 **Model Persistence** | Saved as `.keras`; downloadable from the app |
| 🔔 **Alerts** | Session-persistent prediction notifications |

---

## 📁 Project Structure

```
Stock-Trend-Prediction/
├── data_loader.py      # yfinance data fetching & news loading
├── features.py         # Technical indicator engineering (RSI, MACD, BB, …)
├── sentiment.py        # VADER + FinBERT sentiment with recency weighting
├── preprocessing.py    # Normalisation, LSTM sequences, full pipeline
├── model.py            # LSTM / BiLSTM+Attention / RF build, train, evaluate
├── backtesting.py      # Portfolio simulation, Sharpe ratio, P&L
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

> **Note:** `torch` and `transformers` are required only for FinBERT sentiment.
> `shap` is required only for SHAP feature importance.
> The application runs without these optional packages (VADER is used instead).

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥️ Using the App

1. Enter a **stock ticker** (e.g. `AAPL`, `TSLA`, `GOOG`) in the sidebar.
2. Choose a **historical data period** (1y – 10y).
3. Select the **model architecture** (BiLSTM+Attention or LSTM baseline).
4. Choose the **sentiment mode** (VADER fast or FinBERT advanced).
5. Optionally upload a **news headlines CSV** with `date` and `headline` columns.
6. Click **Run Prediction**.
7. The app will train a model (or load a saved one) and display:
   - Predicted next-day closing price with confidence score
   - Buy / Hold / Sell signal
   - Risk level indicator (Low / Medium / High based on volatility)
   - Technical indicator charts (RSI, MACD, Bollinger Bands)
   - Sentiment trend visualisation
   - Backtest simulation with Sharpe ratio and P&L
   - Model comparison dashboard
   - Feature importance chart

---

## 🛠️ Using the Modules Programmatically

```python
from data_loader import fetch_stock_data, load_news_headlines
from features import add_technical_indicators
from sentiment import merge_sentiment_with_stock
from preprocessing import build_lstm_dataset
from model import build_bilstm_model, train_model, evaluate_model, predict_next_close
from backtesting import run_backtest, summary_report

# 1. Fetch data
stock_df = fetch_stock_data("AAPL", period="5y")
news_df  = load_news_headlines()            # stub headlines if no CSV given

# 2. Preprocess → full LSTM dataset (technical indicators + sentiment)
dataset = build_lstm_dataset(stock_df, news_df=news_df, window_size=60,
                              add_indicators=True)

# 3. Build & train BiLSTM + Attention model
model = build_bilstm_model(input_shape=(60, len(dataset["feature_cols"])))
history = train_model(model, dataset["X_train"], dataset["y_train"],
                      checkpoint_path="saved_models/best.keras")

# 4. Evaluate
results = evaluate_model(model, dataset["X_test"], dataset["y_test"],
                          dataset["scaler"], dataset["feature_cols"])
print(f"MAE={results['mae']:.4f}  RMSE={results['rmse']:.4f}  R²={results['r2']:.4f}")

# 5. Predict next closing price
from preprocessing import normalize_stock_data
scaled_df, _ = normalize_stock_data(dataset["merged_df"],
                                     feature_cols=dataset["feature_cols"])
last_window = scaled_df.values[-60:]
price = predict_next_close(model, last_window, dataset["scaler"],
                            dataset["feature_cols"])
print(f"Predicted next close: ${price:.2f}")

# 6. Backtest
import pandas as pd, numpy as np
test_prices = pd.Series(results["actuals"],
                         index=dataset["merged_df"].index[-len(results["actuals"]):])
bt = run_backtest(test_prices, results["predictions"], initial_capital=10_000)
print(summary_report(bt, label="BiLSTM Strategy"))
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
| `model.build_bilstm_model` | `lstm_units` | `64` | Units per BiLSTM direction |
| `model.build_bilstm_model` | `dropout_rate` | `0.2` | Dropout fraction |
| `model.build_bilstm_model` | `num_attention_heads` | `4` | Attention heads |
| `model.train_model` | `epochs` | `50` | Max training epochs |
| `model.train_model` | `patience` | `10` | Early-stopping patience |
| `features.add_technical_indicators` | `sma_windows` | `[20,50,200]` | SMA periods |
| `backtesting.run_backtest` | `threshold` | `0.5` | Trade trigger (%) |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Historical stock price data |
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | MinMaxScaler, TF-IDF, Random Forest |
| `tensorflow` | LSTM / BiLSTM model (Keras API) |
| `nltk` | VADER sentiment, tokenisation, stopwords |
| `transformers` + `torch` | FinBERT (optional, advanced sentiment) |
| `shap` | Feature importance / explainability (optional) |
| `streamlit` | Web application |
| `plotly` | Interactive charts |
| `matplotlib` | Static plots |

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.  
Predictions made by this model should **not** be used as financial advice.  
Always consult a qualified financial advisor before making investment decisions.

---

## 📄 License

[MIT](LICENSE)


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
