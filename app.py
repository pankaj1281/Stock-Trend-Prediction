"""
app.py
------
Streamlit web application for AI-Driven Stock Market Prediction.

Run with:
    streamlit run app.py

Features
--------
* Input stock ticker symbol (+ optional date range).
* Fetch live OHLCV data, compute VADER sentiment from built-in stub headlines.
* Train (or load) an LSTM model in-app.
* Display predicted next-day closing price and a Buy / Hold / Sell signal.
* Interactive plotly charts:
    - Historical vs. predicted closing prices.
    - Sentiment trend over time.
    - Training loss curve.
"""

import logging
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Trend Predictor",
    page_icon="📈",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["AAPL", "GOOG", "MSFT", "TSLA", "AMZN"]
WINDOW_SIZE = 60
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _model_path(ticker: str) -> str:
    return os.path.join(MODEL_DIR, f"{ticker}_lstm.keras")


def _scaler_path(ticker: str) -> str:
    return os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")


def _feature_cols_path(ticker: str) -> str:
    return os.path.join(MODEL_DIR, f"{ticker}_features.pkl")


@st.cache_data(show_spinner="Fetching stock data …")
def cached_fetch_stock(ticker: str, period: str) -> pd.DataFrame:
    from data_loader import fetch_stock_data  # noqa: PLC0415

    return fetch_stock_data(ticker, period=period)


@st.cache_data(show_spinner="Loading news headlines …")
def cached_load_news(filepath: str | None) -> pd.DataFrame:
    from data_loader import load_news_headlines  # noqa: PLC0415

    return load_news_headlines(filepath)


def build_and_train(ticker: str, stock_df: pd.DataFrame, news_df: pd.DataFrame) -> dict:
    """Run full preprocessing → model build → training pipeline."""
    from preprocessing import build_lstm_dataset  # noqa: PLC0415
    from model import build_lstm_model, train_model  # noqa: PLC0415

    dataset = build_lstm_dataset(
        stock_df,
        news_df=news_df,
        window_size=WINDOW_SIZE,
        test_ratio=0.2,
        sentiment_method="vader",
    )

    keras_model = build_lstm_model(
        input_shape=(WINDOW_SIZE, len(dataset["feature_cols"])),
        lstm_units=50,
        dropout_rate=0.2,
        dense_units=25,
    )

    history = train_model(
        keras_model,
        dataset["X_train"],
        dataset["y_train"],
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        patience=10,
    )

    dataset["model"] = keras_model
    dataset["history"] = history
    return dataset


def save_artefacts(ticker: str, dataset: dict) -> None:
    from model import save_model  # noqa: PLC0415

    save_model(dataset["model"], _model_path(ticker))
    with open(_scaler_path(ticker), "wb") as f:
        pickle.dump(dataset["scaler"], f)
    with open(_feature_cols_path(ticker), "wb") as f:
        pickle.dump(dataset["feature_cols"], f)


def load_artefacts(ticker: str):
    from model import load_model  # noqa: PLC0415

    keras_model = load_model(_model_path(ticker))
    with open(_scaler_path(ticker), "rb") as f:
        scaler = pickle.load(f)
    with open(_feature_cols_path(ticker), "rb") as f:
        feature_cols = pickle.load(f)
    return keras_model, scaler, feature_cols


def artefacts_exist(ticker: str) -> bool:
    return (
        os.path.exists(_model_path(ticker))
        and os.path.exists(_scaler_path(ticker))
        and os.path.exists(_feature_cols_path(ticker))
    )


# ── Plot helpers ───────────────────────────────────────────────────────────────

def plot_predictions(
    merged_df: pd.DataFrame,
    predictions: np.ndarray,
    actuals: np.ndarray,
    ticker: str,
    window_size: int,
) -> go.Figure:
    """Plotly chart of actual vs predicted closing prices (test set)."""
    test_dates = merged_df.index[-(len(actuals)):]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged_df.index,
            y=merged_df["Close"],
            mode="lines",
            name="Historical Close",
            line=dict(color="steelblue", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=actuals,
            mode="lines",
            name="Actual (test)",
            line=dict(color="green", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=predictions,
            mode="lines",
            name="Predicted (test)",
            line=dict(color="crimson", width=1.5, dash="dash"),
        )
    )
    fig.update_layout(
        title=f"{ticker} – Actual vs Predicted Closing Price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def plot_sentiment(merged_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Plotly chart of daily compound sentiment score over time."""
    if "compound" not in merged_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No sentiment data available", showarrow=False)
        return fig

    colors = merged_df["compound"].apply(
        lambda v: "green" if v > 0 else ("red" if v < 0 else "gray")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=merged_df.index,
            y=merged_df["compound"],
            marker_color=colors,
            name="Compound Sentiment",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=merged_df.index,
            y=merged_df["compound"].rolling(window=10).mean(),
            mode="lines",
            name="10-day MA",
            line=dict(color="navy", width=2),
        )
    )
    fig.update_layout(
        title=f"{ticker} – Daily Sentiment Trend (VADER Compound)",
        xaxis_title="Date",
        yaxis_title="Compound Score",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def plot_loss(history) -> go.Figure:
    """Plotly training vs validation loss curve."""
    hist = history.history
    epochs = list(range(1, len(hist["loss"]) + 1))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=epochs, y=hist["loss"], mode="lines", name="Train Loss")
    )
    if "val_loss" in hist:
        fig.add_trace(
            go.Scatter(x=epochs, y=hist["val_loss"], mode="lines", name="Val Loss")
        )
    fig.update_layout(
        title="Training Loss Curve",
        xaxis_title="Epoch",
        yaxis_title="MSE Loss",
    )
    return fig


# ── Main app ───────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("📈 AI-Driven Stock Market Prediction")
    st.markdown(
        "Combines **LSTM deep learning** with **VADER sentiment analysis** "
        "to forecast next-day closing prices and generate trading signals."
    )

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        ticker = st.text_input(
            "Stock ticker",
            value="AAPL",
            help="E.g. AAPL, GOOG, MSFT, TSLA",
        ).upper().strip()

        period = st.selectbox(
            "Historical data period",
            options=["1y", "2y", "3y", "5y", "10y"],
            index=3,
        )

        news_file = st.file_uploader(
            "Upload news headlines CSV (optional)",
            type=["csv"],
            help="CSV must have 'date' and 'headline' columns.",
        )

        retrain = st.checkbox(
            "Force retrain (ignore saved model)",
            value=False,
        )

        run_btn = st.button("🚀 Run Prediction", type="primary")

    if not run_btn:
        st.info("Configure settings in the sidebar and click **Run Prediction**.")
        return

    # ── Data loading ───────────────────────────────────────────────────────────
    with st.spinner(f"Fetching {ticker} data …"):
        try:
            stock_df = cached_fetch_stock(ticker, period)
        except ValueError as exc:
            st.error(str(exc))
            return

    news_path: str | None = None
    if news_file is not None:
        suffix = os.path.splitext(news_file.name)[1] or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(news_file.read())
            news_path = tmp.name

    news_df = cached_load_news(news_path)

    st.subheader(f"📊 {ticker} – Raw Data Preview")
    st.dataframe(stock_df.tail(10), use_container_width=True)

    # ── Model (train or load) ──────────────────────────────────────────────────
    if not retrain and artefacts_exist(ticker):
        st.info(f"Loading saved model for **{ticker}** …")
        keras_model, scaler, feature_cols = load_artefacts(ticker)

        # We still need merged_df for plots, so run preprocessing only
        from preprocessing import build_lstm_dataset  # noqa: PLC0415

        dataset = build_lstm_dataset(
            stock_df,
            news_df=news_df,
            window_size=WINDOW_SIZE,
            test_ratio=0.2,
        )
        dataset["model"] = keras_model
        dataset["scaler"] = scaler
        dataset["feature_cols"] = feature_cols
        history = None
    else:
        progress_bar = st.progress(0, text="Preprocessing …")
        dataset = build_and_train(ticker, stock_df, news_df)
        progress_bar.progress(80, text="Saving model …")
        save_artefacts(ticker, dataset)
        history = dataset.get("history")
        progress_bar.progress(100, text="Done!")

    keras_model = dataset["model"]
    scaler = dataset["scaler"]
    feature_cols = dataset["feature_cols"]

    # ── Evaluation ─────────────────────────────────────────────────────────────
    from model import evaluate_model, predict_next_close  # noqa: PLC0415
    from preprocessing import normalize_stock_data  # noqa: PLC0415

    eval_result = evaluate_model(
        keras_model,
        dataset["X_test"],
        dataset["y_test"],
        scaler,
        feature_cols,
    )

    # ── Next-day prediction ────────────────────────────────────────────────────
    merged_df = dataset["merged_df"]
    scaled_df, _ = normalize_stock_data(merged_df, feature_cols=feature_cols)
    last_window = scaled_df.values[-WINDOW_SIZE:]
    next_price = predict_next_close(keras_model, last_window, scaler, feature_cols)
    last_close = float(merged_df["Close"].iloc[-1])
    pct_change = (next_price - last_close) / last_close * 100

    if pct_change > 1.0:
        signal, signal_color = "🟢 BUY", "green"
    elif pct_change < -1.0:
        signal, signal_color = "🔴 SELL", "red"
    else:
        signal, signal_color = "🟡 HOLD", "orange"

    # ── Metrics panel ──────────────────────────────────────────────────────────
    st.subheader("🎯 Prediction Results")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Last Close", f"${last_close:.2f}")
    col2.metric("Predicted Next Close", f"${next_price:.2f}", f"{pct_change:+.2f}%")
    col3.metric("Signal", signal)
    col4.metric("Test MAE", f"${eval_result['mae']:.2f}")
    col5.metric("Test RMSE", f"${eval_result['rmse']:.2f}")

    # ── Charts ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        ["📉 Price Prediction", "🗞️ Sentiment Trend", "📊 Training Loss"]
    )

    with tab1:
        fig_pred = plot_predictions(
            merged_df,
            eval_result["predictions"],
            eval_result["actuals"],
            ticker,
            WINDOW_SIZE,
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    with tab2:
        fig_sent = plot_sentiment(merged_df, ticker)
        st.plotly_chart(fig_sent, use_container_width=True)

    with tab3:
        if history is not None:
            fig_loss = plot_loss(history)
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("Training loss is only available after retraining.")

    # ── Download model ──────────────────────────────────────────────────────────
    model_file = _model_path(ticker)
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            st.download_button(
                label="⬇️ Download trained model (.keras)",
                data=f,
                file_name=os.path.basename(model_file),
                mime="application/octet-stream",
            )

    st.caption(
        "Disclaimer: This tool is for educational purposes only and does not "
        "constitute financial advice."
    )


if __name__ == "__main__":
    main()
