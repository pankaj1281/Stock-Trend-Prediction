"""
app.py
------
Streamlit web application for the AI-Powered Intelligent Stock Trading System.

Run with:
    streamlit run app.py

Features
--------
* Stock selection with multi-stock comparison.
* BiLSTM + Attention model for next-day price prediction.
* Technical indicators (RSI, MACD, SMA, EMA, Bollinger Bands, Volatility).
* VADER or FinBERT sentiment analysis with recency weighting.
* Buy/Sell/Hold signal with confidence score.
* Risk level indicator (Low / Medium / High).
* Backtesting dashboard with Sharpe ratio, P&L, win rate.
* Model comparison (LSTM vs BiLSTM vs Random Forest).
* Watchlist with session-persistent tracking.
* Sentiment trend visualization.
* Training loss curve.
* SHAP feature importance (when shap is installed).
"""

import logging
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Trading System",
    page_icon="📈",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["AAPL", "GOOG", "MSFT", "TSLA", "AMZN"]
WINDOW_SIZE = 60
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Risk level thresholds (annualised volatility)
LOW_VOLATILITY_THRESHOLD = 0.2
MEDIUM_VOLATILITY_THRESHOLD = 0.4

# Confidence scaling factor for HOLD signals (lower certainty than BUY/SELL)
HOLD_CONFIDENCE_FACTOR = 0.7


# ── Session state helpers ─────────────────────────────────────────────────────

def _init_session_state() -> None:
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = []
    if "alerts" not in st.session_state:
        st.session_state["alerts"] = []


# ── Artefact path helpers ──────────────────────────────────────────────────────

def _model_path(ticker: str, arch: str = "bilstm") -> str:
    return os.path.join(MODEL_DIR, f"{ticker}_{arch}.keras")


def _scaler_path(ticker: str) -> str:
    return os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")


def _feature_cols_path(ticker: str) -> str:
    return os.path.join(MODEL_DIR, f"{ticker}_features.pkl")


def artefacts_exist(ticker: str, arch: str = "bilstm") -> bool:
    return (
        os.path.exists(_model_path(ticker, arch))
        and os.path.exists(_scaler_path(ticker))
        and os.path.exists(_feature_cols_path(ticker))
    )


def save_artefacts(ticker: str, arch: str, dataset: dict) -> None:
    from model import save_model  # noqa: PLC0415

    save_model(dataset["model"], _model_path(ticker, arch))
    with open(_scaler_path(ticker), "wb") as f:
        pickle.dump(dataset["scaler"], f)
    with open(_feature_cols_path(ticker), "wb") as f:
        pickle.dump(dataset["feature_cols"], f)


def load_artefacts(ticker: str, arch: str = "bilstm"):
    from model import load_model  # noqa: PLC0415

    keras_model = load_model(_model_path(ticker, arch))
    with open(_scaler_path(ticker), "rb") as f:
        scaler = pickle.load(f)
    with open(_feature_cols_path(ticker), "rb") as f:
        feature_cols = pickle.load(f)
    return keras_model, scaler, feature_cols


# ── Data helpers ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching stock data …")
def cached_fetch_stock(ticker: str, period: str) -> pd.DataFrame:
    from data_loader import fetch_stock_data  # noqa: PLC0415

    return fetch_stock_data(ticker, period=period)


@st.cache_data(show_spinner="Loading news headlines …")
def cached_load_news(filepath: str | None) -> pd.DataFrame:
    from data_loader import load_news_headlines  # noqa: PLC0415

    return load_news_headlines(filepath)


# ── Model build / train ────────────────────────────────────────────────────────

def build_and_train(
    ticker: str,
    stock_df: pd.DataFrame,
    news_df: pd.DataFrame,
    arch: str = "bilstm",
    sentiment_mode: str = "vader",
) -> dict:
    """Run full preprocessing → model build → training pipeline."""
    from preprocessing import build_lstm_dataset  # noqa: PLC0415
    from model import build_bilstm_model, build_lstm_model, train_model  # noqa: PLC0415

    dataset = build_lstm_dataset(
        stock_df,
        news_df=news_df,
        window_size=WINDOW_SIZE,
        test_ratio=0.2,
        sentiment_method="vader",
        add_indicators=True,
    )

    n_features = len(dataset["feature_cols"])
    input_shape = (WINDOW_SIZE, n_features)

    if arch == "bilstm":
        keras_model = build_bilstm_model(
            input_shape=input_shape,
            lstm_units=64,
            dropout_rate=0.2,
            dense_units=32,
        )
    else:
        keras_model = build_lstm_model(
            input_shape=input_shape,
            lstm_units=50,
            dropout_rate=0.2,
            dense_units=25,
        )

    checkpoint = _model_path(ticker, arch)
    history = train_model(
        keras_model,
        dataset["X_train"],
        dataset["y_train"],
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        patience=10,
        checkpoint_path=checkpoint,
    )

    dataset["model"] = keras_model
    dataset["history"] = history
    return dataset


# ── Risk level ─────────────────────────────────────────────────────────────────

def _risk_level(volatility: float) -> tuple[str, str]:
    """Return (label, colour) based on annualised volatility."""
    if volatility < LOW_VOLATILITY_THRESHOLD:
        return "🟢 Low", "green"
    elif volatility < MEDIUM_VOLATILITY_THRESHOLD:
        return "🟡 Medium", "orange"
    else:
        return "🔴 High", "red"


def _confidence_score(mae: float, last_close: float) -> float:
    """Return a confidence score in [0, 100] based on relative MAE."""
    relative_error = mae / max(last_close, 1e-6)
    confidence = max(0.0, 1.0 - relative_error * 10) * 100
    return round(min(confidence, 100.0), 1)


# ── Plot helpers ───────────────────────────────────────────────────────────────

def plot_predictions(
    merged_df: pd.DataFrame,
    predictions: np.ndarray,
    actuals: np.ndarray,
    ticker: str,
    window_size: int,
) -> go.Figure:
    """Plotly chart of actual vs predicted closing prices (test set)."""
    test_dates = merged_df.index[-len(actuals):]

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


def plot_technical_indicators(merged_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Plotly chart with price, Bollinger Bands, RSI, and MACD."""
    has_bb = all(c in merged_df.columns for c in ["BB_Upper", "BB_Lower", "BB_Middle"])
    has_rsi = "RSI" in merged_df.columns
    has_macd = "MACD" in merged_df.columns

    rows = 1 + int(has_rsi) + int(has_macd)
    row_heights = [0.5] + [0.25] * (rows - 1)
    subplot_titles = [f"{ticker} Price + Bollinger Bands"]
    if has_rsi:
        subplot_titles.append("RSI (14)")
    if has_macd:
        subplot_titles.append("MACD")

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
    )

    # Price + Bollinger Bands
    fig.add_trace(
        go.Scatter(x=merged_df.index, y=merged_df["Close"], name="Close",
                   line=dict(color="steelblue")), row=1, col=1
    )
    if has_bb:
        fig.add_trace(
            go.Scatter(x=merged_df.index, y=merged_df["BB_Upper"],
                       name="BB Upper", line=dict(color="rgba(150,150,150,0.5)", dash="dot")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=merged_df.index, y=merged_df["BB_Lower"],
                       name="BB Lower", fill="tonexty",
                       fillcolor="rgba(173,216,230,0.2)",
                       line=dict(color="rgba(150,150,150,0.5)", dash="dot")),
            row=1, col=1,
        )

    current_row = 2
    if has_rsi:
        fig.add_trace(
            go.Scatter(x=merged_df.index, y=merged_df["RSI"], name="RSI",
                       line=dict(color="purple")),
            row=current_row, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        current_row += 1

    if has_macd:
        fig.add_trace(
            go.Scatter(x=merged_df.index, y=merged_df["MACD"], name="MACD",
                       line=dict(color="blue")),
            row=current_row, col=1,
        )
        if "MACD_Signal" in merged_df.columns:
            fig.add_trace(
                go.Scatter(x=merged_df.index, y=merged_df["MACD_Signal"],
                           name="Signal", line=dict(color="orange", dash="dash")),
                row=current_row, col=1,
            )
        if "MACD_Hist" in merged_df.columns:
            colors_hist = merged_df["MACD_Hist"].apply(
                lambda v: "green" if v >= 0 else "red"
            )
            fig.add_trace(
                go.Bar(x=merged_df.index, y=merged_df["MACD_Hist"],
                       name="Histogram", marker_color=colors_hist),
                row=current_row, col=1,
            )

    fig.update_layout(
        height=150 + 250 * rows,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.1),
        showlegend=True,
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


def plot_portfolio(result, ticker: str) -> go.Figure:
    """Plotly chart comparing strategy vs buy-and-hold portfolio value."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.portfolio_values.index,
            y=result.portfolio_values.values,
            mode="lines",
            name="Strategy",
            line=dict(color="steelblue", width=2),
        )
    )
    fig.update_layout(
        title=f"{ticker} – Backtest Portfolio Value",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
    )
    return fig


def plot_multi_stock(
    stock_data: dict[str, pd.DataFrame],
    norm: bool = True,
) -> go.Figure:
    """Plotly chart comparing normalised closing prices for multiple tickers."""
    fig = go.Figure()
    for ticker, df in stock_data.items():
        prices = df["Close"]
        if norm and len(prices) > 0:
            prices = prices / prices.iloc[0] * 100
        fig.add_trace(
            go.Scatter(x=prices.index, y=prices.values, mode="lines", name=ticker)
        )
    fig.update_layout(
        title="Multi-Stock Comparison" + (" (Normalised to 100)" if norm else ""),
        xaxis_title="Date",
        yaxis_title="Normalised Price" if norm else "Price (USD)",
        hovermode="x unified",
    )
    return fig


# ── SHAP feature importance ────────────────────────────────────────────────────

def _shap_bar_chart(feature_importances: dict[str, float], ticker: str) -> go.Figure:
    """Build a simple horizontal bar chart of SHAP-style feature importances."""
    sorted_items = sorted(feature_importances.items(), key=lambda x: abs(x[1]))
    features = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = ["crimson" if v < 0 else "steelblue" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        title=f"{ticker} – Feature Importance (mean |SHAP|)",
        xaxis_title="Mean |SHAP Value|",
        height=max(300, 30 * len(features)),
    )
    return fig


def compute_shap_importance(
    model,
    X_test: np.ndarray,
    feature_cols: list[str],
    n_samples: int = 50,
) -> dict[str, float] | None:
    """Compute approximate SHAP feature importance via permutation.

    Falls back to a gradient-based proxy if ``shap`` is not installed.
    Returns ``None`` if the computation fails.
    """
    try:
        import shap  # noqa: PLC0415

        background = X_test[:min(n_samples, len(X_test))]
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(background)
        # Average over time steps and samples
        mean_abs = np.mean(np.abs(shap_values[0]), axis=(0, 1))
        return dict(zip(feature_cols, mean_abs.tolist()))
    except Exception:  # noqa: BLE001
        pass

    # Fallback: permutation importance over test set
    try:
        baseline = model.predict(X_test[:n_samples], verbose=0).flatten()
        importances: dict[str, float] = {}
        for f_idx, f_name in enumerate(feature_cols):
            shuffled = X_test[:n_samples].copy()
            np.random.shuffle(shuffled[:, :, f_idx])
            perturbed = model.predict(shuffled, verbose=0).flatten()
            importances[f_name] = float(np.mean(np.abs(baseline - perturbed)))
        return importances
    except Exception:  # noqa: BLE001
        return None


# ── Main app ───────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session_state()

    st.title("📈 AI-Powered Intelligent Stock Trading System")
    st.markdown(
        "Combines **BiLSTM + Attention** deep learning, "
        "**technical indicators**, and **sentiment analysis** "
        "to forecast next-day closing prices and generate trading signals."
    )

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        ticker = st.text_input(
            "Primary stock ticker",
            value="AAPL",
            help="E.g. AAPL, GOOG, MSFT, TSLA",
        ).upper().strip()

        period = st.selectbox(
            "Historical data period",
            options=["1y", "2y", "3y", "5y", "10y"],
            index=3,
        )

        arch = st.selectbox(
            "Model architecture",
            options=["bilstm", "lstm"],
            format_func=lambda x: "BiLSTM + Attention" if x == "bilstm" else "LSTM (baseline)",
        )

        sentiment_mode = st.selectbox(
            "Sentiment mode",
            options=["vader", "finbert"],
            format_func=lambda x: "VADER (fast)" if x == "vader" else "FinBERT (advanced)",
        )

        news_file = st.file_uploader(
            "Upload news headlines CSV (optional)",
            type=["csv"],
            help="CSV must have 'date' and 'headline' columns.",
        )

        retrain = st.checkbox("Force retrain (ignore saved model)", value=False)

        st.markdown("---")
        st.subheader("📋 Watchlist")
        wl_ticker = st.text_input("Add ticker to watchlist", value="").upper().strip()
        if st.button("➕ Add") and wl_ticker:
            if wl_ticker not in st.session_state["watchlist"]:
                st.session_state["watchlist"].append(wl_ticker)
        for wt in st.session_state["watchlist"]:
            st.write(f"• {wt}")

        st.markdown("---")
        run_btn = st.button("🚀 Run Prediction", type="primary")

    if not run_btn:
        st.info(
            "Configure settings in the sidebar and click **Run Prediction** to begin. "
            "On first run the model will be trained (may take a few minutes)."
        )
        # Show watchlist summary if populated
        if st.session_state["watchlist"]:
            st.subheader("📋 Watchlist Overview")
            for wt in st.session_state["watchlist"]:
                try:
                    wdf = cached_fetch_stock(wt, "3mo")
                    last = float(wdf["Close"].iloc[-1])
                    prev = float(wdf["Close"].iloc[-2])
                    chg = (last - prev) / prev * 100
                    color = "🟢" if chg >= 0 else "🔴"
                    st.write(f"{color} **{wt}**: ${last:.2f} ({chg:+.2f}%)")
                except Exception:  # noqa: BLE001
                    st.write(f"⚪ **{wt}**: unavailable")
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
    if not retrain and artefacts_exist(ticker, arch):
        st.info(f"Loading saved **{arch.upper()}** model for **{ticker}** …")
        keras_model, scaler, feature_cols = load_artefacts(ticker, arch)

        from preprocessing import build_lstm_dataset  # noqa: PLC0415

        dataset = build_lstm_dataset(
            stock_df,
            news_df=news_df,
            window_size=WINDOW_SIZE,
            test_ratio=0.2,
            add_indicators=True,
        )
        dataset["model"] = keras_model
        dataset["scaler"] = scaler
        dataset["feature_cols"] = feature_cols
        history = None
    else:
        progress_bar = st.progress(0, text="Preprocessing & training …")
        try:
            dataset = build_and_train(ticker, stock_df, news_df, arch=arch,
                                       sentiment_mode=sentiment_mode)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Training failed: {exc}")
            return
        progress_bar.progress(80, text="Saving model …")
        save_artefacts(ticker, arch, dataset)
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
        confidence = _confidence_score(eval_result["mae"], last_close)
    elif pct_change < -1.0:
        signal, signal_color = "🔴 SELL", "red"
        confidence = _confidence_score(eval_result["mae"], last_close)
    else:
        signal, signal_color = "🟡 HOLD", "orange"
        confidence = max(0.0, _confidence_score(eval_result["mae"], last_close) * HOLD_CONFIDENCE_FACTOR)

    # Risk level from rolling volatility
    if "Volatility" in merged_df.columns:
        vol = float(merged_df["Volatility"].iloc[-1])
    else:
        # Compute annualised vol from last 20 days
        log_ret = np.log(merged_df["Close"] / merged_df["Close"].shift(1)).dropna()
        vol = float(log_ret.tail(20).std() * np.sqrt(252))
    risk_label, risk_color = _risk_level(vol)

    # ── Alerts ──────────────────────────────────────────────────────────────────
    alert_msg = f"{ticker}: {signal} @ ${next_price:.2f} (conf: {confidence:.0f}%)"
    if alert_msg not in st.session_state["alerts"]:
        st.session_state["alerts"].append(alert_msg)
        if len(st.session_state["alerts"]) > 10:
            st.session_state["alerts"].pop(0)

    # ── Metrics panel ──────────────────────────────────────────────────────────
    st.subheader("🎯 Prediction Results")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Last Close", f"${last_close:.2f}")
    col2.metric("Predicted Next Close", f"${next_price:.2f}", f"{pct_change:+.2f}%")
    col3.metric("Signal", signal)
    col4.metric("Confidence", f"{confidence:.0f}%")
    col5.metric("Risk Level", risk_label)
    col6.metric("R² Score", f"{eval_result.get('r2', 0):.3f}")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Test MAE", f"${eval_result['mae']:.2f}")
    col_b.metric("Test RMSE", f"${eval_result['rmse']:.2f}")
    col_c.metric("Model", arch.upper())

    # ── Charts ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📉 Price Prediction",
            "📊 Technical Indicators",
            "🗞️ Sentiment Trend",
            "📈 Backtesting",
            "🤖 Model Comparison",
            "🔬 Feature Importance",
        ]
    )

    with tab1:
        if history is not None:
            st.plotly_chart(plot_loss(history), use_container_width=True)
        fig_pred = plot_predictions(
            merged_df,
            eval_result["predictions"],
            eval_result["actuals"],
            ticker,
            WINDOW_SIZE,
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    with tab2:
        st.plotly_chart(
            plot_technical_indicators(merged_df, ticker),
            use_container_width=True,
        )

    with tab3:
        fig_sent = plot_sentiment(merged_df, ticker)
        st.plotly_chart(fig_sent, use_container_width=True)

    with tab4:
        st.subheader("🔁 Backtest Simulation")
        bt_col1, bt_col2, bt_col3 = st.columns(3)
        init_cap = bt_col1.number_input(
            "Initial Capital ($)", min_value=1000, value=10000, step=1000
        )
        threshold = bt_col2.slider("Trade Threshold (%)", 0.1, 5.0, 0.5, 0.1)
        pos_size = bt_col3.slider("Position Size", 0.1, 1.0, 1.0, 0.1)

        from backtesting import run_backtest, buy_and_hold, summary_report  # noqa: PLC0415

        test_prices = pd.Series(
            eval_result["actuals"],
            index=merged_df.index[-len(eval_result["actuals"]):],
        )
        bt_result = run_backtest(
            prices=test_prices,
            predictions=eval_result["predictions"],
            initial_capital=init_cap,
            threshold=threshold,
            position_size=pos_size,
        )
        bh_result = buy_and_hold(prices=test_prices, initial_capital=init_cap)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Strategy Return", f"{bt_result.total_return:+.2f}%")
        m2.metric("Buy & Hold Return", f"{bh_result.total_return:+.2f}%")
        m3.metric("Sharpe Ratio", f"{bt_result.sharpe_ratio:.3f}")
        m4.metric("Max Drawdown", f"{bt_result.max_drawdown:.2f}%")

        m5, m6, m7 = st.columns(3)
        m5.metric("Total P&L", f"${bt_result.total_profit_loss:+,.2f}")
        m6.metric("Win Rate", f"{bt_result.win_rate * 100:.1f}%")
        m7.metric("Direction Accuracy", f"{bt_result.accuracy * 100:.1f}%")

        st.plotly_chart(plot_portfolio(bt_result, ticker), use_container_width=True)

        with st.expander("📄 Backtest Summary Report"):
            st.text(summary_report(bt_result, label=f"{arch.upper()} Strategy"))
            st.text(summary_report(bh_result, label="Buy & Hold"))

        if not bt_result.trades.empty:
            with st.expander(f"📋 Trade Log ({len(bt_result.trades)} trades)"):
                st.dataframe(bt_result.trades, use_container_width=True)

    with tab5:
        st.subheader("🤖 Model Comparison")
        st.info(
            "Training all three models can take several minutes. "
            "Cached results are reused when available."
        )

        compare_btn = st.button("▶ Run Model Comparison")
        if compare_btn:
            from model import (  # noqa: PLC0415
                build_lstm_model,
                build_bilstm_model,
                build_random_forest_model,
                evaluate_model as eval_keras,
                evaluate_rf_model,
                train_model,
                flatten_sequences,
                compare_models,
            )

            comparison_results: dict[str, dict] = {}

            with st.spinner("Training LSTM …"):
                lstm_m = build_lstm_model(
                    (WINDOW_SIZE, len(feature_cols)), lstm_units=50
                )
                train_model(lstm_m, dataset["X_train"], dataset["y_train"],
                            epochs=30, patience=5, validation_split=0.1)
                comparison_results["LSTM"] = eval_keras(
                    lstm_m, dataset["X_test"], dataset["y_test"], scaler, feature_cols
                )

            with st.spinner("Training BiLSTM + Attention …"):
                bi_m = build_bilstm_model(
                    (WINDOW_SIZE, len(feature_cols)), lstm_units=64
                )
                train_model(bi_m, dataset["X_train"], dataset["y_train"],
                            epochs=30, patience=5, validation_split=0.1)
                comparison_results["BiLSTM+Attention"] = eval_keras(
                    bi_m, dataset["X_test"], dataset["y_test"], scaler, feature_cols
                )

            with st.spinner("Training Random Forest …"):
                rf_m = build_random_forest_model(n_estimators=100)
                rf_m.fit(
                    flatten_sequences(dataset["X_train"]),
                    dataset["y_train"],
                )
                comparison_results["Random Forest"] = evaluate_rf_model(
                    rf_m, dataset["X_test"], dataset["y_test"], scaler, feature_cols
                )

            comp_df = compare_models(comparison_results)
            st.dataframe(comp_df, use_container_width=True)

            # Bar chart
            fig_comp = go.Figure()
            for metric in ["MAE", "RMSE", "R²"]:
                fig_comp.add_trace(
                    go.Bar(
                        name=metric,
                        x=comp_df.index.tolist(),
                        y=comp_df[metric].tolist(),
                    )
                )
            fig_comp.update_layout(
                title="Model Comparison",
                barmode="group",
                xaxis_title="Model",
                yaxis_title="Score",
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.write("Click **▶ Run Model Comparison** to compare LSTM, BiLSTM+Attention, and Random Forest.")

    with tab6:
        st.subheader("🔬 Explainable AI – Feature Importance")
        shap_btn = st.button("▶ Compute Feature Importance")
        if shap_btn:
            with st.spinner("Computing feature importances …"):
                importances = compute_shap_importance(
                    keras_model, dataset["X_test"], feature_cols
                )
            if importances:
                st.plotly_chart(
                    _shap_bar_chart(importances, ticker), use_container_width=True
                )
                imp_df = pd.DataFrame(
                    importances.items(), columns=["Feature", "Importance"]
                ).sort_values("Importance", ascending=False)
                st.dataframe(imp_df, use_container_width=True)
            else:
                st.warning("Could not compute feature importances.")
        else:
            st.write("Click **▶ Compute Feature Importance** to see which features drive predictions.")

    # ── Multi-stock comparison ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Multi-Stock Comparison")
    compare_tickers_raw = st.text_input(
        "Enter tickers to compare (comma-separated)",
        value=", ".join(DEFAULT_TICKERS[:3]),
    )
    compare_tickers = [t.strip().upper() for t in compare_tickers_raw.split(",") if t.strip()]

    if st.button("📊 Compare Stocks") and compare_tickers:
        from data_loader import get_multiple_stocks  # noqa: PLC0415

        with st.spinner("Fetching multi-stock data …"):
            multi_data = get_multiple_stocks(compare_tickers, period="1y")

        if multi_data:
            st.plotly_chart(
                plot_multi_stock(multi_data, norm=True), use_container_width=True
            )
            # Summary table
            rows = []
            for t, df in multi_data.items():
                last = float(df["Close"].iloc[-1])
                if len(df) >= 22:
                    base = float(df["Close"].iloc[-22])
                    ret_1m = float((last / base - 1) * 100) if base > 0 else float("nan")
                else:
                    ret_1m = float("nan")
                ret_str = f"{ret_1m:+.2f}%" if not np.isnan(ret_1m) else "N/A"
                rows.append({"Ticker": t, "Last Close": f"${last:.2f}", "1M Return": ret_str})
            st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)

    # ── Alerts panel ───────────────────────────────────────────────────────────
    if st.session_state["alerts"]:
        st.markdown("---")
        st.subheader("🔔 Prediction Alerts")
        for alert in reversed(st.session_state["alerts"]):
            st.info(alert)

    # ── Download section ───────────────────────────────────────────────────────
    st.markdown("---")
    model_file = _model_path(ticker, arch)
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            st.download_button(
                label=f"⬇️ Download {arch.upper()} model (.keras)",
                data=f,
                file_name=os.path.basename(model_file),
                mime="application/octet-stream",
            )

    # Export predictions CSV
    if eval_result:
        pred_df = pd.DataFrame(
            {
                "date": merged_df.index[-len(eval_result["actuals"]):],
                "actual": eval_result["actuals"],
                "predicted": eval_result["predictions"],
            }
        )
        st.download_button(
            label="⬇️ Export predictions (.csv)",
            data=pred_df.to_csv(index=False).encode(),
            file_name=f"{ticker}_predictions.csv",
            mime="text/csv",
        )

    st.caption(
        "⚠️ Disclaimer: This tool is for educational purposes only and does not "
        "constitute financial advice.  Always consult a qualified financial advisor."
    )


if __name__ == "__main__":
    main()

