"""
features.py
-----------
Technical indicator feature engineering for the Stock Trend Prediction pipeline.

Computes and appends the following indicators to an OHLCV DataFrame:
  * RSI  – Relative Strength Index
  * MACD – Moving Average Convergence Divergence (line, signal, histogram)
  * SMA  – Simple Moving Averages (20, 50, 200 days)
  * EMA  – Exponential Moving Averages (12, 26 days)
  * Bollinger Bands (upper, middle, lower)
  * Volatility  – rolling 20-day standard deviation of daily returns
  * ATR  – Average True Range (14-day)
  * OBV  – On-Balance Volume

All indicators are appended as new columns; the original OHLCV columns are
preserved unchanged.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Low-level indicator helpers ───────────────────────────────────────────────


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with ``min_periods=1``."""
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=window, min_periods=1).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index.

    Parameters
    ----------
    close:
        Closing price series.
    period:
        Look-back window (default 14).

    Returns
    -------
    pd.Series
        RSI values in the range [0, 100], named ``"RSI"``.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral at warm-up period
    return rsi.rename("RSI")


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Compute MACD, signal line, and histogram.

    Returns
    -------
    pd.DataFrame
        Columns: ``MACD``, ``MACD_Signal``, ``MACD_Hist``.
    """
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"MACD": macd_line, "MACD_Signal": signal_line, "MACD_Hist": histogram},
        index=close.index,
    )


def compute_bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Compute Bollinger Bands.

    Returns
    -------
    pd.DataFrame
        Columns: ``BB_Upper``, ``BB_Middle``, ``BB_Lower``.
    """
    middle = _sma(close, window)
    std = close.rolling(window=window, min_periods=1).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame(
        {"BB_Upper": upper, "BB_Middle": middle, "BB_Lower": lower},
        index=close.index,
    )


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Average True Range.

    Parameters
    ----------
    high, low, close:
        OHLCV columns.
    period:
        Smoothing window.

    Returns
    -------
    pd.Series
        ATR values, named ``"ATR"``.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    return atr.rename("ATR")


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume.

    Returns
    -------
    pd.Series
        Cumulative OBV, named ``"OBV"``.
    """
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv.rename("OBV")


def compute_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling annualised volatility (std of daily log returns × √252).

    Returns
    -------
    pd.Series
        Volatility values, named ``"Volatility"``.
    """
    log_returns = np.log(close / close.shift(1))
    vol = log_returns.rolling(window=window, min_periods=2).std() * np.sqrt(252)
    return vol.rename("Volatility")


# ── Composite helper ──────────────────────────────────────────────────────────


def add_technical_indicators(
    df: pd.DataFrame,
    sma_windows: list[int] | None = None,
    ema_spans: list[int] | None = None,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20,
    atr_period: int = 14,
    vol_window: int = 20,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Compute all technical indicators and append them to *df*.

    Parameters
    ----------
    df:
        OHLCV DataFrame with at minimum a ``Close`` column.  ``High``,
        ``Low``, and ``Volume`` are used if present.
    sma_windows:
        Windows for Simple Moving Averages.  Defaults to ``[20, 50, 200]``.
    ema_spans:
        Spans for Exponential Moving Averages.  Defaults to ``[12, 26]``.
    rsi_period, macd_fast, macd_slow, macd_signal, bb_window, atr_period,
    vol_window:
        Indicator parameters (see individual functions for details).
    drop_na:
        Whether to drop rows that contain NaN values after indicator
        computation (caused by warm-up periods).  Default ``True``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with all indicator columns appended.
    """
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    if sma_windows is None:
        sma_windows = [20, 50, 200]
    if ema_spans is None:
        ema_spans = [12, 26]

    df = df.copy()
    close = df["Close"]

    # ── SMAs ──────────────────────────────────────────────────────────────────
    for w in sma_windows:
        df[f"SMA_{w}"] = _sma(close, w)
        logger.debug("Added SMA_%d.", w)

    # ── EMAs ──────────────────────────────────────────────────────────────────
    for s in ema_spans:
        df[f"EMA_{s}"] = _ema(close, s)
        logger.debug("Added EMA_%d.", s)

    # ── RSI ───────────────────────────────────────────────────────────────────
    df["RSI"] = compute_rsi(close, period=rsi_period)

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_df = compute_macd(close, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    df = pd.concat([df, macd_df], axis=1)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_df = compute_bollinger_bands(close, window=bb_window)
    df = pd.concat([df, bb_df], axis=1)

    # ── Volatility ────────────────────────────────────────────────────────────
    df["Volatility"] = compute_volatility(close, window=vol_window)

    # ── ATR (requires High/Low) ───────────────────────────────────────────────
    if "High" in df.columns and "Low" in df.columns:
        df["ATR"] = compute_atr(df["High"], df["Low"], close, period=atr_period)
    else:
        logger.warning("'High'/'Low' columns not found – skipping ATR.")

    # ── OBV (requires Volume) ─────────────────────────────────────────────────
    if "Volume" in df.columns:
        df["OBV"] = compute_obv(close, df["Volume"])
    else:
        logger.warning("'Volume' column not found – skipping OBV.")

    if drop_na:
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped:
            logger.info("Dropped %d rows during indicator warm-up.", dropped)

    logger.info(
        "Added technical indicators. DataFrame shape: %s, columns: %s",
        df.shape,
        list(df.columns),
    )
    return df


def get_indicator_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of technical-indicator column names present in *df*.

    Useful for constructing the ``feature_cols`` list fed to the model.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`add_technical_indicators`.

    Returns
    -------
    list[str]
        Ordered list of indicator column names (excluding OHLCV columns).
    """
    ohlcv = {"Open", "High", "Low", "Close", "Volume"}
    return [c for c in df.columns if c not in ohlcv]
