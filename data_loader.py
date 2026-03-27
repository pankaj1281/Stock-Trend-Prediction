"""
data_loader.py
--------------
Handles fetching historical stock price data via yfinance and loading
financial news headlines from a CSV file or a stub dataset.
"""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_stock_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    period: str = "5y",
) -> pd.DataFrame:
    """Fetch historical OHLCV data for *ticker* using yfinance.

    Parameters
    ----------
    ticker:
        Stock ticker symbol, e.g. ``"AAPL"``.
    start:
        Start date as ``"YYYY-MM-DD"`` string.  If *None*, the *period*
        argument is used instead.
    end:
        End date as ``"YYYY-MM-DD"`` string.  Defaults to today.
    period:
        yfinance period shorthand (``"1y"``, ``"5y"``, …).  Only used when
        *start* is *None*.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Open``, ``High``, ``Low``, ``Close``,
        ``Volume`` indexed by date.

    Raises
    ------
    ValueError
        If no data is returned for the requested ticker / date range.
    """
    logger.info("Fetching stock data for %s …", ticker)

    if start is not None:
        end = end or datetime.today().strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    else:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            "Check the symbol and date range."
        )

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # Keep only the canonical OHLCV columns that are present
    ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[ohlcv_cols].copy()

    # Drop rows where Close is NaN
    df.dropna(subset=["Close"], inplace=True)

    logger.info("Fetched %d rows for %s.", len(df), ticker)
    return df


def load_news_headlines(filepath: str | None = None) -> pd.DataFrame:
    """Load financial news headlines from a CSV file.

    The CSV is expected to have at least two columns:
    ``date`` (parseable date string) and ``headline`` (free text).

    If *filepath* is ``None`` or the file does not exist, a small built-in
    stub dataset is returned so that the rest of the pipeline can run
    without an external file.

    Parameters
    ----------
    filepath:
        Path to the CSV file.  May be ``None``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``date`` (datetime) and ``headline`` (str).
    """
    if filepath and os.path.isfile(filepath):
        logger.info("Loading news headlines from %s …", filepath)
        df = pd.read_csv(filepath, parse_dates=["date"])
        if "date" not in df.columns or "headline" not in df.columns:
            raise ValueError(
                "News CSV must contain 'date' and 'headline' columns."
            )
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "headline"]].dropna()
        logger.info("Loaded %d news headlines.", len(df))
        return df

    # ── Stub / demo headlines ──────────────────────────────────────────────
    logger.warning(
        "No news headlines file provided or found.  "
        "Using built-in stub dataset for demonstration."
    )
    today = datetime.today()
    stub_data = [
        (today - timedelta(days=i), headline)
        for i, headline in enumerate(
            [
                "Markets rally as inflation data comes in lower than expected",
                "Tech stocks surge amid strong earnings season",
                "Fed signals potential rate cuts boosting investor sentiment",
                "Oil prices drop on demand concerns weighing on energy stocks",
                "Strong jobs report sends markets higher on Friday",
                "Trade tensions escalate causing market volatility",
                "Consumer spending beats estimates lifting retail stocks",
                "Recession fears grow as yield curve inverts sharply",
                "Chip makers soar after major AI contract announcement",
                "Profit-taking pulls major indices lower after record highs",
            ]
        )
    ]
    df = pd.DataFrame(stub_data, columns=["date", "headline"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def get_multiple_stocks(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    period: str = "5y",
) -> dict[str, pd.DataFrame]:
    """Fetch historical data for multiple tickers at once.

    Parameters
    ----------
    tickers:
        List of ticker symbols.
    start, end, period:
        Forwarded to :func:`fetch_stock_data`.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker → DataFrame.  Tickers that fail are skipped with
        a warning logged.
    """
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            result[ticker] = fetch_stock_data(ticker, start=start, end=end, period=period)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", ticker, exc)
    return result
