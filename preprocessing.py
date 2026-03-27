"""
preprocessing.py
----------------
Data preprocessing for the Stock Trend Prediction pipeline:

* Normalise stock price data with MinMaxScaler.
* Build LSTM-compatible (X, y) sequences with a configurable window size.
* Clean raw news headlines (lowercase, remove punctuation/stopwords, tokenise).
* Run VADER sentiment analysis on cleaned headlines.
* Optionally run TF-IDF + Logistic Regression sentiment classifier.
* Merge daily sentiment scores into the stock price DataFrame.
"""

import re
import logging
import string
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── NLTK lazy imports (downloaded on first use) ───────────────────────────────
import nltk

logger = logging.getLogger(__name__)

# NLTK resources required by this module
_NLTK_RESOURCES = [
    ("corpora/stopwords", "stopwords"),
    ("sentiment/vader_lexicon.zip", "vader_lexicon"),
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
]


def _ensure_nltk_resources() -> None:
    """Download any missing NLTK resources silently."""
    for resource_path, resource_id in _NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info("Downloading NLTK resource '%s' …", resource_id)
            nltk.download(resource_id, quiet=True)


# ── Stock price preprocessing ─────────────────────────────────────────────────


def normalize_stock_data(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Normalize selected stock price columns to [0, 1].

    Parameters
    ----------
    df:
        Raw OHLCV DataFrame indexed by date.
    feature_cols:
        Columns to normalise.  Defaults to ``["Close"]``.

    Returns
    -------
    tuple[pd.DataFrame, MinMaxScaler]
        * Normalised DataFrame (same index, only *feature_cols*).
        * Fitted scaler (needed to inverse-transform predictions later).
    """
    if feature_cols is None:
        feature_cols = ["Close"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=feature_cols)
    return scaled_df, scaler


def create_sequences(
    data: np.ndarray,
    window_size: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over *data* to create (X, y) pairs for LSTM.

    Parameters
    ----------
    data:
        2-D array of shape ``(n_timesteps, n_features)``.
    window_size:
        Number of past timesteps used as input.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        * ``X`` – shape ``(n_samples, window_size, n_features)``
        * ``y`` – shape ``(n_samples,)`` – next-step Close price (column 0)
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i])
        y.append(data[i, 0])  # target = Close (first feature column)
    return np.array(X), np.array(y)


def train_test_split_sequences(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split (X, y) into train / test sets preserving temporal order.

    Parameters
    ----------
    X, y:
        Sequence arrays from :func:`create_sequences`.
    test_ratio:
        Fraction of samples reserved for testing.

    Returns
    -------
    tuple
        ``X_train, X_test, y_train, y_test``
    """
    split = int(len(X) * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]


# ── Text / headline preprocessing ─────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/digits, strip stopwords, tokenise.

    Parameters
    ----------
    text:
        Raw news headline string.

    Returns
    -------
    str
        Space-joined cleaned tokens.
    """
    _ensure_nltk_resources()
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # punctuation
    text = re.sub(r"\d+", "", text)                   # digits
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


def clean_headlines(df: pd.DataFrame, headline_col: str = "headline") -> pd.DataFrame:
    """Apply :func:`clean_text` to every headline in *df*.

    Returns a copy of *df* with an additional ``cleaned_headline`` column.
    """
    df = df.copy()
    df["cleaned_headline"] = df[headline_col].astype(str).apply(clean_text)
    return df


# ── Sentiment analysis ────────────────────────────────────────────────────────


def compute_vader_sentiment(df: pd.DataFrame, text_col: str = "headline") -> pd.DataFrame:
    """Add VADER sentiment scores to *df*.

    New columns: ``sentiment_neg``, ``sentiment_neu``, ``sentiment_pos``,
    ``compound``.

    Parameters
    ----------
    df:
        DataFrame with a column of raw (uncleaned) headlines.
    text_col:
        Name of the column containing headline text.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with four extra sentiment columns.
    """
    _ensure_nltk_resources()
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sid = SentimentIntensityAnalyzer()

    scores = df[text_col].astype(str).apply(sid.polarity_scores)
    sentiment_df = pd.DataFrame(scores.tolist(), index=df.index)
    sentiment_df.rename(
        columns={
            "neg": "sentiment_neg",
            "neu": "sentiment_neu",
            "pos": "sentiment_pos",
            # "compound" stays as-is
        },
        inplace=True,
    )

    return pd.concat([df, sentiment_df], axis=1)


def compute_tfidf_sentiment(
    df: pd.DataFrame,
    text_col: str = "cleaned_headline",
    labels: pd.Series | None = None,
) -> pd.DataFrame:
    """TF-IDF + Logistic Regression sentiment classifier (optional).

    When *labels* are **not** provided a simple rule-based proxy is used to
    generate training labels (compound VADER score ≥ 0 → positive).  This
    keeps the function self-contained.

    Parameters
    ----------
    df:
        DataFrame with cleaned headline text.
    text_col:
        Column to vectorise.
    labels:
        Binary (0/1) sentiment labels.  If *None*, auto-generated via VADER.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an extra ``tfidf_sentiment`` column (0 = negative,
        1 = positive).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    df = df.copy()

    if labels is None:
        # Auto-label via VADER compound score
        _ensure_nltk_resources()
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        sid = SentimentIntensityAnalyzer()
        compound = df[text_col].astype(str).apply(
            lambda t: sid.polarity_scores(t)["compound"]
        )
        labels = (compound >= 0).astype(int)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(df[text_col].astype(str), labels)
    df["tfidf_sentiment"] = pipeline.predict(df[text_col].astype(str))
    return df


# ── Merge sentiment into stock data ──────────────────────────────────────────


def aggregate_daily_sentiment(news_df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Aggregate per-headline VADER scores to one row per calendar day.

    Parameters
    ----------
    news_df:
        DataFrame from :func:`compute_vader_sentiment` (includes ``date``,
        ``compound``, ``sentiment_neg``, ``sentiment_neu``, ``sentiment_pos``).
    date_col:
        Column containing dates.

    Returns
    -------
    pd.DataFrame
        Daily aggregated sentiment (mean of each score column), indexed by
        date.
    """
    score_cols = [c for c in ["sentiment_neg", "sentiment_neu", "sentiment_pos", "compound"]
                  if c in news_df.columns]
    daily = (
        news_df
        .copy()
        .assign(**{date_col: pd.to_datetime(news_df[date_col]).dt.normalize()})
        .groupby(date_col)[score_cols]
        .mean()
    )
    daily.index.name = "Date"
    return daily


def merge_sentiment_with_stock(
    stock_df: pd.DataFrame,
    news_df: pd.DataFrame,
    date_col: str = "date",
    method: Literal["vader", "tfidf"] = "vader",
) -> pd.DataFrame:
    """Merge daily sentiment scores into the stock price DataFrame.

    Missing trading days (no news coverage) are forward-filled then
    backward-filled so that no NaN rows are introduced.

    Parameters
    ----------
    stock_df:
        OHLCV DataFrame indexed by date (from :func:`data_loader.fetch_stock_data`).
    news_df:
        Headlines DataFrame with at least ``date`` and ``headline`` columns.
    date_col:
        Date column name in *news_df*.
    method:
        ``"vader"`` (default) or ``"tfidf"`` – which sentiment approach to use
        when computing sentiment scores.

    Returns
    -------
    pd.DataFrame
        stock_df extended with sentiment columns.
    """
    # Step 1: clean headlines
    news_df = clean_headlines(news_df, headline_col="headline")

    # Step 2: compute sentiment
    if method == "vader":
        news_df = compute_vader_sentiment(news_df, text_col="headline")
    else:
        # Compute VADER once (used both as labels for LR and as score columns)
        news_df = compute_vader_sentiment(news_df, text_col="headline")
        news_df = compute_tfidf_sentiment(news_df, text_col="cleaned_headline")

    # Step 3: aggregate to daily
    daily_sentiment = aggregate_daily_sentiment(news_df, date_col=date_col)

    # Step 4: left-join onto stock data
    merged = stock_df.join(daily_sentiment, how="left")
    sentiment_cols = daily_sentiment.columns.tolist()
    merged[sentiment_cols] = merged[sentiment_cols].ffill().bfill()

    # Fill any remaining NaNs with neutral values
    for col in ["sentiment_neg", "sentiment_pos", "sentiment_neu"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
    if "compound" in merged.columns:
        merged["compound"] = merged["compound"].fillna(0.0)

    return merged


# ── Full pipeline helper ──────────────────────────────────────────────────────


def build_lstm_dataset(
    stock_df: pd.DataFrame,
    news_df: pd.DataFrame | None = None,
    window_size: int = 60,
    test_ratio: float = 0.2,
    sentiment_method: Literal["vader", "tfidf"] = "vader",
    feature_cols: list[str] | None = None,
    add_indicators: bool = True,
) -> dict:
    """End-to-end preprocessing that returns everything the model needs.

    Parameters
    ----------
    stock_df:
        Raw OHLCV DataFrame from :func:`data_loader.fetch_stock_data`.
    news_df:
        Optional headlines DataFrame.  When *None*, only price features
        are used.
    window_size:
        LSTM look-back window.
    test_ratio:
        Fraction of data reserved for evaluation.
    sentiment_method:
        Which sentiment backend to use.
    feature_cols:
        Which columns to normalise and feed into the model.  When *None*,
        defaults to ``["Close"]`` (price-only) or includes sentiment and
        technical-indicator columns when available.
    add_indicators:
        Whether to compute and include technical indicators (RSI, MACD,
        moving averages, Bollinger Bands, Volatility).  Default ``True``.

    Returns
    -------
    dict with keys:
        ``X_train``, ``X_test``, ``y_train``, ``y_test``,
        ``scaler``, ``merged_df``, ``feature_cols``.
    """
    if news_df is not None:
        df = merge_sentiment_with_stock(stock_df, news_df, method=sentiment_method)
    else:
        df = stock_df.copy()

    # Add technical indicators
    if add_indicators:
        try:
            from features import add_technical_indicators  # noqa: PLC0415

            df = add_technical_indicators(df, drop_na=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not add technical indicators: %s", exc)

    # Default feature columns
    if feature_cols is None:
        candidates = [
            "Close",
            "RSI",
            "MACD",
            "MACD_Signal",
            "BB_Upper",
            "BB_Lower",
            "SMA_20",
            "EMA_12",
            "Volatility",
            "compound",
            "sentiment_pos",
            "sentiment_neg",
        ]
        feature_cols = [c for c in candidates if c in df.columns]
        if not feature_cols:
            feature_cols = ["Close"]

    scaled_df, scaler = normalize_stock_data(df, feature_cols=feature_cols)

    X, y = create_sequences(scaled_df.values, window_size=window_size)
    X_train, X_test, y_train, y_test = train_test_split_sequences(X, y, test_ratio=test_ratio)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "merged_df": df,
        "feature_cols": feature_cols,
    }
