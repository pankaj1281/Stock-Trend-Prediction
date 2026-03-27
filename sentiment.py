"""
sentiment.py
------------
Sentiment analysis for financial news headlines.

Two modes are supported:

a) **VADER** (baseline, fast, no GPU required)
   Uses NLTK's Valence Aware Dictionary for Sentiment Reasoning, a
   lexicon-based analyser tuned for social-media / financial text.

b) **FinBERT** (advanced, transformer-based)
   Uses the ProsusAI/finbert model from Hugging Face Transformers.
   Requires ``transformers`` and ``torch`` (or ``tensorflow``) to be
   installed.  Falls back to VADER automatically if the model cannot be
   loaded.

Public API
----------
* :func:`compute_sentiment`     – compute scores for a DataFrame of headlines
* :func:`weighted_sentiment`    – recency-weighted aggregation per day
* :func:`aggregate_sentiment`   – simple daily mean aggregation
"""

import logging
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── NLTK helpers ──────────────────────────────────────────────────────────────


def _ensure_vader() -> None:
    """Download VADER lexicon if not present."""
    import nltk  # noqa: PLC0415

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        logger.info("Downloading NLTK vader_lexicon …")
        nltk.download("vader_lexicon", quiet=True)


# ── VADER ─────────────────────────────────────────────────────────────────────


def _vader_scores(texts: list[str]) -> list[dict]:
    """Return VADER polarity score dicts for each text."""
    _ensure_vader()
    from nltk.sentiment.vader import SentimentIntensityAnalyzer  # noqa: PLC0415

    sid = SentimentIntensityAnalyzer()
    return [sid.polarity_scores(t) for t in texts]


def compute_vader_sentiment(df: pd.DataFrame, text_col: str = "headline") -> pd.DataFrame:
    """Append VADER sentiment columns to *df*.

    New columns: ``compound``, ``sentiment_pos``, ``sentiment_neg``,
    ``sentiment_neu``, ``sentiment_label``.

    Parameters
    ----------
    df:
        DataFrame containing news headlines.
    text_col:
        Name of the column with raw headline text.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with sentiment columns added.
    """
    df = df.copy()
    texts = df[text_col].astype(str).tolist()
    scores = _vader_scores(texts)
    score_df = pd.DataFrame(scores, index=df.index)
    score_df.rename(
        columns={"neg": "sentiment_neg", "neu": "sentiment_neu", "pos": "sentiment_pos"},
        inplace=True,
    )
    # Derive categorical label from compound score
    score_df["sentiment_label"] = score_df["compound"].apply(
        lambda v: "positive" if v >= 0.05 else ("negative" if v <= -0.05 else "neutral")
    )
    return pd.concat([df, score_df], axis=1)


# ── FinBERT ───────────────────────────────────────────────────────────────────


def _load_finbert():
    """Load FinBERT tokeniser and model (lazy).

    Returns
    -------
    tuple[tokenizer, model, device]

    Raises
    ------
    ImportError
        If ``transformers`` or ``torch`` are not installed.
    RuntimeError
        If the model cannot be downloaded from Hugging Face.
    """
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "FinBERT requires 'transformers' and 'torch'. "
            "Install them with: pip install transformers torch"
        ) from exc

    model_name = "ProsusAI/finbert"
    logger.info("Loading FinBERT model '%s' …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info("FinBERT loaded on device: %s", device)
    return tokenizer, model, device


def _finbert_scores(texts: list[str], batch_size: int = 16) -> list[dict]:
    """Run FinBERT inference and return score dicts.

    Each dict contains keys ``compound``, ``sentiment_pos``,
    ``sentiment_neg``, ``sentiment_neu``, ``sentiment_label``.
    The *compound* value mirrors VADER's scale:
    positive → +score, negative → -score.
    """
    import torch  # noqa: PLC0415

    tokenizer, model, device = _load_finbert()
    all_scores: list[dict] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoding = tokenizer(
            batch,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        # FinBERT label order: positive(0), negative(1), neutral(2)
        for row in probs:
            pos, neg, neu = float(row[0]), float(row[1]), float(row[2])
            compound = pos - neg  # range [-1, 1]
            label = "positive" if compound >= 0.05 else ("negative" if compound <= -0.05 else "neutral")
            all_scores.append(
                {
                    "compound": compound,
                    "sentiment_pos": pos,
                    "sentiment_neg": neg,
                    "sentiment_neu": neu,
                    "sentiment_label": label,
                }
            )

    return all_scores


def compute_finbert_sentiment(df: pd.DataFrame, text_col: str = "headline") -> pd.DataFrame:
    """Append FinBERT sentiment columns to *df*.

    Falls back to VADER if FinBERT cannot be loaded (e.g. transformers not
    installed or no internet access for model download).

    Parameters
    ----------
    df:
        DataFrame containing news headlines.
    text_col:
        Name of the column with raw headline text.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with sentiment columns added (same schema as
        :func:`compute_vader_sentiment`).
    """
    df = df.copy()
    texts = df[text_col].astype(str).tolist()

    try:
        scores = _finbert_scores(texts)
    except (ImportError, Exception) as exc:  # noqa: BLE001
        logger.warning(
            "FinBERT unavailable (%s). Falling back to VADER.", exc
        )
        return compute_vader_sentiment(df, text_col=text_col)

    score_df = pd.DataFrame(scores, index=df.index)
    return pd.concat([df, score_df], axis=1)


# ── Unified entry-point ───────────────────────────────────────────────────────


def compute_sentiment(
    df: pd.DataFrame,
    text_col: str = "headline",
    mode: Literal["vader", "finbert"] = "vader",
) -> pd.DataFrame:
    """Compute sentiment scores for news headlines.

    Parameters
    ----------
    df:
        DataFrame with a column of raw news headlines.
    text_col:
        Column name containing the text.
    mode:
        ``"vader"`` (default, fast) or ``"finbert"`` (transformer-based,
        more accurate for financial text).

    Returns
    -------
    pd.DataFrame
        Copy of *df* extended with columns:
        ``compound``, ``sentiment_pos``, ``sentiment_neg``,
        ``sentiment_neu``, ``sentiment_label``.
    """
    if mode == "finbert":
        return compute_finbert_sentiment(df, text_col=text_col)
    return compute_vader_sentiment(df, text_col=text_col)


# ── Aggregation helpers ───────────────────────────────────────────────────────


def aggregate_sentiment(
    news_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Aggregate per-headline sentiment scores to one row per calendar day
    using a simple mean.

    Parameters
    ----------
    news_df:
        DataFrame from :func:`compute_sentiment`.  Must contain ``date``,
        ``compound``, ``sentiment_pos``, ``sentiment_neg``, ``sentiment_neu``.
    date_col:
        Column containing dates.

    Returns
    -------
    pd.DataFrame
        Daily aggregated sentiment indexed by date (column name ``"Date"``).
    """
    score_cols = [
        c for c in ["compound", "sentiment_pos", "sentiment_neg", "sentiment_neu"]
        if c in news_df.columns
    ]
    daily = (
        news_df.copy()
        .assign(**{date_col: pd.to_datetime(news_df[date_col]).dt.normalize()})
        .groupby(date_col)[score_cols]
        .mean()
    )
    daily.index.name = "Date"
    return daily


def weighted_sentiment(
    news_df: pd.DataFrame,
    date_col: str = "date",
    decay: float = 0.9,
) -> pd.DataFrame:
    """Aggregate daily sentiment with **recency weighting**.

    Headlines from more recent dates within a trading day receive higher
    weight.  The weight for item *i* (0-indexed, most recent last) within a
    given date is ``decay^(n - 1 - i)`` where *n* is the total number of
    items on that day.

    Parameters
    ----------
    news_df:
        DataFrame from :func:`compute_sentiment`.
    date_col:
        Column containing dates.
    decay:
        Exponential decay factor (default 0.9).  Values closer to 1 give
        more uniform weighting; values closer to 0 heavily emphasise the
        most recent headline.

    Returns
    -------
    pd.DataFrame
        Weighted daily sentiment indexed by ``"Date"``.
    """
    score_cols = [
        c for c in ["compound", "sentiment_pos", "sentiment_neg", "sentiment_neu"]
        if c in news_df.columns
    ]

    df = news_df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    records = []
    for date, group in df.groupby(date_col):
        n = len(group)
        weights = np.array([decay ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()  # normalise
        row = {"Date": date}
        for col in score_cols:
            row[col] = float(np.dot(weights, group[col].values))
        records.append(row)

    if not records:
        return pd.DataFrame(columns=["Date"] + score_cols).set_index("Date")

    result = pd.DataFrame(records).set_index("Date")
    result.index = pd.to_datetime(result.index)
    return result


def merge_sentiment_with_stock(
    stock_df: pd.DataFrame,
    news_df: pd.DataFrame,
    date_col: str = "date",
    mode: Literal["vader", "finbert"] = "vader",
    aggregation: Literal["mean", "weighted"] = "weighted",
    decay: float = 0.9,
) -> pd.DataFrame:
    """Compute sentiment, aggregate daily, and left-join onto *stock_df*.

    Missing trading days (no news) are forward-filled then backward-filled.

    Parameters
    ----------
    stock_df:
        OHLCV DataFrame indexed by date.
    news_df:
        Headlines DataFrame with ``date`` and ``headline`` columns.
    date_col:
        Date column in *news_df*.
    mode:
        Sentiment backend: ``"vader"`` or ``"finbert"``.
    aggregation:
        ``"mean"`` for simple daily average; ``"weighted"`` for
        recency-weighted aggregation.
    decay:
        Decay factor used only when ``aggregation="weighted"``.

    Returns
    -------
    pd.DataFrame
        *stock_df* extended with sentiment columns.
    """
    # Compute per-headline scores
    scored_df = compute_sentiment(news_df, text_col="headline", mode=mode)

    # Aggregate to daily
    if aggregation == "weighted":
        daily = weighted_sentiment(scored_df, date_col=date_col, decay=decay)
    else:
        daily = aggregate_sentiment(scored_df, date_col=date_col)

    # Left-join onto stock data
    merged = stock_df.join(daily, how="left")
    sentiment_cols = daily.columns.tolist()

    # Fill gaps (no news on weekends/holidays)
    merged[sentiment_cols] = merged[sentiment_cols].ffill().bfill()

    # Fill any remaining NaN with neutral values
    for col in ["sentiment_neg", "sentiment_pos", "sentiment_neu"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
    if "compound" in merged.columns:
        merged["compound"] = merged["compound"].fillna(0.0)

    return merged
