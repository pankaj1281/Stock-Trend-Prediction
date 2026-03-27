"""
backtesting.py
--------------
Simulates a simple rule-based trading strategy on historical predictions
and computes performance metrics.

Strategy
--------
* **Buy**  when the predicted next-day close > current close by more than
  ``threshold`` percent.
* **Sell** when the predicted next-day close < current close by more than
  ``threshold`` percent.
* **Hold** otherwise.

Each trade uses ``position_size`` fraction of the current portfolio value.
Short selling is not supported; the agent can only hold a position or be
in cash.

Public API
----------
* :class:`BacktestResult`      – named result container
* :func:`run_backtest`         – run strategy and return results
* :func:`compute_sharpe_ratio` – standalone Sharpe computation
* :func:`summary_report`       – pretty-print backtest statistics
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────


@dataclass
class BacktestResult:
    """Container for backtest outputs.

    Attributes
    ----------
    portfolio_values:
        Series of portfolio value at the end of each trading day.
    trades:
        DataFrame with one row per trade (``date``, ``action``, ``price``,
        ``shares``, ``portfolio_value``).
    total_return:
        Total return as a percentage.
    annualised_return:
        Annualised return assuming 252 trading days per year.
    sharpe_ratio:
        Annualised Sharpe ratio (excess return / volatility, risk-free = 0).
    max_drawdown:
        Maximum peak-to-trough drawdown as a percentage.
    win_rate:
        Fraction of closed trades that were profitable.
    total_profit_loss:
        Absolute profit or loss in dollars.
    n_trades:
        Total number of buy/sell transactions.
    accuracy:
        Fraction of days where the direction of price change was predicted
        correctly.
    """

    portfolio_values: pd.Series = field(default_factory=pd.Series)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_return: float = 0.0
    annualised_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_profit_loss: float = 0.0
    n_trades: int = 0
    accuracy: float = 0.0


# ── Metric helpers ────────────────────────────────────────────────────────────


def compute_sharpe_ratio(
    portfolio_values: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute the annualised Sharpe Ratio from a series of portfolio values.

    Parameters
    ----------
    portfolio_values:
        Daily portfolio value series (must have at least 2 observations).
    risk_free_rate:
        Annual risk-free rate (default 0.0).
    periods_per_year:
        Number of trading periods per year (default 252 for daily data).

    Returns
    -------
    float
        Annualised Sharpe Ratio.  Returns 0.0 if the series has no variance.
    """
    returns = portfolio_values.pct_change().dropna()
    if returns.empty or returns.std() == 0:
        return 0.0
    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf
    sharpe = (excess.mean() / excess.std()) * np.sqrt(periods_per_year)
    return float(sharpe)


def _max_drawdown(portfolio_values: pd.Series) -> float:
    """Compute the maximum peak-to-trough drawdown as a percentage."""
    rolling_max = portfolio_values.cummax()
    drawdown = (portfolio_values - rolling_max) / rolling_max
    return float(drawdown.min() * 100)


def _direction_accuracy(
    actuals: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """Fraction of periods where the sign of price change is predicted correctly."""
    actual_dir = np.sign(np.diff(actuals))
    pred_dir = np.sign(predictions[1:] - actuals[:-1])
    correct = np.sum(actual_dir == pred_dir)
    return float(correct / len(actual_dir)) if len(actual_dir) > 0 else 0.0


# ── Core backtest ─────────────────────────────────────────────────────────────


def run_backtest(
    prices: pd.Series,
    predictions: np.ndarray,
    initial_capital: float = 10_000.0,
    threshold: float = 0.5,
    position_size: float = 1.0,
    transaction_cost: float = 0.001,
) -> BacktestResult:
    """Simulate a long-only trading strategy driven by model predictions.

    Parameters
    ----------
    prices:
        Actual daily closing prices (indexed by date).  Length must match
        ``len(predictions)``.
    predictions:
        Predicted next-day closing prices for each day in *prices*.
    initial_capital:
        Starting portfolio value in dollars (default $10,000).
    threshold:
        Minimum predicted percent change to trigger a buy or sell
        (default 0.5%).
    position_size:
        Fraction of available cash to deploy on each buy (default 1.0 =
        fully invested).
    transaction_cost:
        Round-trip cost per trade as a fraction of trade value (default
        0.1% per leg, i.e. 0.001).

    Returns
    -------
    BacktestResult
        See :class:`BacktestResult` for field documentation.
    """
    if len(prices) != len(predictions):
        raise ValueError(
            f"'prices' length ({len(prices)}) must equal "
            f"'predictions' length ({len(predictions)})."
        )

    cash = float(initial_capital)
    shares_held = 0.0
    portfolio_series: list[tuple] = []
    trade_log: list[dict] = []
    buy_price: float | None = None

    dates = prices.index if isinstance(prices, pd.Series) else range(len(prices))
    price_values = prices.values if isinstance(prices, pd.Series) else np.asarray(prices)

    for i, (date, price, pred) in enumerate(
        zip(dates, price_values, predictions)
    ):
        portfolio_value = cash + shares_held * price

        pct_change = (pred - price) / price * 100 if price > 0 else 0.0

        action = "hold"
        if pct_change > threshold and cash > 0:
            # BUY
            amount = cash * position_size
            cost = amount * transaction_cost
            shares_bought = (amount - cost) / price
            cash -= amount
            shares_held += shares_bought
            action = "buy"
            buy_price = price
            trade_log.append(
                {
                    "date": date,
                    "action": action,
                    "price": price,
                    "shares": shares_bought,
                    "portfolio_value": portfolio_value,
                }
            )

        elif pct_change < -threshold and shares_held > 0:
            # SELL – buy_price should always be set when shares_held > 0
            proceeds = shares_held * price
            cost = proceeds * transaction_cost
            cash += proceeds - cost
            effective_buy_price = buy_price if buy_price is not None else price
            sell_profit = (price - effective_buy_price) * shares_held
            shares_held = 0.0
            action = "sell"
            trade_log.append(
                {
                    "date": date,
                    "action": action,
                    "price": price,
                    "shares": 0.0,
                    "portfolio_value": cash,
                    "profit": sell_profit,
                }
            )
            buy_price = None

        portfolio_series.append((date, cash + shares_held * price))

    # Liquidate any remaining position at last price
    if shares_held > 0:
        final_price = float(price_values[-1])
        proceeds = shares_held * final_price * (1 - transaction_cost)
        cash += proceeds
        portfolio_series[-1] = (dates[-1], cash)

    portfolio_values = pd.Series(
        [v for _, v in portfolio_series],
        index=[d for d, _ in portfolio_series],
        name="Portfolio Value",
    )

    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame(
        columns=["date", "action", "price", "shares", "portfolio_value"]
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    final_value = float(portfolio_values.iloc[-1]) if not portfolio_values.empty else initial_capital
    total_return = (final_value - initial_capital) / initial_capital * 100
    n_days = max(len(portfolio_values), 1)
    annualised_return = (
        ((final_value / initial_capital) ** (252 / n_days) - 1) * 100
    )
    sharpe = compute_sharpe_ratio(portfolio_values)
    max_dd = _max_drawdown(portfolio_values) if len(portfolio_values) > 1 else 0.0
    accuracy = _direction_accuracy(price_values, predictions)

    # Win rate: fraction of sell trades with positive profit
    sell_trades = trades_df[trades_df["action"] == "sell"] if not trades_df.empty else pd.DataFrame()
    if "profit" in sell_trades.columns and len(sell_trades) > 0:
        win_rate = float((sell_trades["profit"] > 0).mean())
    else:
        win_rate = 0.0

    n_trades = len(trades_df)
    total_pl = final_value - initial_capital

    logger.info(
        "Backtest complete | Return: %.2f%% | Sharpe: %.3f | "
        "Max DD: %.2f%% | Accuracy: %.2f%% | Trades: %d",
        total_return,
        sharpe,
        max_dd,
        accuracy * 100,
        n_trades,
    )

    return BacktestResult(
        portfolio_values=portfolio_values,
        trades=trades_df,
        total_return=total_return,
        annualised_return=annualised_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        total_profit_loss=total_pl,
        n_trades=n_trades,
        accuracy=accuracy,
    )


# ── Buy-and-hold benchmark ────────────────────────────────────────────────────


def buy_and_hold(
    prices: pd.Series,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.001,
) -> BacktestResult:
    """Compute the buy-and-hold benchmark for comparison.

    Buys at the first price, holds through, sells at the last price.

    Parameters
    ----------
    prices:
        Actual daily closing prices.
    initial_capital:
        Starting capital.
    transaction_cost:
        Round-trip transaction cost fraction.

    Returns
    -------
    BacktestResult
        Portfolio values series and key metrics for the passive strategy.
    """
    price_values = prices.values if isinstance(prices, pd.Series) else np.asarray(prices)

    buy = float(price_values[0])
    shares = (initial_capital * (1 - transaction_cost)) / buy
    portfolio_values = pd.Series(
        shares * price_values,
        index=prices.index,
        name="Portfolio Value",
    )
    # Sell at end
    final = float(portfolio_values.iloc[-1]) * (1 - transaction_cost)
    portfolio_values.iloc[-1] = final

    total_return = (final - initial_capital) / initial_capital * 100
    n_days = max(len(portfolio_values), 1)
    annualised_return = ((final / initial_capital) ** (252 / n_days) - 1) * 100
    sharpe = compute_sharpe_ratio(portfolio_values)
    max_dd = _max_drawdown(portfolio_values)

    return BacktestResult(
        portfolio_values=portfolio_values,
        trades=pd.DataFrame(),
        total_return=total_return,
        annualised_return=annualised_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=0.0,
        total_profit_loss=final - initial_capital,
        n_trades=2,
        accuracy=0.0,
    )


# ── Pretty-print helper ───────────────────────────────────────────────────────


def summary_report(result: BacktestResult, label: str = "Strategy") -> str:
    """Return a formatted multi-line performance summary.

    Parameters
    ----------
    result:
        :class:`BacktestResult` from :func:`run_backtest` or
        :func:`buy_and_hold`.
    label:
        Name of the strategy (used in the header line).

    Returns
    -------
    str
        Human-readable performance summary.
    """
    lines = [
        f"{'─' * 50}",
        f" Backtest Summary: {label}",
        f"{'─' * 50}",
        f"  Total Return        : {result.total_return:+.2f}%",
        f"  Annualised Return   : {result.annualised_return:+.2f}%",
        f"  Sharpe Ratio        : {result.sharpe_ratio:.3f}",
        f"  Max Drawdown        : {result.max_drawdown:.2f}%",
        f"  Total P&L           : ${result.total_profit_loss:+,.2f}",
        f"  Number of Trades    : {result.n_trades}",
        f"  Win Rate            : {result.win_rate * 100:.1f}%",
        f"  Direction Accuracy  : {result.accuracy * 100:.1f}%",
        f"{'─' * 50}",
    ]
    return "\n".join(lines)
