"""Prediction of price change for a ticker.

This module provides a simple wrapper around the existing classification model
to give an estimated price change (percentage and absolute) for the next
trading day.  If a dedicated regression model is not available, we fallback to
using the classification signal and its confidence as a heuristic:

* ``up``   → +1% (scaled by confidence)
* ``down`` → -1% (scaled by confidence)
* ``neutral`` → 0%

The function returns a dictionary that can be printed by the CLI.
"""

from __future__ import annotations

from datetime import date, timedelta
import warnings

import joblib
import pandas as pd
import requests

from src.data.news_fetcher import fetch_news
from src.data.stock_fetcher import fetch_stock_data
from src.features.build_dataset import add_sentiment_features, align_to_trading_days
from src.features.technical_indicators import add_indicators
from src.preprocessing.clean_news import clean_news
from src.preprocessing.clean_stock_data import clean_stock_data
from src.sentiment.sentiment_model import daily_sentiment
from src.utils.config import DISCLAIMER, LABELS, MARKET_FEATURES, MODEL_PATH, SENTIMENT_FEATURES
from src.utils.errors import InsufficientDataError

# Load the same bundle used by ``predict_ticker``
_bundle = None

def _load_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("No trained model found. Run: python main.py train")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_price_change(ticker: str) -> dict:
    """Return an estimated price change for ``ticker``.

    The result contains:
        * ``ticker`` – the ticker symbol.
        * ``as_of`` – date of the latest available price.
        * ``last_close`` – closing price of the latest bar.
        * ``predicted_change_pct`` – estimated percentage change for the next day.
        * ``predicted_change_abs`` – absolute dollar change (based on ``last_close``).
        * ``signal`` – the classification signal (up / down / neutral).
        * ``confidence`` – probability of the chosen signal.
        * ``disclaimer`` – legal disclaimer from config.
    """
    bundle = _load_bundle()
    model, features, classes = bundle["model"], bundle["features"], bundle["classes"]
    uses_sentiment = any(f in SENTIMENT_FEATURES for f in features)

    # Gather recent price data (last ~220 days, same as ``predict_ticker``)
    start = (date.today() - timedelta(days=220)).isoformat()
    prices = clean_stock_data(fetch_stock_data(ticker, start=start))

    # Fetch recent news if the model expects sentiment features
    if uses_sentiment:
        try:
            raw_news = fetch_news(ticker, days=7)
        except requests.RequestException as e:
            warnings.warn(
                f"News fetch failed ({type(e).__name__}); predicting without sentiment."
            )
            raw_news = pd.DataFrame(columns=["date", "title", "description", "source"])
        news = align_to_trading_days(clean_news(raw_news), prices["date"], allow_after_last=True)
        daily = daily_sentiment(news)
    else:
        daily = daily_sentiment(pd.DataFrame())

    # Build the feature dataframe expected by the model
    df = add_sentiment_features(add_indicators(prices), daily)
    # Keep only rows where required market features are present
    valid = df.dropna(subset=[f for f in features if f in MARKET_FEATURES])
    if valid.empty:
        raise InsufficientDataError(
            f"Only {len(prices)} price rows for '{ticker}'; not enough history to compute indicators."
        )
    # Use the most recent row for prediction
    row = valid.iloc[[-1]]
    probas = model.predict_proba(row[features])[0]
    probs = {LABELS[int(c)]: float(p) for c, p in zip(classes, probas)}
    signal = max(probs, key=probs.get)
    confidence = probs[signal]

    # Heuristic change based on the classification signal
    # Scale a base 1% change by the confidence; neutral yields 0.
    base_change = 0.0
    if signal == "up":
        base_change = 1.0
    elif signal == "down":
        base_change = -1.0
    # Apply confidence scaling
    predicted_change_pct = round(base_change * confidence, 2)
    last_close = float(row["close"].iloc[0])
    predicted_change_abs = round(last_close * predicted_change_pct / 100.0, 2)

    return {
        "ticker": ticker.upper(),
        "as_of": str(row["date"].iloc[0].date()),
        "last_close": round(last_close, 2),
        "signal": signal,
        "confidence": round(confidence, 4),
        "predicted_change_pct": predicted_change_pct,
        "predicted_change_abs": predicted_change_abs,
        "disclaimer": DISCLAIMER,
    }
