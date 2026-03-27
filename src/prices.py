from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


def download_prices(
    ticker: str,
    start_date: date,
    end_date: date,
    label: str | None = None,
) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is required to download prices.")

    frame = yf.download(
        tickers=ticker,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "price_symbol",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "daily_return",
                "next_day_return",
            ]
        )

    frame = frame.reset_index()
    flattened_columns: list[str] = []
    for column in frame.columns:
        if isinstance(column, tuple):
            flattened = "_".join(str(part) for part in column if part and part != ticker)
        else:
            flattened = str(column)
        flattened_columns.append(flattened.lower().replace(" ", "_"))
    frame.columns = flattened_columns

    if "adjclose" in frame.columns:
        frame = frame.rename(columns={"adjclose": "adj_close"})
    if "adj_close" not in frame.columns and "adj_close_" in frame.columns:
        frame = frame.rename(columns={"adj_close_": "adj_close"})

    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["ticker"] = label or ticker
    frame["price_symbol"] = ticker
    frame["daily_return"] = frame["close"].pct_change()
    frame["next_day_return"] = frame["daily_return"].shift(-1)

    return frame[
        [
            "date",
            "ticker",
            "price_symbol",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "daily_return",
            "next_day_return",
        ]
    ].copy()
