from __future__ import annotations

from pathlib import Path

import pandas as pd


def compute_lag_correlations(frame: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for lag in range(max_lag + 1):
        working = frame.copy()
        working["lagged_sentiment"] = working["sentiment_score"].shift(lag)
        valid = working[["lagged_sentiment", "next_day_return"]].dropna()
        correlation = valid["lagged_sentiment"].corr(valid["next_day_return"])
        rows.append(
            {
                "lag_days": lag,
                "pearson_r": float(correlation) if pd.notna(correlation) else float("nan"),
                "sample_size": int(len(valid)),
            }
        )

    return pd.DataFrame(rows)


def plot_dual_axis(frame: pd.DataFrame, output_path: Path, title: str) -> Path:
    import matplotlib.pyplot as plt

    fig, axis_left = plt.subplots(figsize=(12, 6))
    axis_right = axis_left.twinx()

    axis_left.plot(frame["date"], frame["sentiment_score"], color="#0f766e", marker="o", label="Sentiment")
    axis_right.plot(frame["date"], frame["close"], color="#b45309", marker="s", label="Close Price")

    axis_left.set_title(title)
    axis_left.set_xlabel("Date")
    axis_left.set_ylabel("Sentiment Score", color="#0f766e")
    axis_right.set_ylabel("Close Price", color="#b45309")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_scatter(frame: pd.DataFrame, output_path: Path, title: str) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns

    valid = frame[["sentiment_score", "next_day_return"]].dropna()

    fig, axis = plt.subplots(figsize=(8, 6))
    sns.regplot(
        data=valid,
        x="sentiment_score",
        y="next_day_return",
        scatter_kws={"alpha": 0.7, "color": "#1d4ed8"},
        line_kws={"color": "#dc2626"},
        ax=axis,
    )

    correlation_text = ""
    if len(valid) >= 2:
        correlation = valid["sentiment_score"].corr(valid["next_day_return"])
        correlation_text = f" | r = {correlation:.3f}"

    axis.set_title(f"{title}{correlation_text}")
    axis.set_xlabel("Daily Sentiment Score")
    axis.set_ylabel("Next-Day Return")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_lag_bars(frame: pd.DataFrame, output_path: Path, title: str, max_lag: int = 3) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns

    lag_frame = compute_lag_correlations(frame, max_lag=max_lag)
    fig, axis = plt.subplots(figsize=(8, 5))

    sns.barplot(data=lag_frame, x="lag_days", y="pearson_r", color="#0ea5e9", ax=axis)
    axis.axhline(0.0, color="#334155", linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("Sentiment Lag (days)")
    axis.set_ylabel("Pearson r vs next-day return")

    for index, row in lag_frame.iterrows():
        axis.text(index, row["pearson_r"] if pd.notna(row["pearson_r"]) else 0.0, f"n={row['sample_size']}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path
