from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .config import CHARTS_DIR, DEFAULT_TICKERS, PROCESSED_DATA_DIR, RAW_DATA_DIR, PipelineConfig, TickerConfig, ensure_project_dirs

if TYPE_CHECKING:
    import pandas as pd


def merge_sentiment_and_prices(sentiment_frame: pd.DataFrame, price_frame: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd

    if sentiment_frame.empty or price_frame.empty:
        return pd.DataFrame()

    working_prices = price_frame.copy()
    working_prices["date"] = pd.to_datetime(working_prices["date"]).dt.normalize()

    merged = sentiment_frame.merge(
        working_prices,
        left_on=["date", "ticker"],
        right_on=["date", "ticker"],
        how="inner",
    )
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def persist_frame(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def run_for_ticker(
    ticker: TickerConfig,
    config: PipelineConfig,
) -> dict[str, str | int]:
    import matplotlib.pyplot as plt

    from .prices import download_prices
    from .scraper import MubasherScraper
    from .sentiment import aggregate_daily_sentiment, score_headlines
    from .visualize import compute_lag_correlations, plot_dual_axis, plot_lag_bars, plot_scatter

    scraper = MubasherScraper(request_delay_seconds=config.request_delay_seconds)
    headline_frame = scraper.scrape(
        symbol=ticker.symbol,
        start_date=config.start_date,
        end_date=config.end_date,
        max_pages=config.max_news_pages,
    )
    persist_frame(headline_frame, RAW_DATA_DIR / f"{ticker.symbol.lower()}_headlines.csv")

    scored_frame = score_headlines(headline_frame)
    persist_frame(scored_frame, PROCESSED_DATA_DIR / f"{ticker.symbol.lower()}_headlines_scored.csv")

    daily_sentiment = aggregate_daily_sentiment(scored_frame)
    persist_frame(daily_sentiment, PROCESSED_DATA_DIR / f"{ticker.symbol.lower()}_daily_sentiment.csv")

    prices_frame = download_prices(
        ticker=ticker.price_symbol,
        start_date=config.start_date,
        end_date=config.end_date,
        label=ticker.symbol,
    )
    persist_frame(prices_frame, RAW_DATA_DIR / f"{ticker.symbol.lower()}_prices.csv")

    merged_frame = merge_sentiment_and_prices(daily_sentiment, prices_frame)
    persist_frame(merged_frame, PROCESSED_DATA_DIR / f"{ticker.symbol.lower()}_merged.csv")

    if not merged_frame.empty:
        plt.style.use(config.charts_style)
        plot_dual_axis(
            frame=merged_frame,
            output_path=CHARTS_DIR / f"{ticker.symbol.lower()}_sentiment_vs_close.png",
            title=f"{ticker.symbol} Sentiment vs Closing Price",
        )
        plot_scatter(
            frame=merged_frame,
            output_path=CHARTS_DIR / f"{ticker.symbol.lower()}_scatter.png",
            title=f"{ticker.symbol} Sentiment vs Next-Day Return",
        )
        lag_frame = compute_lag_correlations(merged_frame, max_lag=3)
        persist_frame(lag_frame, PROCESSED_DATA_DIR / f"{ticker.symbol.lower()}_lag_correlations.csv")
        plot_lag_bars(
            frame=merged_frame,
            output_path=CHARTS_DIR / f"{ticker.symbol.lower()}_lag_correlations.png",
            title=f"{ticker.symbol} Lag Correlations",
            max_lag=3,
        )

    return {
        "ticker": ticker.symbol,
        "headline_rows": int(len(headline_frame)),
        "daily_rows": int(len(daily_sentiment)),
        "merged_rows": int(len(merged_frame)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EGX stock sentiment analysis pipeline.")
    parser.add_argument("--tickers", nargs="+", default=["TMGH", "COMI"], help="Ticker symbols to analyze.")
    parser.add_argument("--start", required=False, default="2024-01-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", required=False, default=date.today().isoformat(), help="End date in YYYY-MM-DD format.")
    parser.add_argument("--pages", type=int, default=6, help="Maximum number of Mubasher news pages to scrape per ticker.")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between scraper requests in seconds.")
    parser.add_argument("--with-llm-analysis", action="store_true", help="Generate an OpenAI-powered findings report after the data pipeline completes.")
    parser.add_argument("--llm-model", default=os.getenv("OPENAI_MODEL", "gpt-5.2"), help="OpenAI model to use for the optional findings report.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    unknown = [symbol.upper() for symbol in args.tickers if symbol.upper() not in DEFAULT_TICKERS]
    if unknown:
        supported = ", ".join(sorted(DEFAULT_TICKERS))
        raise ValueError(f"Unsupported ticker(s): {', '.join(unknown)}. Supported tickers: {supported}")

    selected_tickers = [DEFAULT_TICKERS[symbol.upper()] for symbol in args.tickers]
    return PipelineConfig(
        start_date=datetime.strptime(args.start, "%Y-%m-%d").date(),
        end_date=datetime.strptime(args.end, "%Y-%m-%d").date(),
        max_news_pages=args.pages,
        request_delay_seconds=args.delay,
        tickers=selected_tickers,
    )


def main() -> int:
    import pandas as pd

    args = parse_args()
    ensure_project_dirs()
    config = build_config(args)

    summaries = [run_for_ticker(ticker=ticker, config=config) for ticker in config.tickers]
    summary_frame = pd.DataFrame(summaries)
    persist_frame(summary_frame, PROCESSED_DATA_DIR / "run_summary.csv")
    print(summary_frame.to_string(index=False))

    if args.with_llm_analysis:
        from .llm_analysis import run_llm_analysis_stage

        report_paths = run_llm_analysis_stage(tickers=config.tickers, model=args.llm_model)
        print(f"LLM context saved to: {report_paths['context_path']}")
        print(f"LLM report saved to: {report_paths['report_path']}")

    return 0
