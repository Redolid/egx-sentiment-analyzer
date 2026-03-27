from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PROCESSED_DATA_DIR, REPORTS_DIR, TickerConfig


DEFAULT_OPENAI_MODEL = "gpt-5.2"


@dataclass
class TickerAnalysisPayload:
    ticker: str
    company_name: str
    sector: str
    headline_rows: int
    merged_rows: int
    date_range: dict[str, str | None]
    sentiment_summary: dict[str, float | int | None]
    return_summary: dict[str, float | int | None]
    lag_summary: list[dict[str, float | int | None]]
    notable_days: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "sector": self.sector,
            "headline_rows": self.headline_rows,
            "merged_rows": self.merged_rows,
            "date_range": self.date_range,
            "sentiment_summary": self.sentiment_summary,
            "return_summary": self.return_summary,
            "lag_summary": self.lag_summary,
            "notable_days": self.notable_days,
        }


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _safe_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def build_ticker_analysis_payload(ticker: TickerConfig) -> TickerAnalysisPayload:
    ticker_slug = ticker.symbol.lower()
    scored_path = PROCESSED_DATA_DIR / f"{ticker_slug}_headlines_scored.csv"
    merged_path = PROCESSED_DATA_DIR / f"{ticker_slug}_merged.csv"
    lag_path = PROCESSED_DATA_DIR / f"{ticker_slug}_lag_correlations.csv"

    scored = pd.read_csv(scored_path, parse_dates=["date", "published_at"]) if scored_path.exists() else pd.DataFrame()
    merged = pd.read_csv(merged_path, parse_dates=["date"]) if merged_path.exists() else pd.DataFrame()
    lag_frame = pd.read_csv(lag_path) if lag_path.exists() else pd.DataFrame()

    if merged.empty:
        date_range = {"start": None, "end": None}
        sentiment_summary = {
            "mean_sentiment": None,
            "median_sentiment": None,
            "positive_days": 0,
            "negative_days": 0,
        }
        return_summary = {
            "mean_next_day_return": None,
            "positive_return_days": 0,
            "negative_return_days": 0,
        }
        notable_days: list[dict[str, Any]] = []
    else:
        strongest_sentiment_days = (
            merged.reindex(merged["sentiment_score"].abs().sort_values(ascending=False).index)
            .head(5)
            .copy()
        )
        notable_days = [
            {
                "date": row["date"].date().isoformat(),
                "headline_count": _safe_int(row.get("headline_count")),
                "sentiment_score": _safe_float(row.get("sentiment_score")),
                "next_day_return": _safe_float(row.get("next_day_return")),
            }
            for _, row in strongest_sentiment_days.iterrows()
        ]

        date_range = {
            "start": merged["date"].min().date().isoformat(),
            "end": merged["date"].max().date().isoformat(),
        }
        sentiment_summary = {
            "mean_sentiment": _safe_float(merged["sentiment_score"].mean()),
            "median_sentiment": _safe_float(merged["sentiment_score"].median()),
            "positive_days": int((merged["sentiment_score"] > 0.1).sum()),
            "negative_days": int((merged["sentiment_score"] < -0.1).sum()),
        }
        return_summary = {
            "mean_next_day_return": _safe_float(merged["next_day_return"].mean()),
            "positive_return_days": int((merged["next_day_return"] > 0).sum()),
            "negative_return_days": int((merged["next_day_return"] < 0).sum()),
        }

    lag_summary = []
    if not lag_frame.empty:
        for _, row in lag_frame.iterrows():
            lag_summary.append(
                {
                    "lag_days": _safe_int(row.get("lag_days")),
                    "pearson_r": _safe_float(row.get("pearson_r")),
                    "sample_size": _safe_int(row.get("sample_size")),
                }
            )

    return TickerAnalysisPayload(
        ticker=ticker.symbol,
        company_name=ticker.company_name,
        sector=ticker.sector,
        headline_rows=int(len(scored)),
        merged_rows=int(len(merged)),
        date_range=date_range,
        sentiment_summary=sentiment_summary,
        return_summary=return_summary,
        lag_summary=lag_summary,
        notable_days=notable_days,
    )


def build_analysis_context(tickers: list[TickerConfig]) -> dict[str, Any]:
    payloads = [build_ticker_analysis_payload(ticker) for ticker in tickers]
    return {
        "project": "EGX Stock Sentiment Analyzer",
        "research_question": "Does headline sentiment on day T predict next-day EGX stock price movement on day T+1?",
        "tickers": [payload.to_dict() for payload in payloads],
    }


def write_analysis_context(context: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def build_analysis_messages(context: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "You are a careful financial research assistant. "
                        "Analyze the provided EGX sentiment-study context and return strict JSON. "
                        "Be honest about weak evidence, small sample sizes, and null results. "
                        "Do not invent findings not supported by the data."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Return JSON with this exact top-level shape: "
                        "{"
                        "\"executive_summary\": string, "
                        "\"key_findings\": [string], "
                        "\"ticker_findings\": [{\"ticker\": string, \"finding\": string, \"best_lag\": integer|null, \"best_lag_correlation\": number|null}], "
                        "\"limitations\": [string], "
                        "\"next_steps\": [string]"
                        "}. "
                        "Use only the provided data.\n\n"
                        f"Study context:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
                    ),
                }
            ],
        },
    ]


def call_openai_analysis(
    context: dict[str, Any],
    output_path: Path,
    model: str = DEFAULT_OPENAI_MODEL,
) -> Path:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it before running the LLM analysis stage. "
            f"The analysis context has still been saved to {REPORTS_DIR / 'analysis_context.json'}."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("The openai package is required for the LLM analysis stage.") from exc

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        reasoning={"effort": "medium"},
        input=build_analysis_messages(context),
        text={"format": {"type": "json_object"}},
    )

    response_text = (response.output_text or "").strip()
    if not response_text:
        raise RuntimeError("OpenAI returned an empty analysis response.")

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OpenAI returned analysis text that was not valid JSON.") from exc
    markdown = render_markdown_report(parsed, model=model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def render_markdown_report(analysis: dict[str, Any], model: str) -> str:
    lines = [
        "# LLM Findings Report",
        "",
        f"Model: `{model}`",
        "",
        "## Executive Summary",
        "",
        str(analysis.get("executive_summary", "")).strip(),
        "",
        "## Key Findings",
        "",
    ]

    for finding in analysis.get("key_findings", []):
        lines.append(f"- {finding}")

    lines.extend(["", "## Ticker Findings", ""])
    for item in analysis.get("ticker_findings", []):
        ticker = item.get("ticker", "Unknown")
        finding = item.get("finding", "")
        best_lag = item.get("best_lag")
        best_corr = item.get("best_lag_correlation")
        lag_text = "no clear best lag"
        if best_lag is not None and best_corr is not None:
            lag_text = f"best lag = {best_lag} day(s), r = {best_corr:.3f}"
        lines.append(f"- {ticker}: {finding} ({lag_text})")

    lines.extend(["", "## Limitations", ""])
    for limitation in analysis.get("limitations", []):
        lines.append(f"- {limitation}")

    lines.extend(["", "## Next Steps", ""])
    for step in analysis.get("next_steps", []):
        lines.append(f"- {step}")

    lines.append("")
    return "\n".join(lines)


def render_error_report(message: str, model: str) -> str:
    return "\n".join(
        [
            "# LLM Findings Report",
            "",
            f"Model: `{model}`",
            "",
            "## Status",
            "",
            "The OpenAI analysis stage did not complete.",
            "",
            "## Error",
            "",
            message.strip(),
            "",
            "## Next Step",
            "",
            "Retry this stage after fixing the API key, quota, or billing issue.",
            "",
        ]
    )


def run_llm_analysis_stage(
    tickers: list[TickerConfig],
    model: str = DEFAULT_OPENAI_MODEL,
) -> dict[str, Path]:
    context = build_analysis_context(tickers)
    context_path = write_analysis_context(context, REPORTS_DIR / "analysis_context.json")
    report_path = REPORTS_DIR / "llm_findings.md"
    try:
        report_path = call_openai_analysis(context, report_path, model=model)
    except Exception as exc:
        report_path.write_text(render_error_report(str(exc), model=model), encoding="utf-8")
    return {
        "context_path": context_path,
        "report_path": report_path,
    }
