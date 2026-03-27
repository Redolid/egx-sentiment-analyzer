from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
REPORTS_DIR = OUTPUTS_DIR / "reports"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"


@dataclass(frozen=True)
class TickerConfig:
    symbol: str
    price_symbol: str
    company_name: str
    sector: str


@dataclass
class PipelineConfig:
    start_date: date = field(default_factory=lambda: date.today() - timedelta(days=365))
    end_date: date = field(default_factory=date.today)
    max_news_pages: int = 6
    request_delay_seconds: float = 1.0
    charts_style: str = "seaborn-v0_8-whitegrid"
    output_root: Path = BASE_DIR
    tickers: list[TickerConfig] = field(default_factory=list)


DEFAULT_TICKERS: dict[str, TickerConfig] = {
    "TMGH": TickerConfig(
        symbol="TMGH",
        price_symbol="TMGH.CA",
        company_name="Talaat Moustafa Group Holding",
        sector="Real Estate",
    ),
    "COMI": TickerConfig(
        symbol="COMI",
        price_symbol="COMI.CA",
        company_name="Commercial International Bank",
        sector="Banking",
    ),
    "ETEL": TickerConfig(
        symbol="ETEL",
        price_symbol="ETEL.CA",
        company_name="Telecom Egypt",
        sector="Telecom",
    ),
    "EFID": TickerConfig(
        symbol="EFID",
        price_symbol="EFID.CA",
        company_name="EFG Holding",
        sector="Financial Services",
    ),
}


def ensure_project_dirs() -> None:
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, CHARTS_DIR, REPORTS_DIR, NOTEBOOKS_DIR):
        path.mkdir(parents=True, exist_ok=True)
