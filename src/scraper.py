from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from time import sleep
import re
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import requests


MUBASHER_NEWS_URL = "https://english.mubasher.info/markets/EGX/stocks/{symbol}/news"
DATE_ONLY_PATTERN = re.compile(
    r"^(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+)(?:\s+(?P<year>\d{4}))?\s+"
    r"(?P<time>\d{1,2}:\d{2}\s+[AP]M)$"
)
DATE_WITH_TITLE_PATTERN = re.compile(
    r"^(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+)(?:\s+(?P<year>\d{4}))?\s+"
    r"(?P<time>\d{1,2}:\d{2}\s+[AP]M)\s+(?P<title>.+)$"
)
ARTICLE_URL_PATTERN = re.compile(r"^https://english\.mubasher\.info/news/\d+/.+")


MONTH_LOOKUP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


@dataclass(frozen=True)
class HeadlineRecord:
    date: datetime
    headline: str
    category: str
    source: str
    ticker: str
    url: str | None = None


class MubasherScraper:
    def __init__(
        self,
        request_delay_seconds: float = 1.0,
        session: "requests.Session" | None = None,
    ) -> None:
        self.request_delay_seconds = request_delay_seconds
        self.session = session or self._build_session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            }
        )

    def scrape(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        max_pages: int = 6,
    ) -> pd.DataFrame:
        rows: list[HeadlineRecord] = []

        for page in range(1, max_pages + 1):
            url = self._build_news_url(symbol=symbol, page=page)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            page_rows = self._parse_page(
                html=response.text,
                symbol=symbol,
                default_year=end_date.year,
            )
            if not page_rows:
                break

            rows.extend(page_rows)
            oldest_page_date = min(row.date.date() for row in page_rows)
            if oldest_page_date < start_date:
                break

            sleep(self.request_delay_seconds)

        frame = pd.DataFrame(
            [
                {
                    "date": row.date.date(),
                    "published_at": row.date,
                    "headline": row.headline,
                    "category": row.category,
                    "source": row.source,
                    "ticker": row.ticker,
                    "url": row.url,
                }
                for row in rows
                if start_date <= row.date.date() <= end_date
            ]
        )
        if frame.empty:
            return frame

        frame = (
            frame.sort_values(["published_at", "headline"])
            .drop_duplicates(subset=["ticker", "headline", "date"])
            .reset_index(drop=True)
        )
        return frame

    @staticmethod
    def _build_news_url(symbol: str, page: int) -> str:
        base = MUBASHER_NEWS_URL.format(symbol=symbol.upper())
        if page == 1:
            return base
        return f"{base}/{page}"

    def _parse_page(
        self,
        html: str,
        symbol: str,
        default_year: int,
    ) -> list[HeadlineRecord]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError("beautifulsoup4 is required to parse Mubasher pages.") from exc

        soup = BeautifulSoup(html, "html.parser")
        lines = [line.strip() for line in soup.get_text("\n", strip=True).splitlines() if line.strip()]
        anchors = self._candidate_urls(soup)
        anchor_texts = self._candidate_anchor_texts(soup)

        start_index = self._find_news_section_start(lines)
        if start_index is None:
            return []

        results: list[HeadlineRecord] = []
        url_index = 0
        pending_article: dict[str, datetime | str | None] | None = None

        for line in lines[start_index:]:
            if line in {"Trending", "Markets", "Companies", "Analysis Tools"}:
                break

            date_with_title_match = DATE_WITH_TITLE_PATTERN.match(line)
            if date_with_title_match:
                if pending_article is not None:
                    results.append(
                        HeadlineRecord(
                            date=pending_article["date"],
                            headline=str(pending_article["headline"]),
                            category=str(pending_article["category"] or ""),
                            source="Mubasher",
                            ticker=symbol.upper(),
                            url=str(pending_article["url"]) if pending_article["url"] else None,
                        )
                    )

                published_at = self._parse_datetime(date_with_title_match.groupdict(), default_year)
                headline = date_with_title_match.group("title").strip()
                url = anchors[url_index] if url_index < len(anchors) else None
                url_index += 1
                pending_article = {
                    "date": published_at,
                    "headline": headline,
                    "category": "",
                    "url": url,
                }
                continue

            date_only_match = DATE_ONLY_PATTERN.match(line)
            if date_only_match:
                if pending_article is not None:
                    results.append(
                        HeadlineRecord(
                            date=pending_article["date"],
                            headline=str(pending_article["headline"]),
                            category=str(pending_article["category"] or ""),
                            source="Mubasher",
                            ticker=symbol.upper(),
                            url=str(pending_article["url"]) if pending_article["url"] else None,
                        )
                    )

                published_at = self._parse_datetime(date_only_match.groupdict(), default_year)
                url = anchors[url_index] if url_index < len(anchors) else None
                url_index += 1
                pending_article = {
                    "date": published_at,
                    "headline": "",
                    "category": "",
                    "url": url,
                }
                continue

            if pending_article is not None and not pending_article["headline"]:
                if line in anchor_texts:
                    continue
                pending_article["headline"] = line
                continue

            if pending_article is not None and line in anchor_texts:
                continue

            if pending_article is not None and line not in {"Cairo - Mubasher:", "Cairo - Mubasher"}:
                pending_article["category"] = line

        if pending_article is not None:
            if pending_article["headline"]:
                results.append(
                    HeadlineRecord(
                        date=pending_article["date"],
                        headline=str(pending_article["headline"]),
                        category=str(pending_article["category"] or ""),
                        source="Mubasher",
                        ticker=symbol.upper(),
                        url=str(pending_article["url"]) if pending_article["url"] else None,
                    )
                )

        return results

    @staticmethod
    def _build_session() -> "requests.Session":
        try:
            import requests
        except ImportError as exc:
            raise ImportError("requests is required to scrape Mubasher pages.") from exc

        return requests.Session()

    @staticmethod
    def _find_news_section_start(lines: list[str]) -> int | None:
        news_indexes = [index for index, line in enumerate(lines) if line == "News"]
        for position, index in enumerate(news_indexes):
            next_news_index = news_indexes[position + 1] if position + 1 < len(news_indexes) else len(lines)
            window_end = min(index + 20, next_news_index)
            window = lines[index + 1 : window_end]
            if any(DATE_ONLY_PATTERN.match(candidate) or DATE_WITH_TITLE_PATTERN.match(candidate) for candidate in window):
                return index + 1
        return None

    @staticmethod
    def _candidate_urls(soup: object) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            if href.startswith("/"):
                href = f"https://english.mubasher.info{href}"
            if ARTICLE_URL_PATTERN.match(href) and href not in seen:
                seen.add(href)
                urls.append(href)

        return urls

    @staticmethod
    def _candidate_anchor_texts(soup: object) -> set[str]:
        texts: set[str] = set()

        for anchor in soup.find_all("a", href=True):
            text = anchor.get_text(" ", strip=True)
            if text:
                texts.add(text)

        return texts

    @staticmethod
    def _parse_datetime(match_groups: dict[str, str | None], default_year: int) -> datetime:
        month = MONTH_LOOKUP[match_groups["month"].lower()]
        year = int(match_groups["year"] or default_year)
        hour_text, minute_meridiem = str(match_groups["time"]).split(":", maxsplit=1)
        normalized_hour = "12" if hour_text == "00" else hour_text
        timestamp = datetime.strptime(
            f"{match_groups['day']} {month} {year} {normalized_hour}:{minute_meridiem}",
            "%d %m %Y %I:%M %p",
        )
        return timestamp
