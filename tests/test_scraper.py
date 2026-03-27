from __future__ import annotations

import unittest

from src.scraper import MubasherScraper

try:
    import bs4  # noqa: F401
except ImportError:
    BS4_AVAILABLE = False
else:
    BS4_AVAILABLE = True


SAMPLE_HTML = """
<html>
  <body>
    <div>News</div>
    <div>Top News</div>
    <div>News</div>
    <a href="/news/11111/first-story">Story 1</a>
    <div>17 March 04:34 PM TMG Holding stock tests EGP 74 level amid selling pressure</div>
    <div>Cairo - Mubasher: Technical analysis indicates selling pressure.</div>
    <div>Trading Activities</div>
    <a href="/news/22222/second-story">Story 2</a>
    <div>16 March 10:15 AM TMG Holding profits rise on growth</div>
    <div>Cairo - Mubasher: Profits increased strongly.</div>
    <div>Financial Results</div>
    <div>Trending</div>
  </body>
</html>
"""


@unittest.skipUnless(BS4_AVAILABLE, "beautifulsoup4 is not installed")
class MubasherScraperTests(unittest.TestCase):
    def test_parse_page_extracts_article_category_and_url(self) -> None:
        scraper = MubasherScraper(request_delay_seconds=0.0)

        results = scraper._parse_page(SAMPLE_HTML, symbol="TMGH", default_year=2025)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].headline, "TMG Holding stock tests EGP 74 level amid selling pressure")
        self.assertEqual(results[0].category, "Trading Activities")
        self.assertEqual(results[0].url, "https://english.mubasher.info/news/11111/first-story")
        self.assertEqual(results[1].category, "Financial Results")

    def test_find_news_section_start_skips_generic_navigation_news_labels(self) -> None:
        lines = [
            "Markets",
            "News",
            "Top News",
            "Companies",
            "News",
            "17 March 04:34 PM Headline",
            "Trending",
        ]

        start_index = MubasherScraper._find_news_section_start(lines)

        self.assertEqual(start_index, 5)


if __name__ == "__main__":
    unittest.main()
