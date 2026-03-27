from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src import sentiment


class ArabicSentimentTests(unittest.TestCase):
    def setUp(self) -> None:
        sentiment.get_arabic_sentiment_backend.cache_clear()

    def tearDown(self) -> None:
        sentiment.get_arabic_sentiment_backend.cache_clear()

    def test_normalize_arabic_removes_diacritics_and_normalizes_letters(self) -> None:
        text = "إرتفــاعُ الأرباح"
        self.assertEqual(sentiment.normalize_arabic(text), "ارتفاع الارباح")

    def test_positive_arabic_finance_headline_scores_positive(self) -> None:
        score = sentiment.score_arabic_text("ارتفاع الأرباح ونمو الإيرادات", preferred_backend="lexicon")
        self.assertGreater(score, 0.3)

    def test_negative_arabic_finance_headline_scores_negative(self) -> None:
        score = sentiment.score_arabic_text("تراجع الأرباح وارتفاع الديون", preferred_backend="lexicon")
        self.assertLess(score, -0.3)

    def test_negation_flips_positive_signal(self) -> None:
        positive = sentiment.score_arabic_text("ارتفاع الأرباح", preferred_backend="lexicon")
        negated = sentiment.score_arabic_text("لا ارتفاع في الأرباح", preferred_backend="lexicon")
        self.assertGreater(positive, 0.0)
        self.assertLess(negated, 0.0)

    def test_score_headlines_supports_mixed_languages_without_textblob(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-01"],
                "ticker": ["TMGH", "COMI"],
                "headline": [
                    "ارتفاع الأرباح ونمو الإيرادات",
                    "profits rise on strong growth",
                ],
            }
        )

        with patch("src.sentiment.TextBlob", None):
            scored = sentiment.score_headlines(frame, arabic_backend="lexicon")

        self.assertListEqual(scored["language"].tolist(), ["ar", "en"])
        self.assertTrue((scored["sentiment_score"] > 0).all())
        self.assertTrue((scored["sentiment_label"] == "positive").all())

    def test_aggregate_daily_sentiment_computes_ratios(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-01", "2025-01-02"],
                "ticker": ["TMGH", "TMGH", "TMGH"],
                "headline": ["a", "b", "c"],
                "sentiment_score": [0.4, -0.2, 0.1],
                "sentiment_label": ["positive", "negative", "positive"],
            }
        )

        daily = sentiment.aggregate_daily_sentiment(frame)

        self.assertEqual(len(daily), 2)
        first_day = daily.iloc[0]
        self.assertEqual(first_day["headline_count"], 2)
        self.assertAlmostEqual(first_day["sentiment_score"], 0.1)
        self.assertAlmostEqual(first_day["positive_ratio"], 0.5)
        self.assertAlmostEqual(first_day["negative_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()

