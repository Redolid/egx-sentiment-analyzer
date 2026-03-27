from __future__ import annotations

import unittest

import pandas as pd

from src.pipeline import merge_sentiment_and_prices


class PipelineTests(unittest.TestCase):
    def test_merge_sentiment_and_prices_aligns_on_normalized_dates_and_ticker(self) -> None:
        sentiment_frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "ticker": ["TMGH", "TMGH"],
                "sentiment_score": [0.2, -0.1],
            }
        )
        price_frame = pd.DataFrame(
            {
                "date": ["2025-01-01 15:00:00", "2025-01-02 15:00:00"],
                "ticker": ["TMGH", "TMGH"],
                "close": [10.0, 11.0],
                "next_day_return": [0.1, None],
            }
        )

        merged = merge_sentiment_and_prices(sentiment_frame, price_frame)

        self.assertEqual(len(merged), 2)
        self.assertListEqual(merged["close"].tolist(), [10.0, 11.0])


if __name__ == "__main__":
    unittest.main()

