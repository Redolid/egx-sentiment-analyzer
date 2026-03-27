from __future__ import annotations

import unittest

import pandas as pd

from src.visualize import compute_lag_correlations


class VisualizeTests(unittest.TestCase):
    def test_compute_lag_correlations_returns_expected_shape_and_sample_sizes(self) -> None:
        frame = pd.DataFrame(
            {
                "sentiment_score": [0.1, 0.2, -0.1, 0.3],
                "next_day_return": [0.05, 0.07, -0.02, 0.1],
            }
        )

        lag_frame = compute_lag_correlations(frame, max_lag=2)

        self.assertListEqual(lag_frame["lag_days"].tolist(), [0, 1, 2])
        self.assertListEqual(lag_frame["sample_size"].tolist(), [4, 3, 2])


if __name__ == "__main__":
    unittest.main()

