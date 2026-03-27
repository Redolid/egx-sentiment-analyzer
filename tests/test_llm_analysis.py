from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.config import TickerConfig
from src.llm_analysis import build_analysis_messages, render_markdown_report, write_analysis_context


class LlmAnalysisTests(unittest.TestCase):
    def test_build_analysis_messages_embeds_json_context(self) -> None:
        context = {"project": "EGX", "tickers": [{"ticker": "TMGH", "merged_rows": 17}]}

        messages = build_analysis_messages(context)

        self.assertEqual(len(messages), 2)
        user_text = messages[1]["content"][0]["text"]
        self.assertIn('"ticker": "TMGH"', user_text)
        self.assertIn("Return JSON", user_text)

    def test_render_markdown_report_formats_sections(self) -> None:
        markdown = render_markdown_report(
            {
                "executive_summary": "Signal looks weak.",
                "key_findings": ["Lag 0 is weakly positive."],
                "ticker_findings": [
                    {
                        "ticker": "TMGH",
                        "finding": "Evidence is limited.",
                        "best_lag": 0,
                        "best_lag_correlation": 0.09,
                    }
                ],
                "limitations": ["Small sample."],
                "next_steps": ["Add more tickers."],
            },
            model="gpt-5.2",
        )

        self.assertIn("# LLM Findings Report", markdown)
        self.assertIn("TMGH: Evidence is limited.", markdown)
        self.assertIn("Small sample.", markdown)

    def test_write_analysis_context_persists_json(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "analysis_context.json"
            write_analysis_context({"hello": "world"}, output_path)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), {"hello": "world"})


if __name__ == "__main__":
    unittest.main()
