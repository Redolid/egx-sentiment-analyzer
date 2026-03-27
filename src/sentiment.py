from __future__ import annotations

import math
import re
from functools import lru_cache

import pandas as pd

try:
    from langdetect import detect
except ImportError:  # pragma: no cover
    detect = None

try:
    from textblob import TextBlob
except ImportError:  # pragma: no cover
    TextBlob = None


ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF]")
ARABIC_DIACRITICS_PATTERN = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
ARABIC_TOKEN_PATTERN = re.compile(r"[\u0621-\u064A]+")
ENGLISH_TOKEN_PATTERN = re.compile(r"[A-Za-z]+")

ARABIC_NORMALIZATION_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ؤ": "و",
        "ئ": "ي",
        "ى": "ي",
        "ة": "ه",
        "ـ": "",
    }
)

ARABIC_NEGATIONS = {"لا", "ليس", "لم", "لن", "بدون", "دون", "غير"}
ARABIC_INTENSIFIERS = {"جدا", "قوي", "قويه", "قويا", "حاد", "حادا", "كبير", "كبيره", "قياسي", "ملحوظ"}
ARABIC_DIMINISHERS = {"طفيف", "طفيفه", "محدود", "محدوده", "نسبيا"}

ENGLISH_POSITIVE_WORDS = {
    "growth",
    "rise",
    "gain",
    "profit",
    "profits",
    "improve",
    "improves",
    "improved",
    "strong",
    "surge",
    "record",
    "partnership",
    "expansion",
    "dividend",
}
ENGLISH_NEGATIVE_WORDS = {
    "fall",
    "drop",
    "decline",
    "loss",
    "losses",
    "weak",
    "pressure",
    "pressures",
    "risk",
    "risks",
    "debt",
    "slowdown",
    "lawsuit",
}

ARABIC_LEXICON = {
    "نمو": 0.55,
    "ارتفاع": 0.55,
    "صعود": 0.6,
    "زياده": 0.45,
    "ارباح": 0.65,
    "ربحيه": 0.55,
    "مكاسب": 0.6,
    "تحسن": 0.5,
    "تعافي": 0.5,
    "ايرادات": 0.5,
    "مبيعات": 0.25,
    "توزيعات": 0.4,
    "توسع": 0.45,
    "استثمار": 0.4,
    "شراكه": 0.35,
    "صفقه": 0.3,
    "تمويل": 0.2,
    "استحواذ": 0.35,
    "تطوير": 0.2,
    "قياسي": 0.5,
    "قفزه": 0.65,
    "هبوط": -0.65,
    "انخفاض": -0.55,
    "تراجع": -0.55,
    "خسائر": -0.7,
    "خساره": -0.7,
    "ضغوط": -0.45,
    "ديون": -0.45,
    "مخاطر": -0.45,
    "تباطؤ": -0.4,
    "انكماش": -0.6,
    "تذبذب": -0.25,
    "خفض": -0.35,
    "تحذير": -0.45,
    "غرامه": -0.35,
    "نزاع": -0.3,
    "تراجعها": -0.45,
}

ARABIC_PHRASE_LEXICON = {
    "نمو الارباح": 0.85,
    "ارتفاع الارباح": 0.85,
    "قفزه في الارباح": 0.95,
    "زياده الايرادات": 0.75,
    "تحسن النتائج": 0.75,
    "نتائج قويه": 0.8,
    "شراكه استراتيجيه": 0.7,
    "صفقه تمويليه": 0.45,
    "توزيعات نقديه": 0.65,
    "تراجع الارباح": -0.9,
    "انخفاض الارباح": -0.9,
    "تكبد خسائر": -0.95,
    "ضغوط بيعيه": -0.7,
    "ارتفاع الديون": -0.8,
    "خفض التصنيف": -0.75,
    "تباطؤ النمو": -0.6,
}


def normalize_arabic(text: str) -> str:
    normalized = ARABIC_DIACRITICS_PATTERN.sub("", text or "")
    normalized = normalized.translate(ARABIC_NORMALIZATION_MAP)
    normalized = re.sub(r"[^\u0621-\u064A\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize_arabic(text: str) -> list[str]:
    normalized = normalize_arabic(text)
    return ARABIC_TOKEN_PATTERN.findall(normalized)


def detect_language(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "unknown"

    if ARABIC_CHAR_PATTERN.search(text):
        return "ar"

    if detect is None:
        return "en"

    try:
        return detect(text)
    except Exception:
        return "unknown"


def score_english_text(text: str) -> float:
    if TextBlob is not None:
        return float(TextBlob(text).sentiment.polarity)

    tokens = [token.lower() for token in ENGLISH_TOKEN_PATTERN.findall(text)]
    if not tokens:
        return 0.0

    score = 0.0
    for token in tokens:
        if token in ENGLISH_POSITIVE_WORDS:
            score += 0.35
        elif token in ENGLISH_NEGATIVE_WORDS:
            score -= 0.35

    return max(min(score / max(len(tokens) ** 0.5, 1.0), 1.0), -1.0)


class ArabicSentimentBackend:
    def score(self, text: str) -> float | None:  # pragma: no cover - interface only
        raise NotImplementedError


class CamelToolsArabicSentimentBackend(ArabicSentimentBackend):
    SCORE_MAP = {
        "positive": 0.85,
        "negative": -0.85,
        "neutral": 0.0,
    }

    def __init__(self) -> None:
        self._analyzer = self._load_analyzer()

    @staticmethod
    def _load_analyzer() -> object | None:
        try:
            from camel_tools.sentiment import SentimentAnalyzer
        except Exception:
            return None

        try:
            return SentimentAnalyzer.pretrained()
        except Exception:
            return None

    def score(self, text: str) -> float | None:
        if self._analyzer is None:
            return None

        try:
            labels = self._analyzer.predict([text])
        except Exception:
            return None

        if not labels:
            return None

        return self.SCORE_MAP.get(str(labels[0]).lower())


class LexiconArabicSentimentBackend(ArabicSentimentBackend):
    def score(self, text: str) -> float | None:
        tokens = tokenize_arabic(text)
        if not tokens:
            return 0.0

        normalized_text = " ".join(tokens)
        score = 0.0

        for phrase, weight in ARABIC_PHRASE_LEXICON.items():
            if phrase in normalized_text:
                score += weight

        for index, token in enumerate(tokens):
            base_weight = ARABIC_LEXICON.get(token)
            if base_weight is None:
                continue

            window = tokens[max(0, index - 2) : min(len(tokens), index + 3)]
            adjusted_weight = base_weight

            if any(negation in window[:-1] for negation in ARABIC_NEGATIONS):
                adjusted_weight *= -0.9
            if any(intensifier in window for intensifier in ARABIC_INTENSIFIERS):
                adjusted_weight *= 1.35
            if any(diminisher in window for diminisher in ARABIC_DIMINISHERS):
                adjusted_weight *= 0.7

            score += adjusted_weight

        normalized_score = score / max(len(tokens) ** 0.5, 1.0)
        return max(min(normalized_score, 1.0), -1.0)


@lru_cache(maxsize=1)
def get_arabic_sentiment_backend(preferred_backend: str = "auto") -> ArabicSentimentBackend:
    normalized_preference = preferred_backend.lower()
    if normalized_preference not in {"auto", "camel", "lexicon"}:
        raise ValueError("preferred_backend must be one of: auto, camel, lexicon")

    if normalized_preference in {"auto", "camel"}:
        backend = CamelToolsArabicSentimentBackend()
        if backend._analyzer is not None:
            return backend
        if normalized_preference == "camel":
            raise RuntimeError("camel-tools backend requested but unavailable.")

    return LexiconArabicSentimentBackend()


def score_arabic_text(text: str, preferred_backend: str = "auto") -> float:
    backend = get_arabic_sentiment_backend(preferred_backend)
    score = backend.score(text)
    if score is None and not isinstance(backend, LexiconArabicSentimentBackend):
        score = LexiconArabicSentimentBackend().score(text)
    return float(score or 0.0)


def label_sentiment(score: float, positive_threshold: float = 0.1, negative_threshold: float = -0.1) -> str:
    if math.isnan(score):
        return "unknown"
    if score >= positive_threshold:
        return "positive"
    if score <= negative_threshold:
        return "negative"
    return "neutral"


def score_headlines(frame: pd.DataFrame, arabic_backend: str = "auto") -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    scored = frame.copy()
    scored["language"] = scored["headline"].apply(detect_language)

    def _score_row(row: pd.Series) -> float:
        headline = row["headline"]
        if row["language"] == "ar":
            return score_arabic_text(headline, preferred_backend=arabic_backend)
        return score_english_text(headline)

    scored["sentiment_score"] = scored.apply(_score_row, axis=1)
    scored["sentiment_label"] = scored["sentiment_score"].apply(label_sentiment)
    return scored


def aggregate_daily_sentiment(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "headline_count",
                "sentiment_score",
                "positive_ratio",
                "negative_ratio",
            ]
        )

    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"]).dt.normalize()

    daily = (
        working.groupby(["date", "ticker"], as_index=False)
        .agg(
            headline_count=("headline", "count"),
            sentiment_score=("sentiment_score", "mean"),
            positive_ratio=("sentiment_label", lambda values: (values == "positive").mean()),
            negative_ratio=("sentiment_label", lambda values: (values == "negative").mean()),
        )
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    return daily
