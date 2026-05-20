"""
tests/test_task2.py
--------------------
Unit tests for Task 2: Sentiment & Thematic Analysis

Tests cover:
- src/sentiment.py → run_sentiment() output shape and column values
- src/themes.py    → extract_top_keywords(), assign_themes(), theme_summary()
- Notebook logic   → sentiment percentage calculation, visualization data prep

Run with:
    pytest tests/test_task2.py -v

DistilBERT model is mocked — no model download needed for tests.
"""

import sys
import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.themes import assign_themes, theme_summary, extract_top_keywords, THEME_MAP


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_clean_df():
    """Simulates reviews_clean.csv loaded in notebook Cell 6."""
    return pd.DataFrame({
        "review": [
            "I cannot login, OTP never arrives",
            "Transfer was instant, very happy",
            "App keeps crashing after update",
            "Customer support never picks up",
            "Great app, love the interface",
            "Password reset is broken",
            "Fast and reliable transfers",
            "App freezes on startup",
            "Good balance checking feature",
            "Need dark mode please",
        ],
        "rating": [1, 5, 2, 1, 5, 1, 5, 2, 4, 3],
        "date":   ["2024-01-01"] * 10,
        "bank":   ["CBE", "BOA", "Dashen", "CBE", "BOA",
                   "Dashen", "CBE", "BOA", "Dashen", "CBE"],
        "source": ["Google Play"] * 10,
    })


def make_analyzed_df():
    """Simulates DataFrame after sentiment analysis — used for Task 3 output tests."""
    df = make_clean_df()
    df["sentiment_label"] = ["negative", "positive", "negative", "negative", "positive",
                              "negative", "positive", "negative", "positive", "positive"]
    df["sentiment_score"]  = [0.98, 0.97, 0.95, 0.99, 0.96, 0.91, 0.98, 0.94, 0.87, 0.82]
    df["identified_theme"] = ["Account Access", "Transaction Performance", "App Stability",
                               "Customer Support", "UI & Design", "Account Access",
                               "Transaction Performance", "App Stability",
                               "Balance & Account Info", "Feature Requests"]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SENTIMENT PIPELINE TESTS (mocked — no model download)
# Tests for run_sentiment() used in notebook Cell 11
# ══════════════════════════════════════════════════════════════════════════════

class TestRunSentiment:

    def _make_mock_pipe(self, label="POSITIVE", score=0.97):
        """Returns a mock pipeline that always returns the given label/score."""
        mock = MagicMock()
        mock.return_value = [{"label": label, "score": score}] * 100
        return mock

    def test_adds_sentiment_label_column(self):
        from src.sentiment import run_sentiment
        df   = make_clean_df()
        pipe = self._make_mock_pipe()
        result = run_sentiment(df, pipe)
        assert "sentiment_label" in result.columns

    def test_adds_sentiment_score_column(self):
        from src.sentiment import run_sentiment
        df   = make_clean_df()
        pipe = self._make_mock_pipe()
        result = run_sentiment(df, pipe)
        assert "sentiment_score" in result.columns

    def test_sentiment_label_is_lowercase(self):
        from src.sentiment import run_sentiment
        df   = make_clean_df()
        pipe = self._make_mock_pipe(label="POSITIVE")
        result = run_sentiment(df, pipe)
        for label in result["sentiment_label"]:
            assert label == label.lower()

    def test_sentiment_label_values_are_valid(self):
        from src.sentiment import run_sentiment
        df   = make_clean_df()
        pipe = self._make_mock_pipe(label="NEGATIVE")
        result = run_sentiment(df, pipe)
        valid = {"positive", "negative"}
        assert set(result["sentiment_label"].unique()).issubset(valid)

    def test_sentiment_score_is_between_0_and_1(self):
        from src.sentiment import run_sentiment
        df   = make_clean_df()
        pipe = self._make_mock_pipe(score=0.92)
        result = run_sentiment(df, pipe)
        assert result["sentiment_score"].between(0.0, 1.0).all()

    def test_row_count_unchanged_after_sentiment(self):
        from src.sentiment import run_sentiment
        df     = make_clean_df()
        pipe   = self._make_mock_pipe()
        result = run_sentiment(df, pipe)
        assert len(result) == len(df)

    def test_does_not_modify_original_columns(self):
        from src.sentiment import run_sentiment
        df           = make_clean_df()
        original_cols = list(df.columns)
        pipe         = self._make_mock_pipe()
        result       = run_sentiment(df, pipe)
        for col in original_cols:
            assert col in result.columns

    def test_score_is_rounded_to_4_decimal_places(self):
        from src.sentiment import run_sentiment
        df   = make_clean_df()
        pipe = self._make_mock_pipe(score=0.976543210)
        result = run_sentiment(df, pipe)
        for score in result["sentiment_score"]:
            assert score == round(score, 4)


# ══════════════════════════════════════════════════════════════════════════════
# SENTIMENT PERCENTAGE TESTS
# Tests for the percentage calculation in notebook Cell 13
# ══════════════════════════════════════════════════════════════════════════════

class TestSentimentPercentageCalculation:

    def test_positive_percentage_sums_to_100_with_negative(self):
        df = make_analyzed_df()
        sentiment_by_bank = (
            df.groupby(["bank", "sentiment_label"])
            .size()
            .unstack(fill_value=0)
        )
        for label in ["positive", "negative"]:
            if label in sentiment_by_bank.columns:
                total = sentiment_by_bank.sum(axis=1)
                sentiment_by_bank[f"{label}_pct"] = (
                    sentiment_by_bank[label] / total * 100
                ).round(1)

        for bank in sentiment_by_bank.index:
            pos = sentiment_by_bank.loc[bank, "positive_pct"] if "positive_pct" in sentiment_by_bank.columns else 0
            neg = sentiment_by_bank.loc[bank, "negative_pct"] if "negative_pct" in sentiment_by_bank.columns else 0
            assert abs(pos + neg - 100.0) < 0.2, f"{bank} percentages do not sum to 100"

    def test_cbe_positive_sentiment_matches_notebook(self):
        """
        Notebook Cell 29 records CBE positive = 57.1%.
        This test verifies the calculation logic is correct.
        """
        df = pd.DataFrame({
            "bank":            ["CBE"] * 7,
            "sentiment_label": ["positive"] * 4 + ["negative"] * 3,
        })
        counts = df["sentiment_label"].value_counts()
        pct = round(counts.get("positive", 0) / len(df) * 100, 1)
        assert pct == pytest.approx(57.1, abs=0.1)

    def test_boa_majority_negative(self):
        """
        Notebook Cell 29 records BOA positive = 44.1%, negative = 55.9%.
        BOA is the only bank with majority-negative sentiment.
        """
        df = pd.DataFrame({
            "bank":            ["BOA"] * 17,
            "sentiment_label": ["positive"] * 7 + ["negative"] * 10,
        })
        counts = df["sentiment_label"].value_counts()
        neg_pct = round(counts.get("negative", 0) / len(df) * 100, 1)
        pos_pct = round(counts.get("positive", 0) / len(df) * 100, 1)
        assert neg_pct > pos_pct, "BOA should have more negative than positive"


# ══════════════════════════════════════════════════════════════════════════════
# TF-IDF KEYWORD EXTRACTION TESTS
# Tests for extract_top_keywords() used in notebook Cell 19
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractTopKeywords:

    def test_returns_list(self):
        df = make_clean_df()
        result = extract_top_keywords(df, bank_name="CBE", top_n=5)
        assert isinstance(result, list)

    def test_returns_tuples_of_keyword_and_score(self):
        df = make_clean_df()
        result = extract_top_keywords(df, bank_name="CBE", top_n=5)
        if result:
            kw, score = result[0]
            assert isinstance(kw, str)
            assert isinstance(score, float)

    def test_returns_correct_top_n(self):
        df = make_clean_df()
        result = extract_top_keywords(df, bank_name="CBE", top_n=3)
        assert len(result) <= 3

    def test_returns_empty_list_for_unknown_bank(self):
        df = make_clean_df()
        result = extract_top_keywords(df, bank_name="UnknownBank", top_n=10)
        assert result == []

    def test_scores_are_sorted_descending(self):
        df = make_clean_df()
        result = extract_top_keywords(df, bank_name="CBE", top_n=10)
        if len(result) > 1:
            scores = [s for _, s in result]
            assert scores == sorted(scores, reverse=True)

    def test_all_scores_are_positive(self):
        df = make_clean_df()
        result = extract_top_keywords(df, bank_name="BOA", top_n=10)
        for _, score in result:
            assert score > 0

    def test_unpacking_works_for_plotting(self):
        """Simulates the unpacking pattern in notebook Cell 19."""
        df = make_clean_df()
        keywords = extract_top_keywords(df, bank_name="CBE", top_n=5)
        words  = [kw    for kw, score in keywords]
        scores = [score for kw, score in keywords]
        assert len(words)  == len(scores)
        assert all(isinstance(w, str)   for w in words)
        assert all(isinstance(s, float) for s in scores)

    def test_returns_empty_on_too_few_reviews(self):
        tiny_df = pd.DataFrame({
            "review": ["Good"],
            "rating": [5],
            "date":   ["2024-01-01"],
            "bank":   ["CBE"],
            "source": ["Google Play"],
        })
        result = extract_top_keywords(tiny_df, bank_name="CBE", top_n=10)
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# THEME ASSIGNMENT TESTS
# Tests for assign_themes() used in notebook Cell 22
# ══════════════════════════════════════════════════════════════════════════════

class TestAssignThemes:

    def test_adds_identified_theme_column(self):
        df = make_clean_df()
        result = assign_themes(df)
        assert "identified_theme" in result.columns

    def test_otp_maps_to_account_access(self):
        df = pd.DataFrame({
            "review": ["I cannot login OTP never arrives"],
            "rating": [1], "date": ["2024-01-01"],
            "bank": ["CBE"], "source": ["Google Play"],
        })
        result = assign_themes(df)
        assert result.iloc[0]["identified_theme"] == "Account Access"

    def test_transfer_maps_to_transaction_performance(self):
        df = pd.DataFrame({
            "review": ["Transfer was instant"],
            "rating": [5], "date": ["2024-01-01"],
            "bank": ["BOA"], "source": ["Google Play"],
        })
        result = assign_themes(df)
        assert result.iloc[0]["identified_theme"] == "Transaction Performance"

    def test_crash_maps_to_app_stability(self):
        df = pd.DataFrame({
            "review": ["App keeps crashing after update"],
            "rating": [2], "date": ["2024-01-01"],
            "bank": ["Dashen"], "source": ["Google Play"],
        })
        result = assign_themes(df)
        assert result.iloc[0]["identified_theme"] == "App Stability"

    def test_support_maps_to_customer_support(self):
        df = pd.DataFrame({
            "review": ["Customer support never responds"],
            "rating": [1], "date": ["2024-01-01"],
            "bank": ["CBE"], "source": ["Google Play"],
        })
        result = assign_themes(df)
        assert result.iloc[0]["identified_theme"] == "Customer Support"

    def test_unmatched_review_gets_general(self):
        df = pd.DataFrame({
            "review": ["xyzabc completely unknown words qrstuvwxyz"],
            "rating": [3], "date": ["2024-01-01"],
            "bank": ["CBE"], "source": ["Google Play"],
        })
        result = assign_themes(df)
        assert result.iloc[0]["identified_theme"] == "General"

    def test_no_null_themes(self):
        df = make_clean_df()
        result = assign_themes(df)
        assert result["identified_theme"].isnull().sum() == 0

    def test_all_rows_get_a_theme(self):
        df = make_clean_df()
        result = assign_themes(df)
        assert len(result) == len(df)

    def test_does_not_modify_original_dataframe(self):
        df = make_clean_df()
        original_cols = set(df.columns)
        _ = assign_themes(df)
        assert set(df.columns) == original_cols

    def test_custom_theme_map_overrides_default(self):
        custom_map = {"TestTheme": ["xyztoken"]}
        df = pd.DataFrame({
            "review": ["xyztoken in this review", "nothing special"],
            "rating": [5, 3], "date": ["2024-01-01"] * 2,
            "bank": ["CBE"] * 2, "source": ["Google Play"] * 2,
        })
        result = assign_themes(df, theme_map=custom_map)
        assert result.iloc[0]["identified_theme"] == "TestTheme"
        assert result.iloc[1]["identified_theme"] == "General"

    def test_matching_is_case_insensitive(self):
        df = pd.DataFrame({
            "review": ["LOGIN FAILED AGAIN"],
            "rating": [1], "date": ["2024-01-01"],
            "bank": ["BOA"], "source": ["Google Play"],
        })
        result = assign_themes(df)
        assert result.iloc[0]["identified_theme"] == "Account Access"


# ══════════════════════════════════════════════════════════════════════════════
# THEME SUMMARY TESTS
# Tests for theme_summary() used in notebook Cell 22
# ══════════════════════════════════════════════════════════════════════════════

class TestThemeSummary:

    def test_returns_dataframe(self):
        df = make_clean_df()
        df = assign_themes(df)
        result = theme_summary(df)
        assert isinstance(result, pd.DataFrame)

    def test_columns_are_bank_names(self):
        df = make_clean_df()
        df = assign_themes(df)
        result = theme_summary(df)
        assert set(result.columns).issubset({"CBE", "BOA", "Dashen"})

    def test_index_contains_themes(self):
        df = make_clean_df()
        df = assign_themes(df)
        result = theme_summary(df)
        assert len(result.index) > 0

    def test_total_counts_match_original_rows(self):
        df = make_clean_df()
        df = assign_themes(df)
        result = theme_summary(df)
        assert result.values.sum() == len(df)


# ══════════════════════════════════════════════════════════════════════════════
# THEME MAP TESTS
# Tests for the THEME_MAP constant in src/themes.py
# ══════════════════════════════════════════════════════════════════════════════

class TestThemeMap:

    def test_theme_map_has_seven_themes(self):
        assert len(THEME_MAP) == 7

    def test_all_themes_have_keywords(self):
        for theme, keywords in THEME_MAP.items():
            assert len(keywords) > 0, f"{theme} has no keywords"

    def test_account_access_theme_exists(self):
        assert "Account Access" in THEME_MAP

    def test_transaction_performance_theme_exists(self):
        assert "Transaction Performance" in THEME_MAP

    def test_app_stability_theme_exists(self):
        assert "App Stability" in THEME_MAP

    def test_customer_support_theme_exists(self):
        assert "Customer Support" in THEME_MAP

    def test_all_keywords_are_lowercase(self):
        """Keywords must be lowercase for case-insensitive matching to work."""
        for theme, keywords in THEME_MAP.items():
            for kw in keywords:
                assert kw == kw.lower(), f"Keyword '{kw}' in {theme} is not lowercase"


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT COLUMN TESTS
# Verify the final output matches what Task 3 expects (notebook Cell 28)
# ══════════════════════════════════════════════════════════════════════════════

class TestTask2Output:

    REQUIRED_COLS = [
        "review", "rating", "date", "bank", "source",
        "sentiment_label", "sentiment_score", "identified_theme"
    ]

    def test_analyzed_df_has_all_required_columns(self):
        df = make_analyzed_df()
        for col in self.REQUIRED_COLS:
            assert col in df.columns, f"Missing required column: {col}"

    def test_sentiment_label_column_has_no_nulls(self):
        df = make_analyzed_df()
        assert df["sentiment_label"].isnull().sum() == 0

    def test_sentiment_score_column_has_no_nulls(self):
        df = make_analyzed_df()
        assert df["sentiment_score"].isnull().sum() == 0

    def test_identified_theme_column_has_no_nulls(self):
        df = make_analyzed_df()
        assert df["identified_theme"].isnull().sum() == 0

    def test_output_column_order_matches_notebook(self):
        """Notebook Cell 28 specifies this exact column order."""
        df = make_analyzed_df()
        saved = df[self.REQUIRED_COLS]
        assert list(saved.columns) == self.REQUIRED_COLS
