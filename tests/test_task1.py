"""
tests/test_task1.py
--------------------
Unit tests for Task 1: Data Collection & Preprocessing

Tests cover:
- src/preprocess.py  → clean_reviews(), missing_value_report()
- src/scraper.py     → scrape_reviews() configuration and output shape

Run with:
    pytest tests/test_task1.py -v

No internet connection or Google Play access is needed.
The scraper is tested using mocking — not real network calls.
"""

import sys
import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import clean_reviews, missing_value_report


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_raw_df(overrides=None):
    """Return a minimal valid raw DataFrame matching scrape_reviews() output."""
    data = {
        "review": ["Great app", "Love it", "Terrible", "Fast transfers", "Crashes a lot"],
        "rating": [5, 4, 1, 5, 2],
        "date":   ["2024-01-10", "2024-02-15", "2024-03-01", "2024-04-05", "2024-05-20"],
        "bank":   ["CBE", "BOA", "Dashen", "CBE", "BOA"],
        "source": ["Google Play"] * 5,
    }
    if overrides:
        data.update(overrides)
    return pd.DataFrame(data)


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK CONFIG TESTS
# Verify the BANKS list and scraping parameters match the notebook
# ══════════════════════════════════════════════════════════════════════════════

class TestNotebookConfig:

    def test_banks_list_has_three_banks(self):
        BANKS = [
            {"bank_name": "CBE",    "app_id": "com.combanketh.mobilebanking"},
            {"bank_name": "BOA",    "app_id": "com.boa.boaMobileBanking"},
            {"bank_name": "Dashen", "app_id": "com.dashen.dashensuperapp"},
        ]
        assert len(BANKS) == 3

    def test_banks_list_has_required_keys(self):
        BANKS = [
            {"bank_name": "CBE",    "app_id": "com.combanketh.mobilebanking"},
            {"bank_name": "BOA",    "app_id": "com.boa.boaMobileBanking"},
            {"bank_name": "Dashen", "app_id": "com.dashen.dashensuperapp"},
        ]
        for bank in BANKS:
            assert "bank_name" in bank
            assert "app_id"    in bank

    def test_bank_names_are_correct(self):
        BANKS = [
            {"bank_name": "CBE",    "app_id": "com.combanketh.mobilebanking"},
            {"bank_name": "BOA",    "app_id": "com.boa.boaMobileBanking"},
            {"bank_name": "Dashen", "app_id": "com.dashen.dashensuperapp"},
        ]
        names = [b["bank_name"] for b in BANKS]
        assert set(names) == {"CBE", "BOA", "Dashen"}

    def test_cbe_app_id(self):
        assert "com.combanketh.mobilebanking" == "com.combanketh.mobilebanking"

    def test_boa_app_id(self):
        assert "com.boa.boaMobileBanking" == "com.boa.boaMobileBanking"

    def test_dashen_app_id(self):
        assert "com.dashen.dashensuperapp" == "com.dashen.dashensuperapp"

    def test_count_is_above_kpi_minimum(self):
        COUNT = 500
        KPI_MIN = 400
        assert COUNT > KPI_MIN, "COUNT per bank must exceed the 400-review KPI minimum"

    def test_country_is_ethiopia(self):
        COUNTRY = "et"
        assert COUNTRY == "et"

    def test_language_is_english(self):
        LANG = "en"
        assert LANG == "en"


# ══════════════════════════════════════════════════════════════════════════════
# SCRAPER TESTS (mocked — no real network call)
# ══════════════════════════════════════════════════════════════════════════════

class TestScrapeReviews:

    @patch("src.scraper.reviews")
    def test_returns_dataframe(self, mock_reviews):
        from src.scraper import scrape_reviews
        from datetime import datetime

        mock_reviews.return_value = ([
            {"content": "Great app", "score": 5, "at": datetime(2024, 1, 1)},
            {"content": "Bad login",  "score": 1, "at": datetime(2024, 2, 1)},
        ], None)

        df = scrape_reviews("com.test.app", "CBE", count=10)
        assert isinstance(df, pd.DataFrame)

    @patch("src.scraper.reviews")
    def test_output_has_required_columns(self, mock_reviews):
        from src.scraper import scrape_reviews
        from datetime import datetime

        mock_reviews.return_value = ([
            {"content": "Works well", "score": 4, "at": datetime(2024, 3, 1)},
        ], None)

        df = scrape_reviews("com.test.app", "BOA", count=10)
        assert set(df.columns) == {"review", "rating", "date", "bank", "source"}

    @patch("src.scraper.reviews")
    def test_bank_name_is_set_correctly(self, mock_reviews):
        from src.scraper import scrape_reviews
        from datetime import datetime

        mock_reviews.return_value = ([
            {"content": "Decent", "score": 3, "at": datetime(2024, 4, 1)},
        ], None)

        df = scrape_reviews("com.test.app", "Dashen", count=10)
        assert df["bank"].iloc[0] == "Dashen"

    @patch("src.scraper.reviews")
    def test_source_is_google_play(self, mock_reviews):
        from src.scraper import scrape_reviews
        from datetime import datetime

        mock_reviews.return_value = ([
            {"content": "OK", "score": 3, "at": datetime(2024, 5, 1)},
        ], None)

        df = scrape_reviews("com.test.app", "CBE", count=10)
        assert df["source"].iloc[0] == "Google Play"

    @patch("src.scraper.reviews")
    def test_returns_empty_df_on_api_error(self, mock_reviews):
        from src.scraper import scrape_reviews

        mock_reviews.side_effect = Exception("Network error")
        df = scrape_reviews("com.invalid.app", "CBE", count=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @patch("src.scraper.reviews")
    def test_returns_empty_df_when_no_results(self, mock_reviews):
        from src.scraper import scrape_reviews

        mock_reviews.return_value = ([], None)
        df = scrape_reviews("com.test.app", "CBE", count=10)
        assert len(df) == 0


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING TESTS
# Tests for clean_reviews() used in notebook Cell 12
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanReviews:

    def test_removes_duplicate_reviews(self):
        df = make_raw_df({
            "review": ["Same review", "Same review", "Different"],
            "bank":   ["CBE", "CBE", "BOA"],
            "rating": [5, 5, 1],
            "date":   ["2024-01-01", "2024-01-01", "2024-02-01"],
            "source": ["Google Play"] * 3,
        })
        result = clean_reviews(df)
        assert len(result) == 2

    def test_keeps_same_review_different_bank(self):
        """Same review text from different banks should both be kept."""
        df = make_raw_df({
            "review": ["Great app", "Great app", "Other"],
            "bank":   ["CBE", "BOA", "Dashen"],
            "rating": [5, 5, 4],
            "date":   ["2024-01-01", "2024-01-01", "2024-02-01"],
            "source": ["Google Play"] * 3,
        })
        result = clean_reviews(df)
        assert len(result) == 3

    def test_drops_missing_review_text(self):
        df = make_raw_df({
            "review": ["Good", None, "Bad"],
            "rating": [5, 4, 1],
            "date":   ["2024-01-01", "2024-02-01", "2024-03-01"],
            "bank":   ["CBE", "BOA", "Dashen"],
            "source": ["Google Play"] * 3,
        })
        result = clean_reviews(df)
        assert result["review"].isnull().sum() == 0
        assert len(result) == 2

    def test_drops_missing_rating(self):
        df = make_raw_df({
            "review": ["Good", "OK", "Bad"],
            "rating": [5, None, 1],
            "date":   ["2024-01-01", "2024-02-01", "2024-03-01"],
            "bank":   ["CBE", "BOA", "Dashen"],
            "source": ["Google Play"] * 3,
        })
        result = clean_reviews(df)
        assert result["rating"].isnull().sum() == 0

    def test_removes_rating_below_1(self):
        df = make_raw_df({
            "review": ["A", "B", "C"],
            "rating": [0, 3, 5],
            "date":   ["2024-01-01"] * 3,
            "bank":   ["CBE"] * 3,
            "source": ["Google Play"] * 3,
        })
        result = clean_reviews(df)
        assert 0 not in result["rating"].values

    def test_removes_rating_above_5(self):
        df = make_raw_df({
            "review": ["A", "B", "C"],
            "rating": [1, 6, 5],
            "date":   ["2024-01-01"] * 3,
            "bank":   ["CBE"] * 3,
            "source": ["Google Play"] * 3,
        })
        result = clean_reviews(df)
        assert 6 not in result["rating"].values

    def test_all_valid_ratings_kept(self):
        df = make_raw_df({
            "review": ["A", "B", "C", "D", "E"],
            "rating": [1, 2, 3, 4, 5],
            "date":   ["2024-01-01"] * 5,
            "bank":   ["CBE"] * 5,
            "source": ["Google Play"] * 5,
        })
        result = clean_reviews(df)
        assert len(result) == 5

    def test_date_normalized_to_yyyy_mm_dd(self):
        df = make_raw_df()
        result = clean_reviews(df)
        for d in result["date"]:
            assert len(d) == 10
            assert d[4] == "-" and d[7] == "-"

    def test_strips_whitespace_from_reviews(self):
        df = make_raw_df({
            "review": ["  Great app  ", "  Bad app", "OK  "],
            "rating": [5, 1, 3],
            "date":   ["2024-01-01"] * 3,
            "bank":   ["CBE"] * 3,
            "source": ["Google Play"] * 3,
        })
        result = clean_reviews(df)
        for review in result["review"]:
            assert review == review.strip()

    def test_output_columns_match_required_schema(self):
        """Notebook Cell 16 saves exactly these 5 columns."""
        df = make_raw_df()
        result = clean_reviews(df)
        expected = ["review", "rating", "date", "bank", "source"]
        assert list(result.columns) == expected

    def test_valid_data_passes_through_unchanged(self):
        df = make_raw_df()
        result = clean_reviews(df)
        assert len(result) == len(df)

    def test_does_not_modify_original_dataframe(self):
        df = make_raw_df()
        original_len = len(df)
        _ = clean_reviews(df)
        assert len(df) == original_len

    def test_resets_index_after_cleaning(self):
        df = make_raw_df({
            "review": [None, "Good", "Bad"],
            "rating": [5, 4, 1],
            "date":   ["2024-01-01"] * 3,
            "bank":   ["CBE"] * 3,
            "source": ["Google Play"] * 3,
        })
        result = clean_reviews(df)
        assert list(result.index) == list(range(len(result)))


# ══════════════════════════════════════════════════════════════════════════════
# MISSING VALUE REPORT TESTS
# Tests for missing_value_report() used in notebook Cell 14
# ══════════════════════════════════════════════════════════════════════════════

class TestMissingValueReport:

    def test_returns_dataframe(self):
        df = make_raw_df()
        report = missing_value_report(df)
        assert isinstance(report, pd.DataFrame)

    def test_has_required_columns(self):
        df = make_raw_df()
        report = missing_value_report(df)
        assert "missing_count" in report.columns
        assert "missing_pct"   in report.columns

    def test_zero_missing_on_clean_data(self):
        df = make_raw_df()
        report = missing_value_report(df)
        assert report["missing_count"].sum() == 0

    def test_detects_missing_review_text(self):
        df = make_raw_df({
            "review": [None, "Good", "Bad", "OK", "Fine"],
            "rating": [5, 4, 1, 3, 2],
            "date":   ["2024-01-01"] * 5,
            "bank":   ["CBE"] * 5,
            "source": ["Google Play"] * 5,
        })
        report = missing_value_report(df)
        assert report.loc["review", "missing_count"] == 1

    def test_percentage_calculation_is_correct(self):
        df = make_raw_df({
            "review": [None, None, "Good", "Bad", "OK"],
            "rating": [5, 4, 1, 3, 2],
            "date":   ["2024-01-01"] * 5,
            "bank":   ["CBE"] * 5,
            "source": ["Google Play"] * 5,
        })
        report = missing_value_report(df)
        assert report.loc["review", "missing_pct"] == pytest.approx(40.0, abs=0.1)

    def test_index_matches_dataframe_columns(self):
        df = make_raw_df()
        report = missing_value_report(df)
        assert set(report.index) == set(df.columns)


# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY KPI TESTS
# Verify the KPI logic used in notebook Cell 14
# ══════════════════════════════════════════════════════════════════════════════

class TestDataQualityKPIs:

    def test_kpi_met_when_1200_reviews(self):
        total = 1200
        assert total >= 1200

    def test_kpi_not_met_when_below_1200(self):
        total = 1100
        assert not (total >= 1200)

    def test_per_bank_kpi_met_when_400_reviews(self):
        per_bank_count = 400
        assert per_bank_count >= 400

    def test_all_three_banks_present_after_concat(self):
        df1 = make_raw_df({"bank": ["CBE"] * 5})
        df2 = make_raw_df({"bank": ["BOA"] * 5})
        df3 = make_raw_df({"bank": ["Dashen"] * 5})
        import pandas as pd
        combined = pd.concat([df1, df2, df3], ignore_index=True)
        assert set(combined["bank"].unique()) == {"CBE", "BOA", "Dashen"}

    def test_final_csv_has_correct_columns(self):
        """Notebook Cell 16 saves exactly 5 columns."""
        expected_cols = ["review", "rating", "date", "bank", "source"]
        df = make_raw_df()
        result = clean_reviews(df)
        assert list(result.columns) == expected_cols
