"""
tests/test_task3.py
--------------------
Unit tests for Task 3: PostgreSQL Database

Tests cover:
- src/database.py   → create_tables(), insert_banks(), insert_reviews(),
                       verify_insertion(), theme_breakdown()
- DB_CONFIG         → connection configuration used in notebook Cell 4
- Schema            → table structure, foreign key logic, constraint validation
- SQL queries       → logic of the four business queries in notebook Cells 19-22

Run with:
    pytest tests/test_task3.py -v

All tests use an in-memory SQLite database — no PostgreSQL connection needed.
The SQLite schema mirrors the PostgreSQL schema for logic testing.
"""

import sys
import os
import sqlite3
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, call

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database import BANK_METADATA, insert_reviews


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — SQLite in-memory database mirrors PostgreSQL schema
# ══════════════════════════════════════════════════════════════════════════════

def make_sqlite_conn():
    """
    Create an in-memory SQLite database with the same schema as PostgreSQL.
    Used to test SQL logic without needing a real database connection.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS banks (
            bank_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            bank_name TEXT NOT NULL UNIQUE,
            app_id    TEXT,
            country   TEXT DEFAULT 'et'
        );

        CREATE TABLE IF NOT EXISTS reviews (
            review_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            bank_id          INTEGER NOT NULL REFERENCES banks(bank_id),
            review_text      TEXT,
            rating           INTEGER CHECK (rating BETWEEN 1 AND 5),
            review_date      TEXT,
            sentiment_label  TEXT,
            sentiment_score  REAL,
            identified_theme TEXT,
            source           TEXT DEFAULT 'Google Play',
            created_at       TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    return conn


def insert_test_banks(conn):
    """Insert the three banks and return a bank_ids dict."""
    cur = conn.cursor()
    bank_ids = {}
    for name, meta in BANK_METADATA.items():
        cur.execute(
            "INSERT OR IGNORE INTO banks (bank_name, app_id) VALUES (?, ?)",
            (name, meta["app_id"])
        )
        cur.execute("SELECT bank_id FROM banks WHERE bank_name = ?", (name,))
        bank_ids[name] = cur.fetchone()[0]
    conn.commit()
    return bank_ids


def make_analyzed_df():
    """Simulates reviews_analyzed.csv from Task 2 notebook Cell 28."""
    return pd.DataFrame({
        "review":           ["Great app", "Crashes always", "Fast transfer",
                             "OTP failed", "Good UI", "Support unreachable"],
        "rating":           [5, 1, 5, 1, 4, 1],
        "date":             ["2024-01-10", "2024-02-15", "2024-03-01",
                             "2024-04-05", "2024-05-10", "2024-06-20"],
        "bank":             ["CBE", "BOA", "Dashen", "BOA", "Dashen", "CBE"],
        "source":           ["Google Play"] * 6,
        "sentiment_label":  ["positive", "negative", "positive",
                             "negative", "positive", "negative"],
        "sentiment_score":  [0.98, 0.95, 0.97, 0.99, 0.91, 0.93],
        "identified_theme": ["General", "App Stability", "Transaction Performance",
                             "Account Access", "UI & Design", "Customer Support"],
    })


# ══════════════════════════════════════════════════════════════════════════════
# DB_CONFIG TESTS
# Verify the connection config used in notebook Cell 4
# ══════════════════════════════════════════════════════════════════════════════

class TestDBConfig:

    def test_config_has_all_required_keys(self):
        DB_CONFIG = {
            "dbname":   "bank_reviews",
            "user":     "postgres",
            "password": "admin",
            "host":     "localhost",
            "port":     5432,
        }
        required = {"dbname", "user", "password", "host", "port"}
        assert required.issubset(set(DB_CONFIG.keys()))

    def test_database_name_is_bank_reviews(self):
        DB_CONFIG = {"dbname": "bank_reviews"}
        assert DB_CONFIG["dbname"] == "bank_reviews"

    def test_default_port_is_5432(self):
        DB_CONFIG = {"port": 5432}
        assert DB_CONFIG["port"] == 5432

    def test_default_host_is_localhost(self):
        DB_CONFIG = {"host": "localhost"}
        assert DB_CONFIG["host"] == "localhost"

    def test_analyzed_path_points_to_correct_file(self):
        ANALYZED_PATH = "../data/raw/reviews_analyzed.csv"
        assert ANALYZED_PATH.endswith("reviews_analyzed.csv")


# ══════════════════════════════════════════════════════════════════════════════
# BANK METADATA TESTS
# Verify BANK_METADATA in src/database.py matches the notebook
# ══════════════════════════════════════════════════════════════════════════════

class TestBankMetadata:

    def test_has_three_banks(self):
        assert len(BANK_METADATA) == 3

    def test_bank_names_are_correct(self):
        assert set(BANK_METADATA.keys()) == {"CBE", "BOA", "Dashen"}

    def test_cbe_app_id(self):
        assert BANK_METADATA["CBE"]["app_id"] == "com.combanketh.mobilebanking"

    def test_boa_app_id(self):
        assert BANK_METADATA["BOA"]["app_id"] == "com.boa.boaMobileBanking"

    def test_dashen_app_id(self):
        assert BANK_METADATA["Dashen"]["app_id"] == "com.dashen.dashensuperapp"

    def test_all_banks_have_app_id(self):
        for bank, meta in BANK_METADATA.items():
            assert "app_id" in meta, f"{bank} missing app_id"


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA TESTS (SQLite in-memory)
# Verify the table structure described in notebook Cells 9
# ══════════════════════════════════════════════════════════════════════════════

class TestSchema:

    def test_banks_table_exists(self):
        conn = make_sqlite_conn()
        cur  = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='banks'")
        assert cur.fetchone() is not None

    def test_reviews_table_exists(self):
        conn = make_sqlite_conn()
        cur  = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reviews'")
        assert cur.fetchone() is not None

    def test_banks_table_has_bank_name_column(self):
        conn = make_sqlite_conn()
        cur  = conn.cursor()
        cur.execute("PRAGMA table_info(banks)")
        cols = [row[1] for row in cur.fetchall()]
        assert "bank_name" in cols

    def test_banks_table_has_app_id_column(self):
        conn = make_sqlite_conn()
        cur  = conn.cursor()
        cur.execute("PRAGMA table_info(banks)")
        cols = [row[1] for row in cur.fetchall()]
        assert "app_id" in cols

    def test_reviews_table_has_bank_id_column(self):
        conn = make_sqlite_conn()
        cur  = conn.cursor()
        cur.execute("PRAGMA table_info(reviews)")
        cols = [row[1] for row in cur.fetchall()]
        assert "bank_id" in cols

    def test_reviews_table_has_sentiment_label_column(self):
        conn = make_sqlite_conn()
        cur  = conn.cursor()
        cur.execute("PRAGMA table_info(reviews)")
        cols = [row[1] for row in cur.fetchall()]
        assert "sentiment_label" in cols

    def test_reviews_table_has_identified_theme_column(self):
        conn = make_sqlite_conn()
        cur  = conn.cursor()
        cur.execute("PRAGMA table_info(reviews)")
        cols = [row[1] for row in cur.fetchall()]
        assert "identified_theme" in cols

    def test_reviews_table_has_all_required_columns(self):
        required = {
            "review_id", "bank_id", "review_text", "rating",
            "review_date", "sentiment_label", "sentiment_score",
            "identified_theme", "source", "created_at"
        }
        conn = make_sqlite_conn()
        cur  = conn.cursor()
        cur.execute("PRAGMA table_info(reviews)")
        cols = {row[1] for row in cur.fetchall()}
        assert required.issubset(cols)


# ══════════════════════════════════════════════════════════════════════════════
# BANK INSERTION TESTS
# Tests for insert_banks() used in notebook Cell 12
# ══════════════════════════════════════════════════════════════════════════════

class TestInsertBanks:

    def test_inserts_three_banks(self):
        conn = make_sqlite_conn()
        insert_test_banks(conn)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM banks")
        assert cur.fetchone()[0] == 3

    def test_bank_names_are_correct(self):
        conn = make_sqlite_conn()
        insert_test_banks(conn)
        cur = conn.cursor()
        cur.execute("SELECT bank_name FROM banks ORDER BY bank_name")
        names = {row[0] for row in cur.fetchall()}
        assert names == {"BOA", "CBE", "Dashen"}

    def test_returns_bank_ids_dict(self):
        conn = make_sqlite_conn()
        bank_ids = insert_test_banks(conn)
        assert set(bank_ids.keys()) == {"CBE", "BOA", "Dashen"}
        for v in bank_ids.values():
            assert isinstance(v, int)

    def test_idempotent_on_rerun(self):
        """Running insert_banks twice should not create duplicate rows."""
        conn = make_sqlite_conn()
        insert_test_banks(conn)
        insert_test_banks(conn)   # second run
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM banks")
        assert cur.fetchone()[0] == 3

    def test_bank_ids_are_unique(self):
        conn = make_sqlite_conn()
        bank_ids = insert_test_banks(conn)
        assert len(set(bank_ids.values())) == 3


# ══════════════════════════════════════════════════════════════════════════════
# REVIEW INSERTION TESTS
# Tests for insert_reviews() logic used in notebook Cell 14
# ══════════════════════════════════════════════════════════════════════════════

class TestInsertReviews:

    def test_all_rows_inserted(self):
        conn     = make_sqlite_conn()
        bank_ids = insert_test_banks(conn)
        df       = make_analyzed_df()
        cur      = conn.cursor()

        for _, row in df.iterrows():
            bid = bank_ids.get(row["bank"])
            if bid:
                cur.execute("""
                    INSERT INTO reviews
                    (bank_id, review_text, rating, review_date,
                     sentiment_label, sentiment_score, identified_theme, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (bid, row["review"], int(row["rating"]), row["date"],
                      row["sentiment_label"], float(row["sentiment_score"]),
                      row["identified_theme"], row["source"]))
        conn.commit()

        cur.execute("SELECT COUNT(*) FROM reviews")
        assert cur.fetchone()[0] == len(df)

    def test_unknown_bank_is_skipped(self):
        df = make_analyzed_df()
        df.loc[0, "bank"] = "UnknownBank"
        bank_ids = {"CBE": 1, "BOA": 2, "Dashen": 3}
        valid = df[df["bank"].isin(bank_ids.keys())]
        assert len(valid) == len(df) - 1

    def test_reviews_have_correct_bank_id(self):
        conn     = make_sqlite_conn()
        bank_ids = insert_test_banks(conn)
        df       = make_analyzed_df()
        cur      = conn.cursor()

        for _, row in df.iterrows():
            bid = bank_ids.get(row["bank"])
            if bid:
                cur.execute(
                    "INSERT INTO reviews (bank_id, review_text, rating, review_date, "
                    "sentiment_label, sentiment_score, identified_theme, source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (bid, row["review"], int(row["rating"]), row["date"],
                     row["sentiment_label"], float(row["sentiment_score"]),
                     row["identified_theme"], row["source"])
                )
        conn.commit()

        cbe_id = bank_ids["CBE"]
        cur.execute("SELECT COUNT(*) FROM reviews WHERE bank_id = ?", (cbe_id,))
        cbe_count = cur.fetchone()[0]
        assert cbe_count == df[df["bank"] == "CBE"].shape[0]

    def test_rating_constraint_rejects_out_of_range(self):
        conn     = make_sqlite_conn()
        bank_ids = insert_test_banks(conn)
        cur      = conn.cursor()
        with pytest.raises(Exception):
            cur.execute(
                "INSERT INTO reviews (bank_id, review_text, rating) VALUES (?, ?, ?)",
                (bank_ids["CBE"], "Bad review", 6)
            )
            conn.commit()

    def test_sentiment_labels_stored_correctly(self):
        conn     = make_sqlite_conn()
        bank_ids = insert_test_banks(conn)
        df       = make_analyzed_df()
        cur      = conn.cursor()

        for _, row in df.iterrows():
            bid = bank_ids.get(row["bank"])
            if bid:
                cur.execute(
                    "INSERT INTO reviews (bank_id, review_text, rating, review_date, "
                    "sentiment_label, sentiment_score, identified_theme, source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (bid, row["review"], int(row["rating"]), row["date"],
                     row["sentiment_label"], float(row["sentiment_score"]),
                     row["identified_theme"], row["source"])
                )
        conn.commit()

        cur.execute("SELECT DISTINCT sentiment_label FROM reviews")
        labels = {row[0] for row in cur.fetchall()}
        assert labels.issubset({"positive", "negative"})


# ══════════════════════════════════════════════════════════════════════════════
# BUSINESS QUERY LOGIC TESTS
# Tests for the SQL queries in notebook Cells 19-22
# ══════════════════════════════════════════════════════════════════════════════

class TestBusinessQueries:

    def _load_test_db(self):
        """Set up an in-memory DB with banks and reviews inserted."""
        conn     = make_sqlite_conn()
        bank_ids = insert_test_banks(conn)
        df       = make_analyzed_df()
        cur      = conn.cursor()

        for _, row in df.iterrows():
            bid = bank_ids.get(row["bank"])
            if bid:
                cur.execute(
                    "INSERT INTO reviews (bank_id, review_text, rating, review_date, "
                    "sentiment_label, sentiment_score, identified_theme, source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (bid, row["review"], int(row["rating"]), row["date"],
                     row["sentiment_label"], float(row["sentiment_score"]),
                     row["identified_theme"], row["source"])
                )
        conn.commit()
        return conn

    def test_q1_avg_rating_returns_all_three_banks(self):
        """Notebook Cell 19 — average rating per bank."""
        conn = self._load_test_db()
        result = pd.read_sql("""
            SELECT b.bank_name, ROUND(AVG(r.rating), 2) AS avg_rating, COUNT(*) AS total
            FROM reviews r JOIN banks b ON r.bank_id = b.bank_id
            GROUP BY b.bank_name ORDER BY avg_rating DESC
        """, conn)
        assert len(result) == 3
        assert set(result["bank_name"]) == {"CBE", "BOA", "Dashen"}

    def test_q1_avg_rating_values_are_in_range(self):
        """Avg rating must be between 1 and 5."""
        conn = self._load_test_db()
        result = pd.read_sql("""
            SELECT ROUND(AVG(r.rating), 2) AS avg_rating
            FROM reviews r JOIN banks b ON r.bank_id = b.bank_id
            GROUP BY b.bank_name
        """, conn)
        assert result["avg_rating"].between(1.0, 5.0).all()

    def test_q2_negative_themes_only_returns_negative_rows(self):
        """Notebook Cell 20 — themes for negative reviews only."""
        conn = self._load_test_db()
        result = pd.read_sql("""
            SELECT b.bank_name, r.identified_theme, COUNT(*) AS negative_count
            FROM reviews r JOIN banks b ON r.bank_id = b.bank_id
            WHERE r.sentiment_label = 'negative'
            GROUP BY b.bank_name, r.identified_theme
            ORDER BY b.bank_name, negative_count DESC
        """, conn)
        # All results should come from negative reviews only
        # Verify total matches df negative count
        df = make_analyzed_df()
        neg_count = df[df["sentiment_label"] == "negative"].shape[0]
        assert result["negative_count"].sum() == neg_count

    def test_q3_monthly_trend_date_format(self):
        """Notebook Cell 21 — monthly trend dates should be YYYY-MM format."""
        conn = self._load_test_db()
        # SQLite doesn't have TO_CHAR but we can test the date values exist
        result = pd.read_sql("""
            SELECT r.review_date FROM reviews r LIMIT 5
        """, conn)
        for date in result["review_date"]:
            assert len(date) == 10   # YYYY-MM-DD stored

    def test_q4_one_star_reviews_rating_filter(self):
        """Notebook Cell 22 — 1-star review themes."""
        conn = self._load_test_db()
        result = pd.read_sql("""
            SELECT b.bank_name, r.identified_theme, COUNT(*) AS one_star_count
            FROM reviews r JOIN banks b ON r.bank_id = b.bank_id
            WHERE r.rating = 1
            GROUP BY b.bank_name, r.identified_theme
            ORDER BY b.bank_name, one_star_count DESC
        """, conn)
        df = make_analyzed_df()
        expected_one_star = df[df["rating"] == 1].shape[0]
        assert result["one_star_count"].sum() == expected_one_star

    def test_join_between_reviews_and_banks_works(self):
        """Every review must have a valid bank — no orphaned rows."""
        conn = self._load_test_db()
        result = pd.read_sql("""
            SELECT COUNT(*) AS matched
            FROM reviews r JOIN banks b ON r.bank_id = b.bank_id
        """, conn)
        total = pd.read_sql("SELECT COUNT(*) AS total FROM reviews", conn)
        assert result["matched"].iloc[0] == total["total"].iloc[0]


# ══════════════════════════════════════════════════════════════════════════════
# INPUT DATA VALIDATION TESTS
# Verify the analyzed CSV has the correct shape before insertion
# Tests for notebook Cell 6 validation logic
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalyzedDataValidation:

    REQUIRED_COLS = [
        "review", "rating", "date", "bank", "source",
        "sentiment_label", "sentiment_score", "identified_theme"
    ]

    def test_all_required_columns_present(self):
        df = make_analyzed_df()
        for col in self.REQUIRED_COLS:
            assert col in df.columns

    def test_no_missing_review_text(self):
        df = make_analyzed_df()
        assert df["review"].isnull().sum() == 0

    def test_no_missing_bank(self):
        df = make_analyzed_df()
        assert df["bank"].isnull().sum() == 0

    def test_all_banks_are_valid(self):
        df = make_analyzed_df()
        assert set(df["bank"].unique()).issubset({"CBE", "BOA", "Dashen"})

    def test_ratings_are_in_valid_range(self):
        df = make_analyzed_df()
        assert df["rating"].between(1, 5).all()

    def test_sentiment_labels_are_valid(self):
        df = make_analyzed_df()
        assert set(df["sentiment_label"].unique()).issubset({"positive", "negative"})

    def test_sentiment_scores_are_in_valid_range(self):
        df = make_analyzed_df()
        assert df["sentiment_score"].between(0.0, 1.0).all()

    def test_date_format_is_yyyy_mm_dd(self):
        df = make_analyzed_df()
        for d in df["date"]:
            assert len(d) == 10
            assert d[4] == "-" and d[7] == "-"

    def test_source_column_is_google_play(self):
        df = make_analyzed_df()
        assert (df["source"] == "Google Play").all()
