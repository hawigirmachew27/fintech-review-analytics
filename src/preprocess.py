"""

Reusable preprocessing functions for Play Store review data.
Imported by notebooks, scripts, and unit tests — not run directly.
"""

import pandas as pd


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize a raw reviews DataFrame.

    Steps applied (in order):
    1. Remove duplicate reviews (same text + same bank)
    2. Drop rows missing review text or rating
    3. Validate ratings are in the 1–5 range
    4. Normalize dates to YYYY-MM-DD string format
    5. Strip leading/trailing whitespace from review text

    Parameters
    ----------
    df : Raw DataFrame from scrape_reviews(), must have columns:
         review, rating, date, bank, source

    Returns
    -------
    Cleaned pd.DataFrame with the same five columns.
    """
    df = df.copy()

    original_count = len(df)

    # ── Step 1: Remove duplicates ──────────────────────────────────────────
    df = df.drop_duplicates(subset=["review", "bank"])
    after_dedup = len(df)
    print(f"[clean] Duplicates removed  : {original_count - after_dedup}")

    # ── Step 2: Drop missing review text or rating ─────────────────────────
    df = df.dropna(subset=["review", "rating"])
    after_dropna = len(df)
    print(f"[clean] Missing values dropped: {after_dedup - after_dropna}")

    # ── Step 3: Validate rating range ─────────────────────────────────────
    df = df[df["rating"].between(1, 5)]
    after_rating = len(df)
    print(f"[clean] Invalid ratings removed: {after_dropna - after_rating}")

    # ── Step 4: Normalize dates to YYYY-MM-DD ─────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Drop rows where date could not be parsed
    df = df.dropna(subset=["date"])
    print(f"[clean] Unparseable dates removed: {after_rating - len(df)}")

    # ── Step 5: Strip whitespace from review text ─────────────────────────
    df["review"] = df["review"].str.strip()

    # ── Drop any reviews that are now empty strings ────────────────────────
    df = df[df["review"].str.len() > 0]

    # ── Enforce column order ───────────────────────────────────────────────
    df = df[["review", "rating", "date", "bank", "source"]].reset_index(drop=True)

    print(f"[clean] Final row count    : {len(df)}")
    return df


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame showing missing value counts and percentages
    for each column. Useful for documenting data quality in your report.
    """
    total = len(df)
    report = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_pct":   (df.isnull().sum() / total * 100).round(2),
    })
    return report