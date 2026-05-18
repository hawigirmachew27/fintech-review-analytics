"""
scripts/load_db.py
------------------
Standalone script to load all analyzed reviews into PostgreSQL.
Run from the project root after Task 2 is complete:

    python scripts/load_db.py

Reads  : data/raw/reviews_analyzed.csv
Inserts: into bank_reviews PostgreSQL database
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.database import (
    get_connection,
    create_tables,
    insert_banks,
    insert_reviews,
    verify_insertion,
)

# ── Config — change password to match your PostgreSQL setup ───────────────────
DB_CONFIG = {
    "dbname":   "bank_reviews",
    "user":     "postgres",
    "password": "yourpassword",   # <-- change this
    "host":     "localhost",
    "port":     5432,
}

ANALYZED_PATH = "data/raw/reviews_analyzed.csv"

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load data
    if not os.path.exists(ANALYZED_PATH):
        print(f"ERROR: {ANALYZED_PATH} not found. Run Task 2 notebook first.")
        sys.exit(1)

    df = pd.read_csv(ANALYZED_PATH)
    print(f"Loaded {len(df)} reviews from {ANALYZED_PATH}")

    # Connect
    conn = get_connection(**DB_CONFIG)

    # Setup
    create_tables(conn)
    bank_ids = insert_banks(conn)

    # Insert
    inserted = insert_reviews(conn, df, bank_ids)

    # Verify
    print("\n=== Verification ===")
    summary = verify_insertion(conn)
    print(summary.to_string(index=False))

    conn.close()
    print("\nDone. Database loaded successfully.")
