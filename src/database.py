"""
src/database.py
---------------
Reusable PostgreSQL database module using psycopg2.
Imported by notebooks, scripts, and tests — not run directly.
"""

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd


# ── Connection ─────────────────────────────────────────────────────────────────

def get_connection(dbname="bank_reviews", user="postgres",
                   password="yourpassword", host="localhost", port=5433):
    """
    Open and return a psycopg2 connection to PostgreSQL.

    Parameters
    ----------
    dbname   : Name of the database (default: bank_reviews)
    user     : PostgreSQL username (default: postgres)
    password : PostgreSQL password — change to match your setup
    host     : Database host (default: localhost)
    port     : Database port (default: 5432)

    Returns
    -------
    psycopg2 connection object

    Usage
    -----
    conn = get_connection()
    cur  = conn.cursor()
    ...
    conn.commit()
    cur.close()
    conn.close()
    """
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port,
    )
    print(f"Connected to database: {dbname} @ {host}:{port}")
    return conn


# ── Schema creation ────────────────────────────────────────────────────────────

def create_tables(conn):
    """
    Create the banks and reviews tables if they do not already exist.

    Schema design
    -------------
    banks   — one row per bank (parent table)
    reviews — one row per review, foreign key to banks.bank_id

    Using IF NOT EXISTS means this is safe to run multiple times
    without dropping existing data.
    """
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS banks (
            bank_id   SERIAL PRIMARY KEY,
            bank_name VARCHAR(100) NOT NULL UNIQUE,
            app_id    VARCHAR(200),
            country   VARCHAR(10) DEFAULT 'et'
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id        SERIAL PRIMARY KEY,
            bank_id          INT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
            review_text      TEXT,
            rating           INT CHECK (rating BETWEEN 1 AND 5),
            review_date      DATE,
            sentiment_label  VARCHAR(20),
            sentiment_score  FLOAT,
            identified_theme VARCHAR(100),
            source           VARCHAR(50) DEFAULT 'Google Play',
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    cur.close()
    print("Tables created (or already exist): banks, reviews")


# ── Insert banks ───────────────────────────────────────────────────────────────

BANK_METADATA = {
    "CBE":    {"app_id": "com.combanketh.mobilebanking"},
    "BOA":    {"app_id": "com.boa.boaMobileBanking"},
    "Dashen": {"app_id": "com.dashen.dashensuperapp"},
}

def insert_banks(conn):
    """
    Insert the three banks into the banks table.
    Uses ON CONFLICT DO NOTHING so re-running is safe.

    Returns
    -------
    dict mapping bank_name → bank_id
    """
    cur = conn.cursor()
    bank_ids = {}

    for bank_name, meta in BANK_METADATA.items():
        cur.execute("""
            INSERT INTO banks (bank_name, app_id)
            VALUES (%s, %s)
            ON CONFLICT (bank_name) DO NOTHING;
        """, (bank_name, meta["app_id"]))

        # Fetch the bank_id whether it was just inserted or already existed
        cur.execute("SELECT bank_id FROM banks WHERE bank_name = %s", (bank_name,))
        bank_ids[bank_name] = cur.fetchone()[0]

    conn.commit()
    cur.close()
    print(f"Banks inserted/verified: {bank_ids}")
    return bank_ids


# ── Insert reviews ─────────────────────────────────────────────────────────────

def insert_reviews(conn, df: pd.DataFrame, bank_ids: dict):
    """
    Bulk-insert all reviews into the reviews table using execute_values
    for performance (much faster than looping with cur.execute).

    Parameters
    ----------
    conn     : Open psycopg2 connection
    df       : Cleaned + analyzed DataFrame from Task 2
               Must have columns: review, rating, date, bank,
                                  sentiment_label, sentiment_score,
                                  identified_theme, source
    bank_ids : Dict mapping bank_name → bank_id (from insert_banks)

    Returns
    -------
    int — number of rows inserted
    """
    cur = conn.cursor()

    rows = []
    skipped = 0

    for _, row in df.iterrows():
        bank_id = bank_ids.get(row["bank"])
        if bank_id is None:
            skipped += 1
            continue

        rows.append((
            bank_id,
            row.get("review"),
            int(row["rating"]) if pd.notna(row["rating"]) else None,
            row.get("date"),
            row.get("sentiment_label"),
            float(row["sentiment_score"]) if pd.notna(row.get("sentiment_score")) else None,
            row.get("identified_theme"),
            row.get("source", "Google Play"),
        ))

    execute_values(cur, """
        INSERT INTO reviews
            (bank_id, review_text, rating, review_date,
             sentiment_label, sentiment_score, identified_theme, source)
        VALUES %s
    """, rows)

    conn.commit()
    cur.close()

    print(f"Inserted : {len(rows)} reviews")
    if skipped:
        print(f"Skipped  : {skipped} rows (unrecognised bank name)")

    return len(rows)


# ── Verification queries ───────────────────────────────────────────────────────

def verify_insertion(conn) -> pd.DataFrame:
    """
    Run summary queries to confirm the data loaded correctly.

    Returns
    -------
    pd.DataFrame with one row per bank showing:
    review_count, avg_rating, pct_positive, earliest_date, latest_date
    """
    query = """
        SELECT
            b.bank_name,
            COUNT(*)                                          AS review_count,
            ROUND(AVG(r.rating)::NUMERIC, 2)                 AS avg_rating,
            ROUND(
                100.0 * SUM(CASE WHEN r.sentiment_label = 'positive' THEN 1 ELSE 0 END)
                / COUNT(*), 1
            )                                                 AS pct_positive,
            MIN(r.review_date)                                AS earliest_date,
            MAX(r.review_date)                                AS latest_date
        FROM reviews r
        JOIN banks b ON r.bank_id = b.bank_id
        GROUP BY b.bank_name
        ORDER BY review_count DESC;
    """
    return pd.read_sql(query, conn)


def theme_breakdown(conn) -> pd.DataFrame:
    """
    Return a cross-tab of theme counts per bank — useful for the report.
    """
    query = """
        SELECT
            b.bank_name,
            r.identified_theme,
            COUNT(*) AS review_count
        FROM reviews r
        JOIN banks b ON r.bank_id = b.bank_id
        GROUP BY b.bank_name, r.identified_theme
        ORDER BY b.bank_name, review_count DESC;
    """
    return pd.read_sql(query, conn)
