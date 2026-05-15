"""
Standalone script to clean and merge raw review CSVs for all three banks.
Run from the project root after scripts/scrape.py:

    python scripts/preprocess.py

Reads  : data/raw/{bank}_reviews_raw.csv  (one per bank)
Saves  : data/raw/reviews_clean.csv       (combined, cleaned dataset)
"""

import os
import sys
import glob
import pandas as pd

# Allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import clean_reviews, missing_value_report

RAW_DIR    = "data/raw"
OUTPUT_PATH = os.path.join(RAW_DIR, "reviews_clean.csv")

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Find all raw bank CSV files
    raw_files = glob.glob(os.path.join(RAW_DIR, "*_reviews_raw.csv"))

    if not raw_files:
        print("ERROR: No raw CSV files found in data/raw/")
        print("Run scripts/scrape.py first.")
        sys.exit(1)

    print(f"Found {len(raw_files)} raw file(s): {[os.path.basename(f) for f in raw_files]}\n")

    all_dfs = []

    for path in sorted(raw_files):
        bank = os.path.basename(path).split("_")[0].upper()
        df_raw = pd.read_csv(path)
        print(f"── {bank} ──────────────────────────")
        print(f"   Raw rows: {len(df_raw)}")

        df_clean = clean_reviews(df_raw)
        all_dfs.append(df_clean)
        print()

    # Combine all banks into one dataset
    df_combined = pd.concat(all_dfs, ignore_index=True)

    # ── Final quality report ───────────────────────────────────────────────
    print("═══ Combined Data Quality Report ═══")
    print(f"Total reviews (all banks) : {len(df_combined)}")
    print(f"KPI target (minimum)      : 1200")
    print(f"KPI met                   : {'YES ✓' if len(df_combined) >= 1200 else 'NO — check scraping'}")
    print()
    print("Reviews per bank:")
    print(df_combined["bank"].value_counts().to_string())
    print()
    print("Missing value report:")
    print(missing_value_report(df_combined).to_string())
    print()

    # ── Save ───────────────────────────────────────────────────────────────
    df_combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Clean dataset saved → {OUTPUT_PATH}")
    print(f"Columns: {list(df_combined.columns)}")
    print(f"Shape  : {df_combined.shape}")