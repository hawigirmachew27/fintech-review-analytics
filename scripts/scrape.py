"""

Standalone script to scrape Google Play Store reviews for all three banks.
Run from the project root:

    python scripts/scrape.py

Saves raw CSV files to data/raw/ (excluded from Git via .gitignore).
"""

import os
import sys

# Allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scraper import scrape_reviews

# ── Bank configurations ────────────────────────────────────────────────────────
BANKS = [
    {
        "bank_name": "CBE",
        "app_id":    "com.combanketh.mobilebanking",
    },
    {
        "bank_name": "BOA",
        "app_id":    "com.boa.boaMobileBanking",
    },
    {
        "bank_name": "Dashen",
        "app_id":    "com.dashen.dashensuperapp",
    },
]

LANG    = "en"
COUNTRY = "et"
COUNT   = 500   # request more than 400 as a safety buffer
OUTPUT_DIR = "data/raw"

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for bank in BANKS:
        df = scrape_reviews(
            app_id    = bank["app_id"],
            bank_name = bank["bank_name"],
            lang      = LANG,
            country   = COUNTRY,
            count     = COUNT,
        )

        if df.empty:
            print(f"[{bank['bank_name']}] No data returned — check app ID or network.")
            continue

        # Save one raw CSV per bank
        out_path = os.path.join(OUTPUT_DIR, f"{bank['bank_name'].lower()}_reviews_raw.csv")
        df.to_csv(out_path, index=False)
        print(f"[{bank['bank_name']}] Raw data saved → {out_path}")

    print("\nAll banks scraped. Run scripts/preprocess.py next.")