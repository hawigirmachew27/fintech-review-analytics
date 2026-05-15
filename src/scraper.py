"""

Reusable scraping module for Google Play Store reviews.
Imported by notebooks and scripts — not run directly.
"""

import time
import pandas as pd
from google_play_scraper import reviews, Sort


def scrape_reviews(
    app_id: str,
    bank_name: str,
    lang: str = "en",
    country: str = "et",
    count: int = 500,
    sleep: float = 1.0,
) -> pd.DataFrame:
    """
    Scrape reviews from the Google Play Store for a single app.

    Parameters
    ----------
    app_id    : Google Play app package name, e.g. 'com.combanketh.mobilebanking'
    bank_name : Short label for the bank, e.g. 'CBE'
    lang      : Language code (default 'en')
    country   : Country code (default 'et' for Ethiopia)
    count     : Number of reviews to request (request more than your minimum as buffer)
    sleep     : Seconds to wait after the request to avoid rate limiting

    Returns
    -------
    pd.DataFrame with columns: review, rating, date, bank, source
    """
    print(f"[{bank_name}] Scraping {count} reviews for app: {app_id}")

    try:
        result, _ = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=count,
        )
    except Exception as e:
        print(f"[{bank_name}] ERROR during scraping: {e}")
        print(f"[{bank_name}] Returning empty DataFrame.")
        return pd.DataFrame(columns=["review", "rating", "date", "bank", "source"])

    if not result:
        print(f"[{bank_name}] WARNING: No reviews returned. Check app ID or try a broader date range.")
        return pd.DataFrame(columns=["review", "rating", "date", "bank", "source"])

    records = []
    for r in result:
        records.append({
            "review": r.get("content", ""),
            "rating": r.get("score", None),
            "date":   r.get("at", None),
            "bank":   bank_name,
            "source": "Google Play",
        })

    df = pd.DataFrame(records)
    print(f"[{bank_name}] Fetched {len(df)} reviews.")

    # Polite pause to avoid rate limiting
    time.sleep(sleep)

    return df