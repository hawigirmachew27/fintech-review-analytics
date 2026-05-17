"""
src/sentiment.py
----------------
Reusable sentiment analysis module using DistilBERT.
Imported by notebooks and scripts — not run directly.
"""

import pandas as pd
from transformers import pipeline


def load_sentiment_pipeline():
    """
    Load the DistilBERT sentiment pipeline.

    Uses the SST-2 fine-tuned model:
    'distilbert-base-uncased-finetuned-sst-2-english'

    This downloads the model on first run (~250MB).
    Subsequent runs load it from local cache — no internet needed.

    Returns
    -------
    HuggingFace pipeline object
    """
    print("Loading DistilBERT model (downloads on first run)...")
    pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512,
    )
    print("Model loaded successfully.")
    return pipe


def run_sentiment(df: pd.DataFrame, pipe, batch_size: int = 32) -> pd.DataFrame:
    """
    Run sentiment analysis on a reviews DataFrame.

    Parameters
    ----------
    df         : DataFrame with at least a 'review' column
    pipe       : Loaded HuggingFace sentiment pipeline
    batch_size : Number of reviews to process at once (reduce if you run out of memory)

    Returns
    -------
    Same DataFrame with two new columns added:
        sentiment_label : 'positive' or 'negative'
        sentiment_score : confidence score (0.0 – 1.0)

    How it works
    ------------
    DistilBERT reads each review and outputs:
        - POSITIVE with a score close to 1.0  → the model is confident it is positive
        - NEGATIVE with a score close to 1.0  → the model is confident it is negative
    A score of 0.51 means the model is barely confident — treat those with caution.
    """
    df = df.copy()

    # DistilBERT has a 512-token limit — truncate long reviews
    reviews = df["review"].str[:512].tolist()

    print(f"Running sentiment on {len(reviews)} reviews (batch_size={batch_size})...")
    results = pipe(reviews, batch_size=batch_size)

    df["sentiment_label"] = [r["label"].lower() for r in results]
    df["sentiment_score"]  = [round(r["score"], 4) for r in results]

    # Print summary
    counts = df["sentiment_label"].value_counts()
    print("\nSentiment distribution (all banks):")
    for label, count in counts.items():
        pct = count / len(df) * 100
        print(f"  {label:10s}: {count:>5}  ({pct:.1f}%)")

    return df
