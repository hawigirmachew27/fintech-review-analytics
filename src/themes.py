"""
src/themes.py
-------------
Reusable thematic analysis module using TF-IDF keyword extraction.
Imported by notebooks and scripts — not run directly.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet



# ── Theme keyword map ──────────────────────────────────────────────────────────
# Each theme maps to a list of keywords. If any keyword appears in a review,
# that review is assigned to that theme.
# Edit this map to add or rename themes based on what you find in your data.

THEME_MAP = {
    "Account Access":          ["login", "otp", "password", "sign in", "fingerprint",
                                 "biometric", "unlock", "authenticate", "session", "logout"],
    "Transaction Performance": ["transfer", "send money", "slow", "fast", "instant",
                                 "delay", "pending", "transaction", "payment", "speed"],
    "App Stability":           ["crash", "error", "bug", "freeze", "update", "loading",
                                 "force close", "not working", "glitch", "restart"],
    "Customer Support":        ["support", "helpline", "response", "agent", "contact",
                                 "call center", "service", "complaint", "feedback", "resolve"],
    "UI & Design":             ["interface", "design", "ui", "layout", "button", "screen",
                                 "dark mode", "theme", "font", "display"],
    "Feature Requests":        ["feature", "add", "need", "wish", "want", "please",
                                 "statement", "history", "budgeting", "schedule"],
    "Balance & Account Info":  ["balance", "statement", "account", "history", "check",
                                 "view", "mini statement", "passbook"],
}


from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map NLTK POS tag to WordNet POS tag so lemmatizer works correctly."""
    tag = pos_tag([word])[0][1][0].upper()
    tag_map = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}
    return tag_map.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    tokens = word_tokenize(text.lower())
    return " ".join(lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens if w.isalpha())

def extract_top_keywords(df, bank_name, top_n=20):
    bank_reviews = df[df["bank"] == bank_name]["review"].tolist()

    if len(bank_reviews) < 5:
        print(f"[{bank_name}] Not enough reviews for TF-IDF.")
        return []

    # ── Apply lemmatization before TF-IDF ─────────────────────────────
    lemmatized_reviews = [lemmatize_text(r) for r in bank_reviews]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=200,
        min_df=2,
    )

    tfidf_matrix = vectorizer.fit_transform(lemmatized_reviews)  # use lemmatized

    scores = tfidf_matrix.sum(axis=0).A1
    keywords_scores = list(zip(vectorizer.get_feature_names_out(), scores))
    keywords_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"[{bank_name}] Top {top_n} keywords: {[kw for kw, _ in keywords_scores[:top_n]]}")
    return keywords_scores[:top_n]


def assign_themes(df: pd.DataFrame, theme_map: dict = None) -> pd.DataFrame:
    """
    Assign a theme to each review based on keyword matching.

    Logic
    -----
    For each review, check if any keyword from each theme appears in the text.
    Assign the FIRST matching theme. If no keywords match, assign 'General'.

    Parameters
    ----------
    df        : Reviews DataFrame with a 'review' column
    theme_map : Optional custom theme map. Uses THEME_MAP above by default.

    Returns
    -------
    Same DataFrame with a new 'identified_theme' column.
    """
    if theme_map is None:
        theme_map = THEME_MAP

    df = df.copy()

    def _find_theme(review_text: str) -> str:
        text = review_text.lower()
        for theme, keywords in theme_map.items():
            if any(kw in text for kw in keywords):
                return theme
        return "General"

    df["identified_theme"] = df["review"].apply(_find_theme)

    print("\nTheme distribution (all banks):")
    print(df["identified_theme"].value_counts().to_string())

    return df


def theme_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a cross-tab showing theme counts per bank.
    Useful for your report's thematic analysis section.
    """
    return pd.crosstab(df["identified_theme"], df["bank"])
