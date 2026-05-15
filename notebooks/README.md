# Task 1 — Data Collection & Preprocessing

## Overview

This scrapes Google Play Store reviews for three Ethiopian banks — **Commercial Bank of Ethiopia (CBE)**, **Bank of Abyssinia (BOA)**, and **Dashen Bank** — cleans the raw data, and produces a structured CSV dataset ready for sentiment and thematic analysis.

---



### Apps targeted

| Bank | App ID | Play Store Name |
|------|--------|-----------------|
| CBE | `com.combanketh.mobilebanking` | CBE Mobile Banking |
| BOA | `com.boa.boaMobileBanking` | BOA Mobile Banking |
| Dashen | `com.dashen.dashensuperapp` | Dashen Super App |

### Parameters used

- **Language:** `en` (English)
- **Country:** `et` (Ethiopia)
- **Sort order:** Newest first (`Sort.MOST_RELEVANT`)
- **Reviews requested per bank:** 500 (to ensure at least 400 after cleaning)
- **Date range:** All available reviews up to scrape date

### Fields collected

| Column | Description | Example |
|--------|-------------|---------|
| `review` | Raw user review text | `"Transfers are instant, love it!"` |
| `rating` | Star rating (1–5) | `5` |
| `date` | Review posting date | `2025-11-03` |
| `bank` | Short bank identifier | `CBE` |
| `source` | Data origin | `Google Play` |

---

## Preprocessing Steps



1. **Duplicate removal** — Rows with identical `review` text and `bank` were dropped using `drop_duplicates(subset=["review", "bank"])`.
2. **Missing value handling** — Rows missing `review` text or `rating` were dropped. Counts were logged before and after.
3. **Rating validation** — Ratings outside the 1–5 range were filtered out.
4. **Date normalization** — All dates were standardized to `YYYY-MM-DD` format using `pd.to_datetime().dt.strftime()`.
5. **Whitespace stripping** — Leading and trailing whitespace was removed from all review text.

---


---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## CI/CD

A GitHub Actions workflow (`.github/workflows/unittests.yml`) runs on every push to `main` and on every pull request. It:

1. Installs all dependencies from `requirements.txt`

The workflow must pass before any task branch is merged into `main`.

---

## Git Branching

This task was developed on the `task-1` branch and merged into `main` via a pull request after CI/CD passed. Commits follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard:

```
feat(scrape): add google-play-scraper for all three banks


---

