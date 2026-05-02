---
name: testing-level-detection
description: Test the Level Detection web app end-to-end. Use when verifying frontend display of levels, THOD/TLOD, intraday target, regime cards, or multi-timeframe analysis.
---

# Testing the Level Detection App

## Prerequisites

- Python 3 with all dependencies installed (torch, flask, numpy, pandas, scikit-learn, hdbscan, etc.)
- The app runs on `localhost:5001`

## Server Setup

```bash
cd /home/ubuntu/repos/fartpie
python3 backend.py &
```

Wait ~5 seconds for startup. The server initializes a SQLite database (`users.db`) with default accounts on first run.

## Authentication

- Admin: `rey` / `admin`
- Demo: `user1` / `pw`
- Login at `localhost:5001/login`

## Key Test Scenarios

### 1. Daily Analysis (SPY)
- Select ticker=SPY, timeframe=Daily, click "Analyze Levels"
- Takes ~30-40 seconds to complete
- Verify: Current price, Price Forecast/Intraday Target, THOD/TLOD (1σ/2σ/3σ + Predicted), Regime cards (4), Levels Detected summary, level categories with Support/Resistance split

### 2. 4h Analysis (SPY)
- Select ticker=SPY, timeframe=4 Hours, click "Analyze Levels"
- Takes ~60 seconds (resamples 1h data to 4h)
- Previously had a JSON Infinity bug — verify no "Unexpected token" error
- 4h may show additional categories not seen in Daily (Classical Pivots, Gap Levels, ML Confluence)

### 3. THOD/TLOD Section
- Appears between intraday target and regime cards
- Left card: "Theoretical High of Day" with 1/2/3 StdDev prices (green, should be above current price)
- Right card: "Theoretical Low of Day" with 1/2/3 StdDev prices (red, should be below current price)
- Predicted HOD/LOD with confidence percentage at bottom of each card
- Data comes from `/api/level-constrained-hod-lod` endpoint

### 4. Level Categories
- The API returns levels under various category keys: `structural`, `interaction`, `mlConfluence`, `pivots`, `gaps`, `classicalStructural`, `fallback`, `peakValley`, etc.
- Each category is rendered as a collapsible section with level count
- Levels are split into Resistance (above current price, green) and Support (below current price, red)
- Each level row shows: dollar price, category/method name, strength %, distance %

## Known Quirks

- Classical Pivots and Gap Levels may appear duplicated in 4h view (API might return them under multiple keys)
- The label "Price Forecast" vs "Intraday Target" changes depending on the data source — `mostProbablePath` uses "Intraday Target", forecast data uses "Price Forecast"
- 4h timeframe internally resamples 1h data via `fetch_historical_data_with_resampling()` — if Yahoo Finance 1h data is unavailable, this will fail
- `sanitize_for_json()` in `backend.py` handles both numpy and native Python float inf/nan → null conversion

## API Endpoints

- `POST /api/data` — Main analysis endpoint (ticker, timeframe)
- `GET /api/level-constrained-hod-lod?ticker=SPY&timeframe=1d` — THOD/TLOD data
- `POST /api/train-level-detector` — Retrain the neural network model
- `GET /api/test` — Health check

## Frontend File

- `templates/index.html` — Single-page app with all rendering logic
- Key functions: `analyzeLevels()`, `displayResults()`, `renderThodTlod()`, `renderRegimeInfo()`, `renderLevelCategory()`
