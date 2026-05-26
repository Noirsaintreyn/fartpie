# PRD — LSTM Forecast / Degen Discovery

## Original Problem Statement
> "help me edit the app im trying to add it to where I can put a csv in whenever I login to, get the data whenever I login"
>
> Follow-up: "yes but I want everyone to see the data once I upload it"
>
> Follow-up: "you gotta push the changes so it redeploys in render"
>
> Follow-up: "can you change it so it also edits peepeepoop which is the front end"

## Architecture
- **Backend (fartpie)**: Flask app at `/app/backend.py` — deployed to Render at `https://fartpie.onrender.com`
- **Frontend (peepeepoop)**: Static HTML/JS at `/app/peepeepoop_repo/` — deployed to Cloudflare Pages
  - Main file: `lstm-forecast-example.html` (copied to `dist/index.html` by `build.js`)
- **Database**: SQLite (`users.db`) — uses Render persistent disk

## User Personas
- **Trader/Quant** — uploads CSV exports from MotiveWave to replace yfinance data for backtesting/forecasting

## Core Requirements
1. CSV uploads must persist across server restarts
2. All users see the same uploaded CSV data (global, not user-scoped)
3. UI for upload + view + delete on the deployed frontend
4. Auto-load CSV data on backend startup

## What's Been Implemented

### 2026-01-26 — VbP Level Detection Integration
**New file: `/app/vbp_levels.py`** (908 lines)
- LevelEngine v2 with close-weighted VbP distribution
- Multi-algo: KDE / HDBSCAN / OPTICS / Isolation Forest / Wyckoff / Persistent Homology
- POC / VAH / VAL anchor injection
- `level_holdrate()` walk-forward validation
- InstrumentProfile for tunable parameters per symbol/timeframe
- Patched `ripser` import to be optional (graceful degradation)

**Backend (`/app/backend.py`):**
- Added safe import of `LevelEngine` and `InstrumentProfile` (with `VBP_AVAILABLE` flag)
- Added `calculate_vbp_levels(hist_df, timeframe, current_price)` wrapper that:
  - Auto-picks tick size based on price magnitude
  - Lookback by timeframe (300/400/500 bars)
  - Maps engine output columns (`price`/`score`/`sources`/`type`) → existing level schema
  - Tags POC/VAH/VAL anchors with floor strength 0.80
  - Returns top 12 by strength
- Wired into both level-detection codepaths in `/api/lstm-forecast`
- Added `vbp` count to response `levels_detected` block

**Frontend updates:**
- `/app/templates/index.html`: Added `VbP: ${levelCounts.vbp ?? 0}` to levels grid
- `/app/peepeepoop_repo/lstm-forecast-example.html`: Added `VbP: ${formattedData.levels.vbp ?? 0}` to levels grid

**Files Created:**
- `/app/test_vbp_integration.py` — smoke test (passing: POC=102.40, VAH=108.02, VAL=96.00, 12 levels)

### 2026-01-26 — CSV Persistence Feature
**Backend (`/app/backend.py`):**
- Added `csv_data` SQLite table in `init_db()` (ticker, timeframe, csv_content, bars, dates, uploaded_by)
- Added `save_csv_to_db()` — saves DataFrame as CSV text to DB on upload
- Added `load_all_csv_from_db()` — loads all CSVs into `MOTIVEWAVE_DATA` memory dict on startup
- Modified `/api/motivewave/upload` to also save to DB
- Modified `/api/motivewave/delete` to also delete from DB
- Called `load_all_csv_from_db()` right after `init_db()` on module load

**Frontend (`/app/peepeepoop_repo/lstm-forecast-example.html`):**
- Added new `<section class="mw-section">` with two-card layout (Upload + Active Datasets)
- Added CSS for `.mw-section`, `.mw-card`, `.mw-dataset-row`, etc. (matching gold/dark aesthetic)
- Added JS functions: `uploadMotiveWave()`, `loadMwStatus()`, `deleteMwDataset()`
- Wired up event listeners + auto-load datasets on page load
- Uses `${API_BASE}` (https://fartpie.onrender.com) for all calls

**Files Created:**
- `/app/test_csv_persistence.py` — test suite (all 4 tests passing)
- `/app/CSV_PERSISTENCE_README.md` — documentation

## Prioritized Backlog

### P0 — User must push to deploy
- User to use "Save to Github" → Render will redeploy backend with new DB table
- User to push peepeepoop → Cloudflare Pages will redeploy frontend with upload UI

### P1 — Verify after deploy
- Test CSV upload → confirm row appears in Active Datasets
- Restart Render service → confirm CSV data still loads
- Verify Render uses persistent disk for `users.db` (env var `RENDER_DISK_PATH`)

### P2 — Future enhancements
- Show "uploaded by" + timestamp in the datasets list
- Drag & drop CSV upload
- CSV preview before upload
- Bulk CSV upload (multiple tickers at once)
- Per-user privacy toggle (private vs shared)

## Next Tasks
- User needs to use "Save to Github" feature to deploy backend (fartpie) and frontend (peepeepoop)
- After deploy, smoke test the upload flow end-to-end
