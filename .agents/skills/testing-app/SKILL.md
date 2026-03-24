# Testing the fartpie Flask Backend

## Setup

1. Install Python dependencies:
   ```bash
   pip install flask flask-cors yfinance pandas numpy scikit-learn hdbscan statsmodels arch
   ```
   Optional (non-blocking warnings if missing): `lightgbm`

2. Start the Flask server:
   ```bash
   cd /home/ubuntu/repos/fartpie
   python3 backend.py
   ```
   Server runs on port 5001.

3. On first start, `init_db()` auto-creates a SQLite database (`users.db`) with these test accounts:
   - Admin: `rey` / `flood`
   - Test user 1: `test1` / `pw`
   - Test user 2: `test2` / `pw`

## Authentication

Most API endpoints require session-based auth via `require_auth()`. To authenticate:

- **Via curl:**
  ```bash
  curl -c /tmp/cookies.txt -b /tmp/cookies.txt -X POST http://localhost:5001/api/login \
    -H "Content-Type: application/json" \
    -d '{"username":"test1","password":"pw"}'
  ```

- **Via browser console (on any page served by the app):**
  ```javascript
  fetch('/api/login', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({username:'test1',password:'pw'})}).then(r=>r.json()).then(d=>console.log(d))
  ```
  Then reload the page so subsequent API calls use the session cookie.

## Key Pages and Endpoints

- `GET /` — Health check (no auth)
- `GET /backtest` — Backtest UI page (no auth, but API calls from the page require auth)
- `GET /api/local-data` — List available local CSV instruments/timeframes (requires auth)
- `GET /api/backtest?ticker=ES&timeframe=1d&method=hdbscan&source=local` — Run backtest (requires auth)
- `POST /api/register` — Register new user (no auth)
- `POST /api/login` — Login (no auth)

## Local CSV Data

CSV data files go in `data/{INSTRUMENT}/{TIMEFRAME}/` (e.g., `data/ES/D/D_ES.csv`).
The `data/` directory is `.gitignored`. Files must be placed manually on the server.
Available instruments and timeframes are auto-discovered by scanning the `data/` directory.

## Testing the Backtest Feature

1. Start the server and login (see above)
2. Navigate to `http://localhost:5001/backtest`
3. Switch "Data Source" to "Local CSV Data"
4. Select an instrument (e.g., ES) and timeframe (e.g., Daily)
5. Click "Run Backtest" — ES Daily takes ~60-90 seconds with HDBSCAN
6. Verify results show non-zero Total Levels, Success Rate, and Sample Levels table

Note: Large files (1Min timeframe with millions of rows) may timeout.
