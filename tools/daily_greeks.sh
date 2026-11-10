#!/bin/bash
# Mac cron, ET: 10 16 * * 1-5 (chain at the close) and 35 8 * * 1-5 (prior-close open interest). Needs .env: WEBULL_APP_KEY, WEBULL_APP_SECRET, WEBULL_API_HOST, BACKEND_URL, SITE_TOKEN
cd "$(dirname "$0")/.." || exit 1
if [ "$(date +%H)" -lt 12 ]; then python3 webull_options_log.py --symbols QQQ,SPY --oi-refresh; else python3 webull_options_log.py --symbols QQQ,SPY; fi
python3 webull_greeks.py --symbols QQQ,SPY
