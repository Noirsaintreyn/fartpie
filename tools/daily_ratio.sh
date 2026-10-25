#!/bin/zsh
# Daily options-ratio refresh from IBKR (Mac, TWS / IB Gateway logged in, API on 7497). Weekdays ~16:20 ET via com.rey.dailyratio.plist.
# Appends today's option volume per name, rebuilds the ratio file, commits + pushes -> Render redeploys with it.
cd ~/fartpie || exit 1
PY=.venv311/bin/python; [ -x $PY ] || PY=python3
$PY tools/ibkr_optvol.py || { echo "IBKR pull failed - ratio not refreshed today"; exit 1; }
$PY xs/build_live_ratio.py || exit 1
git pull -q --rebase --autostash origin main
git add live_data/optvol_ratio_live.csv
git commit -m "daily options-ratio refresh (IBKR) $(date +%F)" -q && git push -q origin main && echo "pushed - Render will redeploy" || echo "nothing changed"
