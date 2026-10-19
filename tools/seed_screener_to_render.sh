#!/bin/bash
# ONE-TIME cutover: copy the screener's state + bar cache from this Mac to the Render worker's disk (about 200 MB).
# 1) Stop the Mac job first so it cannot post while you copy:   launchctl unload ~/Library/LaunchAgents/com.rey.vectorscreener.plist
# 2) RENDER_SSH=srv-XXXX@ssh.oregon.render.com tools/seed_screener_to_render.sh
#    (the address is in the Render dashboard: fartpie-screener > Connect > SSH; add your public key under Account Settings > SSH Public Keys first)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${RENDER_SSH:?set RENDER_SSH to the SSH address of the worker}"
LIST=$(mktemp)
python3 - > "$LIST" <<'PY'
import csv, glob, os
tks = sorted({r['tk'] for r in csv.DictReader(open('xs/ext/optvol_ratio.csv'))}) + ['QQQ', 'SPY']
for t in dict.fromkeys(tks):
    for f in sorted(glob.glob(f'lse_ohlcv_cache/{t}_15m_*.parquet'))[-1:]: print(f)
for f in sorted(glob.glob('live/*_15m.parquet')): print(f)
for f in ('live/posted.json', 'live/signal_log.csv', 'xs/ext/optvol_ratio.csv', 'xs/ext/gex/QQQ.csv'):
    if os.path.exists(f): print(f)
PY
echo "copying $(wc -l < "$LIST") files ..."
tar czf - -s ',^xs/ext/,xs_ext/,' -T "$LIST" | ssh "$RENDER_SSH" 'mkdir -p /data && cd /data && tar xzf - && du -sh /data/*'
rm -f "$LIST"
echo "done. Now deploy/restart the fartpie-screener worker and watch its logs."
