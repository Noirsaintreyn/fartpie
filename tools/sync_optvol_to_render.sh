#!/bin/bash
# WEEKLY: after refreshing the options ratio from ThetaData on this Mac, push the file to the Render screener.
#   1) open Theta Terminal, then:  python3 xs/options_volume_ratio.py      (rewrites xs/ext/optvol_ratio.csv)
#   2) RENDER_SSH=srv-XXXX@ssh.oregon.render.com tools/sync_optvol_to_render.sh
# The worker re-reads the file on every scan, so no restart is needed.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${RENDER_SSH:?set RENDER_SSH to the SSH address of the worker}"
python3 - <<'PY'
import pandas as pd
d = pd.read_csv('xs/ext/optvol_ratio.csv', usecols=['date', 'tk']); print('local file: last row', d.date.max(), '| tickers', d.tk.nunique())
PY
scp xs/ext/optvol_ratio.csv "$RENDER_SSH:/data/xs_ext/optvol_ratio.csv"
[ -f xs/ext/gex/QQQ.csv ] && scp xs/ext/gex/QQQ.csv "$RENDER_SSH:/data/xs_ext/gex/QQQ.csv" || true
echo "synced."
