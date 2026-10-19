#!/bin/zsh
# Weekly options-ratio refresh (Mac, needs Theta Terminal running + logged in).
# Pulls this month + any missing months for the live names, rebuilds the ratio, commits live_data/optvol_ratio_live.csv and pushes -> Render redeploys itself.
cd ~/fartpie || exit 1
PY=.venv311/bin/python; [ -x $PY ] || PY=python3
if ! curl -s -m 5 -o /dev/null http://127.0.0.1:25503/v3/option/history/eod; then
  echo "Theta Terminal isn't running - open it (cd ~/ThetaTerminal && java -jar ThetaTerminalv3.jar) and rerun"; exit 1
fi
TKS=($(python3 -c "import pandas as pd; print(' '.join(sorted(pd.read_csv('xs/ext/optvol_ratio.csv', usecols=['tk']).tk.unique())))"))
START=$(date -v-3m +%Y-%m-01)
for s in 0 1 2 3; do FORCE=1 PARTIAL=1 START=$START SHARD=$s NSHARD=4 $PY fetch_theta_volume.py $TKS > /tmp/wr_$s.log 2>&1 & done; wait
$PY xs/build_live_ratio.py || exit 1
git pull -q --rebase --autostash origin main   # pick up anything pushed from elsewhere first
git add live_data/optvol_ratio_live.csv
git commit -m "weekly options-ratio refresh $(date +%F)" -q && git push -q origin main && echo "pushed - Render will redeploy" || echo "nothing changed"
