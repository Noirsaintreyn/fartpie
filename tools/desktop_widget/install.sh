#!/bin/bash
# One-time setup on the Mac (safe to re-run; re-running also reloads the widgets):
#   installs Übersicht (free desktop-widget app) if missing, links the two widgets, unlinks the old ones, and restarts Übersicht so the newest files load.
#   lap-report.widget    La matinée: the 09:00 morning report (market read, indices and sectors, calendar, headlines)  - ~/fartpie/live/analysis.json
#   lap-screener.widget  the live screener holdings (book vs QQQ / SPY, positions, today's signals, ranked names)     - ~/fartpie/live/site_snapshot.json
# Both read files the Mac screener already writes, so they work overnight (last close) and need no website or login.
# Drag either widget anywhere (the spot is remembered); double-click one to put it back.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
APP="/Applications/Übersicht.app"
if [ ! -d "$APP" ]; then
  if command -v brew >/dev/null; then brew install --cask ubersicht; else echo "Install Übersicht from https://tracesof.net/uebersicht/ then run this again"; exit 1; fi
fi
W="$HOME/Library/Application Support/Übersicht/widgets"; mkdir -p "$W"
for old in lap.widget lap-prop.widget; do [ -L "$W/$old" ] && rm "$W/$old"; done      # the combined first version and the prop-accounts widget
for w in lap-report.widget lap-screener.widget; do ln -sfn "$HERE/$w" "$W/$w"; chmod +x "$HERE/$w/fetch.sh"; done
pkill -f "bersicht.app/Contents/MacOS" 2>/dev/null || true          # quit a running copy (matching without the Ü avoids the macOS name-encoding mismatch)
sleep 2
open "$APP"
echo "Übersicht restarted: morning report + screener holdings. Drag them anywhere; double-click one to reset its spot."
