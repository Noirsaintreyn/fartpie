#!/bin/bash
# One-time setup on the Mac (safe to re-run; re-running also reloads the widgets):
#   installs Übersicht (free desktop-widget app) if missing, links both widgets into it, and restarts it so the newest files load.
#   lap-screener.widget  live vector screener        lap-prop.widget  prop accounts: the add-an-account light
# Widgets can be dragged anywhere on the desktop (position is remembered; double-click a widget to reset it).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
APP="/Applications/Übersicht.app"
if [ ! -d "$APP" ]; then
  if command -v brew >/dev/null; then brew install --cask ubersicht; else echo "Install Übersicht from https://tracesof.net/uebersicht/ then run this again"; exit 1; fi
fi
W="$HOME/Library/Application Support/Übersicht/widgets"; mkdir -p "$W"
[ -L "$W/lap.widget" ] && rm "$W/lap.widget"                      # the first combined version
for w in lap-screener.widget lap-prop.widget; do ln -sfn "$HERE/$w" "$W/$w"; chmod +x "$HERE/$w/fetch.sh"; done
rm -f "$HOME/.lap_acct_signals.json"                               # fresh signal read
pkill -f "bersicht.app/Contents/MacOS" 2>/dev/null || true          # quit a running copy (matching without the Ü avoids the macOS name-encoding mismatch)
sleep 2
open "$APP"
echo "Übersicht restarted with the newest widgets. Drag them anywhere; double-click one to reset its spot."
