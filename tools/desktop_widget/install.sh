#!/bin/bash
# One-time setup on the Mac: installs Übersicht (free desktop-widget app) if missing and links both widgets into it:
#   lap-screener.widget  live vector screener (book vs QQQ/SPY, positions, today's signals, ranked names)   bottom-left
#   lap-prop.widget      prop accounts: the ADD ACCOUNT light (account-start signals at the prior close)    next to it
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ ! -d "/Applications/Übersicht.app" ]; then
  if command -v brew >/dev/null; then brew install --cask ubersicht; else echo "Install Übersicht from https://tracesof.net/uebersicht/ then run this again"; exit 1; fi
fi
W="$HOME/Library/Application Support/Übersicht/widgets"; mkdir -p "$W"
[ -L "$W/lap.widget" ] && rm "$W/lap.widget"                      # the first combined version
for w in lap-screener.widget lap-prop.widget; do ln -sfn "$HERE/$w" "$W/$w"; chmod +x "$HERE/$w/fetch.sh"; done
open -a "Übersicht"
echo "Widgets linked in $W (Übersicht menu-bar icon > Refresh All Widgets if they do not show; drag-position via each file's className left/bottom)"
