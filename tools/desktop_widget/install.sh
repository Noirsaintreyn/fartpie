#!/bin/bash
# One-time setup on the Mac: installs Übersicht (free desktop-widget app) if missing and links this widget into it.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ ! -d "/Applications/Übersicht.app" ]; then
  if command -v brew >/dev/null; then brew install --cask ubersicht; else echo "Install Übersicht from https://tracesof.net/uebersicht/ then run this again"; exit 1; fi
fi
W="$HOME/Library/Application Support/Übersicht/widgets"; mkdir -p "$W"
ln -sfn "$HERE/lap.widget" "$W/lap.widget"
chmod +x "$HERE/lap.widget/fetch.sh"
open -a "Übersicht"
echo "Widget linked: $W/lap.widget  (Übersicht menu-bar icon > Refresh All Widgets if it does not show)"
