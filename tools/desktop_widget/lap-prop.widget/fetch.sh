#!/bin/bash
# Account-start signals at the prior close (VOL-HI / VXN / COLD / DIP) for the prop widget; the script caches its own result for 30 minutes.
PYV="$HOME/fartpie/.venv311/bin/python"; [ -x "$PYV" ] || PYV="$(command -v python3)"
out="$("$PYV" "$HOME/fartpie/tools/account_signals.py" --json 2>/dev/null | tail -1)"
case "$out" in \{*) printf '%s' "$out" ;; *) echo '{"error":"signals unavailable - check ~/fartpie/.venv311 (pandas, yfinance)"}' ;; esac
