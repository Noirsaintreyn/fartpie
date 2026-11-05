#!/bin/bash
# Pulls the site's live screener snapshot for the screener widget. Logs in with BACKEND_USER / BACKEND_PASSWORD from ~/fartpie/discord_bot/.env
# (the same login the Discord bot uses), keeps the session cookie in ~/.lap_widget_cookie and only logs in again when the cookie stops working.
ENVF="${LAP_ENV:-$HOME/fartpie/discord_bot/.env}"
val() { grep -E "^$1=" "$ENVF" 2>/dev/null | head -1 | cut -d= -f2- | sed -e 's/^["'\'']//' -e 's/["'\'']$//'; }
URL="$(val BACKEND_URL)"; URL="${URL:-https://fartpie.onrender.com}"; URL="${URL%/}"
JAR="$HOME/.lap_widget_cookie"; touch "$JAR"; chmod 600 "$JAR"
PY=$(command -v python3 || echo "$HOME/fartpie/.venv311/bin/python")
snap() { curl -s --max-time 45 -b "$JAR" -c "$JAR" "$URL/api/site/snapshot"; }
out="$(snap)"
if ! printf '%s' "$out" | grep -q '"success": *true'; then
  U="$(val BACKEND_USER)"; P="$(val BACKEND_PASSWORD)"
  if [ -z "$U" ] || [ -z "$P" ]; then echo '{"error":"no BACKEND_USER / BACKEND_PASSWORD in '"$ENVF"'"}'; exit 0; fi
  body="$("$PY" -c 'import json,sys; print(json.dumps({"username": sys.argv[1], "password": sys.argv[2]}))' "$U" "$P")"
  curl -s --max-time 45 -b "$JAR" -c "$JAR" -H 'Content-Type: application/json' -d "$body" "$URL/api/login" > /dev/null
  out="$(snap)"
fi
case "$out" in \{*) printf '%s' "$out" ;; *) echo '{"error":"site did not answer (Render may be waking up)"}' ;; esac
