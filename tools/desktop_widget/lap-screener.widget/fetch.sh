#!/bin/bash
# Screener widget data. First choice: the Mac screener's own snapshot, ~/fartpie/live/site_snapshot.json (written by the live screener on this Mac,
# no network, no login; overnight it simply holds the last close). Fallback when that file is missing or more than 3 days old: the website
# (https://fartpie.onrender.com) with the bot login from ~/fartpie/discord_bot/.env; the last good copy is kept in ~/.lap_screener_last.json.
SNAP="$HOME/fartpie/live/site_snapshot.json"; CACHE="$HOME/.lap_screener_last.json"; ENVF="$HOME/fartpie/discord_bot/.env"
PY="$HOME/fartpie/.venv311/bin/python"; [ -x "$PY" ] || PY="$(command -v python3)"
if [ -s "$SNAP" ] && [ $(( $(date +%s) - $(stat -f %m "$SNAP") )) -lt 259200 ]; then
  "$PY" -c 'import json,sys; print(json.dumps({"success": True, "source": "mac", "snapshot": json.load(open(sys.argv[1]))}))' "$SNAP" 2>/dev/null && exit 0
fi
val() { grep -E "^$1=" "$ENVF" 2>/dev/null | head -1 | cut -d= -f2- | sed -e 's/^["'\'']//' -e 's/["'\'']$//'; }
URL="https://fartpie.onrender.com"; JAR="$HOME/.lap_widget_cookie"; touch "$JAR"; chmod 600 "$JAR"
snap() { curl -s --max-time 45 -b "$JAR" -c "$JAR" "$URL/api/site/snapshot"; }
out="$(snap)"
if ! printf '%s' "$out" | grep -q '"success": *true'; then
  U="$(val BACKEND_USER)"; P="$(val BACKEND_PASSWORD)"
  if [ -n "$U" ] && [ -n "$P" ]; then
    body="$("$PY" -c 'import json,sys; print(json.dumps({"username": sys.argv[1], "password": sys.argv[2]}))' "$U" "$P")"
    curl -s --max-time 45 -b "$JAR" -c "$JAR" -H 'Content-Type: application/json' -d "$body" "$URL/api/login" > /dev/null
    out="$(snap)"
  fi
fi
printf '%s' "$out" | "$PY" -c '
import json, sys, os
cache = sys.argv[1]
try: new = json.loads(sys.stdin.read())
except Exception: new = None
try: old = json.load(open(cache))
except Exception: old = None
ok = lambda d: isinstance(d, dict) and d.get("success") and d.get("snapshot")
asof = lambda d: (d.get("snapshot") or {}).get("asof") or ""
if ok(new) and (not ok(old) or asof(new) >= asof(old)):
    json.dump(new, open(cache + ".tmp", "w")); os.replace(cache + ".tmp", cache); new["source"] = "site"; print(json.dumps(new))
elif ok(old):
    old["source"] = "saved"; print(json.dumps(old))
else:
    print(json.dumps({"error": "No screener snapshot on this Mac (~/fartpie/live/site_snapshot.json) and the website did not answer. Start the live screener on this Mac."}))
' "$CACHE"
