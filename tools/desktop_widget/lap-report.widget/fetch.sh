#!/bin/bash
# Morning report widget data: the 09:00 ET report the Mac screener writes to ~/fartpie/live/analysis.json
# (market read, indices and sectors with daily / weekly trend, today's calendar, headlines with a one-line read). No network.
F="$HOME/fartpie/live/analysis.json"
if [ -s "$F" ]; then printf '{"report":%s}' "$(cat "$F")"; else echo '{"error":"No morning report yet (~/fartpie/live/analysis.json). The Mac screener writes it at 09:00 ET on weekdays."}'; fi
