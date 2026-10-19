# Running the screener on Render

The screener (`live_screener.py`) runs as the `fartpie-screener` worker (`screener_service.py` starts it at 08:50 ET on weekdays,
restarts it if it dies and warns in Discord). ThetaData stays on the Mac; once a week you refresh the options ratio there and push the file.

## One-time setup
1. Render: create the service from the Blueprint (`render.yaml`), set secrets `DISCORD_TOKEN`, `VECTOR_CHANNEL_ID`, `LSE_API_KEY`.
   Add your SSH public key in Render > Account Settings > SSH Public Keys; the worker's SSH address is under the service's Connect menu.
2. Cutover (after the close, never while both run):
   `launchctl unload ~/Library/LaunchAgents/com.rey.vectorscreener.plist`
   `RENDER_SSH=srv-XXXX@ssh.oregon.render.com tools/seed_screener_to_render.sh`
   then restart the worker. It starts the next weekday at 08:50 ET (or immediately if you restart it during the session).

## Every week
1. Open Theta Terminal and run `python3 xs/options_volume_ratio.py` (rewrites `xs/ext/optvol_ratio.csv`).
2. `RENDER_SSH=srv-XXXX@ssh.oregon.render.com tools/sync_optvol_to_render.sh` (no restart needed).
The worker posts a warning to the channel if the file is more than 10 days old.

## Notes
- The Monday/Friday mid/small book card is off on Render (`MID_BOOK=0`); it needs the xs data pipeline on the Mac.
- To go back to the Mac: `launchctl load ~/Library/LaunchAgents/com.rey.vectorscreener.plist` and suspend the Render worker (one of them at a time).
