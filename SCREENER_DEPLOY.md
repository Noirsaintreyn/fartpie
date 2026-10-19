# Screener + bot on Render

One Render worker (`fartpie-live`) runs the vector screener and the Discord bot together. Every `git push` to main redeploys it.

## One-time setup
1. `git push`
2. In Render, choose New → Blueprint, pick the repo, and paste the 6 secrets it asks for (they're in your `.env` / `discord_bot/.env`).
3. After the close on the day it goes live, stop the Mac copies so nothing posts twice:
   `launchctl unload ~/Library/LaunchAgents/com.rey.vectorscreener.plist` (and stop the Mac bot if it's running).
4. Install the two Mac jobs:
   `cp com.rey.weeklyratio.plist com.rey.midbook.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.rey.weeklyratio.plist ~/Library/LaunchAgents/com.rey.midbook.plist`

## Weekly (automatic)
- **Saturday 10:00:** `tools/weekly_ratio.sh` runs on the Mac. Open Theta Terminal before then.
- **What it does:** pulls the latest options volume, rebuilds the ratio file, then commits and pushes, which makes Render redeploy.
- **If the file gets over 10 days old:** the screener warns in Discord.
- **To run it by hand:** `tools/weekly_ratio.sh`

## Notes
- **First boot:** the worker downloads about 14 months of 15m bars from LSE by itself, so it needs no seeding.
- **Mid/small book:** it stays on the Mac (`com.rey.midbook`). It builds and posts on Friday at 16:45 and re-posts on Monday at 09:00.
- **New S&P names:** they stay out until you OK the universe switch. To add them, run `INCLUDE_NEW=1 .venv311/bin/python xs/build_live_ratio.py`, then commit and push.
- **To go back to the Mac:** suspend the worker in Render, then run `launchctl load ~/Library/LaunchAgents/com.rey.vectorscreener.plist`.
