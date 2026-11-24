"""Post an image to a Discord channel with the bot (Nov 24 2026).   python3 tools/post_chart.py [image] [channel_id] [caption]
Defaults: image xs/neggamma_backtest.png, channel NEGGAMMA_CHANNEL_ID (from .env / discord_bot/.env), no caption.  Uses DISCORD_TOKEN from discord_bot/.env on your Mac."""
import os, sys, requests
from dotenv import load_dotenv
load_dotenv('.env'); load_dotenv('discord_bot/.env')
img = sys.argv[1] if len(sys.argv) > 1 else 'xs/neggamma_backtest.png'; ch = sys.argv[2] if len(sys.argv) > 2 else os.getenv('NEGGAMMA_CHANNEL_ID'); cap = sys.argv[3] if len(sys.argv) > 3 else ''
tok = os.getenv('DISCORD_TOKEN')
if not (tok and ch): sys.exit('need DISCORD_TOKEN (discord_bot/.env) and a channel id (argument 2, or NEGGAMMA_CHANNEL_ID in .env)')
with open(img, 'rb') as f: r = requests.post(f'https://discord.com/api/v10/channels/{ch}/messages', headers={'Authorization': f'Bot {tok}'}, data={'content': cap}, files={'files[0]': (os.path.basename(img), f, 'image/png')}, timeout=30)
print('posted' if r.ok else f'failed: HTTP {r.status_code} {r.text[:200]}')
