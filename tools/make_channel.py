"""Create a Discord text channel from code, in the same category as your vector channel (Nov 24 2026).
Run on the Mac, from ~/fartpie:   python3 tools/make_channel.py neg-gamma-flips
Uses the bot token already in discord_bot/.env (DISCORD_TOKEN, DISCORD_GUILD_ID) and VECTOR_CHANNEL_ID from .env; prints the new channel's ID (idempotent: reuses a channel with that name).
Then set NEGGAMMA_CHANNEL_ID=<that id> on Render (fartpie-live -> Environment).  The bot needs the 'Manage Channels' permission for this one-time step (Server Settings -> Roles -> the bot's role)."""
import os, sys, requests
from dotenv import load_dotenv
load_dotenv('.env'); load_dotenv('discord_bot/.env')
tok, guild, vec = os.getenv('DISCORD_TOKEN'), os.getenv('DISCORD_GUILD_ID'), os.getenv('VECTOR_CHANNEL_ID') or os.getenv('LARGECAP_CHANNEL_ID')
name = (sys.argv[1] if len(sys.argv) > 1 else 'neg-gamma-flips').lower().replace(' ', '-')
if not (tok and guild): sys.exit('DISCORD_TOKEN / DISCORD_GUILD_ID missing from discord_bot/.env')
H = {'Authorization': f'Bot {tok}'}; B = 'https://discord.com/api/v10'
parent = None
if vec:
    r = requests.get(f'{B}/channels/{vec}', headers=H, timeout=20)
    if r.ok: parent = r.json().get('parent_id'); guild = r.json().get('guild_id') or guild          # the vector channel knows its own server: more reliable than DISCORD_GUILD_ID
    else: print(f'note: could not read VECTOR_CHANNEL_ID {vec}: HTTP {r.status_code} (wrong ID, or the bot is not in that server)')
ex = requests.get(f'{B}/guilds/{guild}/channels', headers=H, timeout=20)
if not ex.ok: sys.exit(f'could not list channels in server {guild}: HTTP {ex.status_code}. DISCORD_GUILD_ID is stale or the bot is not in this server: invite the bot to your current server, then set DISCORD_GUILD_ID (Discord: Settings -> Advanced -> Developer Mode, right-click the server icon -> Copy Server ID).')
for c in ex.json():
    if c['name'] == name and c['type'] == 0: print(f'channel already exists: #{name}  ID {c["id"]}'); sys.exit(0)
body = {'name': name, 'type': 0}
if parent: body['parent_id'] = parent
r = requests.post(f'{B}/guilds/{guild}/channels', headers=H, json=body, timeout=20)
if r.status_code == 403: sys.exit('403: the bot lacks Manage Channels. Server Settings -> Roles -> give the bot role "Manage Channels", rerun.')
r.raise_for_status(); print(f'created #{name}' + (' in the same category as your vector channel' if parent else '') + f'  ID {r.json()["id"]}')
