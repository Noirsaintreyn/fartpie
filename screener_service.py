"""Render worker wrapper for live_screener.py (the 15m/1h vector screener that posts BUY/SELL cards to Discord).

What it does: keeps the screener running like the Mac's launchd job did - starts it at 08:50 ET on weekdays (or immediately if the worker is
(re)started during the session), restarts it if it dies, and posts a Discord warning when it does.  It also reminds you in Discord when the
options-ratio file (refreshed weekly from ThetaData on the Mac) is getting old.

Disk layout (persistent disk mounted at DATA_DIR, default /data), linked into the repo folder at start-up:
  /data/live             -> ./live              screener ledger (posted.json, signal_log.csv) + cached 15m bars
  /data/lse_ohlcv_cache  -> ./lse_ohlcv_cache   last 15m LSE file per ticker
  /data/xs_ext           -> ./xs/ext            optvol_ratio.csv (weekly from the Mac) and gex/QQQ.csv
Env: DISCORD_TOKEN, VECTOR_CHANNEL_ID, LSE_API_KEY (secrets), MID_BOOK=0, DATA_DIR, optional OPTVOL_MAX_AGE_DAYS (default 10),
     SCREENER_ARGS (default --loop), FORCE_START=1 / SERVICE_ONCE=1 (testing).
"""
import datetime as dt, os, shutil, subprocess, sys, time
from zoneinfo import ZoneInfo
import requests

ET = ZoneInfo('America/New_York')
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('DATA_DIR', '/data')
ARGS = os.environ.get('SCREENER_ARGS', '--loop').split()
START_MIN, END_MIN = 8 * 60 + 50, 16 * 60 + 50              # start 08:50 ET like the launchd job; never (re)start after 16:50
MAX_AGE = int(os.environ.get('OPTVOL_MAX_AGE_DAYS', '10'))
LINKS = (('live', 'live'), ('lse_ohlcv_cache', 'lse_ohlcv_cache'), (os.path.join('xs', 'ext'), 'xs_ext'))


def log(*a): print(dt.datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S ET'), *a, flush=True)


def notify(text):
    tok, ch = os.environ.get('DISCORD_TOKEN'), os.environ.get('VECTOR_CHANNEL_ID')
    if not (tok and ch): return log('(not posted, no Discord config)', text)
    try: requests.post(f'https://discord.com/api/v10/channels/{ch}/messages', headers={'Authorization': f'Bot {tok}'}, json={'content': text}, timeout=30)
    except Exception as e: log('discord notify failed:', e)


def setup():
    for _, target in LINKS: os.makedirs(os.path.join(DATA, target), exist_ok=True)
    os.makedirs(os.path.join(DATA, 'xs_ext', 'gex'), exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'xs'), exist_ok=True)
    for name, target in LINKS:
        link = os.path.join(ROOT, name)
        if os.path.islink(link): continue
        if os.path.isdir(link) and not os.listdir(link): os.rmdir(link)
        if os.path.exists(link): raise SystemExit(f'{link} exists and is not empty - cannot link it to the persistent disk')
        os.symlink(os.path.join(DATA, target), link)
    log('disk layout ready under', DATA)


def optvol_age_days():
    f = os.path.join(ROOT, 'xs', 'ext', 'optvol_ratio.csv')
    if not os.path.exists(f): return None, None
    import pandas as pd
    last = pd.read_csv(f, usecols=['date'], parse_dates=['date']).date.max()
    return (dt.datetime.now(ET).replace(tzinfo=None) - last.to_pydatetime()).days, last.date()


def in_window(n): return n.weekday() < 5 and START_MIN <= n.hour * 60 + n.minute < END_MIN


def next_start(n):
    c = n.replace(hour=START_MIN // 60, minute=START_MIN % 60, second=0, microsecond=0)
    while c <= n or c.weekday() >= 5: c = (c + dt.timedelta(days=1)).replace(hour=START_MIN // 60, minute=START_MIN % 60)
    return c


def main():
    setup(); log('screener worker up; args', ARGS)
    done, warned = None, None
    while True:
        now = dt.datetime.now(ET)
        if not (in_window(now) or os.environ.get('FORCE_START')) or done == now.date():
            nxt = next_start(now); log('idle until', nxt.strftime('%a %Y-%m-%d %H:%M ET')); time.sleep(max(30, min(1800, (nxt - now).total_seconds()))); continue
        age, last = optvol_age_days()
        if age is None: notify('⚠️ screener: xs/ext/optvol_ratio.csv is missing on the Render disk - run the weekly sync (tools/sync_optvol_to_render.sh).')
        elif age > MAX_AGE and warned != now.date(): notify(f'⚠️ screener: the options-ratio file is {age} days old (last row {last}). Run the weekly ThetaData refresh and tools/sync_optvol_to_render.sh.'); warned = now.date()
        restarts = 0
        while restarts < 8:
            t0 = time.time(); log('starting live_screener.py', ' '.join(ARGS))
            rc = subprocess.run([sys.executable, '-u', 'live_screener.py'] + ARGS, cwd=ROOT).returncode
            if rc == 0: log('screener finished for the day'); break
            restarts += 1; notify(f'⚠️ screener stopped (exit {rc}) after {(time.time() - t0) / 60:.0f} min - restarting ({restarts}/8)'); time.sleep(30)
        else: notify('🛑 screener crashed 8 times today - not restarting until tomorrow. Check the Render logs.')
        done = dt.datetime.now(ET).date()
        if os.environ.get('SERVICE_ONCE'): return


if __name__ == '__main__':
    main()
