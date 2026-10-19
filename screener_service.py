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
    st, seed = os.path.join(DATA, 'live', 'posted.json'), os.path.join(ROOT, 'live_data', 'posted_seed.json')
    if not os.path.exists(st) and os.path.exists(seed):          # first boot: take over the Mac's 'already posted' list so nothing is re-posted
        import shutil; shutil.copy(seed, st); log('seeded posted.json from the Mac snapshot')
    log('disk layout ready under', DATA)


def optvol_age_days():
    f = os.path.join(ROOT, os.environ.get('OPTVOL_FILE', os.path.join('xs', 'ext', 'optvol_ratio.csv')))
    if not os.path.exists(f): return None, None
    import pandas as pd
    last = pd.read_csv(f, usecols=['date'], parse_dates=['date']).date.max()
    return (dt.datetime.now(ET).replace(tzinfo=None) - last.to_pydatetime()).days, last.date()


def in_window(n): return n.weekday() < 5 and START_MIN <= n.hour * 60 + n.minute < END_MIN


def next_start(n):
    c = n.replace(hour=START_MIN // 60, minute=START_MIN % 60, second=0, microsecond=0)
    while c <= n or c.weekday() >= 5: c = (c + dt.timedelta(days=1)).replace(hour=START_MIN // 60, minute=START_MIN % 60)
    return c


def run_bot():
    """Discord bot in the same worker as the screener (same disk), restarted if it exits."""
    import threading
    def loop():
        n = 0
        while True:
            t0 = time.time(); log('starting discord_bot/bot.py')
            rc = subprocess.run([sys.executable, '-u', os.path.join('discord_bot', 'bot.py')], cwd=ROOT).returncode
            n = n + 1 if time.time() - t0 < 300 else 0
            log(f'bot exited ({rc}); restarting'); time.sleep(min(600, 15 * 2 ** min(n, 5)))
    threading.Thread(target=loop, daemon=True).start()


def run_tail(cmd):
    """run cmd, stream its output to the log, return (exit code, last ~12 lines) so crashes can be reported to Discord."""
    from collections import deque
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = deque(maxlen=12)
    for line in p.stdout:
        sys.stdout.write(line); sys.stdout.flush()
        if 'possibly delisted' not in line: last.append(line.rstrip()[:180])
    return p.wait(), '\n'.join(last)[-1500:]


SCREENER_RUNNING = {'on': False}


def run_site_push():
    """Push live/site_snapshot.json to the website whenever the screener rewrites it (and every 10 min regardless)."""
    import threading, requests
    url, tok = os.environ.get('BACKEND_URL', '').rstrip('/'), os.environ.get('SITE_TOKEN', '')
    if not url or not tok: log('site push off (BACKEND_URL / SITE_TOKEN not set)'); return
    f = os.path.join(ROOT, 'live', 'site_snapshot.json')
    def loop():
        last, sent_at, made, err_at = None, 0, 0, 0
        while True:
            try:
                stale = not os.path.exists(f) or time.time() - os.path.getmtime(f) > 1800
                if stale and not SCREENER_RUNNING['on'] and time.time() - made > 1800:
                    made = time.time(); log('building a site snapshot (screener not running)')
                    rc, tail = run_tail([sys.executable, '-u', 'live_screener.py', '--site-snapshot'])
                    if rc != 0 and time.time() - err_at > 6 * 3600:
                        err_at = time.time(); notify(f'⚠️ website snapshot failed (exit {rc}):\n```{tail}```')
                if os.path.exists(f):
                    m = os.path.getmtime(f)
                    if m != last or time.time() - sent_at > 600:
                        r = requests.post(f'{url}/api/site/push', data=open(f, 'rb').read(), headers={'X-Site-Token': tok, 'Content-Type': 'application/json'}, timeout=30)
                        if r.ok: last, sent_at = m, time.time()
                        else: log('site push', r.status_code, r.text[:80])
            except Exception as e: log('site push failed', str(e)[:80])
            time.sleep(30)
    threading.Thread(target=loop, daemon=True).start()


def main():
    setup(); log('screener worker up; args', ARGS)
    if os.environ.get('RUN_BOT') == '1': run_bot()
    run_site_push()
    done, warned = None, None
    while True:
        now = dt.datetime.now(ET)
        if not (in_window(now) or os.environ.get('FORCE_START')) or done == now.date():
            nxt = next_start(now); log('idle until', nxt.strftime('%a %Y-%m-%d %H:%M ET')); time.sleep(max(30, min(1800, (nxt - now).total_seconds()))); continue
        age, last = optvol_age_days()
        if age is None: notify('⚠️ screener: the options-ratio file is missing - run tools/weekly_ratio.sh on the Mac (it commits + pushes it).')
        elif age > MAX_AGE and warned != now.date(): notify(f'⚠️ screener: the options-ratio file is {age} days old (last row {last}). Run tools/weekly_ratio.sh on the Mac.'); warned = now.date()
        restarts = 0; SCREENER_RUNNING['on'] = True
        while restarts < 8:
            t0 = time.time(); log('starting live_screener.py', ' '.join(ARGS))
            rc, tail = run_tail([sys.executable, '-u', 'live_screener.py'] + ARGS)
            if rc == 0: log('screener finished for the day'); break
            restarts += 1; notify(f'⚠️ screener stopped (exit {rc}) after {(time.time() - t0) / 60:.0f} min - restarting ({restarts}/8)' + (f'\n```{tail}```' if restarts in (1, 8) else '')); time.sleep(30)
        else: notify('🛑 screener crashed 8 times today - not restarting until tomorrow. Check the Render logs.')
        SCREENER_RUNNING['on'] = False
        done = dt.datetime.now(ET).date()
        if os.environ.get('SERVICE_ONCE'): return


if __name__ == '__main__':
    main()
