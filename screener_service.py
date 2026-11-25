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
    tok, ch = os.environ.get('DISCORD_TOKEN'), os.environ.get('LARGECAP_CHANNEL_ID') or '1570332365186793522'      # #large-cap (the old VECTOR_CHANNEL_ID channel was deleted)
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
        last, sent_at, made, err_at, ext_at = None, 0, 0, 0, 0
        while True:
            try:
                stale = not os.path.exists(f) or time.time() - os.path.getmtime(f) > 1800
                if stale and not SCREENER_RUNNING['on'] and time.time() - made > 1800:
                    made = time.time(); log('building a site snapshot (screener not running)')
                    rc, tail = run_tail([sys.executable, '-u', 'live_screener.py', '--site-snapshot'])
                    if rc != 0 and time.time() - err_at > 6 * 3600:
                        err_at = time.time(); notify(f'⚠️ website snapshot failed (exit {rc}):\n```{tail}```')
                n = dt.datetime.now(ET); nm = n.hour * 60 + n.minute          # Oct 28 2026: the screener exits at 16:10, so after-hours / premarket prices come from here
                if not SCREENER_RUNNING['on'] and n.weekday() < 5 and (240 <= nm < 570 or 965 <= nm < 1200) and time.time() - ext_at > 300 and os.path.exists(f):
                    ext_at = time.time(); rc, tail = run_tail([sys.executable, '-u', 'live_screener.py', '--ext-refresh'])
                    if rc != 0: log('ext refresh failed', tail[-200:])
                if os.path.exists(f):
                    m = os.path.getmtime(f)
                    if m != last or time.time() - sent_at > 120:          # re-send often: a website redeploy forgets the pushed copy
                        r = requests.post(f'{url}/api/site/push', data=open(f, 'rb').read(), headers={'X-Site-Token': tok, 'Content-Type': 'application/json'}, timeout=30)
                        if r.ok: last, sent_at = m, time.time()
                        else: log('site push', r.status_code, r.text[:80])
            except Exception as e: log('site push failed', str(e)[:80])
            time.sleep(30)
    threading.Thread(target=loop, daemon=True).start()


def run_midsmall():
    """Nov 12 2026: the mid/small-cap ($1-2) screener runs in this same worker (no second paid service): started 09:25 ET on weekdays, it posts to #mid-small-cap and exits itself at the close; restarted if it crashes inside the session."""
    import threading
    args = os.environ.get('MIDSMALL_ARGS', '--loop').split()
    def loop():
        while True:
            now = dt.datetime.now(ET); mins = now.hour * 60 + now.minute
            if os.environ.get('MIDSMALL_OFF') == '1' or now.weekday() >= 5 or not (9 * 60 + 25 <= mins < 16 * 60 + 5) and not os.environ.get('FORCE_START'):
                nxt = (now + dt.timedelta(days=1)).replace(hour=9, minute=25, second=0, microsecond=0) if (now.weekday() >= 5 or mins >= 16 * 60 + 5) else now.replace(hour=9, minute=25, second=0, microsecond=0)
                while nxt.weekday() >= 5: nxt += dt.timedelta(days=1)
                time.sleep(max(60, min(1800, (nxt - now).total_seconds()))); continue
            log('starting midsmall_screener.py', *args); t0 = time.time(); rc, tail = run_tail([sys.executable, '-u', 'midsmall_screener.py'] + args)
            log(f'midsmall screener exited ({rc}) after {time.time() - t0:.0f}s')
            if rc != 0 and dt.datetime.now(ET).hour < 16: notify(f'⚠️ mid/small-cap screener stopped (exit {rc}); restarting in 60s:\n```{tail[-600:]}```')
            if os.environ.get('SERVICE_ONCE'): return
            time.sleep(60 if rc != 0 else 120)
    threading.Thread(target=loop, daemon=True).start()


def webull_env():
    """Nov 24 2026: Render holds the Webull credentials in ONE variable named 'webull_api'; the Webull code reads WEBULL_APP_KEY and WEBULL_APP_SECRET. If those two are missing, split 'webull_api' into them
    (accepts JSON {"app_key":..,"app_secret":..}, 'KEY=..,SECRET=..' pairs, or 'key:secret' / 'key,secret' / 'key secret' / two lines). Child processes inherit the result. Never prints the values."""
    import json as _json, re
    if os.environ.get('WEBULL_APP_KEY') and os.environ.get('WEBULL_APP_SECRET'): return
    raw = (os.environ.get('webull_api') or os.environ.get('WEBULL_API') or '').strip()
    if not raw: return
    k = sc = None
    try:
        j = _json.loads(raw)
        if isinstance(j, dict):
            for a, b in j.items():
                a = str(a).lower().replace('_', '')
                if 'secret' in a: sc = str(b)
                elif 'key' in a or a in ('appid', 'id'): k = str(b)
    except ValueError:
        pairs = dict((p.group(1).lower().replace('_', ''), p.group(2)) for p in re.finditer(r'([A-Za-z_]+)\s*[=:]\s*([^\s,;|]+)', raw))
        if any('secret' in a for a in pairs): sc = next(v for a, v in pairs.items() if 'secret' in a); k = next((v for a, v in pairs.items() if 'key' in a and 'secret' not in a), None)
        else:
            parts = [t for t in re.split(r'[\s,;:|]+', raw) if t]
            if len(parts) == 2: k, sc = parts
            else: return log(f"webull_api found but has {len(parts)} part(s); need an app key AND an app secret (WEBULL_APP_KEY / WEBULL_APP_SECRET)")
    if k and sc: os.environ['WEBULL_APP_KEY'], os.environ['WEBULL_APP_SECRET'] = k, sc; log('Webull credentials loaded from the webull_api variable')
    else: log('webull_api found but could not find both an app key and an app secret in it')


GAMMA_SYMS = os.environ.get('GAMMA_SYMBOLS', 'QQQ,TSLA,NVDA,AAPL,AMD,AMZN,GOOGL,MSFT,META,COST,ADBE,CAT,CRM,DIS,HD,IBM,JPM,MCD,ORCL,SBUX,TGT,UNH,UPS,WMT')   # 24 names: QQQ + the 23 signal-model names (option_models.py)


PCV_EXTRA = os.environ.get('PCV_EXTRA_SYMBOLS', 'ABNB,APP,AVGO,BA,BKNG,C,CMG,COIN,CRWD,CVNA,DLTR,ENPH,FDX,GEV,GS,HOOD,LLY,LMT,LRCX,LULU,MU,NFLX,OXY,PANW,PENN,PLTR,PYPL,QCOM,SMCI,UBER,WYNN')   # Nov 24 2026: 31 more live-universe names for the put/call volume signal (chain pull only, no heatmap)
def run_gamma_job():
    """Nov 24 2026: keep live/neg_gamma.json fresh for the negative-gamma Discord channel. Read-only Webull market data (never orders).
    16:15 ET: log the option chain, then build net gamma (gex_heatmap.py -> live/neg_gamma.json). 08:30 ET: re-pull for the prior-close open interest and rebuild.
    Needs WEBULL_APP_KEY + WEBULL_APP_SECRET in the environment; without them it logs once and stays off. Never raises into the screener."""
    import threading
    if not (os.environ.get('WEBULL_APP_KEY') and os.environ.get('WEBULL_APP_SECRET')): return log('gamma job off (WEBULL_APP_KEY / WEBULL_APP_SECRET not set)')
    def once(tag, oi):
        cmd = [sys.executable, '-u', 'webull_options_log.py', '--symbols', GAMMA_SYMS + ('' if oi or not PCV_EXTRA else ',' + PCV_EXTRA)] + (['--oi-refresh'] if oi else [])
        rc, tail = run_tail(cmd)
        if rc != 0: return notify(f'⚠️ gamma job ({tag}): option chain pull failed (exit {rc})\n```{tail[-500:]}```')
        rc, tail = run_tail([sys.executable, '-u', 'gex_heatmap.py', '--symbols', GAMMA_SYMS])
        log(f'gamma job ({tag}) done, exit {rc}')
        if not oi:                                                       # close run only: the 08:30 refresh has pre-market volume = 0
            rc2, tail2 = run_tail([sys.executable, '-u', 'option_models.py', '--update']); log('option models update:', tail2.strip().splitlines()[-1][:120] if tail2.strip() else rc2)
        if rc != 0: notify(f'⚠️ gamma job ({tag}): gamma build failed (exit {rc})\n```{tail[-500:]}```')
    def loop():
        done = set()
        while True:
            try:
                n = dt.datetime.now(ET); nm = n.hour * 60 + n.minute
                if n.weekday() < 5:
                    for tag, lo, hi, oi in (('close', 16 * 60 + 15, 18 * 60, False), ('oi', 8 * 60 + 30, 9 * 60 + 20, True)):
                        key = (n.date(), tag)
                        if lo <= nm < hi and key not in done: done.add(key); once(tag, oi)
            except Exception as e: log('gamma job error', str(e)[:100])
            time.sleep(60)
    threading.Thread(target=loop, daemon=True).start(); log('gamma job on:', GAMMA_SYMS)


def run_daily_small():
    """Nov 25 2026: sub-$5 DAILY vector flip alert (daily_small.py), weekdays once after the close (16:20-18:00 ET). Posts to DAILYSMALL_CHANNEL_ID (falls back to the mid/small channel). Never raises into the worker; DAILYSMALL_OFF=1 disables it."""
    import threading
    if os.environ.get('DAILYSMALL_OFF') == '1': return log('daily small job off (DAILYSMALL_OFF=1)')
    def loop():
        done = set()
        while True:
            try:
                n = dt.datetime.now(ET); nm = n.hour * 60 + n.minute
                if n.weekday() < 5 and 16 * 60 + 20 <= nm < 18 * 60 and n.date() not in done:
                    done.add(n.date()); import daily_small; fired, exits = daily_small.scan(); log(f'daily small job: {len(fired)} alerts, {len(exits)} exits')
            except Exception as e: log('daily small job error', str(e)[:120]); notify(f'\u26a0\ufe0f daily small job failed: {str(e)[:300]}')
            time.sleep(60)
    threading.Thread(target=loop, daemon=True).start(); log('daily small job on')


def main():
    setup(); webull_env(); log('screener worker up; args', ARGS)
    run_midsmall()
    if os.environ.get('RUN_BOT') == '1': run_bot()
    run_site_push(); run_gamma_job(); run_daily_small()
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
