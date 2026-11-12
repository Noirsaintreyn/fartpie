"""MID/SMALL-CAP GROWTH SCREENER (second Discord screener, separate from live_screener.py = the original mega-cap book).
Rules (found + tested in xs/report_lowvol_combo.py / report_volcheck.py / report_smallcap_book.py, variant C):
  Bars      1h regular-hours bars from Yahoo (no premarket), vector = the same pivot-ratchet state machine as the original screener (xs/vector_pine.py)
  Entry     1h bar closes with a BULL FLIP, AND that bar's volume >= VOL_X (1.5) x the mean of the prior 20 bars, AND the previous day's range is in the
            bottom third of the stock's own history (low-vol day, expanding rank). Entry = next open (alert fires at the bar close).
  Exit      the vector's own bear flip on a 1h close (sell next open). No other stop / target (Rey's rule for flip strategies).
  Slots     SLOTS (8), equal weight, first come first served, higher flip-volume ratio first when several fire on the same bar; skipped = honorable mention (logged).
  Long only. Universe: midsmall_universe.txt (one ticker per line) or the built-in 44-name list.
  Backtest (Dec 2023-Nov 2026, 15 bp round trip, 8 slots): ~+41%/yr, max DD -25%, ~120 trades/yr, 27% win rate, mean hold 10 days / median 6 days;
  2024 +62%, 2025 +97%, 2026 YTD -17%. The universe was picked in hindsight, the filter was chosen on the same data: treat as unproven, paper-trade first.
Usage:
  python3 midsmall_screener.py --once --dry       scan the latest closed 1h bar, print what would be posted
  python3 midsmall_screener.py --replay 60        replay the last 60 days of signals (sanity check, nothing posted)
  python3 midsmall_screener.py --loop             weekday service: scans seconds after every 1h close (10:30 ... 16:00 ET) and posts to Discord
Discord (.env): DISCORD_TOKEN + MIDSMALL_CHANNEL_ID (bot), or MIDSMALL_WEBHOOK_URL (webhook)."""
import argparse, json, os, sys, time, warnings
import numpy as np, pandas as pd, requests
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xs'))
try:
    from dotenv import load_dotenv; load_dotenv('.env'); load_dotenv('discord_bot/.env')
except Exception: pass
from vector_pine import run_state_machine
ET = 'America/New_York'; LIVE = 'live'; os.makedirs(LIVE, exist_ok=True)
STATE_F = f'{LIVE}/midsmall_state.json'; SLOTS = int(os.getenv('MIDSMALL_SLOTS', '8')); VOL_X = float(os.getenv('MIDSMALL_VOL_X', '1.5'))
DEFAULT = ['KGS', 'LCID', 'RIOT', 'Z', 'IREN', 'PLUG', 'UPST', 'CLOV', 'TLRY', 'BB', 'SOFI', 'FUBO', 'OPEN', 'ETSY', 'DKNG', 'ROKU', 'CROX', 'FIVE', 'CHWY', 'MTCH', 'LYFT', 'BBWI',
           'RIVN', 'SNAP', 'PINS', 'HOOD', 'AFRM', 'RKLB', 'SOUN', 'MARA', 'CLSK', 'HIMS', 'CELH', 'JOBY', 'ACHR', 'PATH', 'NIO', 'LI', 'W', 'CHPT', 'RUN', 'ENPH', 'SEDG', 'LUMN']
def universe():
    f = 'midsmall_universe.txt'
    return [l.strip().upper() for l in open(f) if l.strip() and not l.startswith('#')] if os.path.exists(f) else DEFAULT
def now_et(): return pd.Timestamp.now(tz=ET).tz_localize(None)

# ---------------------------------------------------------------- data (one batched Yahoo call)
def fetch_all(tks, period='730d'):
    import yfinance as yf
    t0 = time.time(); D = yf.download(tks, period=period, interval='1h', prepost=False, group_by='ticker', threads=True, progress=False, auto_adjust=False); out = {}
    for tk in tks:
        try:
            Y = (D[tk] if len(tks) > 1 else D).dropna(subset=['Close']).rename(columns=str.lower)[['open', 'high', 'low', 'close', 'volume']]
            Y.index = Y.index.tz_convert(ET).tz_localize(None); Y = Y[~Y.index.duplicated()]
            if len(Y) > 300: out[tk] = Y
        except Exception: pass
    print(f'  fetched {len(out)}/{len(tks)} tickers in {time.time() - t0:.1f}s', flush=True); return out
def closed_only(Y, t=None):
    """drop the bar still forming. 1h RTH bars start at :30 (09:30, 10:30 ... 15:30; the last is a 30-minute bar closing 16:00)."""
    t = now_et() if t is None else t; end = Y.index + pd.Timedelta('60min'); last_open = (Y.index.hour == 15) & (Y.index.minute == 30); end = end.where(~last_open, Y.index + pd.Timedelta('30min'))
    return Y[end <= t + pd.Timedelta('1s')]

# ---------------------------------------------------------------- signals
def features(Y):
    """vector state + volume ratio + low-vol-day flag for every bar (identical definitions to xs/report_lowvol_combo.py)"""
    reg = run_state_machine(Y.assign(datetime=Y.index))[0]; v = Y.volume.values.astype(float)
    vrel = v / pd.Series(v).rolling(20).mean().shift().values; day = Y.index.normalize()
    d = pd.DataFrame({'h': Y.high.groupby(day).max(), 'l': Y.low.groupby(day).min()}); rg = d.h - d.l
    ratio = rg.shift() / rg.shift().rolling(20).mean().shift(); rk = ratio.expanding(min_periods=60).apply(lambda z: (z.iloc[:-1] < z.iloc[-1]).mean(), raw=False)
    lv = (rk.reindex(day) <= 1 / 3).fillna(False).values
    return pd.DataFrame({'reg': reg, 'vrel': vrel, 'lv': lv, 'close': Y.close.values}, index=Y.index)
def events(F):
    """bull entries (flip to +1 passing the filters) and exits (flip to -1) as (bar_open_time, kind, vrel, close)"""
    ev = []; r = F.reg.values
    for i in range(1, len(F)):
        if r[i] == r[i - 1] or r[i] == 0: continue
        if r[i] == 1 and F.vrel.iat[i] >= VOL_X and F.lv.iat[i] and not np.isnan(F.vrel.iat[i]): ev.append((F.index[i], 'BUY', float(F.vrel.iat[i]), float(F.close.iat[i])))
        elif r[i] == 1: ev.append((F.index[i], 'FLIP_FILTERED', float(F.vrel.iat[i]) if not np.isnan(F.vrel.iat[i]) else np.nan, float(F.close.iat[i])))
        elif r[i] == -1: ev.append((F.index[i], 'SELL', float(F.vrel.iat[i]) if not np.isnan(F.vrel.iat[i]) else np.nan, float(F.close.iat[i])))
    return ev
def bar_close(ts):
    ts = pd.Timestamp(ts); return ts + (pd.Timedelta('30min') if (ts.hour == 15 and ts.minute == 30) else pd.Timedelta('60min'))

# ---------------------------------------------------------------- state + Discord
def load_state():
    try: return json.load(open(STATE_F))
    except Exception: return {'pos': {}, 'seen': []}
def save_state(st): st['seen'] = st['seen'][-800:]; json.dump(st, open(STATE_F, 'w'), default=str)
def post(text):
    ch, tok, hook = os.getenv('MIDSMALL_CHANNEL_ID'), os.getenv('DISCORD_TOKEN'), os.getenv('MIDSMALL_WEBHOOK_URL')
    try:
        if ch and tok: r = requests.post(f'https://discord.com/api/v10/channels/{ch}/messages', headers={'Authorization': f'Bot {tok}'}, json={'content': text}, timeout=20)
        elif hook: r = requests.post(hook, json={'content': text}, timeout=20)
        else: print('  ! no Discord target (MIDSMALL_CHANNEL_ID + DISCORD_TOKEN, or MIDSMALL_WEBHOOK_URL)'); return False
        r.raise_for_status(); return True
    except Exception as e: print(f'  ! discord post failed: {e}'); return False

def scan_once(dry, data=None, asof=None):
    tks = universe(); data = data or fetch_all(tks); st = load_state(); fired = []; t = asof
    cand = []
    for tk, Y in data.items():
        Yc = closed_only(Y, t)
        if len(Yc) < 300: continue
        F = features(Yc); last = F.index[-1]
        if (now_et() if t is None else t) - bar_close(last) > pd.Timedelta('3h') and t is None: continue          # stale feed / market closed overnight: ignore
        for (ts, kind, vr, px) in events(F):
            if ts != last: continue                                                                              # only the newest closed bar can fire
            cand.append((tk, ts, kind, vr, px))
    # exits first (free slots), then entries by flip-volume ratio
    for tk, ts, kind, vr, px in cand:
        if kind != 'SELL' or tk not in st['pos']: continue
        key = f'{tk}|SELL|{ts}'
        if key in st['seen']: continue
        p = st['pos'].pop(tk); fired.append(f"⚡ SELL {tk}: 1h vector turned bearish on the {ts:%H:%M} bar (last {px:.2f}). Exit at the next open. Entry was {p['px']:.2f} ({px / p['px'] - 1:+.1%}), held since {p['t'][:16]}.")
        st['seen'].append(key)
    for tk, ts, kind, vr, px in sorted([c for c in cand if c[2] == 'BUY'], key=lambda c: -c[3]):
        key = f'{tk}|BUY|{ts}'
        if key in st['seen'] or tk in st['pos']: continue
        if len(st['pos']) >= SLOTS:
            fired.append(f"⭐ HONORABLE MENTION {tk}: low-vol day + high-volume bull flip on the {ts:%H:%M} bar ({vr:.1f}x volume, last {px:.2f}), but all {SLOTS} slots are full."); st['seen'].append(key); continue
        st['pos'][tk] = {'px': px, 't': str(ts), 'vr': vr}; st['seen'].append(key)
        fired.append(f"⚡ BUY {tk}: 1h vector bull flip on the {ts:%H:%M} bar with {vr:.1f}x volume on a low-volatility day. Entry next open (~{px:.2f}). Exit = the vector's bear flip (no other stop). Slot {len(st['pos'])}/{SLOTS}.")
    for m in fired: print(m); (None if dry else post(m))
    if not dry: save_state(st)
    if not fired: print(f'  no new signals ({len(data)} names scanned, {len(st["pos"])} open: {", ".join(sorted(st["pos"])) or "-"})')
    return fired

def replay(days):
    """walk the last `days` days bar by bar with the same slot logic to see what the live service would have posted"""
    tks = universe(); data = fetch_all(tks); FS = {tk: features(closed_only(Y)) for tk, Y in data.items()}; t0 = now_et().normalize() - pd.Timedelta(days=days)
    evs = []
    for tk, F in FS.items():
        for ts, kind, vr, px in events(F):
            if ts >= t0: evs.append((ts, tk, kind, vr, px))
    evs.sort(key=lambda e: (e[0], -(e[3] if e[3] == e[3] else 0))); pos = {}; nb = ns = nh = 0
    for ts, tk, kind, vr, px in evs:
        if kind == 'SELL' and tk in pos: p = pos.pop(tk); ns += 1; print(f'{ts:%m-%d %H:%M} SELL {tk:5s} {px:8.2f}  ({px / p[0] - 1:+.1%} in {(ts - p[1]).days}d)')
        elif kind == 'BUY' and tk not in pos:
            if len(pos) < SLOTS: pos[tk] = (px, ts); nb += 1; print(f'{ts:%m-%d %H:%M} BUY  {tk:5s} {px:8.2f}  vol {vr:.1f}x  slot {len(pos)}/{SLOTS}')
            else: nh += 1; print(f'{ts:%m-%d %H:%M} HM   {tk:5s} {px:8.2f}  vol {vr:.1f}x  (slots full)')
    print(f'\nreplay {days}d: {nb} buys, {ns} sells, {nh} honorable mentions, {len(pos)} still open: {", ".join(sorted(pos)) or "-"}')

def wait_for_bar(target, timeout=120):
    """poll Yahoo (SPY 1h) until the bar closing at `target` is published"""
    import yfinance as yf; t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            y = yf.Ticker('SPY').history(period='5d', interval='1h', prepost=False)
            if len(y):
                ix = y.index[-1]; last = ix.tz_convert(ET).tz_localize(None) if ix.tzinfo else ix
                if bar_close(last) >= target and now_et() >= target + pd.Timedelta('3s'): return True
        except Exception: pass
        time.sleep(3)
    return False
def loop():
    print(f'midsmall screener up: {len(universe())} names, {SLOTS} slots, volume >= {VOL_X}x'); post(f'\U0001f7e2 Mid/small-cap screener online: {len(universe())} names, {SLOTS} slots (1h vector bull flips on low-vol days with >= {VOL_X}x volume; exit = bear flip).')
    while True:
        t = now_et(); mins = t.hour * 60 + t.minute
        if t.dayofweek >= 5 or mins > 16 * 60 + 5: print('market closed - exiting'); return
        marks = [10 * 60 + 30, 11 * 60 + 30, 12 * 60 + 30, 13 * 60 + 30, 14 * 60 + 30, 15 * 60 + 30, 16 * 60]
        nxt = next((m for m in marks if m > mins - 1), None)
        if nxt is None: return
        tg = t.normalize() + pd.Timedelta(minutes=nxt); time.sleep(max(0, (tg - now_et()).total_seconds()))
        wait_for_bar(tg)
        try: scan_once(False)
        except Exception as e: print(f'  ! scan failed: {e}')
        time.sleep(5)
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--once', action='store_true'); ap.add_argument('--dry', action='store_true'); ap.add_argument('--loop', action='store_true'); ap.add_argument('--replay', type=int, default=0)
    a = ap.parse_args()
    if a.replay: replay(a.replay)
    elif a.loop: loop()
    else: scan_once(a.dry)
