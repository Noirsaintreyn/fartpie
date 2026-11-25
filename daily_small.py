"""Nov 25 2026: DAILY vector flip alert for SUB-$5 names (Rey: 'only for the small cap sub 5'). After the close (16:20 ET job in screener_service, or `python daily_small.py`):
 1. daily bars (Yahoo) for the sub-$5 universe (midsmall_universe.txt), 20d median dollar volume >= $500k, last close < $5;
 2. fresh DAILY bull flip of the vector state machine on today's bar;
 3. base5 score (mean ECDF rank of ATRP, F11, VOLX, LTURN, -LMC; frozen train cuts in daily_base5_ref.json) >= the frozen 70th-percentile cut = the 'high-vol group' of reports 477-483;
 4. alert: 5m chart (daily vector regime colored), distance to the vector level (sizing input, report 480), backtest note, catalyst/news/insider report signed Valiant. Exit alert when the daily vector flips bear on a held name.
Shares outstanding (LTURN, LMC) come from SEC cover pages (dei:EntityCommonStockSharesOutstanding), cached in live/shares.json; a name without shares is scored on the other features. Discord: DAILYSMALL_CHANNEL_ID (falls back to the mid/small channel)."""
import json, os, sys, time, argparse, warnings
import numpy as np, pandas as pd, requests
warnings.filterwarnings('ignore'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'xs'))
try:
    from dotenv import load_dotenv; load_dotenv('.env'); load_dotenv('discord_bot/.env')
except Exception: pass
from vector_pine import run_state_machine
ET = 'America/New_York'; LIVE = os.path.join(HERE, 'live'); os.makedirs(LIVE, exist_ok=True); STATE_F = f'{LIVE}/daily_small_state.json'; SH_F = f'{LIVE}/shares.json'
REF = json.load(open(os.path.join(HERE, 'daily_base5_ref.json'))); QS = np.linspace(0, 1, 101); H = {'User-Agent': 'research rey@example.com'}
MAX_PX = 5.0; MIN_DV = 5e5; SLOT = 0.125
def now_et(): return pd.Timestamp.now(tz=ET).tz_localize(None)
def universe():
    f = os.path.join(HERE, 'midsmall_universe.txt'); return [l.strip().upper() for l in open(f) if l.strip() and not l.startswith('#')]
def score(F):
    R = []
    for f, sg in REF['signs'].items():
        x = F.get(f); r = np.nan if x is None or x != x else float(np.interp(x, REF['ref'][f], QS)); R.append(r if sg > 0 else 1 - r)
    R = [x for x in R if x == x]; return float(np.mean(R)) if R else np.nan
def shares(tk, cache):
    if tk in cache: return cache[tk]
    try:
        import catalysts as C; c = C._cik(tk); j = requests.get(f'https://data.sec.gov/api/xbrl/companyconcept/CIK{c:010d}/dei/EntityCommonStockSharesOutstanding.json', headers=H, timeout=25).json()
        u = j['units']['shares']; u = sorted(u, key=lambda z: z['filed'])[-1]; cache[tk] = float(u['val'])
    except Exception: cache[tk] = None
    return cache[tk]
def get_daily(tks):
    import yfinance as yf; out = {}
    for i in range(0, len(tks), 80):
        ch = tks[i:i + 80]
        try:
            D = yf.download(ch, period='2y', interval='1d', group_by='ticker', progress=False, auto_adjust=False, threads=False)
            for tk in ch:
                try:
                    q = (D[tk] if len(ch) > 1 else D).dropna(subset=['Close']).rename(columns=str.lower)[['open', 'high', 'low', 'close', 'volume']]; q.index = pd.DatetimeIndex(pd.to_datetime(q.index).date)
                    if len(q) > 150: out[tk] = q[~q.index.duplicated()]
                except Exception: pass
        except Exception as e: print('  ! download', str(e)[:80])
        time.sleep(1)
    return out
def analyze(tk, q, cache, need_flip=True):
    reg, lev, atr, _ = run_state_machine(q.assign(datetime=q.index)); i = len(q) - 1; c = q.close.values; v = q.volume.values.astype(float)
    if need_flip and not (reg[i] == 1 and reg[i - 1] != 1): return None
    dv = float(pd.Series(c * v).rolling(20).median().iat[-1]); vm = float(pd.Series(v).rolling(20).mean().shift(1).iat[-1]); s = shares(tk, cache) if (need_flip and c[i] < MAX_PX) else None
    F = dict(ATRP=atr[i] / c[i], F11=(c[i] - lev[i]) / c[i] if lev[i] == lev[i] else np.nan, VOLX=v[i] / vm if vm > 0 else np.nan, LTURN=np.log(max(v[i] / s, 1e-9)) if s else np.nan, LMC=np.log(c[i] * s) if s else np.nan)
    return dict(tk=tk, day=q.index[i], px=float(c[i]), lev=float(lev[i]), prev=float(lev[i - 1]) if lev[i - 1] == lev[i - 1] else np.nan, reg=int(reg[i]), dv=dv, F=F, score=score(F), shares=s, age=None)
def load_state():
    try: return json.load(open(STATE_F))
    except Exception: return {'pos': {}, 'seen': []}
def save_state(st): st['seen'] = st['seen'][-500:]; json.dump(st, open(STATE_F, 'w'), default=str)
def _target():
    return os.getenv('DAILYSMALL_CHANNEL_ID') or os.getenv('MIDSMALL_CHANNEL_ID') or '1570332394714562681', os.getenv('DISCORD_TOKEN'), os.getenv('DAILYSMALL_WEBHOOK_URL') or os.getenv('MIDSMALL_WEBHOOK_URL')
def post(text, path=None):
    ch, tok, hook = _target(); parts = []; cur = ''
    for ln in text.split('\n'):
        if len(cur) + len(ln) + 1 > 1850 and cur: parts.append(cur); cur = ''
        cur += ln + '\n'
    parts.append(cur.rstrip('\n')); ok = True
    for k, p in enumerate(parts):
        try:
            if path and k == 0:
                with open(path, 'rb') as fh:
                    d = {'payload_json': json.dumps({'content': p})}; fl = {'files[0]': (os.path.basename(path), fh, 'image/png')}
                    r = requests.post(f'https://discord.com/api/v10/channels/{ch}/messages', headers={'Authorization': f'Bot {tok}'}, data=d, files=fl, timeout=40) if (ch and tok) else requests.post(hook, data=d, files=fl, timeout=40)
            else: r = requests.post(f'https://discord.com/api/v10/channels/{ch}/messages', headers={'Authorization': f'Bot {tok}'}, json={'content': p}, timeout=20) if (ch and tok) else requests.post(hook, json={'content': p}, timeout=20)
            r.raise_for_status()
        except Exception as e: print('  ! discord post failed:', str(e)[:100]); ok = False
    return ok
def alert_text(a):
    F = a['F']; bt = REF['bt']; risk = F['F11']; size = SLOT * float(np.clip(0.10 / risk, 0.3, 1.0)) if risk == risk and risk > 0 else SLOT
    return (f"\U0001f4c8 DAILY BUY {a['tk']} (sub-$5 high-vol group) · daily vector bull flip on the {a['day']:%a %b %d} close {a['px']:.2f}\n"
            f"Entry next open. Exit = the DAILY vector's bear flip (no other stop). Broke above {a['prev']:.2f}; the new vector level is {a['lev']:.2f} ({risk:.0%} below price = what a failed flip costs; risk sizing input).\n"
            f"Size: ~{size:.1%} of the book (1/8 slot = {SLOT:.1%}, scaled by 10%/distance-to-level, floor 30%).\n"
            f"Score {a['score']:.2f} (cut {REF['cut']:.2f}) · ATRP {F['ATRP']:.1%} · volume {F['VOLX']:.1f}x" + (f" · mcap ${np.exp(F['LMC'])/1e6:,.0f}M" if F['LMC'] == F['LMC'] else '') + "\n"
            f"Backtest (sub-$5 high-vol group, 2022+): {bt['reach']:.0%} of these flips reach +10% at some point, usually within about 3 trading days (median); the rest lose. Not a promise.")
def scan(dry=False, asof_check=True):
    st = load_state(); cache = {}
    try: cache = json.load(open(SH_F))
    except Exception: pass
    tks = universe(); D = get_daily(tks); today = now_et().normalize(); fired = []; exits = []
    for tk, q in D.items():
        if asof_check and q.index[-1] != today and not dry: continue                                        # only today's completed bar
        try:
            if tk in st['pos']:
                a = analyze(tk, q, cache, need_flip=False)
                if a and a['reg'] != 1: exits.append((tk, a, st['pos'][tk]))
                continue
            if q.close.iat[-1] >= MAX_PX: continue
            a = analyze(tk, q, cache)
            if a is None or a['dv'] < MIN_DV: continue
            key = f"{tk}|{a['day']:%Y-%m-%d}"
            if key in st['seen']: continue
            st['seen'].append(key)
            if a['score'] == a['score'] and a['score'] >= REF['cut']: fired.append(a)
            else: print(f"  skipped {tk}: daily flip, score {a['score']:.2f} < cut {REF['cut']:.2f}")
        except Exception as e: print(f'  ! {tk}: {str(e)[:70]}')
    json.dump(cache, open(SH_F, 'w'))
    for tk, a, p in exits:
        t = f"\U0001f534 DAILY SELL {tk}: the daily vector flipped bear on the {a['day']:%a %b %d} close {a['px']:.2f}. Exit next open. In from {p['px']:.2f} ({p['day'][:10]}): {a['px']/p['px']-1:+.1%} on the close (before costs).\n" + __import__('catalysts').SIGN
        print(t); (None if dry else post(t)); st['pos'].pop(tk, None)
    for a in fired:
        t = alert_text(a); path = os.path.join(LIVE, f"d5m_{a['tk']}.png")
        try:
            import chart5m; chart5m.make_chart_5m(a['tk'], path, flip_day=a['day'], info={'DAILY flip': f"{a['day']:%b %d}", 'level': f"{a['lev']:.2f}"})
        except Exception as e: print('  ! chart', str(e)[:80]); path = None
        try:
            import catalysts as C; rep = C.report(a['tk'], asof=a['day'])
        except Exception as e: rep = ''; print('  ! catalyst report', str(e)[:60])
        full = t + ('\n\n' + rep if rep else ''); print(full); (None if dry else post(full, path)); st['pos'][a['tk']] = dict(px=a['px'], day=str(a['day']), lev=a['lev'])
    print(f'daily_small: {len(D)} names, {len(fired)} alerts, {len(exits)} exits')
    if not dry: save_state(st)
    return fired, exits
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--dry', action='store_true'); ap.add_argument('--any-day', action='store_true', help='ignore the bar-is-today check (testing)'); a = ap.parse_args(); scan(a.dry, asof_check=not a.any_day)
