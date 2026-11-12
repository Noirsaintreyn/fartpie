"""DOES THE VECTOR (pivot-ratchet, the live 'SVM - Fractals' port xs/vector_pine.py) HAVE PREDICTIVE EDGE VS RANDOM, ESPECIALLY ON HIGH-VOLATILITY DAYS? Fixed indicator, nothing fitted. DECLARED BEFORE RESULTS.
Universe A: 10 stocks drawn at random (seed 11), stratified 4 large / 3 mid / 3 small cap from a hand-made pool spanning sectors; Yahoo bars (1h: Dec 2023+, 15m: last 60 days only). Universe B: the 8 mega-cap names with long 15m history (lse_clean, 2019-26), 15m and 1h (resampled like the live 1h track), split 2019-21 / 2022+.
Bars: 04:00-16:00 ET (premarket + RTH, as live). State at bar close t-1 -> next bar's open-to-close return. TEST 1 'per-bar edge' = mean(sign(state) x ret) in bp vs 1000 circular shifts of the state series (keeps persistence and long/short share, so drift is controlled).
TEST 2 'flip trades': enter next open after a flip, exit next open after the opposite flip (long on bull flip, short on bear flip), mean return per trade vs 1000 random entries of the same direction and holding length on the same ticker.
Day classes: ALL | HV-pre = previous day's range / its trailing 20-day mean is in the top third of the ticker's history so far (known at the open, tradable) | HV-post = today's range in the top third (known only after the fact; look-ahead, shown for reference)."""
import os, sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
from vector_pine import run_state_machine
import yfinance as yf
rng = np.random.default_rng(11)
POOL = {'large': ['JPM', 'XOM', 'UNH', 'WMT', 'HD', 'CAT', 'LIN', 'NEE', 'T', 'PLD', 'KO', 'GE'], 'mid': ['ETSY', 'WSM', 'DKNG', 'ROKU', 'CROX', 'FIVE', 'CHWY', 'MTCH', 'LYFT', 'BBWI'], 'small': ['PLUG', 'UPST', 'CLOV', 'TLRY', 'BB', 'SOFI', 'FUBO', 'OPEN']}
SECT = dict(JPM='Fin', XOM='Energy', UNH='Health', WMT='Staples', HD='Disc', CAT='Indus', LIN='Mat', NEE='Util', T='Comm', PLD='REIT', KO='Staples', GE='Indus', ETSY='Disc', WSM='Disc', DKNG='Disc', ROKU='Comm', CROX='Disc', FIVE='Disc', CHWY='Disc', MTCH='Comm', LYFT='Indus', BBWI='Disc', PLUG='Energy', UPST='Fin', CLOV='Health', TLRY='Health', BB='Tech', SOFI='Fin', FUBO='Comm', OPEN='RE')
PICK = []
for k, n in (('large', 4), ('mid', 3), ('small', 3)): PICK += list(rng.choice(POOL[k], n, replace=False))
print('Universe A (random, stratified):', ', '.join(f'{t}({SECT[t]})' for t in PICK), flush=True)
def clean(df):
    df = df.copy(); df.index = pd.to_datetime(df.index); df.index = df.index.tz_convert('America/New_York').tz_localize(None) if df.index.tz is not None else df.index
    df.columns = [str(c[0]).lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]; df = df[['open', 'high', 'low', 'close']].dropna()
    m = df.index.hour * 60 + df.index.minute; return df[(m >= (570 if os.environ.get('RTH_ONLY') == '1' else 240)) & (m < 960)]
def yf_bars(tk, interval, period):
    d = yf.download(tk, period=period, interval=interval, prepost=(os.environ.get('RTH_ONLY') != '1'), progress=False, auto_adjust=False); return clean(d) if len(d) else None
def local_bars(tk):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_2018-01-01_*.parquet')[0]; H = pd.read_parquet(f, columns=['datetime', 'open', 'high', 'low', 'close'])
    H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); return clean(H.drop(columns='datetime')).sort_index()
def to1h(B): return B.resample('60min', origin='start_day', offset='30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
def prep(B):
    B = B[~B.index.duplicated()].sort_index(); reg, _, _, flips = run_state_machine(B.assign(datetime=B.index)); n = len(B)
    day = B.index.normalize(); m = B.index.hour * 60 + B.index.minute; rth = (m >= 570) & (m < 960)
    d = pd.DataFrame({'h': B.high.where(rth).groupby(day).max(), 'l': B.low.where(rth).groupby(day).min()}).dropna(); rg = d.h - d.l
    pre = (rg.shift() / rg.shift().rolling(20).mean().shift()); pre_rk = pre.expanding(min_periods=60).apply(lambda s: (s.iloc[:-1] < s.iloc[-1]).mean(), raw=False)
    post = rg / rg.shift().rolling(20).mean(); post_rk = post.expanding(min_periods=60).apply(lambda s: (s.iloc[:-1] < s.iloc[-1]).mean(), raw=False)
    cls = {'ALL': np.ones(n, bool), 'HV-pre': (pre_rk.reindex(day) >= 2 / 3).fillna(False).values, 'HV-post': (post_rk.reindex(day) >= 2 / 3).fillna(False).values}
    o, c = B.open.values, B.close.values; r = c / o - 1; s = np.r_[0, reg[:-1]].astype(float)          # state at t-1 close drives bar t (only RTH bars scored)
    return dict(B=B, reg=reg, s=s, r=r, rth=rth, cls=cls, flips=flips, yr=B.index.year.values, o=o, n=n, day=day)
def edge_null(P, cls, mask_extra=None, nsh=1000):
    M = P['cls'][cls] & P['rth'] & (np.ones(P['n'], bool) if mask_extra is None else mask_extra); s, r = P['s'], P['r']; k = M.sum()
    if k < 30: return None
    obs = (s * r)[M].mean() * 1e4; idx = np.where(M)[0]; nul = []
    for _ in range(nsh):
        sh = rng.integers(60, P['n'] - 60); nul.append((np.roll(s, sh) * r)[idx].mean() * 1e4)
    nul = np.array(nul); return dict(n=int(k), obs=obs, null_mean=nul.mean(), p=(nul >= obs).mean(), z=(obs - nul.mean()) / (nul.std() + 1e-12), nul=nul, w=k)
def trades(P):
    reg = P['reg']; o = P['o']; n = P['n']; ch = [i for i in range(1, n) if reg[i] != reg[i - 1] and reg[i] != 0]; out = []
    for a, i in enumerate(ch):
        j = ch[a + 1] if a + 1 < len(ch) else None
        if j is None or i + 1 >= n or j + 1 >= n: continue
        e, x = o[i + 1], o[j + 1]; out.append((i + 1, j + 1, int(reg[i]), reg[i] * (x / e - 1)))
    return out
def trade_null(P, cls, nsh=500):
    T = [t for t in trades(P) if P['cls'][cls][t[0]]]
    if len(T) < 8: return None
    obs = np.mean([t[3] for t in T]) * 1e4; o = P['o']; n = P['n']; cand = np.where(P['cls'][cls] & P['rth'])[0]; nul = []
    for _ in range(nsh):
        v = []
        for (a, b, d, rr) in T:
            L = b - a; st = rng.choice(cand); st = min(st, n - L - 1); v.append(d * (o[st + L] / o[st] - 1))
        nul.append(np.mean(v) * 1e4)
    nul = np.array(nul); return dict(n=len(T), obs=obs, null_mean=nul.mean(), p=(nul >= obs).mean())
def report(name, PS, tf):
    print(f'\n######## {name} | {tf}', flush=True)
    for cls in ('ALL', 'HV-pre', 'HV-post'):
        res = [(tk, edge_null(P, cls)) for tk, P in PS.items()]; res = [(tk, x) for tk, x in res if x]
        if not res: continue
        W = np.array([x['w'] for _, x in res], float); obs = np.average([x['obs'] for _, x in res], weights=W)
        pooled = np.average(np.vstack([x['nul'] for _, x in res]), axis=0, weights=W); p = (pooled >= obs).mean()
        pos = sum(x['p'] < .1 for _, x in res)
        tr = [(tk, trade_null(P, cls)) for tk, P in PS.items()]; tr = [(tk, x) for tk, x in tr if x]
        tn = sum(x['n'] for _, x in tr); tobs = np.average([x['obs'] for _, x in tr], weights=[x['n'] for _, x in tr]) if tr else np.nan
        tnul = np.average([x['null_mean'] for _, x in tr], weights=[x['n'] for _, x in tr]) if tr else np.nan; tp = np.average([x['p'] for _, x in tr], weights=[x['n'] for _, x in tr]) if tr else np.nan
        print(f'  {cls:8s} per-bar edge {obs:+6.2f} bp (random-state mean {pooled.mean():+.2f}) pooled p={p:.3f} | tickers with p<.10: {pos}/{len(res)} | flip trades n {tn}: avg {tobs:+.0f} bp vs random-entry {tnul:+.0f} bp (avg p {tp:.2f})', flush=True)
        if cls == 'ALL' or True: print('       ' + ' '.join(f'{tk}:{x["obs"]:+.1f}' for tk, x in res), flush=True)
# ---------------- Universe A: random 10, Yahoo
for tf, interval, period in (('1h (Dec 2023+)', '1h', '730d'), ('15m (last 60 days, small sample)', '15m', '60d')):
    PS = {}
    for tk in PICK:
        B = yf_bars(tk, interval, period)
        if B is None or len(B) < 300: print('  skip', tk); continue
        PS[tk] = prep(B)
    report('UNIVERSE A: 10 random stocks', PS, tf)
# ---------------- Universe B: local mega-caps, long history, by era
B15 = {tk: local_bars(tk) for tk in ('AAPL', 'AMZN', 'AVGO', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA')}
for tf in ('15m', '1h'):
    PS = {tk: prep(B if tf == '15m' else to1h(B)) for tk, B in B15.items()}
    report('UNIVERSE B: 8 mega-caps 2019-26 (all years)', PS, tf)
    for era, f in (('2019-21', lambda y: y <= 2021), ('2022+', lambda y: y >= 2022)):
        print(f'\n######## UNIVERSE B | {tf} | {era}', flush=True)
        for cls in ('ALL', 'HV-pre', 'HV-post'):
            res = [(tk, edge_null(P, cls, f(P['yr']))) for tk, P in PS.items()]; res = [(tk, x) for tk, x in res if x]
            W = np.array([x['w'] for _, x in res], float); obs = np.average([x['obs'] for _, x in res], weights=W); pooled = np.average(np.vstack([x['nul'] for _, x in res]), axis=0, weights=W)
            print(f'  {cls:8s} per-bar edge {obs:+6.2f} bp (random-state mean {pooled.mean():+.2f}) pooled p={(pooled>=obs).mean():.3f} | tickers p<.10: {sum(x["p"]<.1 for _,x in res)}/{len(res)}', flush=True)
