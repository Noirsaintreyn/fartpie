"""WHAT ADDS ON TOP OF 'FLIP-BAR VOLUME >= 1.5x PRIOR-20-BAR MEAN'? Ideas from the intraday-momentum / noise-area paper (Zarattini-Aziz-Barbon: bands around the open scaled by the typical time-of-day move of the last 14 days; VWAP as trend side) plus volume variants: time-of-day relative volume (vs same bar of the last 14 days), volume vs its EMA20, up-volume/down-volume ratio over the last 4 bars, signed volume delta. All fixed before results. Base = vector flip trades (both directions, vector exit, no TP) whose flip bar volume >= 1.5x its prior-20-bar mean. Each add-on is compared with random entries placed on bars that ALSO satisfy the base volume condition (same direction/hold length), so any (obs-null) beyond the volume baseline is the add-on's own information. Panels: 44 small/mid 1h halves (Dec23-24 / 2025+), 8 mega-caps 15m (train 2019-21 / test 2022+). Interesting only if the sign agrees in all four."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')

from vector_pine import run_state_machine
rng = np.random.default_rng(9); KS = (None, .25, .5, .75, 1.0, 1.5)
def clean(H):
    m = H.index.hour * 60 + H.index.minute; return H[(m >= 240) & (m < 960)]
def local(tk):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_2018-01-01_*.parquet')[0]; H = pd.read_parquet(f, columns=['datetime', 'open', 'high', 'low', 'close'])
    H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.drop(columns='datetime').dropna().sort_index(); return clean(H[~H.index.duplicated()])
def yf15(tk):
    import yfinance as yf
    d = yf.download(tk, period='60d', interval='15m', prepost=True, progress=False, auto_adjust=False); d.columns = [c[0].lower() for c in d.columns]
    d.index = d.index.tz_convert('America/New_York').tz_localize(None); d = d[['open', 'high', 'low', 'close']].dropna(); return clean(d[~d.index.duplicated()])
def sparse(a, fn):
    T = [a]; k = 1
    while 2 * k <= len(a): p = T[-1]; T.append(fn(p[:-k], p[k:])); k *= 2
    return T
def rq(T, fn, l, r):                                                           # inclusive l..r arrays
    ln = r - l + 1; k = np.floor(np.log2(ln)).astype(int); out = np.empty(len(l))
    for kk in np.unique(k): m = k == kk; out[m] = fn(T[kk][l[m]], T[kk][r[m] - (1 << kk) + 1])
    return out
def prep(B):
    n = len(B); reg = run_state_machine(B.assign(datetime=B.index))[0]; day = B.index.normalize(); m = B.index.hour * 60 + B.index.minute; rth = (m >= 570) & (m < 960)
    d = pd.DataFrame({'h': B.high.where(rth).groupby(day).max(), 'l': B.low.where(rth).groupby(day).min(), 'c': B.close.where(rth).groupby(day).last()}).dropna(); rgp = (d.h - d.l) / d.c
    adr = rgp.rolling(20).mean().shift(); ratio = rgp.shift() / rgp.shift().rolling(20).mean().shift(); rk = ratio.expanding(min_periods=60).apply(lambda z: (z.iloc[:-1] < z.iloc[-1]).mean(), raw=False)
    P = dict(n=n, o=B.open.values, h=B.high.values, l=B.low.values, reg=reg, m=m, adr=adr.reindex(day).values, hv=(rk.reindex(day) >= 2 / 3).fillna(False).values, lv=(rk.reindex(day) <= 1 / 3).fillna(False).values, yr=B.index.year.values)
    P['Th'] = sparse(P['h'], np.maximum); P['Tl'] = sparse(P['l'], np.minimum); return P
def trades(P):
    reg, n = P['reg'], P['n']; ch = [i for i in range(1, n) if reg[i] != reg[i - 1] and reg[i] != 0]; T = []
    for a in range(len(ch) - 1):
        i, j = ch[a], ch[a + 1]
        if j + 1 >= n: continue
        if P['m'][i + 1] >= 480 and P['m'][i + 1] < 945 and not np.isnan(P['adr'][i + 1]): T.append((i + 1, j + 1, int(reg[i])))
    return np.array(T).reshape(-1, 3)
def ret(P, st, en, d, k):
    e = P['o'][st]; x = P['o'][en]; base = d * (x / e - 1)
    if k is None: return base
    a = P['adr'][st]; hi = rq(P['Th'], np.maximum, st, en - 1); lo = rq(P['Tl'], np.minimum, st, en - 1)
    tp_long = e * (1 + k * a); tp_short = e * (1 - k * a); hit = np.where(d > 0, hi >= tp_long, lo <= tp_short)
    return np.where(hit, k * a, base)
def run(PS, tag, eras, classes, nul=200):
    for cls in classes:
        for dirs in ('both', 'long'):
            print(f'\n#### {tag} | day class {cls} | directions {dirs}', flush=True)
            hdr = 'era        k     n  obs bp  null bp   p   win%   (obs - null)'; print(hdr)
            for en, (lo, hi) in eras.items():
                for k in KS:
                    N = 0; obs = 0; nl = np.zeros(nul); win = 0
                    for tk, P in PS.items():
                        T = trades(P)
                        if not len(T): continue
                        msk = (P['yr'][T[:, 0]] >= lo) & (P['yr'][T[:, 0]] <= hi)
                        if cls == 'HV-pre': msk &= P['hv'][T[:, 0]]
                        if dirs == 'long': msk &= T[:, 2] > 0
                        T = T[msk]
                        if not len(T): continue
                        r = ret(P, T[:, 0], T[:, 1], T[:, 2], k); L = T[:, 1] - T[:, 0]; N += len(T); obs += r.sum(); win += (r > 0).sum()
                        cand = np.where((P['m'] >= 480) & (P['m'] < 945) & ~np.isnan(P['adr']) & (P['yr'] >= lo) & (P['yr'] <= hi) & (P['hv'] if cls == 'HV-pre' else True))[0]
                        for z in range(nul):
                            st = rng.choice(cand, len(T)); en_ = np.minimum(st + L, P['n'] - 1); st = np.minimum(st, en_ - 1)
                            nl[z] += ret(P, st, en_, T[:, 2], k).sum()
                    if N < 20: print(f'{en:9s} {str(k):5s} {N:5d}  (too few)'); continue
                    o = obs / N * 1e4; nm = nl / N * 1e4; print(f'{en:9s} {str(k):5s} {N:5d} {o:+7.1f} {nm.mean():+8.1f} {(nm >= o).mean():5.2f} {win / N * 100:5.1f}   {o - nm.mean():+6.1f}', flush=True)

import yfinance as yf
A1 = ['KGS','LCID','RIOT','Z','IREN','PLUG','UPST','CLOV','TLRY','BB','SOFI','FUBO','OPEN','ETSY','DKNG','ROKU','CROX','FIVE','CHWY','MTCH','LYFT','BBWI','RIVN','SNAP','PINS','HOOD','AFRM','RKLB','SOUN','MARA','CLSK','HIMS','CELH','JOBY','ACHR','PATH','NIO','LI','W','CHPT','RUN','ENPH','SEDG','LUMN']
def h1v(tk):
    d = yf.download(tk, period='730d', interval='1h', prepost=False, progress=False, auto_adjust=False); d.columns = [c[0].lower() for c in d.columns]
    d.index = d.index.tz_convert('America/New_York').tz_localize(None); d = d[['open', 'high', 'low', 'close', 'volume']].dropna(); return d[~d.index.duplicated()]
def localv(tk):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_2018-01-01_*.parquet')[0]; H = pd.read_parquet(f, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.drop(columns='datetime').dropna().sort_index(); return clean(H[~H.index.duplicated()])
def addfeat(P, B):
    c = B.close.values; v = B.volume.values.astype(float); day = B.index.normalize(); m = P['m']; rth = (m >= 570) & (m < 960)
    d = pd.DataFrame({'o': B.open.where(rth).groupby(day).first(), 'c': B.close.where(rth).groupby(day).last()}).dropna(); gp = (d.o / d.c.shift() - 1)
    P['gap'] = (gp.reindex(day).values / P['adr']); P['vrel'] = v / pd.Series(v).rolling(20).mean().shift().values
    P['ma'] = {nm: np.sign(c - fn(c, L)) for nm, fn, L in (('EMA20', lambda x, L: pd.Series(x).ewm(span=L, adjust=False).mean().values, 20), ('EMA50', lambda x, L: pd.Series(x).ewm(span=L, adjust=False).mean().values, 50), ('SMA50', lambda x, L: pd.Series(x).rolling(L).mean().values, 50), ('SMA200', lambda x, L: pd.Series(x).rolling(L).mean().values, 200))}
    return P
def build(names, loader):
    PS = {}
    for tk in names:
        try:
            B = loader(tk)
            if len(B) < 1500: continue
            PS[tk] = addfeat(prep(B), B)
        except Exception as e: print('skip', tk, str(e)[:50])
    return PS

def addfeat2(P, B):
    addfeat(P, B); c = B.close.values; o = B.open.values; h = B.high.values; l = B.low.values; v = B.volume.values.astype(float); m = P['m']; day = pd.Series(B.index.normalize()); rth = (m >= 570) & (m < 960); n = len(B)
    df = pd.DataFrame({'day': B.index.normalize(), 'm': m, 'c': c, 'o': o, 'v': v, 'rth': rth}); r = df[df.rth]
    dopen = r.groupby('day').o.first(); dclose = r.groupby('day').c.last(); pc = dclose.shift()
    mv = (r.c / r.day.map(dopen) - 1).abs(); piv = pd.DataFrame({'day': r.day, 'm': r.m, 'mv': mv, 'vv': r.v}).pivot_table(index='day', columns='m', values=['mv', 'vv'], aggfunc='last')
    sg = piv['mv'].rolling(14, min_periods=10).mean().shift(1); rv = piv['vv'].rolling(14, min_periods=10).mean().shift(1)
    sig = np.full(n, np.nan); rvm = np.full(n, np.nan)
    for q in np.where(rth)[0]:
        d_, m_ = df.day.iat[q], m[q]
        if d_ in sg.index and m_ in sg.columns: sig[q] = sg.at[d_, m_]; rvm[q] = rv.at[d_, m_]
    op = df.day.map(dopen).values; pcv = df.day.map(pc).values
    up = np.maximum(op, pcv) * (1 + sig); lo = np.minimum(op, pcv) * (1 - sig)
    P['nz'] = np.where(c > up, 1, np.where(c < lo, -1, 0)) * (~np.isnan(sig))        # +1 broke above noise area, -1 below
    P['rvol'] = np.where(np.isnan(rvm) | (rvm <= 0), np.nan, v / rvm)
    P['vema'] = v / pd.Series(v).ewm(span=20, adjust=False).mean().shift().values
    s_ = np.sign(c - o); upv = pd.Series(v * (s_ > 0)).rolling(4).sum().values; dnv = pd.Series(v * (s_ < 0)).rolling(4).sum().values; P['udv'] = (upv + 1) / (dnv + 1)
    P['delta'] = pd.Series(s_ * v).rolling(4).sum().values / (pd.Series(v).rolling(4).sum().values + 1)
    tp = (h + l + c) / 3; vw = (pd.Series(np.where(rth, tp * v, 0)).groupby(df.day.values).cumsum() / (pd.Series(np.where(rth, v, 0)).groupby(df.day.values).cumsum() + 1e-9)).values
    P['vwap'] = np.where(rth, np.sign(c - vw), 0)
    return P
def buildn(names, loader):
    PS = {}
    for tk in names:
        try:
            B = loader(tk)
            if len(B) < 1500: continue
            PS[tk] = addfeat2(prep(B), B)
        except Exception as e: print('skip', tk, str(e)[:60])
    return PS
F = lambda P, T: T[:, 0] - 1
CON = [('BASE: flip vol >= 1.5x (ref)', lambda P, T: np.ones(len(T), bool)),
 ('+ outside noise area (with trade)', lambda P, T: P['nz'][F(P, T)] == T[:, 2]), ('+ inside noise area / against', lambda P, T: P['nz'][F(P, T)] != T[:, 2]),
 ('+ price on VWAP side of trade', lambda P, T: P['vwap'][F(P, T)] == T[:, 2]), ('+ price on wrong side of VWAP', lambda P, T: P['vwap'][F(P, T)] == -T[:, 2]),
 ('+ time-of-day RVOL >= 1.5', lambda P, T: P['rvol'][F(P, T)] >= 1.5), ('+ time-of-day RVOL < 1.5', lambda P, T: P['rvol'][F(P, T)] < 1.5),
 ('+ vol >= 1.5x its EMA20', lambda P, T: P['vema'][F(P, T)] >= 1.5), ('+ vol >= 2.5x prior-20 mean', lambda P, T: P['vrel'][F(P, T)] >= 2.5),
 ('+ up/down vol (4b) favours trade >= 1.5', lambda P, T: np.where(T[:, 2] > 0, P['udv'][F(P, T)], 1 / P['udv'][F(P, T)]) >= 1.5), ('+ up/down vol (4b) against trade', lambda P, T: np.where(T[:, 2] > 0, P['udv'][F(P, T)], 1 / P['udv'][F(P, T)]) < 1 / 1.5),
 ('+ signed-volume delta (4b) > +0.2 with trade', lambda P, T: T[:, 2] * P['delta'][F(P, T)] > .2), ('+ signed-volume delta (4b) < -0.2 against', lambda P, T: T[:, 2] * P['delta'][F(P, T)] < -.2),
 ('+ noise breakout AND tod RVOL >= 1.5', lambda P, T: (P['nz'][F(P, T)] == T[:, 2]) & (P['rvol'][F(P, T)] >= 1.5))]
def runn(PS, lo, hi, label, nul=150):
    print(f'\n######## {label}', flush=True); print('  condition                                      n   obs bp  null bp(base-vol rand)   p   win%  (obs-null)')
    for cn, fn in CON:
        N = 0; obs = 0; nl = np.zeros(nul); win = 0
        for tk, P in PS.items():
            T = trades(P)
            if not len(T): continue
            T = T[(P['yr'][T[:, 0]] >= lo) & (P['yr'][T[:, 0]] <= hi)]
            if not len(T): continue
            T = T[P['vrel'][T[:, 0] - 1] >= 1.5]
            if not len(T): continue
            T = T[fn(P, T)]
            if not len(T): continue
            r = ret(P, T[:, 0], T[:, 1], T[:, 2], None); L = T[:, 1] - T[:, 0]; N += len(T); obs += r.sum(); win += (r > 0).sum()
            ci = np.arange(1, P['n']); cand = ci[(P['m'][ci] >= 480) & (P['m'][ci] < 945) & ~np.isnan(P['adr'][ci]) & (P['yr'][ci] >= lo) & (P['yr'][ci] <= hi) & (P['vrel'][ci - 1] >= 1.5)]
            for z in range(nul):
                st = rng.choice(cand, len(T)); en_ = np.minimum(st + L, P['n'] - 1); st = np.minimum(st, en_ - 1); nl[z] += ret(P, st, en_, T[:, 2], None).sum()
        if N < 30: print(f'  {cn:44s} {N:5d} (too few)'); continue
        o = obs / N * 1e4; nm_ = nl / N * 1e4; print(f'  {cn:44s} {N:5d} {o:+7.1f} {nm_.mean():+9.1f}          {(nm_ >= o).mean():5.2f} {win / N * 100:5.1f}  {o - nm_.mean():+6.1f}', flush=True)
PSS = buildn(A1, h1v); print('small/mid', len(PSS), flush=True)
runn(PSS, 2023, 2024, 'SMALL/MID 1h HALF 1 (Dec23-2024)'); runn(PSS, 2025, 2026, 'SMALL/MID 1h HALF 2 (2025+)')
PSM = buildn(['AAPL', 'AMZN', 'AVGO', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA'], localv)
runn(PSM, 2019, 2021, 'MEGA-CAPS 15m TRAIN 2019-21'); runn(PSM, 2022, 2026, 'MEGA-CAPS 15m TEST 2022+')
