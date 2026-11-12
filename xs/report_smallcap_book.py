"""MID/SMALL-CAP GROWTH BOOK (separate from the mega-cap original screener) - first look at it as a portfolio. Fixed before results.
Universe: the 44 small/mid names (Yahoo 1h RTH ~Dec 2023-Nov 2026). LONG-ONLY vector flips (bull flip enter next open, exit at the open after the bear flip; the vector's own flip is the only exit). K slots (default 4, equal share of realised equity each), when more signals than free slots take the highest flip-bar volume ratio first. Costs: 15 bp round trip (30 bp stress). Entry filters: A = all bull flips, B = flip-bar volume >= 1.5x, C = low-vol day AND volume >= 1.5x (as found in report_lowvol_combo.py / report_volcheck.py). Compared with equal-weight buy-and-hold of the same names over the same dates. 
CAVEAT stated up front: the 44 names were picked today (many are 2024-25 high-flyers = survivorship/hindsight in the universe), so absolute returns are inflated; read the A/B/C comparison and the drawdown, not the level."""
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
    P['ts'] = B.index.values; P['cl'] = B.close.values; P['gap'] = (gp.reindex(day).values / P['adr']); P['vrel'] = v / pd.Series(v).rolling(20).mean().shift().values
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
CONDS = [('LV-day (reference)', lambda P, T: np.ones(len(T), bool))]
CONDS += [('LV + gap >= +0.5 ADR (with trade)', lambda P, T: T[:, 2] * P['gap'][T[:, 0] - 1] >= .5), ('LV + gap <= -0.5 ADR (against)', lambda P, T: T[:, 2] * P['gap'][T[:, 0] - 1] <= -.5), ('LV + |gap| < 0.5 ADR', lambda P, T: np.abs(P['gap'][T[:, 0] - 1]) < .5),
          ('LV + flip vol >= 1.5x', lambda P, T: P['vrel'][T[:, 0] - 1] >= 1.5), ('LV + flip vol < 1.5x', lambda P, T: P['vrel'][T[:, 0] - 1] < 1.5)]
for nm in ('EMA20', 'EMA50', 'SMA50', 'SMA200'):
    CONDS += [(f'LV + {nm} agrees', (lambda nm: lambda P, T: P['ma'][nm][T[:, 0] - 1] == T[:, 2])(nm)), (f'LV + {nm} disagrees', (lambda nm: lambda P, T: P['ma'][nm][T[:, 0] - 1] == -T[:, 2])(nm))]
def runc(PS, lo, hi, label, nul=200):
    print(f'\n######## {label}', flush=True); print('  condition                           n   obs bp  null bp(LV rand)   p   win%  (obs-null)')
    for cn, fn in CONDS:
        N = 0; obs = 0; nl = np.zeros(nul); win = 0
        for tk, P in PS.items():
            T = trades(P)
            if not len(T): continue
            T = T[(P['yr'][T[:, 0]] >= lo) & (P['yr'][T[:, 0]] <= hi) & P['lv'][T[:, 0]]]
            if not len(T): continue
            T = T[fn(P, T)]
            if not len(T): continue
            r = ret(P, T[:, 0], T[:, 1], T[:, 2], None); L = T[:, 1] - T[:, 0]; N += len(T); obs += r.sum(); win += (r > 0).sum()
            cand = np.where((P['m'] >= 480) & (P['m'] < 945) & ~np.isnan(P['adr']) & (P['yr'] >= lo) & (P['yr'] <= hi) & P['lv'])[0]
            for z in range(nul):
                st = rng.choice(cand, len(T)); en_ = np.minimum(st + L, P['n'] - 1); st = np.minimum(st, en_ - 1); nl[z] += ret(P, st, en_, T[:, 2], None).sum()
        if N < 30: print(f'  {cn:34s} {N:5d} (too few)'); continue
        o = obs / N * 1e4; nm_ = nl / N * 1e4; print(f'  {cn:34s} {N:5d} {o:+7.1f} {nm_.mean():+9.1f}      {(nm_ >= o).mean():5.2f} {win / N * 100:5.1f}  {o - nm_.mean():+6.1f}', flush=True)

PSS = build(A1, h1v); print('universe', len(PSS), flush=True)
def events(filt):
    E = []
    for tk, P in PSS.items():
        T = trades(P)
        if not len(T): continue
        T = T[T[:, 2] > 0]
        if not len(T): continue
        T = T[filt(P, T)] if len(T) else T
        if not len(T): continue
        r = ret(P, T[:, 0], T[:, 1], T[:, 2], None)
        for (st, en, d), rr in zip(T, r): E.append((P['ts'][st], P['ts'][en], float(rr), float(P['vrel'][st - 1]), tk))
    return sorted(E, key=lambda e: (e[0], -e[3]))
def sim(E, K, cost, t0, t1):
    eq = 1.0; open_ = []; curve = [(t0, 1.0)]; n = 0; wins = 0; hold = []
    for (a, b, rr, vr, tk) in E:
        if a < t0 or a >= t1: continue
        # close finished positions
        still = []
        for p in open_:
            if p[1] <= a: eq += p[3] * (p[2] - cost); curve.append((p[1], eq)); n += 1; wins += p[2] - cost > 0; hold.append((p[1] - p[0]) / np.timedelta64(1, 'D'))
            else: still.append(p)
        open_ = still
        if len(open_) < K and tk not in [p[4] for p in open_]: open_.append((a, b, rr, eq / K, tk))
    for p in open_: eq += p[3] * (p[2] - cost); curve.append((p[1], eq)); n += 1; wins += p[2] - cost > 0; hold.append((p[1] - p[0]) / np.timedelta64(1, 'D'))
    cv = np.array([v for _, v in sorted(curve, key=lambda x: x[0])]); dd = (cv / np.maximum.accumulate(cv) - 1).min()
    return eq - 1, dd, n, (wins / n if n else np.nan), (np.mean(hold) if hold else np.nan)
def bh(t0, t1):
    out = []
    for tk, P in PSS.items():
        ts = pd.to_datetime(P['ts']); m = (ts >= t0) & (ts < t1)
        if m.sum() > 50: out.append(P['cl'][np.where(m)[0][-1]] / P['o'][np.where(m)[0][0]] - 1)
    return np.mean(out)
VARS = {'A all bull flips': lambda P, T: np.ones(len(T), bool), 'B flip vol >= 1.5x': lambda P, T: P['vrel'][T[:, 0] - 1] >= 1.5, 'C low-vol day + vol >= 1.5x': lambda P, T: (P['vrel'][T[:, 0] - 1] >= 1.5) & P['lv'][T[:, 0]]}
EV = {k: events(f) for k, f in VARS.items()}
H = {'half 1 Dec23-Dec24': (pd.Timestamp('2023-12-01'), pd.Timestamp('2025-01-01')), 'half 2 2025+': (pd.Timestamp('2025-01-01'), pd.Timestamp('2027-01-01'))}
for K in (4, 8):
    for cost in (.0015, .003):
        print(f'\n######## K={K} slots, cost {cost * 1e4:.0f} bp round trip')
        for hn, (t0, t1) in H.items():
            print(f'  {hn}: buy&hold equal-weight same names {bh(t0, t1) * 100:+.0f}%')
            for vn in VARS:
                tr_, dd, n, w, hd = sim(EV[vn], K, cost, t0, t1); print(f'     {vn:28s} total {tr_ * 100:+7.1f}%  maxDD {dd * 100:6.1f}%  trades {n:4d}  win {w * 100:4.1f}%  avg hold {hd:4.1f}d')
