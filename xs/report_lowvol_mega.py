"""CHECK of LV-day on the 8 mega-caps 15m 2019-26 (train 2019-21 / test 2022+). Fixed before results. Universe: 22 earlier names + ~22 more small/mid names (Yahoo 1h RTH ~730d, halves Dec23-24 / 2025+; the extras are named in the log, picked without looking at results). Trades = 1h vector flips, vector exit (as before), optional 1.0 x ADR TP.
Low-vol classes (all known at the entry bar): LV-day = previous day's range in the bottom third of the ticker's own history so far; LV-atr = the ticker's 1h ATR in the bottom third of its trailing 500 bars; LV-both. Compared with the complement (not low) and with random entries (same direction, hold length, TP rule, same class mask; 200 draws)."""
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

CL = {'ALL': lambda P: np.ones(P['n'], bool), 'LV-day': lambda P: P['lv'], 'NOT low (day)': lambda P: ~P['lv'], 'HV-day (top third)': lambda P: P['hv']}
PS = {tk: prep(local(tk)) for tk in ('AAPL', 'AMZN', 'AVGO', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA')}
def run2(lo, hi, label, nul=200):
    print(f'\n######## {label}', flush=True); print('  class                 n   obs bp  null bp   p   win%  (obs-null)')
    for cn, fn in CL.items():
        N = 0; obs = 0; nl = np.zeros(nul); win = 0
        for tk, P in PS.items():
            T = trades(P); msk = (P['yr'][T[:, 0]] >= lo) & (P['yr'][T[:, 0]] <= hi) & fn(P)[T[:, 0]]; T = T[msk]
            if not len(T): continue
            r = ret(P, T[:, 0], T[:, 1], T[:, 2], None); L = T[:, 1] - T[:, 0]; N += len(T); obs += r.sum(); win += (r > 0).sum()
            cand = np.where((P['m'] >= 480) & (P['m'] < 945) & ~np.isnan(P['adr']) & (P['yr'] >= lo) & (P['yr'] <= hi) & fn(P))[0]
            for z in range(nul):
                st = rng.choice(cand, len(T)); en_ = np.minimum(st + L, P['n'] - 1); st = np.minimum(st, en_ - 1); nl[z] += ret(P, st, en_, T[:, 2], None).sum()
        o = obs / N * 1e4; nm = nl / N * 1e4; print(f'  {cn:20s} {N:5d} {o:+7.1f} {nm.mean():+8.1f} {(nm >= o).mean():5.2f} {win / N * 100:5.1f}  {o - nm.mean():+6.1f}', flush=True)
run2(2019, 2021, 'MEGA-CAPS 15m TRAIN 2019-21 (vector exit, no TP)'); run2(2022, 2026, 'MEGA-CAPS 15m TEST 2022+ (vector exit, no TP)')
