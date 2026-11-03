"""Nov 2 2026: attribution controls (outside review #4). Not strategies to pick from; they split the edge into parts.
Same universe, budget, 6 slots, breaker, costs. Each control keeps everything about a live trade except one thing:
 C1 timing within the day   same ticker, same exit time, entry at a random 9:30-13:45 bar of the entry day   -> what the exact flip bar adds
 C2 timing at all           same ticker, same holding length (bars), entry at a random bar within +-60 days     -> what flip timing adds over 'hold this name'
 C3 stock choice            same entry and exit times, a random OTHER name traded in the universe that quarter -> what picking THIS name adds
 C4 both                    random other name AND random time within +-60 days, same holding length            -> universe + holding mechanics only
40 runs: run i uses the same 5% signal drops as before and random seed i for the control's draws. 2019-21 / 2022-26."""
import glob, zlib, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
LSE = '/mnt/user-data/uploads/fartpie/lse_clean'; C2 = 2e-4
X = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet').sort_values('entry').reset_index(drop=True)
B = {}
for tk in X.tk.unique():
    f = glob.glob(f'{LSE}/{tk}_15m_*.parquet'); H = pd.read_parquet(f[0])
    H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H['m'] = H.dt.dt.hour * 60 + H.dt.dt.minute
    H = H[(H.m >= 570) & (H.m < 960) & (H.volume > 0)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
    B[tk] = (H.dt.values, H.open.values, H.m.values, H.dt.dt.normalize().values)
A = np.zeros(len(X), int); Bb = np.zeros(len(X), int); ok = np.zeros(len(X), bool)
for i, (tk, e, x) in enumerate(zip(X.tk, X.entry.values, X.exit.values)):
    t = B[tk][0]; a, b = np.searchsorted(t, e), np.searchsorted(t, x)
    if a < len(t) and b < len(t) and b > a and t[a] == e: A[i], Bb[i], ok[i] = a, b, True
X = X[ok].reset_index(drop=True); A, Bb = A[ok], Bb[ok]; NB = Bb - A
X['q'] = X.entry.dt.to_period('Q').astype(str); QT = {k: list(v) for k, v in X.groupby('q').tk.unique().items()}; QV = X.q.values; TKV = X.tk.values
print(f'{len(X)} trades; live avg trade {X.ret.mean():+.3%}', flush=True)
def draw(kind, rng):
    E, Xx, R = X.entry.values.copy(), X.exit.values.copy(), X.ret.values.copy(); EV = X.entry.values
    for i in range(len(X)):
        tk = TKV[i]; nb = NB[i]
        if kind in ('C3', 'C4'):
            pool = [p for p in QT[QV[i]] if p != tk]
            if not pool: continue
            tk = pool[rng.integers(len(pool))]
        t, o, m, dy = B[tk]
        if kind == 'C1':
            day = dy[A[i]]; lo = np.searchsorted(dy, day); hi = min(np.searchsorted(dy, day, side='right'), Bb[i])
            idx = [k for k in range(lo, hi) if m[k] < 840]
            if not idx: continue
            a = idx[rng.integers(len(idx))]; b = Bb[i]
        elif kind == 'C3':
            a, b = np.searchsorted(t, E[i]), np.searchsorted(t, Xx[i])
            if a >= len(t) or b >= len(t) or b <= a: continue
        else:
            c = np.searchsorted(t, E[i]); lo, hi = np.searchsorted(t, E[i] - np.timedelta64(60, 'D')), np.searchsorted(t, E[i] + np.timedelta64(60, 'D'))
            hi = min(hi, len(t) - nb - 1)
            if hi <= lo: continue
            for _ in range(20):
                a = rng.integers(lo, hi)
                if m[a] < 840: break
            b = a + nb
        E[i], Xx[i], R[i] = t[a], t[b], o[b] / o[a] - 1 - C2
    return E, Xx, R
HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(X.tk, X.entry.astype('int64'))], dtype=np.uint64)
DROPS = [((HSH * 2654435761 + i * 40503) % 1000) < 50 for i in range(1, 41)]
IDX = pd.bdate_range('2019-01-01', X.exit.max() + pd.Timedelta(days=200))
P2 = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def stat2(s):
    eq = s.groupby(s.index.normalize()).last().reindex(IDX, method='ffill').fillna(0); out = []
    for a, b in P2:
        e = eq[(eq.index >= a) & (eq.index < b)]; e = np.exp(e - e.iloc[0]); yr = (e.index[-1] - e.index[0]).days / 365.25
        c = e.iloc[-1] ** (1 / yr) - 1; dd = (e / e.cummax() - 1).min(); out += [c, dd, c / -dd]
    return out
def sim(E, Xx, R, K, TK):
    o = np.lexsort((TK, -K, E)); op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0
    for e, x, r in zip(E[o], Xx[o], R[o]):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / 6); hist.append((xt, real)); win.append((xt, real))
        op = [q for q in op if q > e]
        if len(op) >= 6: continue
        if hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-.08): continue
        op.append(x); pend.append((x, r))
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr / 6); hist.append((xt, real))
    return pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
K = X.ratio.values.astype(float); TK = X.tk.values
def run(kind):
    out, avg, top = [], [], []
    for i, dl in enumerate(DROPS):
        if kind == 'live': E, Xx, R = X.entry.values, X.exit.values, X.ret.values
        else: E, Xx, R = draw(kind, np.random.default_rng(1000 + i))
        out.append(stat2(sim(E[~dl], Xx[~dl], R[~dl], K[~dl], TK[~dl]))); avg.append(R.mean()); top.append(np.sort(R)[-len(R) // 20:].sum() / R.sum() if R.sum() > 0 else np.nan)
    return np.array(out), np.mean(avg), np.nanmedian(top)
L0, la, lt = run('live')
def line(nm, a, av, b=None):
    m = np.median(a, axis=0); s = f'  {nm:<44} avg trade {av:+.3%}   ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[:, 2] > b[:, 2]).mean():.0%} / {(a[:, 5] > b[:, 5]).mean():.0%}'
    print(s, flush=True)
line('live', L0, la)
import os
for kind, nm in [t for t in (('C1', 'C1 random bar same day, same exit'), ('C2', 'C2 same name, random time +-60d'), ('C3', 'C3 random universe name, same times'), ('C4', 'C4 random name + random time')) if t[0] in os.getenv('ONLY', 'C1,C2,C3,C4').split(',')]:
    a, av, _ = run(kind); line(nm, a, av, L0)
