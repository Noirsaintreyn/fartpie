"""Nov 2 2026: risk-parity slot sizing (the trend-following standard: Moskowitz-Ooi-Pedersen 2012, Baltas & Kosowski, Hurst-Ooi-Pedersen).
Same entries/exits as live; each position's weight = (1/6) x (sigma_ref / sigma_i)^a, sigma_i = stock's 20d daily vol at the prior close,
sigma_ref = median sigma of the book's past-year trades (trailing, no look-ahead); weight capped at [lo, hi] x 1/6 and at the idle cash
(gross never above 100%, no margin). a = 1 full inverse vol, 0.5 square-root. Also the opposite (bigger in high vol) as a sanity check.
6 slots, 8% breaker, 40 matched runs, 2019-21 / 2022-26."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
X = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet').sort_values('entry').reset_index(drop=True)
SD = {}
for tk in X.tk.unique():
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0], columns=['datetime', 'close']); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m_ = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m_ >= 570) & (m_ < 960)]; c = H.groupby(H.dt.dt.normalize()).close.last()
    SD[tk] = c.pct_change().rolling(20).std().shift(1)
X['sig'] = [SD[t].asof(d) if t in SD else np.nan for t, d in zip(X.tk, X.d)]
ref = X.set_index('entry').sig.rolling('365D', min_periods=200).median().shift(1).values; X['ref'] = ref
X = X[X.sig.notna() & X.ref.notna()].reset_index(drop=True)
print(f'{len(X)} trades from {X.entry.min():%Y-%m}; sigma ratio (ref/own) quartiles: {np.round((X.ref / X.sig).quantile([.1, .25, .5, .75, .9]).values, 2)}')
def simv(Z, a, lo, hi, n=6, brk=.08):
    o = np.lexsort((Z.tk.values, Z.h1.values, -Z.ratio.values, Z.entry.values)); E, Xx, R = Z.entry.values[o], Z.exit.values[o], Z.ret.values[o]
    W = (np.clip((Z.ref.values / Z.sig.values) ** a, lo, hi) / n)[o]
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0
    for e, x, r_, w in zip(E, Xx, R, W):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr); hist.append((xt, real)); win.append((xt, real))
        op = [q for q in op if q[0] > e]; gross = sum(q[1] for q in op)
        if len(op) >= n: continue
        w = min(w, 1 - gross)
        if w < .02: continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
        op.append((x, w)); pend.append((x, r_ * w))
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index(); return s.groupby(s.index.normalize()).last()
rng = np.random.default_rng(101); DR = [rng.random(len(X)) < .05 if i else np.zeros(len(X), bool) for i in range(41)]
def ev(a, lo, hi): return np.array([[v_ for cd in stat(dret(simv(X[~dl], a, lo, hi)).cumsum()) for v_ in (cd[0], cd[1], cd[0] / -cd[1])] for dl in DR])
def line(nm, A, b=None):
    m = np.median(A[1:], axis=0); s = f'  {nm:<40} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(A[1:, 2] > b[1:, 2]).mean():.0%} / {(A[1:, 5] > b[1:, 5]).mean():.0%}  (return {(A[1:, 0] > b[1:, 0]).mean():.0%} / {(A[1:, 3] > b[1:, 3]).mean():.0%})'
    print(s, flush=True)
B0 = ev(0, 1, 1); line('live (equal 1/6)', B0)
for a, lo, hi in ((1, .5, 1.5), (1, .5, 1.0), (1, .33, 2), (.5, .5, 1.5), (.5, .7, 1.3), (1, .25, 1.0)): line(f'inverse vol a={a} cap [{lo}, {hi}]x', ev(a, lo, hi), B0)
for a, lo, hi in ((-1, .5, 1.5), (-.5, .7, 1.3)): line(f'PRO vol (sanity) a={a} cap [{lo}, {hi}]x', ev(a, lo, hi), B0)
