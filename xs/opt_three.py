"""Nov 2 2026: three untested options angles to pick the best names INSIDE the options-ratio universe. All known at the prior close
(OI-based ones lagged an extra day to be safe about the reporting convention). DECLARED BEFORE RESULTS:
 T1 term     near (30d) ATM IV / far (91d) ATM IV  (> 1 = an event is priced; hypothesis: higher = better)       [opt_features, all names]
 T2 opening  call OI change / call volume  minus  put OI change / put volume  (net new bullish positioning)     [opt_summary, 38 names]
 T3 reach    implied 1-day move (30d ATM IV / sqrt 252) / distance from entry to the vector's stop level      [all names]
 Each: avg trade + big-winner ratio by quintile, then the book with ratio + feature as the tie-break (40 matched runs).
 Pass: >= 70% of runs beat live in BOTH 2019-21 and 2022-26 (T2: book on its 38-name subset vs live on the same subset)."""
import sys, os, glob, zlib, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
src = open('/home/claude/fartpie/xs/vec_state.py').read(); exec(src.split("if __name__ == '__main__':")[0])
X = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet').sort_values('entry').reset_index(drop=True); BA_ = BASE['BREAK_ATR']
# --- stop distance at entry (vector level after the bullish flip)
CACHE = '/tmp/claude-0/ml/opt3_stopd.npy'
SD = np.load(CACHE) if os.path.exists(CACHE) else np.full(len(X), np.nan)
for tk, g in ([] if os.path.exists(CACHE) else X.groupby('tk')):
    H = load(glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')[0]); S15 = vec_i(H[['dt', 'open', 'high', 'low', 'close']])
    r = agg(H, '1h', '9h30min'); S1h = vec_i(r); kn1 = (r.dt + pd.Timedelta('1h')).values; t = H.dt.values; op = H.open.values
    for i, x in g.iterrows():
        e = np.datetime64(x.entry); j = np.searchsorted(t, e)
        if j >= len(t): continue
        if not x.h1:
            k = j - 1
            while k > j - 120 and k > 0 and not (S15.reg.values[k] == 1 and S15.reg.values[k - 1] == -1): k -= 1
            S = S15
        else:
            k = np.searchsorted(kn1, e, side='right') - 1
            while k > 0 and not (S1h.reg.values[k] == 1 and S1h.reg.values[k - 1] == -1): k -= 1
            S = S1h
        if k + 1 < len(S) and pd.notna(S.lvl.values[k + 1]): SD[i] = (op[j] - S.lvl.values[k + 1]) / op[j]
X['stopd'] = SD; np.save(CACHE, SD)
print('stop distance located for', f'{np.isfinite(SD).mean():.0%}', 'of trades; median', f'{np.nanmedian(SD):.1%}', flush=True)
# --- options features
OF = pd.read_csv('/mnt/user-data/uploads/fartpie/xs/ext/opt_features.csv', parse_dates=['date']).sort_values('date')
OF['iv'] = OF[['iv_c50', 'iv_p50']].mean(axis=1); OF['ivf'] = OF[['iv_c50_far', 'iv_p50_far']].mean(axis=1); OF['term'] = OF.iv / OF.ivf
def asof(df, key, col, lag_days):
    out = np.full(len(X), np.nan)
    for tk, g in X.groupby('tk'):
        s = df[df[key] == tk].set_index('date')[col].dropna(); s = s[~s.index.duplicated(keep='last')]
        if s.empty: continue
        v = s.reindex(s.index.union(pd.DatetimeIndex(g.d.unique()))).ffill().shift(lag_days); out[g.index] = v.reindex(g.d).values
    return out
X['iv'] = asof(OF, 'sym', 'iv', 1); X['term'] = asof(OF, 'sym', 'term', 1)
X['reach'] = (X.iv / np.sqrt(252)) / X.stopd.where(X.stopd > 0)
OS = []
for f in glob.glob('/mnt/user-data/uploads/fartpie/xs/ext/opt_summary/*.csv'):
    d = pd.read_csv(f, parse_dates=['date']).sort_values('date'); d['tk'] = os.path.basename(f)[:-4]
    d['open_c'] = d.call_oi.diff() / d.call_vol.shift(1); d['open_p'] = d.put_oi.diff() / d.put_vol.shift(1)
    d['opening'] = (d.open_c.clip(-2, 2) - d.open_p.clip(-2, 2)).rolling(5).mean(); OS.append(d)
OS = pd.concat(OS); X['opening'] = asof(OS, 'tk', 'opening', 2)
X['h'] = np.where(X.entry.dt.year <= 2021, '2019-21', '2022-26'); X['yr'] = X.entry.dt.year
top = X.groupby('h').ret.transform(lambda s: s >= s.quantile(.95))
for c in ('term', 'opening', 'reach'):
    q = X[c].groupby(X.yr).rank(pct=True); out = []
    for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        m = (q > lo) & (q <= hi); row = []
        for h in ('2019-21', '2022-26'):
            z = (X.h == h) & q.notna(); mm = m & z; row.append(f'{X.ret[mm].mean():+.2%} j{(top[mm].sum() / top[z].sum()) / max(mm.sum() / z.sum(), 1e-9):.2f}')
        out.append('/'.join(row))
    print(f'  {c:<8} coverage {X[c].notna().mean():.0%}  q1..q5 (avg trade, big-winner ratio; 2019-21/2022-26): ' + '  |  '.join(out), flush=True)
HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(X.tk, X.entry.astype('int64'))], dtype=np.uint64)
DROPS = [((HSH * 2654435761 + i * 40503) % 1000) < 50 for i in range(1, 41)]
IDX = pd.bdate_range('2019-01-01', X.exit.max())
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
def ev(key, sub=None):
    sub = np.ones(len(X), bool) if sub is None else sub; K = key.fillna(key.median()).values.astype(float)
    return np.array([stat2(sim(X.entry.values[sub & ~dl], X.exit.values[sub & ~dl], X.ret.values[sub & ~dl], K[sub & ~dl], X.tk.values[sub & ~dl])) for dl in DROPS])
def line(nm, a, b=None):
    m = np.median(a, axis=0); s = f'  {nm:<44} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None:
        w1, w2 = (a[:, 2] > b[:, 2]).mean(), (a[:, 5] > b[:, 5]).mean(); s += f'   beats live {w1:.0%} / {w2:.0%}' + ('  PASS' if w1 >= .7 and w2 >= .7 else '')
    print(s, flush=True)
pr = lambda s: s.groupby(X.yr).rank(pct=True).fillna(.5)
print('\nbook, 40 matched runs, 2019-21 / 2022-26'); B0 = ev(X.ratio); line('live (ratio tie-break)', B0)
line('T1 ratio + IV term (near/far)', ev(pr(X.ratio) + pr(X.term)), B0)
line('T3 ratio + reach (implied move / stop dist)', ev(pr(X.ratio) + pr(X.reach)), B0)
sub = X.tk.isin(set(OS.tk)).values; B1 = ev(X.ratio, sub); line(f'live on the {int(len(set(OS.tk) & set(X.tk)))} names with OI', B1)
line('T2 ratio + opening (same names)', ev(pr(X.ratio) + pr(X.opening), sub), B1)
