"""VECTOR-FLIP EXIT vs TIME EXIT on the same entries (RA and V15a legs; direction saved only for these). Same stop (0.15*ADR), cost 1.0 pt (run_sig2 convention). Nov 10 2026.
A = exit on the first adverse 15m vector regime flip (next 5m open) else 15:55 close; B = time exit 12:00; C = time exit at the 15:55 close (no vector exit); D = vector exit only after 12:00 (hold to noon minimum)."""
import sys, os; sys.path.insert(0, '/home/claude/fartpie/xs'); os.chdir('/home/claude/fartpie')
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
src = open('/home/claude/fartpie/xs/dte_season2.py').read(); exec(src[:src.rindex("\nif __name__ == '__main__':")])
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
O_, N_ = pickle.load(open(SC + 's6.pkl', 'rb')); legs = dict(zip(N_, O_['V2']))
regmap = dict(zip(B15.index.values, B15.reg.values))
hv = H.dt.values
def sim(d, mm, dr, mode, stop_k=.15, cost=1.0):
    ix = gi[d]; mi = m[ix]; w = np.where(mi == mm)[0]
    if len(w) == 0: return np.nan
    j = int(w[0]); i0 = ix[j]; e = o[i0]; S = e - dr * stop_k * DY.adr[d]
    for p in range(j, len(ix)):
        ii = ix[p]
        if m[ii] >= 955: return ((c[ix[p - 1]] - e) * dr) - cost
        if (l[ii] <= S) if dr > 0 else (h[ii] >= S): return ((S - e) * dr) - cost
        if mode == 'B' and m[ii] >= 715: return ((c[ii] - e) * dr) - cost
        if mode in ('A', 'D') and (m[ii] + 5) % 15 == 0 and p + 1 < len(ix) and (mode == 'A' or m[ii] >= 720):
            r = regmap.get(hv[ii] - np.timedelta64(10, 'm'))
            if r is not None and r == -dr: return ((o[ix[p + 1]] - e) * dr) - cost
    return ((c[ix[-1]] - e) * dr) - cost
def tt(x): x = pd.Series(x).dropna(); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 15 and x.std() > 0 else np.nan
for nm in ('V15a', 'RA'):
    T = legs[nm][0]; T = T[(T.sz > 0) & (pd.to_datetime(T.day) >= '2018-02-01') & (pd.to_datetime(T.day) <= '2026-01-30')].copy(); T['day'] = pd.to_datetime(T.day); yr = T.day.dt.year.values
    print(f'\n=== {nm}: {len(T)} entries (recorded leg pts/trade {T.pts.mean():+.2f}) ===')
    for mode, lab in (('A', 'A vector-flip exit (else close)'), ('B', 'B time exit 12:00'), ('C', 'C hold to close, no vector exit'), ('D', 'D vector exit only after 12:00')):
        r = np.array([sim(d, int(mm), int(dr), mode) for d, mm, dr in zip(T.day, T.m, T.dir)]); ok = np.isfinite(r); w, ls = r[ok & (r > 0)], r[ok & (r <= 0)]
        a, b = r[ok & (yr < 2022)], r[ok & (yr >= 2022)]
        print(f' {lab:34} n {ok.sum():4d} mean {np.nanmean(r):+6.2f} win {np.mean(r[ok]>0):4.0%} avg win {w.mean():+6.1f} avg loss {ls.mean():+6.1f} (RR {abs(w.mean()/ls.mean()):.2f}) | 18-21 {a.mean():+6.2f} (t {tt(a):+.1f}) 22-26 {b.mean():+6.2f} (t {tt(b):+.1f}) | worst {r[ok].min():+.0f} | total {np.nansum(r):+.0f}')

def sim2(d, mm, dr, mode, stop_k=.15, cost=1.0):
    ix = gi[d]; mi = m[ix]; w = np.where(mi == mm)[0]
    if len(w) == 0: return (np.nan, np.nan)
    j = int(w[0]); i0 = ix[j]; e = o[i0]; S = e - dr * stop_k * DY.adr[d]; worst = 0.0
    for p in range(j, len(ix)):
        ii = ix[p]
        if m[ii] >= 955: return (((c[ix[p - 1]] - e) * dr) - cost, worst - cost)
        if (l[ii] <= S) if dr > 0 else (h[ii] >= S): worst = min(worst, (S - e) * dr); return (((S - e) * dr) - cost, worst - cost)
        worst = min(worst, ((l[ii] if dr > 0 else h[ii]) - e) * dr)
        if mode in ('A', 'D') and (m[ii] + 5) % 15 == 0 and p + 1 < len(ix) and (mode == 'A' or m[ii] >= 720):
            r = regmap.get(hv[ii] - np.timedelta64(10, 'm'))
            if r is not None and r == -dr: return (((o[ix[p + 1]] - e) * dr) - cost, worst - cost)
    return (((c[ix[-1]] - e) * dr) - cost, worst - cost)
RA = legs['RA'][0].copy(); RA['day'] = pd.to_datetime(RA.day); rows = []
for r_ in RA.itertuples():
    pts, mae = sim2(r_.day, int(r_.m), int(r_.dir), 'D'); rows.append((r_.day, int(r_.m), pts, mae))
pd.DataFrame(rows, columns=['day', 'm', 'pts_D', 'mae_D']).to_pickle(SC + 'ra_hold.pkl'); print('saved ra_hold.pkl', len(rows))
