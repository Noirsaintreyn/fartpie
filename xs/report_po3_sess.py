"""Session model: Asia (19:00 prev-eve-03:00 ET) consolidates, London (03:00-08:00) manipulates, NY distributes. Nov 11 2026.
Sweeps (wick past H/L, close back inside) of: Asia range by London bars (AL), Asia range by 08:00->entry bars (AN), London range by 08:00->entry bars (LN).
aligned = dir*state (+1 = trade in the direction a sweep implies: low swept -> long). Classic = London swept Asia and trade aligned with it. Leverage test as before."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
O, N = pickle.load(open(SC + 's6.pkl', 'rb')); L = dict(zip(N, O['V2']))
X = pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet')[['dt', 'high', 'low', 'close']].set_index('dt').sort_index()
H, Lw, C = X.high.values, X.low.values, X.close.values; IDX = X.index
def win(t0, t1): a, b = IDX.searchsorted(t0), IDX.searchsorted(t1); return a, b
def rng(t0, t1):
    a, b = win(t0, t1)
    return (H[a:b].max(), Lw[a:b].min()) if b - a > 20 else None
def sweep(lv, t0, t1):
    if lv is None: return 0
    a, b = win(t0, t1); ev = []
    for i in range(a, b):
        if H[i] > lv[0] and C[i] < lv[0]: ev.append((i, -1))
        if Lw[i] < lv[1] and C[i] > lv[1]: ev.append((i, +1))
    return max(ev)[1] if ev else 0
def tstat(x): x = np.asarray(x, float); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
def rngc(t0, t1, lv):   # did the window trade outside lv at all (any extension)?
    r = rng(t0, t1); return r
for nm in ('V15a', 'RA'):
    T = L[nm][0]; T = T[T.sz > 0].copy(); T['day'] = pd.to_datetime(T.day); rows = []
    for r in T.itertuples():
        d = r.day; D = pd.Timestamp(d).normalize(); te = D + pd.Timedelta(minutes=int(r.m))
        as0, as1, lo1, ny0 = D - pd.Timedelta(hours=5), D + pd.Timedelta(hours=3), D + pd.Timedelta(hours=8), D + pd.Timedelta(hours=8)
        asia, lon = rng(as0, as1), rng(as1, lo1)
        if asia is None or lon is None: continue
        dr = int(r.dir); al = sweep(asia, as1, lo1); an = sweep(asia, lo1, te); ln = sweep(lon, lo1, te)
        rows.append(dict(day=d, m=int(r.m), pts=r.pts, dir=dr, AL=dr * al, AN=dr * an, LN=dr * ln, al=al,
                         asia_w=asia[0] - asia[1], lon_w=lon[0] - lon[1]))
    R = pd.DataFrame(rows); R['era'] = np.where(R.day < '2022-01-01', 'A', 'B')
    print(f'\n== {nm}: {len(R)} trades, mean {R.pts.mean():+.1f}, win {np.mean(R.pts>0):.0%}')
    def line(lab, s):
        a = s[s.era == 'A'].pts; b = s[s.era == 'B'].pts
        ok = len(a) >= 30 and len(b) >= 30 and np.sign(a.mean()) == np.sign(b.mean()) and abs(tstat(a)) >= 2 and abs(tstat(b)) >= 2
        print(f' {lab:44} n {len(s):4d} win {np.mean(s.pts>0) if len(s) else 0:3.0%} mean {s.pts.mean() if len(s) else 0:+6.1f} | A n{len(a):3d} {a.mean():+6.1f} ({tstat(a):+.1f}) | B n{len(b):3d} {b.mean():+6.1f} ({tstat(b):+.1f}) {"PASS" if ok else ""}')
    for col, lab in (('AL', 'London swept Asia'), ('AN', 'NY-side swept Asia'), ('LN', 'NY-side swept London')):
        for v, t in ((1, 'trade WITH sweep dir'), (0, 'no sweep'), (-1, 'AGAINST')): line(f'{lab}: {t}', R[R[col] == v])
    print(' -- classic: London swept Asia, then trade...')
    line('classic: with London sweep (NY distributes opposite of the sweep wick)', R[R.AL == 1])
    line('London swept Asia AND NY then swept London, with LN', R[(R.AL == 1) & (R.LN == 1)])
    line('London did NOT sweep Asia (Asia kept / continuation)', R[R.al == 0])
    line('  ...and trade is with the larger-window drift: skip', R[R.al == 0].iloc[0:0])
    for q, lab in (('asia_w', 'tight Asia (below median) & AL=with'), ):
        m = R[q].median(); line(lab, R[(R[q] < m) & (R.AL == 1)]); line('wide Asia (above median) & AL=with', R[(R[q] >= m) & (R.AL == 1)])
    print(' -- leverage 2x when AL==1 or (AL==1 & LN>=0) ; 1x else')
    for lab, w in (('2x on AL==1', R.AL == 1), ('2x on AL==1 or AN==1', (R.AL == 1) | (R.AN == 1))):
        y = pd.Series(np.where(w, 2 * R.pts, R.pts), index=R.index); ey = y.groupby(R.day.dt.year).sum()
        print(f'   {lab:22} total {y.sum():+8.0f} mean {y.mean():+5.1f} std {y.std():5.1f} worst {y.min():+6.0f} yrs-neg {int((ey<0).sum())}/{len(ey)} (1x total {R.pts.sum():+.0f})')
    R.to_pickle(SC + f'po3sess_{nm}.pkl')
