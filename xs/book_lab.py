"""Oct 26 2026: study the 6-slot book's selection itself.
 noise    the live book re-run with random ordering of simultaneous signals (instead of the options-ratio tie-break), and with 1% of
          signals randomly dropped -> the luck band any rule change must clear
 mine     every feature we built, measured only on the trades the live book actually admits, by period
 state    book-state at each admission (slots already used, same-sector positions open, 15m/1h mix, minutes since the last exit)
          and joint stop-outs on the worst days; rules on book state through a state-aware scheduler
python3 xs/book_lab.py noise SEED N OUT | mine | state"""
import sys, os, json, glob, numpy as np, pandas as pd
from collections import deque
exec(open('xs/book_mtm.py').read().split("if __name__")[0])
src = open('xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
EXT = 'xs/ext'
L = prep(glob.glob(f'{EXT}/tm/*.parquet')); L = L[L['var'] == 'vec'].reset_index(drop=True)
PER = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
MODE = sys.argv[1]
if MODE == 'noise':
    seed, N, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; rng = np.random.default_rng(seed); R_ = []
    for i in range(N):
        z = L.copy(); z['ratio'] = rng.random(len(z)); a = [sim(z, x, y) for x, y in PER]
        k = L.sample(frac=.99, random_state=int(rng.integers(1e9))); b = [sim(k, x, y) for x, y in PER]
        R_.append([v for p in a for v in p] + [v for p in b for v in p]); print(i, flush=True)
    pd.DataFrame(R_, columns=[f'rand_{p}_{m}' for p in ('19', '22', 'ytd') for m in ('cagr', 'dd')] + [f'drop_{p}_{m}' for p in ('19', '22', 'ytd') for m in ('cagr', 'dd')]).to_parquet(out); raise SystemExit
T = taken(L); key = set(zip(T.tk, T.entry)); L['inbook'] = [k in key for k in zip(L.tk, L.entry)]; L['half'] = np.where(L.d.dt.year >= 2022, '22-26', '19-21')
if MODE == 'mine':
    def merge(df, cols):
        global L; L = L.merge(df[['tk', 'h1', 'entry'] + cols].drop_duplicates(['tk', 'h1', 'entry']), on=['tk', 'h1', 'entry'], how='left')
    merge(pd.concat(pd.read_parquet(f) for f in glob.glob(f'{EXT}/fx/*.parquet')), ['age1h', 'dage', 'dstate', 'mkt20', 'sec20', 'res20', 'stk20', 'cp5', 'dse', 'rse'])
    A = pd.concat(pd.read_parquet(f) for f in glob.glob(f'{EXT}/adv/ft/*.parquet')); A['nest'] = A[['n2', 'n5', 'n12', 'n30']].sum(axis=1)
    merge(A, ['nsw', 'n2', 'n5', 'n12', 'n30', 'nest'] + [f'sw{i}' for i in range(10)])
    merge(pd.read_parquet(f'{EXT}/px/feat.parquet'), ['peer_bull', 'peer_new', 'lead', 'opt_vol_trend', 'call_share_trend', 'iv_trend'])
    RV = pd.concat(pd.read_parquet(f) for f in glob.glob(f'{EXT}/rv/*.parquet')); merge(RV[RV['var'] == 'live'], ['used', 'both', 'wk', 'age'])
    L['tmin'] = L.entry.dt.hour * 60 + L.entry.dt.minute; L['h1i'] = L.h1.astype(int)
    B = L[L.inbook]; feats = [c for c in L.columns if c not in ('tk', 'var', 'h1', 'entry', 'exit', 'ret', 'd', 'sp', 'w', 'inbook', 'half', 'liq')]
    rows = []
    for c in feats:
        x = B[c].astype(float)
        if x.notna().sum() < 600 or x.nunique() < 2: continue
        q = pd.qcut(x.rank(method='first'), 3, labels=False) if x.nunique() > 3 else x.rank(method='dense') - 1
        g = B.groupby([q, B.half]).ret.agg(['mean', 'std', 'size']).unstack()
        try:
            lo19, hi19, lo22, hi22 = (g[('mean', '19-21')].iloc[0], g[('mean', '19-21')].iloc[-1], g[('mean', '22-26')].iloc[0], g[('mean', '22-26')].iloc[-1])
            se = lambda h, i: g[('std', h)].iloc[i] / np.sqrt(g[('size', h)].iloc[i])
            t19 = (hi19 - lo19) / np.hypot(se('19-21', 0), se('19-21', -1)); t22 = (hi22 - lo22) / np.hypot(se('22-26', 0), se('22-26', -1))
        except Exception: continue
        rows.append((c, x.notna().sum(), lo19 * 100, hi19 * 100, t19, lo22 * 100, hi22 * 100, t22, np.sign(t19) == np.sign(t22) and min(abs(t19), abs(t22)) >= 2))
    R = pd.DataFrame(rows, columns=['feature', 'n', 'low 19-21', 'high 19-21', 't 19-21', 'low 22-26', 'high 22-26', 't 22-26', 'consistent']).sort_values('t 22-26')
    print('admitted trades only: avg trade % in the lowest vs highest third of each feature, and the t-stat of the gap, by period'); print(R.round(2).to_string(index=False))
    C = R[R.consistent]; print('\nconsistent (same sign, |t| >= 2 in both periods):', list(C.feature))
    P4 = PER[:2] + [(np.datetime64('2019-01-01'), np.datetime64('2027-01-01')), PER[2]]
    def show(nm, Z): print(f'  {nm:<56}' + ''.join(f'{a:>+9.1%} /{b:>5.0%}' for a, b in (sim(Z, x, y) for x, y in P4)), flush=True)
    show('live', L)
    for c in C.feature:
        x = L[c].astype(float); bt = L[L.inbook & (L.half == '19-21')][c]; good_high = R.set_index('feature').loc[c, 't 19-21'] > 0
        cut = bt.quantile(1 / 3 if good_high else 2 / 3); bad = (x < cut) if good_high else (x > cut)
        z = L.copy(); z.loc[bad, 'ret'] = 0.0; show(f'{c}: A bad third held as cash', z); show(f'{c}: B drop bad third, rerun', L[~bad])
        z = L.copy(); z.loc[bad, 'w'] = .5; show(f'{c}: C bad third half size', z)
if MODE == 'state':
    SEC = json.load(open('sp500_sectors.json'))
    L['sec'] = L.tk.map(SEC).fillna('?')
    def sim_state(Z, a, b, rule=None, n=6, brk=.08):
        Z = Z[(Z.entry >= a) & (Z.entry < b)].sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
        op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; last_x = None; rec = []
        for r in Z.itertuples():
            e = r.entry.to_datetime64(); done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
            for xt, rr, w in done: real += np.log1p(rr * w); hist.append((xt, real)); win.append((xt, real)); last_x = xt
            op = [o for o in op if o[0] > e]
            if len(op) >= n: continue
            if brk and hist:
                t0 = e - np.timedelta64(28, 'D')
                while win and win[0][0] < t0: lvl0 = win.popleft()[1]
                if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
            st = dict(used=len(op), same=sum(o[1] == r.sec for o in op), n1h=sum(o[2] for o in op), since=(e - last_x) / np.timedelta64(1, 'm') if last_x is not None else 1e9)
            if rule is not None and not rule(st, r): continue
            op.append((r.exit.to_datetime64(), r.sec, r.h1)); pend.append((r.exit.to_datetime64(), r.ret, r.w / n)); rec.append((r.tk, r.entry, r.ret, r.half, *st.values()))
        for xt, rr, w in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr * w); hist.append((xt, real))
        s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index(); eq = np.exp(s.groupby(s.index.normalize()).last()); y = (eq.index[-1] - eq.index[0]).days / 365.25
        return (eq.iloc[-1] ** (1 / y) - 1, (eq / eq.cummax() - 1).min()), pd.DataFrame(rec, columns=['tk', 'entry', 'ret', 'half', 'used', 'same', 'n1h', 'since'])
    _, REC = sim_state(L, np.datetime64('2019-01-01'), np.datetime64('2027-01-01'))
    print(f'admitted trades {len(REC)} (state recorded at admission)')
    for c, bins, labs in (('used', [-1, 0, 2, 4, 5], ['1st slot', '2nd-3rd', '4th-5th', '6th']), ('same', [-1, 0, 1, 9], ['no same-sector', '1 same', '2+ same']),
                          ('n1h', [-1, 0, 2, 9], ['0 hourly open', '1-2', '3+']), ('since', [-1, 15, 240, 1440, 1e12], ['<=15 min after an exit', '15m-4h', '4h-1d', '>1d / none'])):
        g = REC.groupby([pd.cut(REC[c], bins, labels=labs), 'half']).ret.agg(['mean', 'size']).unstack(); g[[('mean', '19-21'), ('mean', '22-26')]] *= 100
        print(f'\n{c}:'); print(g.round(3).to_string())
    P4 = PER[:2] + [(np.datetime64('2019-01-01'), np.datetime64('2027-01-01')), PER[2]]
    def show(nm, rule=None): print(f'  {nm:<56}' + ''.join(f'{a:>+9.1%} /{b:>5.0%}' for a, b in (sim_state(L, x, y, rule)[0] for x, y in P4)), flush=True)
    print(f'\n{"":<58}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}'); show('live (state scheduler)')
    show('sector cap: max 2 open per sector', lambda s, r: s['same'] < 2); show('sector cap: max 1 open per sector', lambda s, r: s['same'] < 1)
    show('max 3 hourly-track positions', lambda s, r: not (r.h1 and s['n1h'] >= 3))
    show('no re-fill within 15 min of an exit', lambda s, r: s['since'] > 15)
    show('6th slot only for 1h-track signals', lambda s, r: s['used'] < 5 or r.h1)
    # joint stop-outs: worst days of the MTM book
    X = pd.read_parquet(f'{EXT}/vt/series.parquet')['tm:vec']; worst = X.nsmallest(15)
    T2 = taken(L); T2['sec'] = T2.tk.map(SEC)
    print('\n15 worst book days: positions open, how many lost, sectors')
    for d_, v in worst.items():
        o = T2[(T2.entry.dt.normalize() <= d_) & (T2.exit.dt.normalize() >= d_)]
        print(f'  {d_:%Y-%m-%d} {v:+.1%}  open {len(o)}  top sector share {o.sec.value_counts(normalize=True).max() if len(o) else 0:.0%}  {", ".join(o.tk)}')
