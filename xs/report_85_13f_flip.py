"""Nov 14 2026 13F POSITIONING x VECTOR FLIP. Rey: fund positions (Renaissance etc.) then enter on the vector flip. Declared before results. Funds fixed in fetch_13f.py: CONC (Berkshire, Pershing, Appaloosa, Baupost, ThirdPoint, Greenlight, TigerGlobal, LonePine, Viking, Coatue, Fairholme) and QUANT (Renaissance, Citadel, DEShaw, Point72). Equity long rows only (puts/calls excluded). A fund-quarter BUY on a ticker = new position, or shares up >=25% vs that fund's previous 13F. Flag is live from the day AFTER the filing date (not period end) for 90 calendar days. Flip trades = daily vector long flips (as elsewhere: entry next open, exit open after opposite flip, 30bp). Flagged = flip date inside a flag window for that ticker. Groups: CONC_BUY, QUANT_BUY, ANY. Compare flagged vs unflagged flips: mean, median, share >=+100%, ex-top-3 mean, per era (2019-21 / 2022-26Q1), month-clustered t of flagged-minus-unflagged. PASS = flagged beats unflagged by >=2pp mean in BOTH eras, pooled t>=2, n flagged >=150 per era. Second test (no flips): 60-day forward excess return vs equal-weight universe from flag day open, flagged vs not. Coverage caveat: only CUSIPs that map to universe tickers; survivorship-biased universe; 13F covers quarter-end longs only."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'
A = pd.read_pickle(X + 'f13_holdings.pkl'); A = A[A.putcall.isna()]; M = pickle.load(open(X + 'cusip_map.pkl', 'rb')); A['tk'] = A.cusip.map(M); A = A[A.tk.fillna('') != '']
A['filed'] = pd.to_datetime(A.filed); A['period'] = pd.to_datetime(A.period); H = A.groupby(['fund', 'grp', 'period', 'filed', 'tk']).shares.sum().reset_index()
print('mapped holdings rows', len(H), 'tickers', H.tk.nunique(), 'fund-quarters', H.groupby(['fund', 'period']).ngroups)
ev = []
for f, z in H.groupby('fund'):
    pers = sorted(z.period.unique()); prev = None
    for p in pers:
        cur = z[z.period == p].set_index('tk'); fd = cur.filed.iloc[0]
        if prev is not None:
            for tk, r in cur.iterrows():
                s0 = prev.shares.get(tk, 0.0)
                if s0 == 0 or r.shares >= 1.25 * s0: ev.append((f, r.grp, tk, fd))
        prev = cur
E_ = pd.DataFrame(ev, columns=['fund', 'grp', 'tk', 'filed']); print('buy events', len(E_), E_.groupby('grp').size().to_dict())
D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); win = {}
for g in ('CONC', 'QUANT', 'ANY'):
    s = E_ if g == 'ANY' else E_[E_.grp == g]
    for tk, d in s.groupby('tk'): win.setdefault((g, tk), []).extend([(x + pd.Timedelta(days=1), x + pd.Timedelta(days=91)) for x in d.filed])
opn = pd.DataFrame({tk: B.open[~B.index.duplicated()] for tk, B in D.items()}).sort_index(); ro = opn.pct_change(fill_method=None).clip(-.5, 1.0); Lm = np.log1p(ro.mean(axis=1).fillna(0)).cumsum()
rows = []
for tk, B in D.items():
    B = B[~B.index.duplicated()]; B = B[B.index >= '2018-06-01']
    if len(B) < 300: continue
    o = B.open.values; idx = B.index; reg, lev, atr, _ = run_state_machine(B.assign(datetime=B.index)); n = len(B); i = 1
    while i < n - 2:
        if reg[i] == 1 and reg[i - 1] != 1:
            j = i + 1
            while j < n and reg[j] != -1: j += 1
            if j + 1 < n and idx[i] >= pd.Timestamp('2019-01-01') and idx[i] < pd.Timestamp('2026-03-01'):
                fl = {g: any(a <= idx[i] <= b for a, b in win.get((g, tk), [])) for g in ('CONC', 'QUANT', 'ANY')}
                rows.append((tk, idx[i], o[j + 1] / o[i + 1] - 1 - 0.003, fl['CONC'], fl['QUANT'], fl['ANY']))
            i = j
        i += 1
F = pd.DataFrame(rows, columns=['tk', 'd', 'r', 'CONC', 'QUANT', 'ANY']); F['era'] = np.where(F.d < '2022-01-01', '2019-21', '2022-26Q1'); F['mo'] = F.d.dt.to_period('M'); print('\nflips', len(F), 'flagged ANY', int(F.ANY.sum()), 'CONC', int(F.CONC.sum()), 'QUANT', int(F.QUANT.sum()))
def st(x): return f'n {len(x):5d} mean {x.mean()*100:+6.1f}% med {x.median()*100:+5.1f}% huge {(x>=1).mean()*100:4.1f}% ex-top3 {x.sort_values().iloc[:-3].mean()*100 if len(x)>5 else np.nan:+5.1f}%'
def mct(z, g):
    a = z[z[g]].groupby('mo').r.mean(); b = z[~z[g]].groupby('mo').r.mean(); d = (a - b).dropna(); return (d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 5 else np.nan), (z[z[g]].r.mean() - z[~z[g]].r.mean()) * 100
res = {}
for g in ('ANY', 'CONC', 'QUANT'):
    print(f'\n== {g} 13F-buy flag ==')
    for en in ('2019-21', '2022-26Q1', 'POOLED'):
        z = F if en == 'POOLED' else F[F.era == en]; t, d = mct(z, g); res[(g, en)] = (d, t, int(z[g].sum()))
        print(f'  {en:9s} FLAGGED   {st(z[z[g]].r)}\n  {"":9s} UNFLAGGED {st(z[~z[g]].r)}\n  {"":9s} diff {d:+.1f}pp  month-clustered t {t:+.1f}')
    print('  PASS' if all(res[(g, e)][0] >= 2 and res[(g, e)][2] >= 150 for e in ('2019-21', '2022-26Q1')) and res[(g, 'POOLED')][1] >= 2 else '  no pass')
# second test: 60d forward excess from flag day, per flagged event vs all stock-days (sampled)
mk = {}; fw = 60; rows2 = []
for _, r in E_.iterrows():
    B = D.get(r.tk)
    if B is None: continue
    B = B[~B.index.duplicated()]; k = B.index.searchsorted(r.filed + pd.Timedelta(days=1))
    if k + fw + 1 >= len(B) or B.index[k] < pd.Timestamp('2019-01-01') or B.index[k] > pd.Timestamp('2026-02-01'): continue
    d0 = B.index[k]; ra = B.open.iloc[k + fw] / B.open.iloc[k] - 1; m = np.exp(Lm.get(B.index[k + fw], np.nan) - Lm.get(d0, np.nan)) - 1; rows2.append((r.grp, r.fund, d0, ra - m - 0.003))
G = pd.DataFrame(rows2, columns=['grp', 'fund', 'd', 'ex']); G['era'] = np.where(G.d < '2022-01-01', '2019-21', '2022-26Q1'); G['mo'] = G.d.dt.to_period('M')
print('\n== 60-day excess return after a 13F buy becomes public (vs equal-weight universe, net 30bp) ==')
for g in ('CONC', 'QUANT'):
    for en in ('2019-21', '2022-26Q1'):
        z = G[(G.grp == g) & (G.era == en)]; c = z.groupby('mo').ex.mean(); print(f'  {g:5s} {en:9s} n {len(z):6d} mean {z.ex.mean()*100:+5.2f}% med {z.ex.median()*100:+5.2f}% month-clustered t {c.mean()/(c.std()/np.sqrt(len(c))):+.1f}')
print('  by fund (pooled):'); 
for f, z in G.groupby('fund'): print(f'    {f:12s} n {len(z):5d} mean {z.ex.mean()*100:+5.2f}%')
F.to_pickle(X + 'f13_flips.pkl')
