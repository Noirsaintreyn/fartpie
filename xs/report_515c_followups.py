"""515c follow-ups: (1) is HMM state-2 a precursor or just 'already running'? (2) big runs (peak>=+300%) vs ordinary doubles: catalyst share. (3) what the unexplained runs look like."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; X = SP + 'ext/'
P = pd.read_pickle(X + 'runs515_panel2.pkl'); TR = (P.yr >= 2019) & (P.yr <= 2021); P['ab'] = pd.qcut(P.ATRP.rank(method='first'), 5, labels=False)
K = pd.read_csv(X + '8k_wide.csv'); KG = {t: 1 for t in K.ticker.unique()}
def lift(flag, mask):
    d = P[mask].assign(f=np.asarray(flag)[mask.values]); d = d[d.f.isin([0, 1])]
    out = []
    for (m, b), g in d.groupby(['mon', 'ab']):
        a, c = g[g.f == 1].y, g[g.f == 0].y
        if len(a) >= 3 and len(c) >= 3: out.append((m, b, a.mean() - c.mean(), len(a)))
    if not out: return np.nan, np.nan, np.nan, 0
    o = pd.DataFrame(out, columns=['m', 'b', 'x', 'n']); mm = o.groupby('m').apply(lambda z: np.average(z.x, weights=z.n)); t = mm.mean() / (mm.std() / np.sqrt(len(mm))) if len(mm) > 2 else np.nan
    return d[d.f == 1].y.mean(), d[d.f == 0].y.mean(), mm.mean(), t


E = pd.read_csv('xs/report_515_episodes.csv'); has = E.tk.isin(KG.keys())
print('\n== (2) catalyst share by run size (names with 8-K coverage)')
E['size'] = pd.cut(E.peak40, [0.99, 1.5, 2.99, 99], labels=['100-150%', '150-300%', '300%+'])
print(E[has].groupby('size').cat.value_counts(normalize=True).unstack().round(2).to_string()); print(E[has].groupby('size').size().to_dict())
print('trigger-day gap>=+15% share (opening gap = overnight news):', round((E.trig_gap >= .15).mean(), 2), '| trig_ret>=+50%:', round((E.trig_ret >= .5).mean(), 2))
print('start year of 300%+ runs:', E[E.peak40 >= 3].start.astype(str).str[:4].value_counts().sort_index().to_dict())
print('\n== (1) HMM state-2 split by whether the stock had ALREADY run (r20 >= +30%)')
P['ran'] = (P.r20 >= .3).astype(int)
for lab, m in (('hs=2 & not yet run (r20<30%)', (P.hs == 2) & (P.ran == 0)), ('hs=2 & already run', (P.hs == 2) & (P.ran == 1)), ('hs<2 (all)', P.hs < 2)):
    print(f'{lab:32s} n={m.sum():6d}', {e: round(P[m & (P.era == e)].y.mean(), 3) for e in ['2018', '2019-21', '2022', '2023-26']}, 'ATRP med', round(P[m].ATRP.median(), 3))
f = ((P.hs == 2) & (P.ran == 0)).astype(int)
for nm, mk in (('train', TR), ('2018', P.yr == 2018), ('2022', P.yr == 2022), ('2023-26', P.yr >= 2023)):
    p1, p0, d, t = lift(f, mk); print('  within-ATRP lift hs=2&notrun', nm, f'{d:+.3f} t {t:.1f}')
print('\n== 8 largest runs'); print(E.sort_values('peak40', ascending=False).head(10)[['tk', 'start', 'px', 'days_to_2x', 'peak40', 'trig_day', 'trig_gap', 'items', 'cat']].to_string(index=False))
