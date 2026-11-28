"""515d: the news samples (33 hand-checked runs) show earnings/guidance and scheduled events among the triggers. Scheduled events are forecastable, so test: does an earnings release (8-K item 2.02, filed in the NEXT 1-14 calendar days) raise P(double<=40d) within ATRP quintile? Uses the actual 2.02 date as the 'calendar' (earnings dates are announced ~2-4 weeks ahead and repeat quarterly, so this is a mild foresight proxy, disclosed)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; X = SP + 'ext/'
P = pd.read_pickle(X + 'runs515_panel2.pkl'); TR = (P.yr >= 2019) & (P.yr <= 2021); P['ab'] = pd.qcut(P.ATRP.rank(method='first'), 5, labels=False)
K = pd.read_csv(X + '8k_wide.csv').dropna(subset=['items']); K = K[K['items'].astype(str).str.contains('2.02')]; K['filed'] = pd.to_datetime(K.filed).astype('datetime64[ns]'); EG = {t: np.sort(g.filed.values) for t, g in K.groupby('ticker')}
cov = P.tk.isin(EG.keys()); P = P[cov].copy(); out = np.zeros(len(P), bool); nxt = np.full(len(P), np.nan)
for tk, idx in P.reset_index(drop=True).groupby('tk').indices.items():
    e = EG[tk]; d = P.d.values[idx]; j = np.searchsorted(e, d, side='right'); has = j < len(e); gap = np.where(has, (e[np.clip(j, 0, len(e) - 1)] - d) / np.timedelta64(1, 'D'), np.nan); nxt[idx] = gap
P['nxt'] = nxt; P['E14'] = ((P.nxt >= 1) & (P.nxt <= 14)).astype(int); P['E40'] = ((P.nxt >= 1) & (P.nxt <= 40)).astype(int)
def lift(flagcol, mask):
    d = P[mask & P[flagcol].isin([0, 1])]; r = []
    for (m, b), g in d.groupby(['mon', 'ab']):
        a, c = g[g[flagcol] == 1].y, g[g[flagcol] == 0].y
        if len(a) >= 3 and len(c) >= 3: r.append((m, a.mean() - c.mean(), len(a)))
    o = pd.DataFrame(r, columns=['m', 'x', 'n']); mm = o.groupby('m').apply(lambda z: np.average(z.x, weights=z.n)); return d[d[flagcol] == 1].y.mean(), d[d[flagcol] == 0].y.mean(), mm.mean(), mm.mean() / (mm.std() / np.sqrt(len(mm)))
print('names with 2.02 coverage', P.tk.nunique(), 'days', len(P), 'share E14', round(P.E14.mean(), 2), 'E40', round(P.E40.mean(), 2))
for c in ('E14', 'E40'):
    for nm, mk in (('train 2019-21', TR), ('2018', P.yr == 2018), ('2022', P.yr == 2022), ('2023-26', P.yr >= 2023)):
        a, b, d, t = lift(c, mk); print(f'{c} {nm:14s} P1 {a:.3f} P0 {b:.3f}  within-ATRP lift {d:+.3f} t {t:.1f}')
