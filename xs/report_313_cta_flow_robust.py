"""Nov 18 2026 report 313. Robustness of the report-312 finding (algo ALL pts/day sorts by sign of ex-ante CTA-proxy NQ position change: +31|+53 after increases, -27|-16 after decreases, +13|+22 unchanged). EXPLORATORY: sign-of-dpos was chosen after seeing terciles degenerate, so this is the pre-registration run for it. Declared checks: (a) ES-based dpos, (b) alternate windows 10/42/252 with stops 2/3/4/5, (c) shuffled-position placebo (shift dpos by 7 days), (d) tilt value: ALL pts/day with size x1.5 after increases and x0.5 after decreases, mean and std ratio, train|test. Pass = same sign ordering in both eras for (a),(b), placebo ~0."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
D = pd.read_parquet(SP + 'd312.parquet')
def split(s, lab):
    out = []
    for nm, m in (('train', D.tr), ('test', ~D.tr)):
        z = D[m].assign(s=np.sign(s[m].round(3))); out.append(f'{nm} up {z[z.s > 0].ALL.mean():+6.1f} (n{int((z.s > 0).sum())}) down {z[z.s < 0].ALL.mean():+6.1f} (n{int((z.s < 0).sum())}) flat {z[z.s == 0].ALL.mean():+6.1f}')
    print(f'{lab:28s}', ' | '.join(out))
split(D.NQ_dpos, 'NQ dpos (report 312)'); split(D.ES_dpos, '(a) ES dpos'); split(D.NQ_dpos.shift(7).fillna(0), '(c) placebo NQ dpos lag 7'); split(D.NQ_dpos.sample(frac=1, random_state=1).set_axis(D.index), '(c) placebo shuffled')
# (b) alternate proxy
def daily(path):
    H = pd.read_parquet(path); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H['day'] = H.dt.dt.normalize(); g = H.groupby('day'); return pd.DataFrame({'c': g.close.last(), 'h': g.high.max(), 'l': g.low.min()})
x = daily(SP + 'lsefut/lse_ohlcv_cache/NASDAQ100.F_15m_2017-04-01_2026-11-11.parquet'); c = x.c; atr = (x.h - x.l).rolling(14).mean(); out = []
for w in (10, 42, 252):
    tr = np.sign(c - c.shift(w)); pk = c.rolling(w).max(); tg = c.rolling(w).min()
    for k in (2, 3, 4, 5): out.append(((tr > 0) & (c > pk - k * atr)).astype(float) - ((tr < 0) & (c < tg + k * atr)).astype(float))
alt = pd.concat(out, axis=1).mean(axis=1).diff().shift(1).reindex(D.index); split(alt, '(b) alt windows/stops')
for w in ((21,), (63,), (126,)):
    out = []; 
    for ww in w:
        tr = np.sign(c - c.shift(ww)); pk = c.rolling(ww).max(); tg = c.rolling(ww).min()
        for k in (3, 4, 5, 6): out.append(((tr > 0) & (c > pk - k * atr)).astype(float) - ((tr < 0) & (c < tg + k * atr)).astype(float))
    split(pd.concat(out, axis=1).mean(axis=1).diff().shift(1).reindex(D.index), f'(b2) single window {w[0]}d')
# (d) tilt value
s = np.sign(D.NQ_dpos.round(3)); wt = np.where(s > 0, 1.5, np.where(s < 0, .5, 1.0))
for nm, m in (('train', D.tr), ('test', ~D.tr)):
    a = D[m].ALL; b = a * wt[m.values]; print(f'(d) tilt {nm}: mean {a.mean():+.1f} -> {b.mean():+.1f} pts/day; std {a.std():.0f} -> {b.std():.0f}; worst day {a.min():.0f} -> {b.min():.0f}; mean/std {a.mean() / a.std():.3f} -> {b.mean() / b.std():.3f}')
