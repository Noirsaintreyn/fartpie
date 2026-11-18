"""Nov 18 2026 report 329. PAPER CLAIM (desk D1a composite Z 'revisions' leg; Citi/GS notes; Harvest 'event/expectation'): analyst revisions lead returns.  Declared before running.
THEORY: sell-side upgrades convey information and move slowly-updating clients (herding, 'revisions are autocorrelated'); small caps have few analysts, so each upgrade is a larger share of the information set.  Direction: upgrade -> positive abnormal drift; downgrade -> negative.
Why it fails: upgrades chase price (momentum already in, the same news as the earnings print), and the rating-change tape is sparse and partly paid-for coverage in tiny names.
DATA LIMIT: yfinance rating changes only go back to ~2021 (no 2019-21 train under the strict split) -> split INTERNALLY: 2021-23 vs 2024-26 (flagged; this cannot satisfy the stated train/test rule).
Entry = close of the rating date (reaction already seen), abnormal vs equal-weight universe, price>=$1, 20d $vol>=$300k.  Windows d5/d20/d60.  Also: net upgrade breadth over the prior 90 days (ups - downs) quintiles -> next-20d abnormal."""
import glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
C = pd.read_parquet(SP + 'sm_close.parquet'); V = pd.read_parquet(SP + 'sm_vol.parquet'); C = C.dropna(how='all', axis=1); idx = C.index
R = C.pct_change().replace([np.inf, -np.inf], np.nan).clip(-.9, 3); EW = R.mean(axis=1); CA = R.sub(EW, axis=0).fillna(0).cumsum(); DV = (C * V).rolling(20, min_periods=15).median()
rows = []
for f in glob.glob('data/fund/yf_small/*_ud.parquet'):
    tk = os.path.basename(f)[:-11]
    if tk not in C.columns: continue
    u = pd.read_parquet(f)
    if u.empty or 'Action' not in u: continue
    u = u[u.Action.isin(['up', 'down'])]
    for t, a in zip(u.index, u.Action):
        i = idx.searchsorted(pd.Timestamp(t).tz_localize(None).normalize())
        if i < 2 or i + 62 >= len(idx) or not (C[tk].iloc[i] >= 1 and DV[tk].iloc[i] >= 3e5): continue
        ca = CA[tk].values; rows.append(dict(tk=tk, day=idx[i], up=(a == 'up'), pre=ca[i] - ca[i - 20], d5=ca[i + 5] - ca[i], d20=ca[i + 20] - ca[i], d60=ca[i + 60] - ca[i], px=C[tk].iloc[i]))
E = pd.DataFrame(rows); E['m'] = E.day.dt.to_period('M'); E = E[E.day >= '2021-01-01']; E['h1'] = E.day < '2024-01-01'; print(f'rating changes {len(E)} (up {int(E.up.sum())}); 2021-23 {int(E.h1.sum())}, 2024-26 {int((~E.h1).sum())}')
def tt(s, m):
    g = s.groupby(m.loc[s.index]).mean(); return f'{s.mean() * 100:+5.2f}% (t{g.mean() / (g.std() / np.sqrt(len(g))) if len(g) > 5 else np.nan:+4.1f},n{len(s):5d})'
print('abnormal drift after a rating change from the date close;   [2021-23 | 2024-26]')
for w in ('d5', 'd20', 'd60'):
    for lab, m in (('up', E.up), ('down', ~E.up)):
        a, b = E[m & E.h1], E[m & ~E.h1]; print(f'  {w} {lab:4s}: {tt(a[w], a.m)} | {tt(b[w], b.m)}')
print('  prior-20d abnormal return before the change: up', round(E[E.up].pre.mean() * 100, 2), '% down', round(E[~E.up].pre.mean() * 100, 2), '%  (chasing price if up>0)')
for lab, s in (('price<$5', E.px < 5), ('price>=$5', E.px >= 5)):
    for w in ('d20',):
        a, b = E[E.up & s & E.h1], E[E.up & s & ~E.h1]; print(f'  {w} UP {lab}: {tt(a[w], a.m)} | {tt(b[w], b.m)}')
E.to_parquet(SP + 'e329.parquet')
