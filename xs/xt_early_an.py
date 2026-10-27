import pandas as pd, numpy as np
A = pd.read_parquet('/tmp/claude-0/xt_early.parquet'); A['h2'] = np.where(A.yr <= 2021, 'a19-21', 'b22-26')
nd = A.groupby(['tk', 'd']).ngroups
def first(M):
    F = A[M].sort_values('t').groupby(['tk', 'd', 'side']).head(1); return F
def stat(nm, F):
    r = [nm, f'{len(F) / nd:.2f}', f'{F.t.median() // 60:.0f}:{F.t.median() % 60:02.0f}']
    for _, G in F.groupby('h2'):
        r += [f"{G.held.mean():.0%}", f"{G.fav.mean() * 1e4:.0f}", f"{G.adv.mean() * 1e4:.0f}", f"{G.close.mean() * 1e4:+.1f}"]
    print('  '.join(f'{x:>7}' for x in r))
print('rule                per-day  med t | 19-21: held  fav  risk  close(bp) | 22-26: held fav risk close')
lv = A.ctx != 'fresh'
stat('LIVE p85 10:15+', first((A.p >= 85) & lv & (A.t >= 615) & (A.t <= 945)))
for t in (630, 690, 750, 810):
    stat(f'BASE clock {t//60}:{t%60:02d}', A[A.t == t])
for T in (60, 70, 75, 80, 85):
    for cap in (660, 720, 780):
        stat(f'p{T} by {cap//60}:{cap%60:02d}', first((A.p >= T) & lv & (A.t >= 600) & (A.t <= cap)))
for cs, nm in ((('15m', 'both'), '15m/both'), (('1h', 'both'), '1h/both'), (('15m', '1h', 'both'), 'any flip'), (('both',), 'both')):
    for cap in (660, 720, 780):
        stat(f'{nm} by {cap//60}:{cap%60:02d}', first(A.ctx.isin(cs) & (A.t >= 600) & (A.t <= cap)))
