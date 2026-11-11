"""Score-dependent STOP on V15a: tight stop when the narrative is bad, wide/none when good. Nov 11 2026.
Same 15m vector-flip entries; exit = opposite flip or the stop (k x ADR20). Score = declared narrative count 0-6 (sweep, MSS, displacement, FVG, ADX not low, RSI>median; thresholds from 2014-21), known at entry.
Per-trade table: pts for each stop k, by score bucket, both eras; then policies stop(score)."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
# trades for each stop (reuse run() from report_v15_nostop)
src = open('/home/claude/fartpie/xs/report_v15_nostop.py').read(); NS1 = {'__name__': 'x'}; exec(src[:src.index("for nm, k in (('with 0.15")], NS1); run = NS1['run']
KS = {'k05': .05, 'k08': .08, 'k10': .10, 'k15': .15, 'k25': .25, 'k50': .50, 'pure': None}
T = {n: run(k).set_index(['day', 'm', 'dir']) for n, k in KS.items()}
keys = T['k15'].index
for n in T: T[n] = T[n][~T[n].index.duplicated()]
common = keys
for n in T: common = common.intersection(T[n].index)
print('trades per stop', {n: len(T[n]) for n in T}, 'common', len(common))
# features (exec ict + htf feature code slices)
ict = open('/home/claude/fartpie/xs/report_ict.py').read(); N2 = {'__name__': 'x'}; exec(ict[:ict.index("for nm in ('V15a', 'RA'):")], N2)
htf = open('/home/claude/fartpie/xs/report_po3_htf.py').read(); N3 = {'__name__': 'x'}; exec(htf[:htf.index("for nm in ('V15a', 'RA'):")], N3)
rows = []
for (D, m, dr) in common:
    D = pd.Timestamp(D).normalize(); f = N2['feats'](D, D + pd.Timedelta(minutes=int(m)), int(dr))
    if not f: continue
    jb = int((int(m) - 570) // 5) - 1
    if jb < 2 or D not in N3['A']: continue
    S = {k: N3['sweeps'](D, jb, k) for k in N3['KINDS']}; st = {k: int(dr) * N3['st'](S[k]) for k in S}
    f['n_with'] = sum(1 for k in st if st[k] == 1); f['day'] = D; f['m'] = int(m); f['dir'] = int(dr); rows.append(f)
F = pd.DataFrame(rows).set_index(['day', 'm', 'dir']); F['era'] = np.where(F.index.get_level_values(0) < '2022-01-01', 'A', 'B')
A_ = F[F.era == 'A']; adx_lo = A_.adx.quantile(1/3); rsi_md = A_.rsi.median()
F['score'] = ((F.n_with >= 1).astype(int) + (F.mss == 1) + (F.disp == 1) + (F.fvg_with == 1) + (F.adx > adx_lo) + (F.rsi > rsi_md)).astype(int)
for n in T: F['pts_' + n] = T[n].pts.reindex(F.index).values; F['mae_' + n] = T[n].mae.reindex(F.index).values
F = F.dropna(subset=['pts_k15']); F.to_pickle(SC + 'stopscore.pkl'); print('trades with features', len(F))
def tstat(x): x = np.asarray(x, float); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
print('\n-- mean pts per trade by score bucket x stop (n; both eras same sign marked *) --')
B = {'score<=2': F.score <= 2, 'score 3-4': (F.score >= 3) & (F.score <= 4), 'score>=5': F.score >= 5, 'ALL': F.score >= 0}
print(f'{"":10} n   ' + ' '.join(f'{n:>10}' for n in KS))
for bn, msk in B.items():
    g = F[msk]; line = f'{bn:10} {len(g):4d} '
    for n in KS:
        a, b = g[g.era == 'A']['pts_' + n], g[g.era == 'B']['pts_' + n]; line += f' {g["pts_" + n].mean():+7.1f}{"*" if np.sign(a.mean()) == np.sign(b.mean()) else " "}  '
    print(line)
print('\n-- worst-case / average adverse excursion by bucket (mae_k15 mean) --')
for bn, msk in B.items(): g = F[msk]; print(f'{bn:10} mean MAE (0.15 stop) {g.mae_k15.mean():+6.1f} | pure {g.mae_pure.mean():+6.1f} | loss rate k15 {np.mean(g.pts_k15<0):.0%} pure {np.mean(g.pts_pure<0):.0%}')
pol = {}
def P(lo, mid, hi): return np.select([F.score <= 2, F.score >= 5], [F['pts_' + lo], F['pts_' + hi]], F['pts_' + mid]), np.select([F.score <= 2, F.score >= 5], [F['mae_' + lo], F['mae_' + hi]], F['mae_' + mid])
print('\n-- policies stop(score): low / mid / high --')
print(f'{"policy":28} {"mean":>6} {"win":>4} {"worst":>6} {"A mean(t)":>14} {"B mean(t)":>14} {"sum":>7} {"yrs<0":>6}')
for lo, mid, hi in (('k15', 'k15', 'k15'), ('k08', 'k15', 'k15'), ('k05', 'k15', 'k15'), ('k05', 'k10', 'k15'), ('k08', 'k15', 'k25'), ('k08', 'k15', 'k50'), ('k05', 'k15', 'k50'), ('k05', 'k15', 'pure'), ('k08', 'k15', 'pure'), ('k10', 'k15', 'k25'), ('k15', 'k15', 'k25'), ('k15', 'k15', 'k50'), ('k15', 'k15', 'pure'), ('k25', 'k25', 'k25')):
    y, mae = P(lo, mid, hi); y = pd.Series(y, index=F.index); a, b = y[F.era == 'A'], y[F.era == 'B']; ey = y.groupby(F.index.get_level_values(0).year).sum()
    print(f'{lo + "/" + mid + "/" + hi:28} {y.mean():+6.1f} {np.mean(y>0):4.0%} {y.min():+6.0f} {a.mean():+7.1f} ({tstat(a):+.1f}) {b.mean():+7.1f} ({tstat(b):+.1f}) {y.sum():+7.0f} {int((ey<0).sum()):>3}/{len(ey)}')
