"""Nov 15 2026 TIME OF YEAR AND LIQUIDITY ROTATION (Rey: 'time of year' and 'companies move more when they take money out of others: long when other companies are in a negative vector'). Declared before results. Daily flips (report_90 file, 30,323 bull flips, 2019-Mar 2026, entry next open, exit open after the bear flip, 30bp, survivorship-biased), TRAIN 2019-21 / TEST 2022+, plus a LARGE-CAP subset (20d median dollar volume >= $100M, price >= $10). TEST A SEASON: net return by calendar month of the flip; declared hypothesis: Nov-Apr flips beat May-Oct flips (sign + on BOTH eras); also report the month-mean rank correlation between eras (is any month pattern stable?). TEST B ROTATION: bear share = share of the universe (all names with a vector state that prior close, self excluded by construction of the share) in the vector BEAR state at the close BEFORE the flip day; declared hypothesis: flips fire better when many OTHER names are bearish (money leaving others), sign = higher bear share -> higher return, on both eras. B2 SECTOR rotation: share of same-2-digit-SIC names in bear state (needs >= 5 peers), same declared sign. Metrics: Spearman IC of the share vs net return, quintiles by TRAIN-era cut points (mean/median net, win10, huge share >=+100%), month-clustered t of top-minus-bottom quintile, and a control: the equal-weight universe's forward 20-day open-to-open return by bear-share bucket (separates 'good market timing' from 'good flip selection'). Lead bar: sign as declared on BOTH eras, TEST month-clustered t >= 2, and the effect is not just the market control."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from vector_pine import run_state_machine
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); F = pd.read_pickle(X + 'stack_flips.pkl'); sic = pd.read_csv(X + 'sic.csv').drop_duplicates('tk').set_index('tk'); COST = .003
regs = {}; opens = {}
for tk, q in D.items():
    q = q[['open', 'high', 'low', 'close', 'volume']].copy(); q = q[~q.index.duplicated()].dropna(); q.index = pd.to_datetime(q.index).tz_localize(None).normalize()
    if len(q) < 300: continue
    r, _, _, _ = run_state_machine(q.assign(datetime=q.index)); regs[tk] = pd.Series(r, index=q.index); opens[tk] = q.open
R = pd.DataFrame(regs); O = pd.DataFrame(opens); print('universe matrix', R.shape)
nb = (R == -1).sum(axis=1); nt = R.notna().sum(axis=1); bear_share = (nb / nt).shift(1); ew = (O.shift(-20) / O - 1).mean(axis=1)
sec = sic.sic.floordiv(100); grp = {s: [t for t in R.columns if sec.get(t, np.nan) == s] for s in sec.dropna().unique()}
secshare = {}
for s, tks in grp.items():
    if len(tks) >= 6: S = R[tks]; secshare[s] = ((S == -1).sum(axis=1) / S.notna().sum(axis=1)).shift(1)
F = F.copy(); F['bear'] = F.d.map(bear_share); F['sec2'] = F.tk.map(sec); F['secbear'] = [secshare.get(s, pd.Series(dtype=float)).get(d, np.nan) if s == s else np.nan for s, d in zip(F.sec2, F.d)]; F['ew20'] = F.d.map(ew)
F['n'] = F.r - COST; F['mo'] = F.d.dt.to_period('M').astype(str); F['m'] = F.d.dt.month; F['era'] = np.where(F.test, 'TEST', 'TRAIN'); F['lc'] = (F.dv >= 1e8) & (F.px >= 10)
def stt(z): s = z.n.sort_values(ascending=False); return f'n {len(z):5d} mean {z.n.mean()*100:+6.2f}% med {z.n.median()*100:+6.2f}% win10 {(z.n>=.1).mean()*100:5.1f}% huge {(z.n>=1).mean()*100:4.2f}% ex-top3 {s.iloc[3:].mean()*100:+6.2f}%'
def mdiff(z, a, b): g = z.assign(x=0).groupby(['mo']).apply(lambda w: (w[a].mean() if len(w[a]) else np.nan)); return g
def topbot(z, col, lo, hi):
    zz = z.dropna(subset=[col]); t = zz[zz[col] >= hi]; b = zz[zz[col] <= lo]; mt = t.groupby('mo').n.mean(); mb = b.groupby('mo').n.mean(); d = (mt - mb).dropna()
    return len(t), len(b), t.n.mean() * 100, b.n.mean() * 100, (d.mean() * 100, d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 5 else np.nan)
for universe, U in (('ALL', F), ('LARGE-CAP (dv>=$100M, px>=$10)', F[F.lc])):
    print(f'\n######## {universe} ########')
    print('\nTEST A SEASON: mean net by flip month (train | test) and winter(Nov-Apr) vs summer(May-Oct)')
    for m in range(1, 13): a = U[(U.era == 'TRAIN') & (U.m == m)].n; b = U[(U.era == 'TEST') & (U.m == m)].n; print(f'  month {m:2d}  train n {len(a):4d} {a.mean()*100:+6.2f}% | test n {len(b):4d} {b.mean()*100:+6.2f}%')
    mm = pd.DataFrame({'a': U[U.era == 'TRAIN'].groupby('m').n.mean(), 'b': U[U.era == 'TEST'].groupby('m').n.mean()}).dropna(); print(f'  month-mean rank corr train vs test {spearmanr(mm.a, mm.b)[0]:+.2f}')
    for era in ('TRAIN', 'TEST'):
        z = U[U.era == era]; w = z[z.m.isin([11, 12, 1, 2, 3, 4])]; s = z[z.m.isin([5, 6, 7, 8, 9, 10])]; g = z.assign(w=z.m.isin([11, 12, 1, 2, 3, 4])).groupby(['mo', 'w']).n.mean().unstack(); d = (g[True] - g[False]).dropna(); print(f'  {era}: winter {stt(w)} | summer {stt(s)} | winter-summer month-t {d.mean()/(d.std()/np.sqrt(len(d))):+.2f}')
    print('\nTEST B ROTATION: bear share (other names in vector BEAR at the prior close), market and sector')
    cutb = U[U.era == 'TRAIN'].bear.quantile([.2, .8]).values; cuts = U[U.era == 'TRAIN'].secbear.quantile([.2, .8]).values
    for col, cuts_ in (('bear', cutb), ('secbear', cuts)):
        print(f' -- {col}: train 20/80 cut {cuts_.round(3)}')
        for era in ('TRAIN', 'TEST'):
            z = U[U.era == era].dropna(subset=[col]); ic = spearmanr(z[col], z.n)[0]; nt_, nb_, mt, mb, (dm, dt) = topbot(z, col, cuts_[0], cuts_[1]); print(f'   {era} IC {ic:+.3f} | HIGH bear share n {nt_} {mt:+.2f}% vs LOW n {nb_} {mb:+.2f}% | top-minus-bottom {dm:+.2f}pp month-t {dt:+.2f}')
    z = U.dropna(subset=['bear', 'ew20']); print(' -- control: equal-weight universe fwd 20d open-to-open by bear-share bucket (all dates a flip occurred):')
    z = z.assign(b=pd.qcut(z.bear, 5, labels=False))
    for era in ('TRAIN', 'TEST'): zz = z[z.era == era]; print(f'   {era} ew20 mean by bear quintile 1..5: ' + ' '.join(f'{zz[zz.b==k].ew20.mean()*100:+5.1f}%' for k in range(5)) + ' | flip net by quintile: ' + ' '.join(f'{zz[zz.b==k].n.mean()*100:+5.1f}%' for k in range(5)))
