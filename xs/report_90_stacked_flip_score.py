"""Nov 14 2026 STACKED FLIP SCORE on the big universe (Rey: 'a great signal, big universe'; imagine classifying good flips from bad). Strict split: TRAIN 2019-21 chooses which features enter and their SIGN; TEST 2022-Mar 2026 is scored once with frozen choices. Candidate features (all known at the flip-bar close): VOLX flip-bar volume / prior 20d mean (validated earlier, high), PEERS other same-2-digit-SIC flips in prior 5 calendar days (post-hoc lead, low), VR ATR10/ATR60 (post-hoc lead, high), EXT close/60d low-1, R60 60d return, BLOW 20d/120d realized vol, LDV log 20d median dollar volume, LPX log price (supply). Rule (no fitting of weights): a feature enters iff its TRAIN month-clustered Spearman IC with the trade return has |t|>=1.5; sign = sign of the train IC; score = mean of the within-month percentile ranks (signed). Also a ridge logistic (target trade return >= +10%) trained on 2019-21 and frozen. Evaluate on TEST only: top quintile vs bottom quintile by score (within month): n, mean, median, huge (>=+100%) rate, win10 rate, AUC for win10, month-clustered t of top-minus-bottom; by price bucket (all, $1-10, $10+). PASS = top-bottom > 0 on TEST with t>=2 and top-quintile mean beats the unconditional test mean, in all three price buckets or at least 'all' and one bucket, with AUC(win10) >= 0.55. Survivorship-flattered universe; 30bp costs (small-cap costs understated)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; F = pd.read_pickle(X + 'flip_context.pkl'); D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl')
vr = {}
for tk, B in D.items():
    B = B[~B.index.duplicated()]; vr[tk] = (B.volume / B.volume.rolling(20).mean().shift(1)).replace([np.inf, -np.inf], np.nan)
F['VOLX'] = [vr.get(t, pd.Series(dtype=float)).get(d, np.nan) for t, d in zip(F.tk, F.d)]
sic = pd.read_csv(X + 'sic.csv').drop_duplicates('tk').set_index('tk').sic; F['g'] = F.tk.map(sic) // 100; F['dn'] = F.d.dt.normalize(); by = {g: np.sort(z.dn.values.astype('datetime64[D]').astype(int)) for g, z in F.dropna(subset=['g']).groupby('g')}
F['PEERS'] = [((by[g] >= np.datetime64(d, 'D').astype(int) - 5) & (by[g] < np.datetime64(d, 'D').astype(int))).sum() if g == g else np.nan for g, d in zip(F.g, F.dn)]
F['LDV'] = np.log(F.dv); F['LPX'] = np.log(F.px); FE = ['VOLX', 'PEERS', 'VR', 'EXT', 'R60', 'BLOW', 'LDV', 'LPX']; F = F.dropna(subset=FE + ['r']); F['win10'] = F.r >= .10; F['test'] = F.d >= '2022-01-01'
for f in FE: F['k_' + f] = F.groupby('mo')[f].rank(pct=True)
tr = F[~F.test]; use = {}
print('TRAIN (2019-21) month-clustered Spearman IC of feature vs trade return:')
for f in FE:
    ic = tr.groupby('mo').apply(lambda z: spearmanr(z[f], z.r)[0] if len(z) > 8 else np.nan).dropna(); t = ic.mean() / (ic.std() / np.sqrt(len(ic))); print(f'  {f:6s} IC {ic.mean():+.3f} t {t:+.1f}' + ('  -> IN' if abs(t) >= 1.5 else '')); 
    if abs(t) >= 1.5: use[f] = np.sign(ic.mean())
print('features used:', use)
F['score'] = sum((F['k_' + f] if s > 0 else 1 - F['k_' + f]) for f, s in use.items()) / max(len(use), 1)
# ridge logistic frozen on train
from sklearn.linear_model import LogisticRegression
Z = lambda d: np.column_stack([d['k_' + f] for f in FE]); m = LogisticRegression(C=0.5, max_iter=500).fit(Z(F[~F.test]), F[~F.test].win10); F['lr'] = m.predict_proba(Z(F))[:, 1]; print('ridge-logistic coefs (rank features):', dict(zip(FE, np.round(m.coef_[0], 2))))
def auc(s, y): r = s.rank(); n1 = y.sum(); n0 = len(y) - n1; return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def ev(z, col, lab):
    z = z.copy(); z['q'] = z.groupby('mo')[col].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) if len(s) >= 10 else np.nan); a = z[z.q == 4]; b = z[z.q == 0]; d = (a.groupby('mo').r.mean() - b.groupby('mo').r.mean()).dropna(); t = d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 5 else np.nan
    st = lambda x: f'n {len(x):5d} mean {x.r.mean()*100:+6.1f}% med {x.r.median()*100:+5.1f}% huge {(x.r>=1).mean()*100:4.1f}% win10 {x.win10.mean()*100:4.1f}%'
    print(f'  {lab:22s} ALL {st(z)} || TOP-Q {st(a)} || BOT-Q {st(b)} || top-bot {(a.r.mean()-b.r.mean())*100:+.1f}pp t {t:+.1f} AUC(win10) {auc(z[col], z.win10):.3f}')
T = F[F.test]; print(f'\nTEST 2022-26Q1 flips {len(T)}')
for lab, z in (('ALL prices', T), ('$1-10', T[T.px < 10]), ('$10+', T[T.px >= 10])):
    print(f' {lab}'); ev(z, 'score', 'rank-sum (frozen)'); ev(z, 'lr', 'ridge logistic (frozen)')
print('\nIN-SAMPLE (train) for reference, same rules:'); ev(F[~F.test], 'score', 'rank-sum train'); ev(F[~F.test], 'lr', 'ridge train')
print('\nTEST single-feature top-minus-bottom quintile (for comparison to the stack):')
for f in FE: ev(T.assign(sg=T['k_' + f] * (use.get(f, 1))), 'sg', f + (' (train sign)' if f in use else ' (+ sign)'))
F.to_pickle(X + 'stack_flips.pkl')
