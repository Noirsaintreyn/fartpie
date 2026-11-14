"""Nov 14 2026 ANATOMY OF THE HUGE WINNERS (descriptive first, then one strict test). Wide daily mid/small universe (report_60 flips: long vector flips, price>=$1, 20d median dollar volume>=$500k, 2019-2026Q1, survivorship-biased). HUGE = trade return >= +100% (also +50% shown). For every flip: vector features (initial stop distance = (close-level)/close and in ATR, prior-leg return, days since previous flip, flips in last 60 bars), volume (flip-bar BV, 5d volume trend), volatility (ATR%, ATR% vs its 120d median = expansion/compression, flip-bar range/ATR), location (drawdown from 252d high, position in 252d range, prior 20/60/120d returns), size/liquidity (price, dollar volume), market (share of universe in bull regime that day = breadth, median 20d return, dispersion), insider (NI). Part 1: single-feature AUC for HUGE vs rest, per era (2019-21 / 2022-26Q1): a hint = same direction both eras and |AUC-0.5|>=0.05. Part 2: the 25 biggest winners. Part 3: gradient boosting trained 2019-21 only, tested 2022+, label HUGE>=+50%; AUC and precision/recall in the top 5% / 10% scores, permutation importance, ablations (no insider, BV only). No tuning on the test era."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/claude/fartpie/xs')
from vector_pine import run_state_machine
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score as AUC
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(SC + 'wide_daily_ohlcv.pkl'); W = pd.read_pickle(SC + 'wide_flips_ins.pkl')[['tk', 't', 'NI']]
rows = []; regs = {}
for tk, B in D.items():
    B = B[~B.index.duplicated()]; n = len(B)
    if n < 400: continue
    o = B.open.values; h = B.high.values; l = B.low.values; c = B.close.values; v = B.volume.values.astype(float); idx = B.index; reg, lev, atr, _ = run_state_machine(B.assign(datetime=B.index)); regs[tk] = pd.Series(reg, idx)
    dv = pd.Series(c * v).rolling(20).median().shift(1).values; vm = pd.Series(v).rolling(20).mean().shift(1).values; v60 = pd.Series(v).rolling(60).mean().shift(1).values; hi252 = pd.Series(h).rolling(252, min_periods=120).max().values; lo252 = pd.Series(l).rolling(252, min_periods=120).min().values
    atrp = atr / c; atr_med = pd.Series(atrp).rolling(120, min_periods=60).median().shift(1).values; ch = [i for i in range(1, n) if reg[i] != reg[i - 1] and reg[i] != 0]; chA = np.array(ch)
    for a in range(1, len(ch) - 1):
        i, j = ch[a], ch[a + 1]
        if reg[i] != 1 or j + 1 >= n or i + 1 >= n or i < 130: continue
        if c[i] < 1 or np.isnan(dv[i]) or dv[i] < 5e5 or vm[i] <= 0 or np.isnan(hi252[i]) or np.isnan(lev[i]) or atr[i] <= 0: continue
        p = ch[a - 1]
        rows.append(dict(tk=tk, t=idx[i + 1], d0=idx[i], ret=o[j + 1] / o[i + 1] - 1, hold=j - i, STOP=(c[i] - lev[i]) / c[i], STOPATR=(c[i] - lev[i]) / atr[i], PLR=c[i] / c[p] - 1, DSL=np.log1p(i - p), NFL=int(((chA > i - 60) & (chA <= i)).sum()), BV=v[i] / vm[i], VT=vm[i] / v60[i] if v60[i] > 0 else np.nan,
            ATRP=atrp[i], ATREXP=atrp[i] / atr_med[i] if atr_med[i] > 0 else np.nan, RNG=(h[i] - l[i]) / atr[i], DD52=c[i] / hi252[i] - 1, POS52=(c[i] - lo252[i]) / max(hi252[i] - lo252[i], 1e-9), R20=c[i] / c[i - 20] - 1, R60=c[i] / c[i - 60] - 1, R120=c[i] / c[i - 120] - 1, LPX=np.log(c[i]), LDV=np.log(dv[i])))
P = pd.DataFrame(rows); P = P[P.t < '2026-04-01'].merge(W, on=['tk', 't'], how='left'); P['NI'] = P.NI.fillna(0); P = P[P.t >= '2019-01-01'].sort_values('t').reset_index(drop=True)
R = pd.DataFrame(regs); br = (R > 0).where(R.notna()).mean(axis=1); cl = pd.DataFrame({tk: D[tk].close[~D[tk].index.duplicated()] for tk in regs}); m20 = cl.pct_change(20).median(axis=1); ds = cl.pct_change(20).std(axis=1)
P['BREADTH'] = br.reindex(P.d0).values; P['MKT20'] = m20.reindex(P.d0).values; P['DISP'] = ds.reindex(P.d0).values; P['HUGE'] = (P.ret >= 1.0).astype(int); P['BIG50'] = (P.ret >= 0.5).astype(int)
print('flips', len(P), '| HUGE (>=+100%)', int(P.HUGE.sum()), f'({P.HUGE.mean()*100:.2f}%)', '| >=+50%', int(P.BIG50.sum()), f'({P.BIG50.mean()*100:.2f}%)', '| share of total long return from HUGE:', f'{P.ret[P.HUGE==1].sum()/P.ret.sum()*100:.0f}%')
F = ['STOP', 'STOPATR', 'PLR', 'DSL', 'NFL', 'BV', 'VT', 'ATRP', 'ATREXP', 'RNG', 'DD52', 'POS52', 'R20', 'R60', 'R120', 'LPX', 'LDV', 'BREADTH', 'MKT20', 'DISP', 'NI']
E1 = (P.t >= '2019') & (P.t < '2022'); E2 = P.t >= '2022'
print('\nPART 1: HUGE (>=+100%) vs rest. median(HUGE) | median(rest) | AUC 2019-21 | AUC 2022+  (AUC>0.5: higher value -> more likely HUGE). ** = same direction both eras and |AUC-0.5|>=0.05 in both')
for f in F:
    x = P[f]; au = []
    for e in (E1, E2):
        s = P[e & x.notna()]; au.append(AUC(s.HUGE, s[f]) if s.HUGE.sum() > 5 else np.nan)
    flag = '**' if all(abs(a - 0.5) >= 0.05 for a in au) and (au[0] - 0.5) * (au[1] - 0.5) > 0 else ''
    print(f' {f:8s} huge {P[P.HUGE==1][f].median():9.3f} | rest {P[P.HUGE==0][f].median():9.3f} | AUC {au[0]:.3f} | {au[1]:.3f} {flag}')
print('\nPART 2: 25 biggest winners'); cols = ['tk', 'd0', 'ret', 'hold', 'BV', 'ATRP', 'ATREXP', 'DD52', 'R60', 'DSL', 'BREADTH', 'NI']; T = P.nlargest(25, 'ret')[cols].copy(); T['ret'] = (T.ret * 100).round(0); T['ATRP'] = (T.ATRP * 100).round(1); T['DD52'] = (T.DD52 * 100).round(0); T['R60'] = (T.R60 * 100).round(0); T['d0'] = T.d0.dt.strftime('%Y-%m-%d'); print(T.round(2).to_string(index=False))
print('\nPART 3: gradient boosting, train 2019-21 -> test 2022+, label >= +50%')
tr = P[E1]; te = P[E2]
def fit(cols, label='BIG50'):
    m = HGB(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=2.0, min_samples_leaf=40, random_state=0).fit(tr[cols].values, tr[label].values); return m, m.predict_proba(te[cols].values)[:, 1]
def rep(name, s, label='BIG50'):
    y = te[label].values; o = np.argsort(-s); line = f' {name:30s} AUC {AUC(y, s):.3f}'
    for q in (0.05, 0.10):
        k = int(len(s) * q); top = o[:k]; line += f' | top{int(q*100)}%: hit {y[top].mean()*100:4.1f}% recall {y[top].sum()/y.sum()*100:4.1f}% mean ret {te.ret.values[top].mean()*100:+6.1f}%'
    print(line + f' | base hit {y.mean()*100:.1f}%')
m, s = fit(F); rep('all features', s); imp = {}
y = te.BIG50.values; base = AUC(y, s); rng = np.random.default_rng(0)
for k, f in enumerate(F):
    d = []
    for _ in range(3): X = te[F].values.copy(); X[:, k] = rng.permutation(X[:, k]); d.append(base - AUC(y, m.predict_proba(X)[:, 1]))
    imp[f] = np.mean(d)
print('   permutation importance (AUC drop):', ' '.join(f'{k} {v:+.3f}' for k, v in sorted(imp.items(), key=lambda z: -z[1])[:10]))
F2 = [f for f in F if f != 'NI']; _, s2 = fit(F2); rep('no insider', s2); _, s3 = fit(['BV']); rep('BV only', s3); _, s4 = fit(['BV', 'NI']); rep('BV + insider only', s4)
_, s5 = fit(['STOP', 'STOPATR', 'PLR', 'DSL', 'NFL']); rep('vector-state features only', s5); _, s6 = fit(['ATRP', 'ATREXP', 'RNG']); rep('volatility features only', s6); _, s7 = fit(['DD52', 'POS52', 'R20', 'R60', 'R120']); rep('location features only', s7); _, s8 = fit(['BREADTH', 'MKT20', 'DISP']); rep('market-state features only', s8)
P.to_pickle(SC + 'wide_flips_feat.pkl')
