"""Nov 18 2026 report 288: ENVIRONMENT CHARACTER of each model (Rey: assign every model an environment). Legs rebuilt through Oct 2026 (xs/run_legs_ext.py; FIG-V and NZ dropped).
Day-level environment features known at the open (all use prior days + overnight): ATRP (ADR20 pct of 252d), expansion (prior range/ADR20), acceleration (ADR5/ADR20), trend efficiency (10d), |z| of prior close vs 20d,
52-wk location, gap/ADR, open outside prior range, overnight range/ADR, overnight inventory (ON close vs prior close, /ADR), prior-day volume/20d (heavy), QQQ net-GEX sign.
Terciles cut on TRAIN 2019-2021 only. Per model per feature: mean pts/trade in low/mid/high tercile, train | test (2018 + 2022-Oct 2026). A 'character' = best tercile same sign & rank in both and test t>1.5 vs the model's own mean."""
import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
L = pickle.load(open(SP + 'legs287_ext.pkl', 'rb')); Y = pd.read_parquet('/tmp/claude-0/ml/nq5_real_ext.parquet')
R = Y[Y.rth]; D = R.groupby('day').agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last'), v=('volume', 'sum'), n=('open', 'size'))
D = D[(D.n >= 70) & (D.index != '2026-11-16')]
N = Y[~Y.rth & (Y.day >= D.index[0])].groupby('day').agg(oh=('high', 'max'), ol=('low', 'min'), oc=('close', 'last'))
D = D.join(N); rng = D.h - D.l; adr = rng.rolling(20).mean().shift(1); pc = D.c.shift(1)
F = pd.DataFrame(index=D.index)
F['ATRP'] = (rng.rolling(20).mean()).rolling(252, min_periods=120).rank(pct=True).shift(1)
F['expansion'] = (rng / rng.rolling(20).mean()).shift(1)
F['accel'] = (rng.rolling(5).mean() / rng.rolling(20).mean()).shift(1)
F['trend_eff'] = ((D.c - D.c.shift(10)).abs() / D.c.diff().abs().rolling(10).sum()).shift(1)
F['absz'] = ((D.c - D.c.rolling(20).mean()).abs() / D.c.rolling(20).std()).shift(1)
F['loc52'] = ((D.c - D.l.rolling(252, min_periods=120).min()) / (D.h.rolling(252, min_periods=120).max() - D.l.rolling(252, min_periods=120).min())).shift(1)
F['gap'] = ((D.o - pc).abs() / adr)
F['ONrange'] = ((D.oh - D.ol) / adr)
F['ONinv'] = ((D.oc - pc) / adr)
F['vol_heavy'] = (D.v / D.v.rolling(20).mean()).shift(1)
G = pd.read_csv('/home/claude/fartpie/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').net_gex.sort_index(); F['gex'] = G.reindex(D.index, method='ffill').shift(1)
F.loc[F.index > '2026-10-08', 'gex'] = np.nan
tr = (F.index >= '2019-01-01') & (F.index <= '2021-12-31'); te = ((F.index >= '2018-01-01') & (F.index < '2019-01-01')) | ((F.index >= '2022-01-01') & (F.index <= '2026-10-31'))
CUT = {c: F.loc[tr, c].quantile([1 / 3, 2 / 3]).values for c in F.columns}
def terc(c): x = F[c]; return pd.Series(np.where(x.isna(), np.nan, np.where(x <= CUT[c][0], 0, np.where(x <= CUT[c][1], 1, 2))), index=F.index)
TC = {c: terc(c) for c in F.columns if c != 'gex'}; TC['gex'] = pd.Series(np.where(F.gex.isna(), np.nan, (F.gex > 0).astype(float) * 2), index=F.index)  # 0 neg, 2 pos
rows = []
for nm in ('RA', 'M1', 'R1', 'ON+', 'ON-', 'RULES', 'V15a', 'S1'):
    T = L[nm]; T = T[(T.sz > 0) if nm not in ('ON+', 'ON-') else slice(None)].copy(); T['d'] = pd.to_datetime(T.day); T = T[T.d <= '2026-10-31']
    if nm == 'S1': T = T[T.reg > 0] if 'reg' in T.columns else T
    T = T.set_index('d'); a_tr, a_te = T[T.index.isin(F.index[tr])], T[T.index.isin(F.index[te])]
    print(f'\n{nm}: n train {len(a_tr)} test {len(a_te)} | mean pts/trade train {a_tr.pts.mean():+.1f} test {a_te.pts.mean():+.1f}')
    for c in TC:
        t = TC[c].reindex(T.index).values; out = []
        for k in (0, 1, 2):
            if c == 'gex' and k == 1: continue
            mtr = a_tr.pts[TC[c].reindex(a_tr.index).values == k]; mte = a_te.pts[TC[c].reindex(a_te.index).values == k]
            out.append((k, len(mtr), mtr.mean() if len(mtr) else np.nan, len(mte), mte.mean() if len(mte) else np.nan, mte.std() / np.sqrt(max(len(mte), 1)) if len(mte) > 1 else np.nan))
        best = max(out, key=lambda z: (z[2] if np.isfinite(z[2]) else -9e9)); worst = min(out, key=lambda z: (z[2] if np.isfinite(z[2]) else 9e9))
        bt, wt = [z for z in out if z[0] == best[0]][0], [z for z in out if z[0] == worst[0]][0]
        cons = bt[4] > a_te.pts.mean() and wt[4] < a_te.pts.mean() and bt[3] >= 20 and wt[3] >= 20
        rows.append(dict(model=nm, feat=c, best=best[0], wrst=worst[0], trB=round(bt[2], 1), teB=round(bt[4], 1), nteB=bt[3], trW=round(wt[2], 1), teW=round(wt[4], 1), nteW=wt[3], base_te=round(a_te.pts.mean(), 1), tB=round((bt[4] - a_te.pts.mean()) / bt[5], 1) if bt[5] else np.nan, cons=cons))
        print(f'  {c:10s} lo/mid/hi train ' + ' '.join(f'{z[2]:+6.1f}' for z in out) + ' | test ' + ' '.join(f'{z[4]:+6.1f}(n{z[3]})' for z in out) + ('  <== consistent' if cons else ''))
pd.DataFrame(rows).to_csv(SP + 'env288.csv', index=False)
