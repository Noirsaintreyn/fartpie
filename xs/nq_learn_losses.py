"""Nov 4 2026: let the system LEARN FROM ITS LOSERS, walk-forward. DECLARED BEFORE RESULTS.
Base = final system (Rule A trades, Yang-Zhang + VWAP sizing 1 / 0.5 / 0.3 NQ, 50 / 75, flat 12:00, gap-gated eval). Lucid 50K ($88) + Topstep.
META MODELS: every month, refit on ALL earlier Rule A trades (>= 120), target = trade won; predict this month's trades. Features known at entry:
  direction, 60-day path thinness, monthly node (HVN / LVN), VWAP distance, flip-bar range / close location / volume, Yang-Zhang and HAR percentiles, QQQ 20d z, 20d return,
  VXN vs 20d, negative gamma, 1h / 4h / gap / SMA20 / SMA50 / VXN agreement, vanna-flow agreement (0 when missing), entry minute, position in today's and the 20-day range.
  LOGIT (L2, C = 0.1, standardised) | GBM (depth 2, 100 trees, lr .05, min 20 per leaf).  Use: bottom third of predictions (vs that month's training predictions) -> x0.5, or SKIP.
LOSS RULES (no model): LS2 half size after 2 losses in a row until the next winner | EQ20 half size while the last 20 trades' net is below 0.
Report: AUC by year 2022-26, Rule A $ by year, firm sims."""
_fs = open('xs/nq_final_system.py').read(); exec(_fs[:_fs.rindex("if __name__ == '__main__':")])
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
RL = run_l(XA).join(XA[['dir']]); RL = RL.loc[XA.sort_values(['day', 'j']).index.intersection(RL.index)]
F_ = pd.DataFrame(index=XA.index)
F_['dir'] = XA.dir; F_['thin'] = XA['ahead_60D']; F_['hvn'] = (XA.NODE_M == 'HVN').astype(float); F_['lvn'] = (XA.NODE_M == 'LVN').astype(float)
for c in ('vwapd', 'disp', 'clv', 'vol5'): F_[c] = X.loc[XA.index, c]
F_['yz'] = prior(yz.dropna().rolling(252, min_periods=120).rank(pct=True), XA.day); F_['har'] = prior(HP, XA.day)
xq = XQ.reindex(pd.DatetimeIndex(XA.day) - pd.Timedelta(days=1), method='ffill'); F_['z20'] = xq.qz.values; F_['r20q'] = xq.qr20.values; F_['vrel'] = xq.vrel.values
F_['neg'] = XA.neg.astype(float)
for g in ('H1', 'H4', 'GAP', 'SMA20', 'SMA50', 'VXN', 'VFLOW'): v = XA[f'g_{g}']; F_[f'a_{g}'] = np.where(v.isna() | (v == 0), 0.0, np.where(v == XA.dir, 1.0, -1.0))
F_['minute'] = m5[XA.j.values]; F_['rd'] = X.loc[XA.index, 'EXTD'].astype(float); F_['mid20'] = X.loc[XA.index, 'MID20'].astype(float)
F_ = F_.astype(float).fillna(F_.median(numeric_only=True))
Y = (RL.net > 0).astype(int).reindex(F_.index)
pred = {'LOGIT': pd.Series(np.nan, index=XA.index), 'GBM': pd.Series(np.nan, index=XA.index)}; cut = {k: pd.Series(np.nan, index=XA.index) for k in pred}
mon = XA.day.dt.to_period('M')
for mo in sorted(mon.unique()):
    te = (mon == mo).values; tr = (XA.day < mo.start_time).values & Y.notna().values
    if tr.sum() < 120 or te.sum() == 0: continue
    Xtr, ytr, Xte = F_.values[tr], Y.values[tr], F_.values[te]
    sc = StandardScaler().fit(Xtr); lg = LogisticRegression(C=.1, max_iter=500).fit(sc.transform(Xtr), ytr)
    gb = HistGradientBoostingClassifier(max_depth=2, max_iter=100, learning_rate=.05, min_samples_leaf=20, random_state=0).fit(Xtr, ytr)
    for k, mdl, Xa, Xb in (('LOGIT', lg, sc.transform(Xtr), sc.transform(Xte)), ('GBM', gb, Xtr, Xte)):
        ptr = mdl.predict_proba(Xa)[:, 1]; pred[k][te] = mdl.predict_proba(Xb)[:, 1]; cut[k][te] = np.quantile(ptr, 1 / 3)
print('walk-forward AUC (Rule A trades, target = win)', flush=True)
for k in pred:
    ok = pred[k].notna() & Y.notna(); yy = XA.day.dt.year
    print(f'  {k:<5} all {roc_auc_score(Y[ok], pred[k][ok]):.3f} | ' + ' | '.join(f'{y} {roc_auc_score(Y[ok & (yy == y)], pred[k][ok & (yy == y)]) if (ok & (yy == y)).sum() > 10 and Y[ok & (yy == y)].nunique() > 1 else float("nan"):.2f} (n{(ok & (yy == y)).sum()})' for y in range(2021, 2027)), flush=True)
low = {k: pd.Series((pred[k] < cut[k]).fillna(False).values, index=XA.index) for k in pred}
seq = RL.net; ls = pd.Series(1.0, index=XA.index); eq = pd.Series(1.0, index=XA.index); streak = 0; hist = []
for idx in RL.index:
    ls.loc[idx] = .5 if streak >= 2 else 1.0; eq.loc[idx] = .5 if len(hist) >= 20 and sum(hist[-20:]) < 0 else 1.0
    n_ = seq.loc[idx]; streak = streak + 1 if n_ <= 0 else 0; hist.append(n_)
MUL = {'normal': pd.Series(1.0, index=XA.index), 'LOGIT x0.5 bottom third': pd.Series(np.where(low['LOGIT'], .5, 1.0), index=XA.index),
       'GBM x0.5 bottom third': pd.Series(np.where(low['GBM'], .5, 1.0), index=XA.index), 'LS2 half after 2 losses': ls, 'EQ20 half when last 20 < 0': eq}
SKIP = {'LOGIT skip bottom third': low['LOGIT'], 'GBM skip bottom third': low['GBM']}
bs = pd.Series([(1.0, .5, .3)[min(int(bool(YZF.reindex([pd.Timestamp(d)]).iloc[0])) + int(bool(X.at[i, 'VWAPX'])), 2)] if pd.Timestamp(d) in YZF.index else 1.0 for i, d in zip(XA.index, XA.day)], index=XA.index)
yrs = XA.day.dt.year.reindex(RL.index)
print('\nRule A $ by year (final sizing x rule; 2022-26)', flush=True)
for nm, m in list(MUL.items()) + [(k, pd.Series(np.where(v, 0.0, 1.0), index=XA.index)) for k, v in SKIP.items()]:
    usd = RL.net * 20 * bs.reindex(RL.index) * m.reindex(RL.index)
    print(f'  {nm:<26} ' + ' | '.join(f'{y} {usd[yrs == y].sum():+7,.0f}' for y in range(2022, 2027)) + f' | total {usd[yrs >= 2022].sum():+8,.0f}', flush=True)
VAR = {}
for nm, m in MUL.items(): DS[f'e_{nm}'] = build(~AG['GAP'], mult=m); DS[f'f_{nm}'] = build(mult=m); VAR[nm] = (f'e_{nm}', f'f_{nm}', f'f_{nm}')
for nm, sk in SKIP.items(): DS[f'e_{nm}'] = build(~AG['GAP'] & ~sk); DS[f'f_{nm}'] = build(~sk); VAR[nm] = (f'e_{nm}', f'f_{nm}', f'f_{nm}')
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in VAR for fn in ('LUCID 50K', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job10, jobs)
    print('\nLEARNING-FROM-LOSSES SYSTEMS', flush=True)
    for r in RES:
        print(f'  {r["nm"]:<26} {r["fn"]:<9} | eval pass {r["passed"]:4.0%} | funded paid {r["paid_f"]:4.0%} | evals -> payout {r["paid_e"]:4.0%} | net/eval {r["net_eval"]:+6,.0f} | 12m median {r["k_med"]:+6,.0f} positive {r["k_pos"]:4.0%} | 2022-Oct26 {r["full"]:+7,.0f} | 2026 {r["y26"]:+5,.0f}', flush=True)
