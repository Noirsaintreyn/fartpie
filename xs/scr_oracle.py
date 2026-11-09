"""Nov 9 2026: HINDSIGHT (ORACLE) PICKS FOR THE SCREENER'S SLOTS, AND WHAT THE WINNERS LOOKED LIKE BEFORE THEY MOVED (Rey). DECLARED BEFORE RESULTS.
1) Ceiling and floor: same 6-slot sim, but when several flips arrive in one bar the slots go to the ones that TURNED OUT best (oracle) / worst (anti-oracle);
   live ratio order and random for reference. (Slots, timing and the breaker unchanged - only who gets the slot uses hindsight.)
2) Can the winners be spotted beforehand? For every pre-entry feature (options ratio, Yang-Zhang, HAR, HAR vs normal, prior-day relative volume, gap, 52-week high,
   20-day relative strength, prior-day return, idiosyncratic vol, entry time, SPY 20-day trend, how many flips arrived together; put/call and VRP where options exist):
   a) in bars with >= 3 flips: how often the feature's top pick is the bar's best trade, vs chance (1 / n), 2019-22 / 2023-26
   b) top-decile winners vs the rest: average feature value
   c) all features together: gradient boosting fit on 2019-22 to flag a bar's best trade, rank AUC within bars on 2023-26, and the slot sim with that order."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/scr_rank_lit.py').read(); exec(_s[:_s.index("\nif __name__ == '__main__':")])
X = base2.copy(); X['tm'] = X.entry.dt.hour * 60 + X.entry.dt.minute; X['n'] = X.groupby('entry').tk.transform('size')
spy20 = SPYd.close.pct_change(20); sp_nx = pd.Series(spy20.values, index=SPYd.index).shift(1); X['spy20'] = X.day.map(sp_nx)
X['ratio_rk'] = X.groupby('entry').ratio.rank(ascending=False); X['per'] = np.where(X.day < '2023-01-01', 'a', 'b')
FEATS = ['ratio', 'yz', 'har', 'harrel', 'rvol', 'gap', 'hi52', 'rs20', 'rev1', 'ivol', 'tm', 'spy20', 'n', 'pcr', 'vrp']
def run_order(key):
    global T0
    T0 = X.assign(_k=key).sort_values(['entry', '_k', 'tk'], kind='stable'); tk_, _ = taken(); s_, _ = daily(tk_, 6); return s_
if __name__ == '__main__':
    print('=== 1) who gets the slot when flips arrive together | CAGR / max DD 2019-22 | 2023-26 | all ===', flush=True)
    for lab, key in (('ORACLE: best actual return first', -X.ret), ('live: highest options ratio', -X.ratio), ('random', np.random.default_rng(1).random(len(X))),
                     ('ANTI-ORACLE: worst return first', X.ret)):
        s_ = run_order(key.values if hasattr(key, 'values') else key); e = (1 + s_).cumprod(); print(f'  {lab:<36} {st2(s_)} | $1k -> ${1000 * e.iloc[-1]:,.0f}', flush=True)
    B = X[X.n >= 3].copy(); B['best'] = B.groupby('entry').ret.transform('max') == B.ret
    print(f'\n=== 2a) bars with >= 3 flips ({B.entry.nunique():,} bars): how often each feature\'s top pick is the bar\'s best trade | chance {np.mean(1 / B.groupby("entry").n.first()):.0%} ===', flush=True)
    rows = []
    for f in FEATS + ['ratio_rk']:
        if f in ('tm', 'n', 'spy20', 'ratio_rk'): continue
        for d_ in (-1, 1):
            Z = B.dropna(subset=[f]); top = Z.loc[Z.groupby('entry')[f].idxmax() if d_ == -1 else Z.groupby('entry')[f].idxmin()]
            ra, rb = top[top.per == 'a'].best.mean(), top[top.per == 'b'].best.mean(); rows.append((f, 'highest' if d_ == -1 else 'lowest', ra, rb, top[top.per == 'a'].ret.mean(), top[top.per == 'b'].ret.mean()))
    for f, d_, ra, rb, xa, xb in sorted(rows, key=lambda r: -(r[2] + r[3])):
        print(f'  {d_:<7} {f:<7} picks the best {ra:4.0%} / {rb:4.0%} of the time | its pick averages {xa:+.2%} / {xb:+.2%}', flush=True)
    q = X[X.per == 'a'].ret.quantile(.9); X['top'] = X.ret >= q
    print(f'\n=== 2b) TOP-DECILE winners (return >= {q:.2%}) vs the rest: average feature before entry, 2019-22 | 2023-26 ===', flush=True)
    for f in FEATS + ['ratio_rk']:
        cells = [f'{X[(X.per == p_) & X.top][f].mean():8.3f} vs {X[(X.per == p_) & ~X.top][f].mean():8.3f}' for p_ in 'ab']
        print(f'  {f:<8} ' + ' | '.join(cells), flush=True)
    from sklearn.ensemble import HistGradientBoostingClassifier
    FX = [f for f in FEATS] + ['ratio_rk']; tr = B[B.per == 'a']; te = B[B.per == 'b']
    gb = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=.05, min_samples_leaf=200, random_state=0).fit(tr[FX], tr.best)
    B['p'] = gb.predict_proba(B[FX])[:, 1]
    def within_auc(Z, col):
        out = []
        for _, g_ in Z.groupby('entry'):
            if g_.best.sum() == 1 and len(g_) > 1: out.append((g_[col].rank().loc[g_.best].iloc[0] - 1) / (len(g_) - 1))
        return np.mean(out)
    print(f'\n=== 2c) all features together (boosting fit on 2019-22): within-bar AUC for spotting the best trade | 2019-22 (fit) / 2023-26 (unseen) ===', flush=True)
    B['neg_rk'] = -B.ratio_rk
    print(f'  model {within_auc(B[B.per == "a"], "p"):.3f} / {within_auc(B[B.per == "b"], "p"):.3f} | options ratio alone {within_auc(B[B.per == "a"], "ratio"):.3f} / {within_auc(B[B.per == "b"], "ratio"):.3f} | chance 0.500', flush=True)
    X['p'] = gb.predict_proba(X[FX])[:, 1]; s_ = run_order(-X.p.values); print(f'  slot sim ordered by the model: {st2(s_)}', flush=True)
