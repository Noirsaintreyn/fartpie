"""Nov 15 2026 WHICH FLIP CLASSES ARE WORTH TRADING (Rey: 'the model is a classifier but we need to find which classifications are worth trading'). Declared before results. Daily flips (report_90 file, 30,323 bull flips, 2019-Mar 2026, entry next open, hold to bear flip, 30bp, survivorship-biased). Classes = a pre-declared grid (no search): price bucket {<$2, $2-5, $5-10, $10-30, >=$30} x frozen-score tier {top 20% by the TRAIN-era cut, middle 40%, bottom 40%} x flip-bar volume {VOLX >= 1.5, < 1.5} = 30 cells. SELECTION on TRAIN 2019-21 only: trade a cell if n >= 80, mean net > 0, AND ex-top-3 mean > 0 (a cell that is positive only because of 3 winners is not selected). JUDGED on TEST 2022-26: the union of selected cells (every flip in them) vs all flips, mean/median/win10/huge/ex-top3, flips per year, and year-by-year means. PERMUTATION CONTROL: draw the same number of cells at random from those with train n >= 80, 5,000 times; report the share of draws whose test mean >= the selected union's test mean (small = the selection carries information beyond chance). Also report the whole class map (train / test n, mean, ex-top3) so the decision is visible, and the same exercise restricted to the LARGE-CAP subset (dv >= $100M, px >= $10; grid = score tier x volume only, 6 cells, n >= 40). Lead bar: union beats all flips on test mean and ex-top3, positive in >= 3 of 5 test years, permutation share <= 10%."""
import numpy as np, pandas as pd
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; F = pd.read_pickle(X + 'stack_flips.pkl'); COST = .003; F['n'] = F.r - COST; F['y'] = F.d.dt.year; rng = np.random.default_rng(7)
cut80 = F[~F.test].lr.quantile(.8); cut40 = F[~F.test].lr.quantile(.4)
F['tier'] = np.where(F.lr >= cut80, 'top20', np.where(F.lr >= cut40, 'mid40', 'bot40')); F['pb'] = pd.cut(F.px, [0, 2, 5, 10, 30, 1e9], labels=['<2', '2-5', '5-10', '10-30', '30+']).astype(str); F['vb'] = np.where(F.VOLX >= 1.5, 'vol>=1.5', 'vol<1.5'); F['lc'] = (F.dv >= 1e8) & (F.px >= 10)
def ex3(s): s = s.sort_values(ascending=False); return s.iloc[3:].mean() if len(s) > 6 else np.nan
def run(U, keys, nmin, lab):
    tr = U[~U.test]; te = U[U.test]; rows = []
    for k, g in tr.groupby(keys):
        t = te
        for c, v in zip(keys, (k if isinstance(k, tuple) else (k,))): t = t[t[c] == v]
        rows.append(dict(cell=k, ntr=len(g), mtr=g.n.mean(), x3tr=ex3(g.n), nte=len(t), mte=t.n.mean() if len(t) else np.nan, x3te=ex3(t.n) if len(t) else np.nan))
    M = pd.DataFrame(rows); sel = M[(M.ntr >= nmin) & (M.mtr > 0) & (M.x3tr > 0)]; elig = M[M.ntr >= nmin]
    print(f'\n######## {lab}: cells {len(M)}, eligible (train n>={nmin}) {len(elig)}, SELECTED {len(sel)} ########'); pd.set_option('display.width', 220)
    for _, r in M.sort_values('mtr', ascending=False).iterrows(): print(f"  {str(r.cell):34s} train n {r.ntr:5.0f} mean {r.mtr*100:+7.2f}% ex3 {r.x3tr*100:+6.2f}% | test n {r.nte:5.0f} mean {r.mte*100:+7.2f}% ex3 {r.x3te*100:+6.2f}%" + ('  <== SELECTED' if r.cell in set(sel.cell) else ''))
    cells = set(sel.cell); mask = lambda z: pd.Series([tuple(r) in cells if len(keys) > 1 else r[0] in cells for r in z[keys].values], index=z.index)
    ut = te[mask(te)]; print(f'\n TEST all flips   n {len(te):5d} mean {te.n.mean()*100:+6.2f}% med {te.n.median()*100:+6.2f}% win10 {(te.n>=.1).mean()*100:5.1f}% huge {(te.n>=1).mean()*100:4.2f}% ex3 {ex3(te.n)*100:+6.2f}%')
    if len(ut): print(f' TEST selected    n {len(ut):5d} mean {ut.n.mean()*100:+6.2f}% med {ut.n.median()*100:+6.2f}% win10 {(ut.n>=.1).mean()*100:5.1f}% huge {(ut.n>=1).mean()*100:4.2f}% ex3 {ex3(ut.n)*100:+6.2f}% | flips/yr {len(ut)/ (te.d.max()-te.d.min()).days*365:.0f}'); print('  by year  all: ' + ' '.join(f'{y}:{g.n.mean()*100:+.1f}' for y, g in te.groupby('y')) + '\n  by year  sel: ' + ' '.join(f'{y}:{g.n.mean()*100:+.1f}(n{len(g)})' for y, g in ut.groupby('y')))
    pool = elig.cell.tolist(); k = len(sel)
    if len(ut) and k and len(pool) > k:
        te2 = te.copy(); te2['cellk'] = list(map(tuple, te2[keys].values)) if len(keys) > 1 else te2[keys[0]]; byc = te2.groupby('cellk').n.agg(['sum', 'count']); hit = 0; N = 5000
        for _ in range(N): pick = [pool[i] for i in rng.choice(len(pool), k, replace=False)]; s = byc.reindex(pick).fillna(0); hit += (s['sum'].sum() / max(s['count'].sum(), 1)) >= ut.n.mean()
        print(f' PERMUTATION: share of {N} random {k}-cell draws with test mean >= selected union: {hit/N:.3f}')
run(F, ['pb', 'tier', 'vb'], 80, 'ALL NAMES (price x score tier x volume)')
run(F[F.lc], ['tier', 'vb'], 40, 'LARGE-CAP (score tier x volume)')
