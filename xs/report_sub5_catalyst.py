"""Was the move 'explained'? No historical news feed is reachable (SEC ticker map blocked, Yahoo news is recent-only and mixes in other companies' stories), so use PROXIES known at entry: (EARN) an earnings reaction in the last 2 sessions (dse<=2); (GAP) overnight gap >= +5% on the flip day or the day before (news gaps); CAT = EARN or GAP; NOCAT = neither. Mechanism (Rey): breakouts with no reason cannot hold new value. Sub-$5 sets C and C+E+V, B. Mean/trade net 40bp, win rate, random-split p, H1/H2."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import midsmall_screener as M
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb'))
E = pd.read_pickle(SC + 'earnmac_T2.pkl'); V = pd.read_pickle(SC + 'sub5_T.pkl')[['tk', 'et', 'vwx']]; T = E[E.sub5].merge(V, on=['tk', 'et'], how='left'); COST = .004; gap = []; DG = {}
for tk, g in T.groupby('tk'):
    Yc = M.closed_only(data[tk]); d = Yc.index.normalize(); o = Yc.open.groupby(d).first(); c = Yc.close.groupby(d).last(); gp = (o / c.shift(1) - 1); DG[tk] = (gp, d)
for _, r in T.iterrows():
    gp, d = DG[r.tk]; fd = pd.Timestamp(r.et).normalize(); pos = gp.index.searchsorted(fd); gs = gp.iloc[max(0, pos - 1):pos + 1]; gap.append(gs.max() if len(gs) else np.nan)
T['gap'] = gap; T['EARN'] = (T.dse <= 2) & (T.dse >= 0); T['GAP'] = T.gap >= .05; T['CAT'] = T.EARN | T.GAP
H1 = (pd.Timestamp('2023-12-01'), pd.Timestamp('2025-04-01')); rng = np.random.default_rng(2)
for bn, bm in (('B vol>=1.5x', T.vr >= 1.5), ('C B+low-vol-day', (T.vr >= 1.5) & T.lv), ('C+E+V', (T.vr >= 1.5) & T.lv & ~(T.dte <= 3) & ~(T.vwx > 1))):
    b = T[bm]; print(f'\n=== sub-$5 {bn}: {len(b)} flips, mean {(b.ret - COST).mean() * 100:+.2f}%  win {((b.ret - COST) > 0).mean() * 100:.0f}%')
    for nm, m in (('CAT (earn reaction<=2 sessions OR gap>=+5%)', b.CAT), ('  EARN only', b.EARN), ('  GAP only', b.GAP & ~b.EARN), ('NOCAT (nothing visible)', ~b.CAT)):
        k = b[m.values]; rm = np.array([(b.ret.values[rng.permutation(len(b))[:len(k)]] - COST).mean() for _ in range(2000)]) * 100; km = (k.ret - COST).mean() * 100
        h1 = k[k.et < H1[1]]; h2 = k[k.et >= H1[1]]
        print(f'   {nm:46s} n {len(k):4d}  mean {km:+6.2f}%  win {((k.ret - COST) > 0).mean() * 100:4.0f}%  median {k.ret.sub(COST).median() * 100:+5.2f}%  H1 {(h1.ret - COST).mean() * 100:+6.2f}% (n{len(h1)}) H2 {(h2.ret - COST).mean() * 100:+6.2f}% (n{len(h2)})  p(rand>=) {(rm >= km).mean():.3f}')
    w = b[(b.ret - COST) > .05]; l = b[(b.ret - COST) < 0]; print(f'   share with a visible catalyst: big winners (>5%) {w.CAT.mean() * 100:.0f}% (n{len(w)}) | losers {l.CAT.mean() * 100:.0f}% (n{len(l)}) | all {b.CAT.mean() * 100:.0f}%')
