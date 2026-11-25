"""Report 479 (Nov 25 2026): Rey - 'does a close below [the vector on the] 4h cut them too early?' DAILY vector flips (daily bars built from the 1h RTH bars of uni_cache), Dec 2023-Nov 2026, px<20, entry next session open. Level in force on session t = the vector level of the last COMPLETED daily bar (bull floor).
Exit variants: A = baseline (daily vector turns bear -> exit at the next session open); B4 = exit at the close of the first 4h bar (09:30-13:30 / 13:30-16:00 halves, same split as the screeners) that closes below the level; B1 = first 1h close below the level. All costs 30bp. Report: share of trades each variant exits EARLIER than A, mean/median/win, avg hold (sessions), and among the trades it cut: how A ended (cut early = A later better) plus what it does to the big winners (A return >= +30%). Groups: all flips and top-third ATRP (daily ATR14/close of the flip day, ranked within this sample, descriptive). DECLARED BEFORE RESULTS; no thresholds fitted."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.'); sys.path.insert(0, 'xs')
import midsmall_screener as M
from vector_pine import run_state_machine
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb')); CD = .003; rows = []
for tk, Y in data.items():
    Y = M.closed_only(Y)
    if len(Y) < 500: continue
    day = Y.index.normalize(); g = Y.groupby(day); D = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), volume=('volume', 'sum')); D.index = pd.DatetimeIndex(D.index)
    if len(D) < 150: continue
    reg, lev, _, fl = run_state_machine(D.assign(datetime=D.index)); reg = pd.Series(reg, index=D.index); lev = pd.Series(lev, index=D.index)
    h, l, c = D.high.values, D.low.values, D.close.values; pc = np.r_[c[0], c[:-1]]; atrp = pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).rolling(14).mean().values / c
    for i in range(120, len(D) - 2):
        if not (reg.iat[i] == 1 and reg.iat[i - 1] != 1) or D.index[i] < pd.Timestamp('2023-12-15') or c[i] >= 20: continue
        e = D.open.iat[i + 1]; a_exit = None; b4 = None; b1 = None
        for j in range(i + 1, len(D)):
            L = lev.iat[j - 1]; dj = D.index[j]; bars = Y[day == dj]
            if b1 is None:
                bb = bars[bars.close < L]
                if len(bb): b1 = (j, bb.close.iloc[0])
            if b4 is None:
                am = bars[bars.index.hour < 13]; pm = bars[bars.index.hour >= 13]
                for part in (am, pm):
                    if len(part) and part.close.iloc[-1] < L: b4 = (j, part.close.iloc[-1]); break
            if reg.iat[j] != 1 and a_exit is None:
                a_exit = (j, D.open.iat[j + 1] if j + 1 < len(D) else np.nan); break
        if a_exit is None or not np.isfinite(a_exit[1]): continue
        ra = a_exit[1] / e - 1 - CD; r4 = (b4[1] / e - 1 - CD) if b4 and b4[0] <= a_exit[0] else ra; r1 = (b1[1] / e - 1 - CD) if b1 and b1[0] <= a_exit[0] else ra
        h4 = (b4[0] - (i + 1) if b4 and b4[0] <= a_exit[0] else a_exit[0] + 1 - (i + 1)); h1 = (b1[0] - (i + 1) if b1 and b1[0] <= a_exit[0] else a_exit[0] + 1 - (i + 1))
        rows.append(dict(tk=tk, d=D.index[i], atrp=atrp[i], ra=ra, r4=r4, r1=r1, ha=a_exit[0] + 1 - (i + 1), h4=h4, h1=h1, c4=bool(b4 and b4[0] <= a_exit[0] and abs(r4 - ra) > 1e-9), c1=bool(b1 and b1[0] <= a_exit[0] and abs(r1 - ra) > 1e-9)))
R = pd.DataFrame(rows); print('daily flips', len(R), 'names', R.tk.nunique()); R['top'] = R.atrp >= R.atrp.quantile(2 / 3)
for nm, S in (('ALL flips', R), ('top-third ATRP', R[R.top])):
    print(f'\n=== {nm}: {len(S)} trades')
    for v, lab in (('ra', 'A daily vector bear -> next open'), ('r4', 'B4 first 4h close below level'), ('r1', 'B1 first 1h close below level')):
        k = {'ra': None, 'r4': 'c4', 'r1': 'c1'}[v]; hh = {'ra': 'ha', 'r4': 'h4', 'r1': 'h1'}[v]; x = S[v]; cut = S[S[k]] if k else S.iloc[0:0]
        extra = f' | exits EARLIER than A in {len(cut) / len(S):.0%}; of those, A later ended better in {(cut.ra > cut[v]).mean():.0%} (A mean {cut.ra.mean():+.1%} vs {cut[v].mean():+.1%}); big winners (A>=+30%) cut: {int(((S.ra >= .3) & S[k]).sum()) if k else 0} of {int((S.ra >= .3).sum())}, their return {S[(S.ra >= .3) & S[k]][v].mean() if k and ((S.ra >= .3) & S[k]).any() else float("nan"):+.0%} vs {S[(S.ra >= .3) & S[k]].ra.mean() if k and ((S.ra >= .3) & S[k]).any() else float("nan"):+.0%}' if k else ''
        print(f'  {lab:34s} win {(x > 0).mean():.0%} mean {x.mean():+.1%} median {x.median():+.1%} p10 {x.quantile(.1):+.0%} avg hold {S[hh].mean():.1f}d{extra}')
