"""Nov 14 2026 WHICH FLIPS BECOME THE BIG MOVE (declared before results). Same 374-name 1h universe/flips as report_50, BOTH directions (d=+1 long on bull flip, d=-1 short on bear flip), entry next-bar open, exit open after opposite flip. REAL = direction-adjusted trade return in the top decile of its direction (the fat tail that carries the book). Features known at the flip bar only, direction-adjusted where signed: LV prior-day range / 20d mean (LOW = quiet day before; hypothesis: low -> real), BV flip-bar volume / prior-20-bar mean (hypothesis: high -> real), CMP mean bar range last 10 / last 60 bars (hypothesis: low = compressed -> real), MOM direction-signed return over prior 100 bars (hypothesis: from the prior-trend side? tested both ways), DH distance of price from trailing 500-bar extreme in the trade direction in % (0 = at the extreme / breaking out). PIT expanding terciles within direction; report share REAL and mean return low | high; halves by time; PASS = same sign both halves and pooled t >= 2 on mean-return high-low."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
from vector_pine import run_state_machine
D = pd.read_pickle('live/midsmall_bars.pkl')['d']; rows = []
for tk, B in D.items():
    B = B.dropna(); B = B[~B.index.duplicated()]
    if len(B) < 600 or B.close.median() < 0.3: continue
    reg = run_state_machine(B.assign(datetime=B.index))[0]; n = len(B); o = B.open.values; h = B.high.values; l = B.low.values; c = B.close.values; v = B.volume.values.astype(float); idx = B.index; day = idx.normalize()
    dd = pd.DataFrame({'h': B.high.groupby(day).max(), 'l': B.low.groupby(day).min(), 'c': B.close.groupby(day).last()}); rg = (dd.h - dd.l) / dd.c; lv = (rg.shift(1) / rg.shift(1).rolling(20).mean().shift(0)); LVs = pd.Series(lv.reindex(day).values)
    rng = h - l; rs = pd.Series(rng); r10 = rs.rolling(10).mean().values; r60 = rs.rolling(60).mean().values; hx = pd.Series(h).rolling(500, min_periods=200).max().values; lx = pd.Series(l).rolling(500, min_periods=200).min().values
    ch = [i for i in range(1, n) if reg[i] != reg[i - 1] and reg[i] != 0]
    for a in range(len(ch) - 1):
        i, j = ch[a], ch[a + 1]; st, en = i + 1, j + 1; d = int(reg[i])
        if en >= n or i < 100 or np.isnan(LVs.iloc[i]) or np.isnan(hx[i]) or r60[i] <= 0: continue
        bv = v[i] / max(v[i - 20:i].mean(), 1); ent = o[st]
        rows.append(dict(t=idx[st], d=d, ret=d * (o[en] / ent - 1), LV=LVs.iloc[i], BV=bv, CMP=r10[i] / r60[i], MOM=d * (c[i] / c[i - 100] - 1), DH=(1 - c[i] / hx[i]) if d > 0 else (c[i] / lx[i] - 1)))
P = pd.DataFrame(rows).sort_values('t').reset_index(drop=True); print('flips', len(P))
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 5 and len(b) > 5 else np.nan
for dd_, nm in ((1, 'LONG'), (-1, 'SHORT')):
    S = P[P.d == dd_].copy().reset_index(drop=True); thr = S.ret.quantile(0.9); S['real'] = S.ret >= thr; print(f'\n{nm}: n {len(S)} base mean {S.ret.mean()*100:+.2f}% | REAL = ret >= {thr*100:.1f}% (10% of flips) carries {S.ret[S.real].sum()/S.ret.sum()*100:.0f}% of total return' if S.ret.sum() > 0 else f'\n{nm}: n {len(S)} base mean {S.ret.mean()*100:+.2f}% | REAL threshold {thr*100:.1f}%')
    mid = S.t.iloc[len(S) // 2]
    for c_ in ('LV', 'BV', 'CMP', 'MOM', 'DH'):
        v_ = S[c_].values; T = np.full(len(v_), np.nan)
        for i in range(150, len(v_)): q1, q2 = np.quantile(v_[:i], [1 / 3, 2 / 3]); T[i] = 0 if v_[i] <= q1 else 2 if v_[i] >= q2 else 1
        S['T'] = T; line = f' {c_:3s}'
        for lab, m in (('h1', S.t < mid), ('h2', S.t >= mid), ('all', S.t >= S.t.min())):
            s = S[m]; lo = s[s['T'] == 0]; hi = s[s['T'] == 2]; line += f' | {lab}: real% low {lo.real.mean()*100:4.1f} high {hi.real.mean()*100:4.1f} mean {lo.ret.mean()*100:+.2f}/{hi.ret.mean()*100:+.2f} t {tt(hi.ret, lo.ret):+.1f}'
        print(line)
