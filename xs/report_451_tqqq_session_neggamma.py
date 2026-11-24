"""Nov 24 2026 report 451. Rey: if neg-gamma flips work on QQQ they should work on TQQQ long-only. Report 441's QQQ pass used same-session exits; 449/450 held TQQQ overnight on vector exits. DECLARED BEFORE RESULTS.
REAL TQQQ 15m bars (LSE vault, gaps fixed). Signal = QQQ 15m vector bull flip (real OI gamma, NEG = prior-close net_gex < 0). Entry = next bar open if the bar is 09:45-14:30 ET; exit = open after the opposite (bear) flip, else the 15:45 bar open (same-session flat, a session boundary not a hold parameter). Cost 4bp round trip on TQQQ (1bp on QQQ for comparison). 2018-07..2026-09. Long only.
PASS: NEG beats POS on TQQQ per-trade mean with month-clustered t >= 2 pooled and same sign in train (2019-21) and test (2018 + 2022+) and in 2023-26; and a NEG-only book beats the ungated book on Calmar in both halves.
BOOK: all of equity into each TQQQ trade (flat otherwise), daily marks = realized by day (intraday DD not marked). Report by era 18-19 | 20-21 | 22 | 23-26."""
import os; os.environ['TF'] = '15m'
src = open('xs/report_450_tqqq_real.py').read(); exec(src[:src.index("trades2 = ")])
import numpy as np, pandas as pd
mm = (H.dt.dt.hour * 60 + H.dt.dt.minute).values; rows = []
for i, dd in fls:
    if dd != 1: continue
    fb = i + 1; e = fb + 1
    if e + 1 >= n or not np.isfinite(gprev[fb - 1]) or not (585 <= mm[e] <= 870) or day[e] != day[fb - 1]: continue
    t0 = H.dt[e]
    if t0 < pd.Timestamp('2018-07-01') or t0 > pd.Timestamp('2026-09-30'): continue
    x = None
    for j in range(e + 1, n):
        if day[j] != day[e]: x = j - 1; break
        if regs[j - 1] == -1 or mm[j] >= 945: x = j; break
    if x is None or np.isnan(oT[e]) or np.isnan(oT[x]): continue
    rows.append(dict(entry=t0, day=pd.Timestamp(day[e]), neg=bool(gprev[fb - 1] < 0), t=oT[x] / oT[e] - 1 - 4e-4, q=o_[x] / o_[e] - 1 - 1e-4, hold=(H.dt[x] - H.dt[e]).total_seconds() / 3600))
T = pd.DataFrame(rows); T['mon'] = T.entry.dt.to_period('M'); T['era'] = pd.cut(T.entry.dt.year, [2017, 2019, 2021, 2022, 2100], labels=['18-19', '20-21', '22', '23-26']); T['half'] = np.where((T.entry >= '2019-01-01') & (T.entry < '2022-01-01'), 'train', 'test')
def tst(A, B, c):
    dd = pd.concat([A.assign(s=1), B.assign(s=-1)]); gm = dd.groupby('mon').apply(lambda x: pd.Series(dict(n1=(x.s == 1).sum(), n0=(x.s == -1).sum(), s1=x[c][x.s == 1].sum(), s0=x[c][x.s == -1].sum()))); m1, m0 = A[c].mean(), B[c].mean()
    se = np.sqrt((((gm.s1 - gm.n1 * m1) / len(A) - (gm.s0 - gm.n0 * m0) / len(B)) ** 2).sum()); return (m1 - m0) * 1e4, (m1 - m0) / se
A, B = T[T.neg], T[~T.neg]; print(f'trades {len(T)} NEG {len(A)} POS {len(B)}; avg hold {T.hold.mean():.1f}h')
for c, lab in (('t', 'TQQQ'), ('q', 'QQQ ')):
    d, t_ = tst(A, B, c); print(f"{lab} ALL {T[c].mean()*1e4:6.1f}bp win {(T[c]>0).mean()*100:3.0f}% | NEG {A[c].mean()*1e4:6.1f}bp win {(A[c]>0).mean()*100:3.0f}% | POS {B[c].mean()*1e4:6.1f}bp win {(B[c]>0).mean()*100:3.0f}% | NEG-POS {d:+6.1f}bp t {t_:+.2f} | train {(A[A.half=='train'][c].mean()-B[B.half=='train'][c].mean())*1e4:+.1f} test {(A[A.half=='test'][c].mean()-B[B.half=='test'][c].mean())*1e4:+.1f}")
    print('   by era NEG/POS bp: ' + ' | '.join(f"{e} {A[A.era==e][c].mean()*1e4:+.0f}(n{(A.era==e).sum()}) / {B[B.era==e][c].mean()*1e4:+.0f}(n{(B.era==e).sum()})" for e in ('18-19', '20-21', '22', '23-26')))
idx = pd.bdate_range('2018-07-01', '2026-09-30')
def stat(sel, nm):
    r = T[sel].groupby('day').t.apply(lambda p: (1 + p).prod() - 1).reindex(idx).fillna(0); out = []
    for k, mask in (('all', np.ones(len(idx), bool)), ('train', (idx >= '2019-01-01') & (idx < '2022-01-01')), ('test', (idx < '2019-01-01') | (idx >= '2022-01-01'))):
        rr = r[mask]; e = (1 + rr).cumprod(); cg = e.iloc[-1] ** (252 / len(rr)) - 1; dd = (e / e.cummax() - 1).min(); out.append(f"{k}: CAGR {cg*100:5.1f}% DD {dd*100:6.1f}% Calmar {cg/abs(dd):4.2f} Sh {rr.mean()/rr.std()*np.sqrt(252):4.2f}")
    print(f"{nm:18s} exposure {(T[sel].hold.sum()/ (len(idx)*6.5))*100:4.1f}% | " + ' | '.join(out))
print('\nBOOKS (TQQQ, all equity per trade)'); stat(T.neg | ~T.neg, 'A ungated'); stat(T.neg, 'B NEG only'); stat(~T.neg, 'C POS only')
bh = pd.Series(np.r_[0, dcT[1:] / dcT[:-1] - 1], index=pd.DatetimeIndex(ud)).reindex(idx).fillna(0); out = []
for k, mask in (('all', np.ones(len(idx), bool)), ('train', (idx >= '2019-01-01') & (idx < '2022-01-01')), ('test', (idx < '2019-01-01') | (idx >= '2022-01-01'))):
    rr = bh[mask]; e = (1 + rr).cumprod(); cg = e.iloc[-1] ** (252 / len(rr)) - 1; out.append(f"{k}: CAGR {cg*100:5.1f}% DD {(e/e.cummax()-1).min()*100:6.1f}% Calmar {cg/abs((e/e.cummax()-1).min()):4.2f}")
print(f"{'B&H TQQQ':18s} exposure 100% | " + ' | '.join(out))
