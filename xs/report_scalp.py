"""SCALPS at VWAP sigma bands (1 sigma / 2 sigma), fade vs continuation, TP 10-20 pts, stop 10-30, within 60 min; conditions: high-vol env (YZ pct top third / HAR>1), VP confluence (prior POC/VAH/VAL/H/L within 10 pts of the band).
Nov 10 2026, declared before results. PASS: pooled expectancy > 0 net of 0.5 pt cost, same sign and |t|>=2 in 2018-21 and 2022-26."""
import sys, os; sys.path.insert(0, '/home/claude/fartpie/xs'); os.chdir('/home/claude/fartpie/xs')
import numpy as np, pandas as pd, warnings, itertools; warnings.filterwarnings('ignore')
src = open('magnitude.py').read(); src = src[:src.index("# GEX / vxn / balance")]
exec(src)                                  # D, yzp, har_ratio, gjr_ratio, prof (idea3) ...
import nz_core as Z
LV = {d: prof(d) for d in Z.ALLD}; prev = {d: Z.ALLD[i - 1] for i, d in enumerate(Z.ALLD) if i > 0}
_o = open('orb_noise_vector.py').read(); NS_ = {}; exec(_o[:_o.index("OUT, NAMES = pickle.load")], NS_); Gd = NS_['G']
COST = 0.5; TPS = (10, 15, 20); STS = (10, 20, 30); HOR = 12
ev = []
for d in sorted(Gd):
    if d < pd.Timestamp('2018-02-01') or d > pd.Timestamp('2026-01-30'): continue
    g = Gd[d]; lv = LV.get(prev.get(d))
    if len(g) < 60: continue
    o, h, l, c, v, tp = [g[x].values.astype(float) for x in ('o', 'h', 'l', 'c', 'v', 'tp')]; vw = g.vw.values.astype(float); mins = g.index.values
    cv = np.nancumsum(np.nan_to_num(v)); s1 = np.nancumsum(np.nan_to_num(v) * tp); s2 = np.nancumsum(np.nan_to_num(v) * tp * tp)
    with np.errstate(all='ignore'): m1 = s1 / cv; sd = np.sqrt(np.maximum(s2 / cv - m1 * m1, 0))
    seen = set(); n = len(c)
    for j in range(6, n - 2):
        if mins[j] + 5 > 900 or not np.isfinite(sd[j]) or sd[j] < 1: continue
        for k in (1, 2):
            for side in (1, -1):
                lvl = vw[j] + side * k * sd[j]; hit = (h[j] >= lvl) if side > 0 else (l[j] <= lvl)
                if not hit or (k, side) in seen: continue
                seen.add((k, side)); e = o[j + 1]; res = {}
                conf = float(lv is not None and any(abs(x - lvl) <= 10 for x in lv[:5])) if lv is not None else np.nan
                for mode in ('fade', 'cont'):
                    dr = -side if mode == 'fade' else side
                    for T, S in itertools.product(TPS, STS):
                        out = None
                        for q in range(j + 1, min(j + 1 + HOR, n)):
                            hi_, lo_ = h[q], l[q]
                            tpl = e + dr * T; stl = e - dr * S
                            hit_tp = (hi_ >= tpl) if dr > 0 else (lo_ <= tpl); hit_st = (lo_ <= stl) if dr > 0 else (hi_ >= stl)
                            if hit_st: out = -S; break
                            if hit_tp: out = T; break
                        if out is None: out = (c[min(j + HOR, n - 1)] - e) * dr
                        res[(mode, T, S)] = out - COST
                ev.append(dict(day=d, j=j, k=k, side=side, conf=conf, yz=yzp.get(d, np.nan), har=har_ratio.get(d, np.nan), mm=mins[j], **{f'{a}|{b}|{c_}': r for (a, b, c_), r in res.items()}))
E = pd.DataFrame(ev); E['half'] = np.where(E.day.dt.year < 2022, 'A', 'B'); cut = E[E.half == 'A'].yz.quantile(2 / 3); E['hivol'] = (E.yz > cut)
E.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/scalp_events.pkl')
print(f'events {len(E)} over {E.day.nunique()} days | per day {len(E)/E.day.nunique():.1f} | band1 {np.mean(E.k==1):.0%} | hivol cut yz_pct>{cut:.2f}: {E.hivol.mean():.0%} | VP-confluent {np.nanmean(E.conf):.0%}')
cols = [c_ for c_ in E.columns if c_.count('|') == 2]
def tt(x): x = x.dropna(); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 20 and x.std() > 0 else np.nan
def rep(mask, label, top=6):
    X = E[mask]; rows = []
    for c_ in cols:
        a, b = X[X.half == 'A'][c_], X[X.half == 'B'][c_]
        rows.append((c_, X[c_].mean(), np.mean(X[c_] > 0), a.mean(), tt(a), b.mean(), tt(b)))
    R = pd.DataFrame(rows, columns=['combo', 'mean', 'win', 'A', 'tA', 'B', 'tB']).sort_values('mean', ascending=False)
    print(f'\n--- {label}: n {len(X)} events | best combos by pooled expectancy (pts per trade, net of cost) ---')
    for r in R.head(top).itertuples():
        ok = 'PASS' if r.A > 0 and r.B > 0 and r.tA >= 2 and r.tB >= 2 else ''
        print(f'   {r.combo:14} mean {r.mean:+5.2f} win {r.win:4.0%} | A {r.A:+5.2f} (t {r.tA:+4.1f}) B {r.B:+5.2f} (t {r.tB:+4.1f}) {ok}')
    return R
for k in (1, 2):
    for md in ('fade', 'cont'):
        sub = [c_ for c_ in cols if c_.startswith(md)]
        pass
for nm, mk in (('ALL events band 1', E.k == 1), ('ALL events band 2', E.k == 2), ('HIGH-VOL band 1', (E.k == 1) & E.hivol), ('HIGH-VOL band 2', (E.k == 2) & E.hivol),
               ('VP-confluent band 1', (E.k == 1) & (E.conf == 1)), ('VP-confluent band 2', (E.k == 2) & (E.conf == 1)), ('HIGH-VOL + VP-confluent band 1', (E.k == 1) & E.hivol & (E.conf == 1)), ('HIGH-VOL + VP-confluent band 2', (E.k == 2) & E.hivol & (E.conf == 1))):
    rep(mk, nm)
