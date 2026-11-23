"""Nov 22 2026 report 368. Rey: 'we have the options data; look at the last few weeks, separate good flips from bad and see if you notice anything; and chart 10 stocks that flipped last month on 15m and 1h with the vector and 50/200 EMA'.
DATA FACT: the ThetaData options files end Oct 8 (optvol ratio), Oct 16 (live ratio) and Oct 8-20 (open interest, 17 of 35 names); nothing for November.  So the options look-back is the 15m bull flips of Sep 8 - Oct 16 2026 (all 54 names; options features only where the name has data); outcomes run through Nov 20 (exit at the next bear flip of the same timeframe else marked at the last bar), 4bp cost.  Charts cover the last month (Oct 22 - Nov 20) price-only.
EXPLORATORY: ~15 features x good(ret>0) vs bad, Mann-Whitney p; ~0.7 'p<0.05' are expected by luck.  A fishing trip to generate hypotheses, not a test."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
from scipy.stats import mannwhitneyu
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'; COST = .0004
O1 = pd.read_csv(U + 'xs/ext/optvol_ratio.csv', usecols=['tk', 'date', 'opt_contracts', 'sh', 'call_c']); O2 = pd.read_csv(U + 'xs/ext/optvol_ratio_live.csv'); OP = pd.concat([O1, O2]).drop_duplicates(['tk', 'date'], keep='last'); OP['date'] = pd.to_datetime(OP.date); OP['ratio'] = OP.opt_contracts * 100 / OP.sh; OP['callsh'] = OP.call_c / OP.opt_contracts
OI = {}
for f in glob.glob(U + 'xs/ext/oi_daily/*.csv'):
    d = pd.read_csv(f, parse_dates=['date']).set_index('date'); OI[f.split('/')[-1][:-4]] = pd.DataFrame({'c_turn': d.call_vol / d.call_oi, 'c_build': d.c_dnet / d.call_oi, 'p_build': d.p_dnet / d.put_oi, 'pc_vol': d.put_vol / d.call_vol, 'oi_cp': np.log(d.call_oi / d.put_oi)})
def agg1h(H): g = H.resample('60min', origin='start_day', offset='30min'); return pd.DataFrame({'open': g.open.first(), 'high': g.high.max(), 'low': g.low.min(), 'close': g.close.last(), 'volume': g.volume.sum()}).dropna()
def daily(H): g = H.groupby(H.index.normalize()); return pd.DataFrame({'o': g.open.first(), 'h': g.high.max(), 'l': g.low.min(), 'c': g.close.last()})
def zs(d):
    r = np.log(d.c).diff(); n = 20; k = .34 / (1.34 + (n + 1) / (n - 1)); on = np.log(d.o / d.c.shift()); oc = np.log(d.c / d.o); rs = np.log(d.h / d.c) * np.log(d.h / d.o) + np.log(d.l / d.c) * np.log(d.l / d.o)
    yz = np.sqrt((on.rolling(n).var() + k * oc.rolling(n).var() + (1 - k) * rs.rolling(n).mean()).clip(lower=1e-10)); reg = r.rolling(20).std(); r5 = np.log(d.c / d.c.shift(5))
    return pd.DataFrame({'z_yz': r5 / (yz.shift(5) * np.sqrt(5)), 'r5': r5, 'atrp': ((d.h - d.l) / d.c).rolling(14).mean()})
def flips_for(H, tf, t0, t1):
    regs, lv, atr, fl = run_state_machine(H); last = len(H) - 1; c = H.close; e50 = c.ewm(span=50, adjust=False).mean().values; e200 = c.ewm(span=200, adjust=False).mean().values; out = []
    tod = H.volume.groupby(H.index.strftime('%H:%M')).transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean()).values
    for i, d in fl:
        if d != 1 or i + 1 > last: continue
        t = H.index[i]
        if not (t0 <= t < t1): continue
        ent = H.open.values[i + 1]; nxt = [j for j, dd in fl if dd == -1 and j > i]
        if nxt and nxt[0] + 1 <= last: ix = nxt[0] + 1; px = H.open.values[ix]; hold = ix - i
        else: px = H.close.values[last]; hold = last - i
        out.append(dict(tf=tf, flip=t, ret=px / ent - 1 - COST, hold_bars=hold, d50=(c.values[i] - e50[i]) / atr[i], d200=(c.values[i] - e200[i]) / atr[i], e50_gt_200=float(e50[i] > e200[i]), rvol=np.log(H.volume.values[i] / tod[i]) if tod[i] > 0 else np.nan, hour=t.hour + t.minute / 60, lvl_dist=(c.values[i] - lv[i]) / atr[i]))
    return out
rows = []
for f in sorted(glob.glob(SP + 'lastweek/*.parquet')):
    tk = f.split('/')[-1][:-8]; H = pd.read_parquet(f); Z = zs(daily(H))
    for r in flips_for(H, '15m', pd.Timestamp('2026-09-08'), pd.Timestamp('2026-10-17')):
        pdays = Z.index[Z.index < r['flip'].normalize()]; z = Z.loc[pdays[-1]]; o = OP[(OP.tk == tk) & (OP.date < r['flip'].normalize()) & (OP.date >= r['flip'].normalize() - pd.Timedelta(days=5))].sort_values('date'); r.update(tk=tk, z_yz=z.z_yz, r5=z.r5, atrp=z.atrp, ratio=np.log(o.ratio.iloc[-1]) if len(o) else np.nan, callsh=o.callsh.iloc[-1] if len(o) else np.nan)
        if tk in OI:
            q = OI[tk]; q = q[(q.index < r['flip'].normalize()) & (q.index >= r['flip'].normalize() - pd.Timedelta(days=5))]
            if len(q): r.update(q.iloc[-1].to_dict())
        rows.append(r)
A = pd.DataFrame(rows); A['good'] = A.ret > 0; print('Sep 8-Oct 16 15m bull flips: %d | good %d (%.0f%%) | mean %+.2f%% median %+.2f%% | with live ratio %d | with OI %d' % (len(A), A.good.sum(), A.good.mean() * 100, A.ret.mean() * 100, A.ret.median() * 100, A.ratio.notna().sum(), A.c_turn.notna().sum()))
print('best 5: ', A.nlargest(5, 'ret')[['tk', 'flip', 'ret']].assign(ret=lambda x: (x.ret * 100).round(1)).to_string(index=False, header=False).replace('\n', ' | '))
print('worst 5:', A.nsmallest(5, 'ret')[['tk', 'flip', 'ret']].assign(ret=lambda x: (x.ret * 100).round(1)).to_string(index=False, header=False).replace('\n', ' | '))
F = ['ratio', 'callsh', 'c_turn', 'c_build', 'p_build', 'pc_vol', 'oi_cp', 'z_yz', 'r5', 'atrp', 'd50', 'd200', 'e50_gt_200', 'rvol', 'hour', 'lvl_dist']
print('\n%-11s %8s %8s %8s %8s %8s' % ('feature', 'n good', 'n bad', 'med good', 'med bad', 'MWU p')); ps = []
for k in F:
    g = A[A.good][k].dropna(); b_ = A[~A.good][k].dropna()
    if len(g) < 8 or len(b_) < 8: print('%-11s too few (%d/%d)' % (k, len(g), len(b_))); continue
    p = mannwhitneyu(g, b_).pvalue; ps.append(p); print('%-11s %8d %8d %8.3f %8.3f %8.3f%s' % (k, len(g), len(b_), g.median(), b_.median(), p, '  <-' if p < .05 else ''))
print('(%d features tested; ~%.1f expected below 0.05 by luck)' % (len(ps), len(ps) * .05)); A.to_parquet(SP + 'A368.parquet')
