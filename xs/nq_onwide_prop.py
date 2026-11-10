"""Nov 10 2026: OVERNIGHT-RANGE SIZING IN THE FULL LUCID 50K EVAL / PAYOUT SIM. DECLARED BEFORE RESULTS.
Why: xs/nq_edge_hunt.py found that Rule A trades averaged ~+25 pts when the Globex range (18:00 prior -> 09:30, real NQ 5m) was in the top third of its last 20 sessions
vs ~0 otherwise (2018-21 and 2022-26 both, 8 of 9 years). It was 1 pass out of ~37 flags on the trade list, so it must survive the real account sim before it is used.
Base = final system S1 (xs/nq_final_system.py): Rule A trades, 50 / 75, flat 12:00, <= 2 a day, Yang-Zhang + VWAP sizing 1 / 0.5 / 0.3 NQ, eval skips flips against the 09:30 gap.
WIDE = Globex range > its 20-day 2/3 quantile (known at 09:30, before every Rule A entry). Variants (eval / funded use the same rule unless stated):
  W0  base S1 (gap-gated eval, funded all trades)
  W1  narrow x0.5 (size cut on narrow-overnight days, eval and funded)
  W2  eval: skip narrow days | funded: narrow x0.5
  W3  wide x1.5, narrow x0.5
  W4  wide x1.0, narrow x0.3
Report as nq_final_system: eval pass, funded paid, evals -> payout, payouts per funded account, average payout, net per eval, 12-month runs (median, 10th pct, % positive), Jan 2022 - Oct 2026 net, 2026 net.
Also split by period via the run's own windows. Decision rule: adopt only if net per eval AND 12m 10th percentile improve vs W0 and the gain is not all in one period."""
import sys, os, numpy as np, pandas as pd, multiprocessing as mp, warnings; warnings.filterwarnings('ignore')
import yfinance as yf
Dq = yf.download('QQQ', start='2015-06-01', interval='1d', progress=False, auto_adjust=False); Dq.columns = [c[0] if isinstance(c, tuple) else c for c in Dq.columns]; Dq.index = Dq.index.normalize()
_fs = open('xs/nq_final_system.py').read(); exec(_fs[:_fs.rindex("if __name__ == '__main__':")])
_eh = open('xs/nq_edge_hunt.py').read(); exec(_eh[:_eh.index("if __name__ == '__main__':")])
_NQ = days(load()); onr = _NQ.OH - _NQ.OL; WIDE = (onr > onr.shift(1).rolling(20).quantile(2 / 3))
WX = pd.Series(WIDE.reindex(pd.DatetimeIndex(XA.day)).fillna(False).values, index=XA.index)
print(f'Rule A flips {len(XA)}, on wide-overnight days {WX.mean():.0%}', flush=True)
def mul(w, n): return pd.Series(np.where(WX, w, n), index=XA.index)
DS['W1'] = build(~AG['GAP'], mult=mul(1.0, .5)); DS['W2e'] = build(~AG['GAP'] & WX); DS['W2f'] = build(mult=mul(1.0, .5))
DS['W3'] = build(~AG['GAP'], mult=mul(1.5, .5)); DS['W3f'] = build(mult=mul(1.5, .5)); DS['W4'] = build(~AG['GAP'], mult=mul(1.0, .3)); DS['W4f'] = build(mult=mul(1.0, .3))
VAR = {'W0 base S1': ('GAPg', 'base', 'base'), 'W1 narrow x0.5': ('W1', 'W2f', 'W2f'), 'W2 eval skip narrow / funded x0.5': ('W2e', 'W2f', 'W2f'),
       'W3 wide x1.5 narrow x0.5': ('W3', 'W3f', 'W3f'), 'W4 narrow x0.3': ('W4', 'W4f', 'W4f')}
if __name__ == '__main__':
    jobs = [(nm, 'LUCID 50K') for nm in VAR]
    with mp.get_context('fork').Pool(4) as pool: RES = pool.map(job10, jobs)
    print('\nOVERNIGHT-RANGE SIZING, Lucid 50K', flush=True)
    for r in RES:
        print(f'  {r["nm"]:<38} | eval pass {r["passed"]:4.0%} | funded paid {r["paid_f"]:4.0%} | evals -> payout {r["paid_e"]:4.0%} | payouts/funded {r["npay_f"]:3.1f} | avg payout ${r["avg_pay"]:,.0f} | net/eval {r["net_eval"]:+6,.0f} '
              f'| 12m median {r["k_med"]:+6,.0f} (10th {r["k_p10"]:+5,.0f}) positive {r["k_pos"]:4.0%} paid {r["k_paid"]:4.0%} | 2022-Oct26 {r["full"]:+7,.0f} | 2026 {r["y26"]:+5,.0f}', flush=True)
