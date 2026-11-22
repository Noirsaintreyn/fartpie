"""Nov 22 2026 report 357. Rey: rank the slots by SPECULATION (lottery/attention) if relative volatility and the options ratio do not work.  Declared before running.
Only the PICK ORDER among flips arriving on the same bar changes (6 slots, 8% breaker, 2+2bp, 54 names, all else = live).  Proxies known at entry (b302 columns): vol20 (20d realised vol), beta, rvol (flip-bar volume vs normal), rngatr (flip-bar range / ATR), mom20 (prior 20d return = lottery/MAX-like), composite SPEC = mean percentile rank of vol20, beta, rvol, rngatr, mom20.  Each HIGH-first and LOW-first; baseline = live order (ratio).
THEORY (Kumar 2009, Bali-Cakici-Whitelaw 2011, Barberis-Huang): speculative/lottery names are overpriced and earn LOWER later returns (so LOW-first should win) - but in a breakout screener attention can add follow-through (HIGH-first).  Both directions are shown so neither is cherry-picked; prior evidence (136, 295) says high ATRP/rvol hurt 2022-26.
Measured: CAGR / DD / Sharpe, train 2019-21 | test 2022-26, mean taken-trade return."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
_s = open('/home/claude/fartpie/xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
for c in ('vol20', 'beta', 'rvol', 'rngatr', 'mom20'): B0[c + '_p'] = B0[c].rank(pct=True)
B0['spec'] = B0[['vol20_p', 'beta_p', 'rvol_p', 'rngatr_p', 'mom20_p']].mean(axis=1)
def run(col, hi):
    global order
    def order(Bs):
        k = (-Bs[col].values if hi else Bs[col].values) if col else Bs.brank.values + 1e-6 * Bs.ratio.rank(ascending=False).values
        return Bs.assign(_k=k).sort_values(['entry', '_k', 'tk'], kind='stable')
    TK, _ = sim_swap(B0, None); return TK
def line(lab, TK): show(lab, TK); print(f'{"":22s} mean taken-trade {np.mean([t[3] for t in TK]):+.3%}')
line('LIVE (ratio order)', run(None, True))
for col in ('vol20', 'beta', 'rvol', 'rngatr', 'mom20', 'spec'):
    for hi in (True, False): line(f'{col} {"HIGH" if hi else "LOW"} first', run(col, hi))
