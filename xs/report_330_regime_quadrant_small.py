"""Nov 18 2026 report 330. PAPER CLAIM (Harvest 'Does a regime label improve the model?', Macro Framework growth-inflation quadrants, D3 'regime-driven structure'). Declared before running.
Harvest's own rule: a taxonomy earns a place only if it improves decisions OUT OF SAMPLE after costs, and the label must be FILTERED (known at the time), not smoothed.  So: label at month-end t from price proxies only (no revised macro data): GROWTH up if 3m return of XLI/XLU ratio > 0 (cyclical leadership); INFLATION up if 3m return of GLD+DBC mean > 0 vs IEF (real-asset leadership over duration).  Four quadrants.  Claim tested: the quadrant at t predicts next-month IWM and IWM-SPY, small caps doing best in growth-up/inflation-down ('goldilocks', credit easy) and worst in growth-down/inflation-up ('stagflation').  Decision test: IWM-SPY spread top quadrant minus bottom quadrant, train (2019-21) | test (2018, 2022-26), plus a 'does it beat a coin' check = the same spread with the label shuffled across months (1,000 draws).
Why it fails: proxies are noisy (gold/commodities have their own drivers), only ~4 quadrants x 100 months = ~25 obs per cell, the label is persistent so effective n is small, and growth leadership is itself the thing being predicted."""
import numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
P = pd.read_parquet(SP + 'macro_px.parquet'); X = yf.download(['GLD', 'DBC', 'IEF'], start='2015-01-01', end='2026-11-17', auto_adjust=True, progress=False)['Close']; P = P.join(X)
M = P.resample('ME').last(); r = M.pct_change(); fw = r.shift(-1)
g = ((M.XLI / M.XLU) / (M.XLI / M.XLU).shift(3) - 1) > 0; inf = ((M.GLD / M.IEF).pipe(lambda s: s / s.shift(3) - 1) + (M.DBC / M.IEF).pipe(lambda s: s / s.shift(3) - 1)) > 0
Q = np.where(g & ~inf, 'G+I-', np.where(g & inf, 'G+I+', np.where(~g & ~inf, 'G-I-', 'G-I+'))); Q = pd.Series(Q, index=M.index)[M.XLI.notna() & M.DBC.notna()]
Y = pd.DataFrame({'q': Q, 'iwm': fw.IWM, 'rel': fw.IWM - fw.SPY}).dropna(); Y['tr'] = (Y.index >= '2019-01-01') & (Y.index < '2022-01-01'); Y['te'] = (Y.index >= '2018-01-01') & ~Y.tr
def tt(s): return '   n/a' if len(s) < 5 else f'{s.mean() * 100:+5.2f}% (t{s.mean() / (s.std() / np.sqrt(len(s))):+4.1f},n{len(s):3d})'
print('next-month IWM-SPY by quadrant [train | test]   (G=growth leadership, I=real-asset/inflation leadership)')
for q in ('G+I-', 'G+I+', 'G-I-', 'G-I+'): print(f'  {q}: rel {tt(Y[Y.tr & (Y.q == q)].rel)} | {tt(Y[Y.te & (Y.q == q)].rel)}      IWM abs {tt(Y[Y.tr & (Y.q == q)].iwm)} | {tt(Y[Y.te & (Y.q == q)].iwm)}')
def spr(d): return d[d.q == 'G+I-'].rel.mean() - d[d.q == 'G-I+'].rel.mean()
rng = np.random.default_rng(1)
for nm, m in (('train', Y.tr), ('test ', Y.te)):
    d = Y[m]; s0 = spr(d); sh = [spr(d.assign(q=rng.permutation(d.q.values))) for _ in range(1000)]; print(f'  goldilocks minus stagflation spread, {nm}: {s0 * 100:+.2f}%/mo; shuffled-label 95th pct {np.nanpercentile(sh, 95) * 100:+.2f}%; p={np.mean(np.array(sh) >= s0):.2f}')
