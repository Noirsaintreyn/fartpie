"""Nov 22 2026 report 347. Report 346 says shorting under the vector loses; the cheaper use of Rey's observation is as a GATE on the long book: skip / keep long flips by the QQQ WEEKLY (and DAILY) vector regime at entry (last COMPLETED bar only).  Declared before running.
Measured: per-trade return of the 12,320 large-cap flips by QQQ weekly regime at entry (bull / bear), same for the daily regime; train 2019-21 | test 2018 + 2022-26; and the 6-slot book skipping flips taken while the weekly (or daily) regime is bear.  THEORY: a trailing-stop long flip needs the index to trend up; flips taken while the index's own vector is bear are counter-trend and whipsaw (report 302 found persistence matters).  Why it fails: bear regimes include the sharp V rebounds where the best flips happen (Apr 2020, Apr 2025)."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from stress_lib import *; from vector_pine import run_state_machine; import yfinance as yf
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; B0 = pd.read_parquet(SP + 'b302.parquet'); B0 = B0[B0.exit < END].copy(); B0['entry'] = pd.to_datetime(B0.entry); B0['exit'] = pd.to_datetime(B0.exit); B0['h1'] = False; B0 = B0.reset_index(drop=True)
def sim(Bs, N=6, brk=.08):
    global T0
    key = Bs.brank.values + 1e-6 * Bs.ratio.rank(ascending=False).values; T0 = Bs.assign(_k=key).sort_values(['entry', '_k', 'tk'], kind='stable'); tk_, _ = taken(N, brk); R = np.zeros(len(D))
    for tk, e, x, r, ra in tk_:
        a, b = D.searchsorted(e.normalize()), D.searchsorted(x.normalize())
        if a >= len(D): continue
        b = min(b, len(D) - 1); n = b - a + 1; w = 1 / N; R[a:b + 1] += w * ((1 + r) * (1 - 2e-4)) ** (1 / n) - w
    return pd.Series(R, index=D)
def cagr(s): e = (1 + s).cumprod(); y = (s.index[-1] - s.index[0]).days / 365.25; return e.iloc[-1] ** (1 / y) - 1
def mdd(s): e = (1 + s).cumprod(); return float((e / e.cummax() - 1).min())
Q = yf.download('QQQ', start='2005-01-01', auto_adjust=True, progress=False); Q.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in Q.columns]; Q = Q.dropna()
W = Q.resample('W-FRI').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
def regat(df, dates):
    regs = run_state_machine(df.reset_index(drop=True))[0]; k = df.index.searchsorted(dates.dt.normalize().values, side='left') - 1; return np.where(k >= 0, regs[np.clip(k, 0, len(regs) - 1)], 0)
B0['wk'] = regat(W, B0.entry); B0['dy'] = regat(Q, B0.entry); tr = ((B0.entry >= '2019-01-01') & (B0.entry < '2022-01-01')).values; te = ~tr
def f(m): s = B0.ret.values[m]; return f'{s.mean():+.3%} (t{s.mean() / (s.std() / np.sqrt(len(s))):+4.1f}, n{len(s)}, win {(s > 0).mean():.0%})'
print(f'share of flips taken in a WEEKLY bear regime {np.mean(B0.wk == -1):.1%}, DAILY bear {np.mean(B0.dy == -1):.1%}')
for col, lab in (('wk', 'QQQ WEEKLY'), ('dy', 'QQQ DAILY')):
    print(f'{lab}: bull   train {f(tr & (B0[col] == 1))} | test {f(te & (B0[col] == 1))}\n{lab}: BEAR   train {f(tr & (B0[col] == -1))} | test {f(te & (B0[col] == -1))}')
s0 = sim(B0); s0 = s0[s0.index >= '2019-01-01']; print(f'\nBOOK baseline: CAGR {cagr(s0):.1%}, DD {mdd(s0):.1%}, Sharpe {sharpe(s0.values):.2f}')
for lab, keep in (('skip while weekly bear', B0.wk != -1), ('skip while daily bear', B0.dy != -1), ('skip if either bear', (B0.wk != -1) & (B0.dy != -1))):
    z = sim(B0[keep]); z = z[z.index >= '2019-01-01']; print(f'BOOK {lab:24s}: {keep.sum()} flips, CAGR {cagr(z):.1%}, DD {mdd(z):.1%}, Sharpe {sharpe(z.values):.2f}')
print('\nCurrent state: QQQ weekly regime at last completed bar', run_state_machine(W.reset_index(drop=True))[0][-1], '| daily', run_state_machine(Q.reset_index(drop=True))[0][-1], '(1 bull, -1 bear)')
