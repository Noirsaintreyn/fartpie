"""Nov 22 2026 report 342. Rey: apply the Po3 idea (true consolidation vs slow trend) to the vector.  Declared before running.
For each of the 12,320 candidate large-cap bull flips (b302, 54 names, 15m), measured on the 20 closed 15m bars BEFORE the flip entry bar (no look-ahead): ER20 = |net|/path (Kaufman); COMP = Donchian-20 width / price, percentile rank against that ticker's previous 200 bars (low = compressed).  THEORY: a flip out of a tight, directionless base (low ER, low COMP) is a fresh break with stored energy; a flip after a smooth run (high ER) is a late chase of an extended move - report 339 showed the strong flips are the ones that keep going, so extension may help instead.  Either sign is reported; terciles cut on TRAIN (2019-21); test 2022-26; also the 6-slot book dropping the worst tercile.  Per-trade return = b302 ret (2bp+2bp cost already in)."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from stress_lib import *
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
n = len(B0); ER = np.full(n, np.nan); CP = np.full(n, np.nan)
for tk in B0.tk.unique():
    f = sorted(glob.glob(f'/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')); H = pd.read_parquet(f[-1]); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index()
    c, h, l = H.close.values, H.high.values, H.low.values; w = pd.Series(h).rolling(20).max().values - pd.Series(l).rolling(20).min().values; wp = w / c
    rk = pd.Series(wp).rolling(201).apply(lambda x: (x[:-1] <= x[-1]).mean() * 100, raw=True).values
    m = np.where((B0.tk == tk).values)[0]; ie = H.index.searchsorted(B0.entry.values[m])
    for j, i in zip(m, ie):
        if i < 221 or i >= len(H): continue
        a = i - 20; b = i - 1; path = np.abs(np.diff(c[a:b + 1])).sum(); ER[j] = abs(c[b] - c[a]) / path if path > 0 else np.nan; CP[j] = rk[b]
B0['er'] = ER; B0['cp'] = CP; X = B0.dropna(subset=['er', 'cp']); tr = (X.entry >= '2019-01-01') & (X.entry < '2022-01-01'); te = X.entry >= '2022-01-01'; print(f'{len(X)} of {n} flips have features; train {tr.sum()} test {te.sum()}')
def f(m): s = X.loc[m, 'ret']; return f'{s.mean():+.3%} (t{s.mean() / (s.std() / np.sqrt(len(s))):+4.1f}, n{len(s)}, win {(s > 0).mean():.0%})'
for col, lab in (('er', 'ER20 (low = directionless base)'), ('cp', 'COMP pct (low = compressed)')):
    q = X.loc[tr, col].quantile([1 / 3, 2 / 3]).values; L, M, Hh = X[col] <= q[0], (X[col] > q[0]) & (X[col] < q[1]), X[col] >= q[1]; print(f'\n{lab}: cuts {q.round(2)}\n   train low {f(tr & L)} | mid {f(tr & M)} | high {f(tr & Hh)}\n   test  low {f(te & L)} | mid {f(te & M)} | high {f(te & Hh)}')
print(f'\nall flips: train {f(tr)} | test {f(te)}')
q1 = X.loc[tr, 'er'].quantile([1 / 3, 2 / 3]).values; q2 = X.loc[tr, 'cp'].quantile(1 / 3)
for lab, m in (('low ER & compressed', (X.er <= q1[0]) & (X.cp <= q2)), ('high ER & wide', (X.er >= q1[1]) & (X.cp > q2))): print(f'{lab:22s} train {f(tr & m)} | test {f(te & m)}')
s0 = sim(B0); s0 = s0[s0.index >= '2019-01-01']; print(f'\nBOOK baseline: CAGR {cagr(s0):.1%}, DD {mdd(s0):.1%}, Sharpe {sharpe(s0.values):.2f}')
for lab, keep in (('drop high-ER third', ~(B0.er >= q1[1])), ('drop low-ER third', ~(B0.er <= q1[0])), ('drop compressed third', ~(B0.cp <= q2))):
    z = sim(B0[keep | B0.er.isna()]); z = z[z.index >= '2019-01-01']; print(f'BOOK {lab:22s}: CAGR {cagr(z):.1%}, DD {mdd(z):.1%}, Sharpe {sharpe(z.values):.2f}')
