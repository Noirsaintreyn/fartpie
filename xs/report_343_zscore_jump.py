"""Nov 22 2026 report 343. Rey: try a Z-score and a jump-diffusion decomposition; remodel the data by its statistical properties.  Declared before running.
Z-SCORE (from the attached design, simplified to what we have): e = ln(close) - ln(EMA20(close)); z = (e - prior median)/ prior scale in the SAME time-of-day bucket (previous 60 sessions, shifted so the current bar is excluded); direction-aligned (all vector flips are long, a = +z).  Known at the flip-bar close (entry = next bar open).
JUMP-DIFFUSION: Lee-Mykland style stat on the flip bar, J = r_t / sqrt(BPV), BPV = (pi/2) x mean(|r_j||r_{j-1}|) over the previous 16 bars (bipower variation = diffusive variance robust to jumps).  Signed up-jump (J >= 3) = a JUMP-DRIVEN flip; |J| < 1.5 = a DIFFUSIVE (grind) flip.  Also RJ = share of the previous 32 bars' variance that is jumps = max(0, RV-BPV)/RV.
THEORY (stated): a flip through a jump is an information/liquidity shock - impact partly reverses at the next bars (late entry worse) or continues if the information is real; a diffusive grind flip is slow accumulation that tends to persist.  Either sign is reported.  Terciles on TRAIN (2019-21); test 2022-26; then the 6-slot book dropping the worst bucket.
NQ ORB side: gap measured in ADR14 (orb335.parquet), direction-aligned (a = gap x break direction): continuous Z-version of the gap-alignment survivor; terciles on train; points per trade.
Why it should fail: 40 flips a day cannot all be shocks; the z is largely a restatement of the flip itself (the flip happens when price is extended above its pivot ratchet)."""
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
n = len(B0); Z = np.full(n, np.nan); J = np.full(n, np.nan); RJ = np.full(n, np.nan)
for tk in B0.tk.unique():
    f = sorted(glob.glob(f'/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')); H = pd.read_parquet(f[-1]); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index()
    c = H.close.values; lc = np.log(c); r = np.r_[np.nan, np.diff(lc)]; e = lc - np.log(pd.Series(c).ewm(span=20, adjust=False).mean().values)
    bk = (H.index.hour * 60 + H.index.minute).values; E = pd.DataFrame({'e': e, 'bk': bk}); g = E.groupby('bk').e
    med = g.transform(lambda x: x.shift(1).rolling(60, min_periods=20).median()); sc = g.transform(lambda x: 1.4826 * (x.shift(1) - x.shift(1).rolling(60, min_periods=20).median()).abs().rolling(60, min_periods=20).median())
    z = ((e - med.values) / sc.values)
    a = np.abs(r); bp = (np.pi / 2) * pd.Series(a * np.r_[np.nan, a[:-1]]).rolling(16).mean().shift(1).values; jj = r / np.sqrt(bp)
    rv = pd.Series(r ** 2).rolling(32).sum().shift(1).values; bv32 = (np.pi / 2) * pd.Series(a * np.r_[np.nan, a[:-1]]).rolling(32).sum().shift(1).values; rj = np.maximum(0, rv - bv32) / rv
    m = np.where((B0.tk == tk).values)[0]; ie = H.index.searchsorted(B0.entry.values[m]) - 1
    ok = (ie >= 0) & (ie < len(H)); Z[m[ok]] = z[ie[ok]]; J[m[ok]] = jj[ie[ok]]; RJ[m[ok]] = rj[ie[ok]]
B0['z'] = Z; B0['J'] = J; B0['RJ'] = RJ; X = B0.dropna(subset=['z', 'J', 'RJ']); tr = (X.entry >= '2019-01-01') & (X.entry < '2022-01-01'); te = X.entry >= '2022-01-01'; print(f'{len(X)} of {n} flips with features (train {tr.sum()}, test {te.sum()}); share of flips with J>=3: {(X.J >= 3).mean():.0%}, |J|<1.5: {(X.J.abs() < 1.5).mean():.0%}')
def f(m): s = X.loc[m, 'ret']; return '   n/a' if len(s) < 25 else f'{s.mean():+.3%} (t{s.mean() / (s.std() / np.sqrt(len(s))):+4.1f}, n{len(s)}, win {(s > 0).mean():.0%})'
print(f'all flips: train {f(tr)} | test {f(te)}')
for col, lab in (('z', 'aligned Z of displacement (TOD-adjusted)'), ('J', 'jump stat J of the flip bar (signed)'), ('RJ', 'jump share RJ of prior 32 bars')):
    q = X.loc[tr, col].quantile([1 / 3, 2 / 3]).values; L, M, Hh = X[col] <= q[0], (X[col] > q[0]) & (X[col] < q[1]), X[col] >= q[1]; print(f'\n{lab}: cuts {q.round(2)}\n   train low {f(tr & L)} | mid {f(tr & M)} | high {f(tr & Hh)}\n   test  low {f(te & L)} | mid {f(te & M)} | high {f(te & Hh)}')
print('\nFixed jump classes (flip bar):')
for lab, m in (('jump-driven J>=3', X.J >= 3), ('diffusive |J|<1.5', X.J.abs() < 1.5), ('between', (X.J.abs() >= 1.5) & (X.J < 3)), ('down-bar flip J<=-1.5', X.J <= -1.5)): print(f'   {lab:22s} train {f(tr & m)} | test {f(te & m)}')
s0 = sim(B0); s0 = s0[s0.index >= '2019-01-01']; print(f'\nBOOK baseline: CAGR {cagr(s0):.1%}, DD {mdd(s0):.1%}, Sharpe {sharpe(s0.values):.2f}')
q = X.loc[tr, 'z'].quantile([1 / 3, 2 / 3]).values; qj = X.loc[tr, 'J'].quantile([1 / 3, 2 / 3]).values
for lab, drop in (('drop low-Z third', B0.z <= q[0]), ('drop high-Z third', B0.z >= q[1]), ('drop top-J third', B0.J >= qj[1]), ('drop jump-driven J>=3', B0.J >= 3)):
    z_ = sim(B0[~drop.fillna(False)]); z_ = z_[z_.index >= '2019-01-01']; print(f'BOOK {lab:22s}: CAGR {cagr(z_):.1%}, DD {mdd(z_):.1%}, Sharpe {sharpe(z_.values):.2f}')
O = pd.read_parquet(SP + 'orb335.parquet'); O['a'] = O.gap * O.dir; O = O.dropna(subset=['a', 'hold']); t1 = (O.index >= '2019-01-01') & (O.index < '2022-01-01'); t2 = ~t1
def g(m): s = O.loc[m, 'hold']; return f'{s.mean():+6.2f} (t{s.mean() / (s.std() / np.sqrt(len(s))):+4.1f}, n{len(s)})'
q = O.loc[t1, 'a'].quantile([1 / 3, 2 / 3]).values; print(f'\nNQ ORB, direction-aligned gap in ADR (a = gap x dir), cuts {q.round(2)}, pts/trade:\n   train low {g(t1 & (O.a <= q[0]))} | mid {g(t1 & (O.a > q[0]) & (O.a < q[1]))} | high {g(t1 & (O.a >= q[1]))}\n   test  low {g(t2 & (O.a <= q[0]))} | mid {g(t2 & (O.a > q[0]) & (O.a < q[1]))} | high {g(t2 & (O.a >= q[1]))}')
for lab, lo, hi in (('a < -0.5 (big gap against)', -9, -.5), ('-0.5..0', -.5, 0), ('0..0.5', 0, .5), ('0.5..1', .5, 1), ('> 1 (big gap with)', 1, 99)): m = (O.a > lo) & (O.a <= hi); print(f'   {lab:28s} train {g(t1 & m)} | test {g(t2 & m)}')
