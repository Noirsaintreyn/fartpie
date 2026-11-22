"""Nov 22 2026 report 351. Harvey et al. 'Breaking bad trends': split each market into BULL / CORRECTION / BEAR / REBOUND by slow vs fast trend and treat rebounds differently.  Rey: the vector short in 2008 did not move lower with the decline (no pivot highs in a waterfall so the pivot ratchet never tightens).  Declared before running.
A) STATES for the long flips: slow = QQQ WEEKLY vector regime, fast = QQQ DAILY vector regime at the last completed bars: bull/bull, bull/bear = CORRECTION, bear/bear = BEAR, bear/bull = REBOUND.  Per-trade return by state (train 2019-21 | test 2018+2022-26) and the 6-slot book skipping each state.  THEORY: the best long flips are rebounds out of a bear (report 347: weekly-bear flips +0.43% vs +0.17%).
B) SHORT with its own trail (QQQ 1999-2026, daily bars): enter at the next open after the daily bear flip; exit at the first of (vector bull flip next open) or (CLOSE above lowest-low-since-entry + k x ATR14, exit next open), k in {2,3,4}; and the STATE FILTER: short only when the weekly regime is also bear (BEAR state, never a REBOUND/CORRECTION), k=3.  Also on WEEKLY bars with k=3.  THEORY: a trail anchored on the lowest low follows a waterfall down where the pivot ratchet cannot; the state filter drops the rebound/correction shorts that gave back gains (report 346).  Reported: short-only CAGR / DD, long vector + this short, per-trade mean/win, episodes 2000, 2008, 2022, 2026; and the same short rule on the 54 large caps (2018-26, per trade) vs vector short -2.77%.
Costs 2bp per side.  Why it fails: V-rebounds (2009, 2020, Apr 2025) gap through any trail; the sample of true bears is 3 episodes."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
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
Q = yf.download('QQQ', start='1999-03-10', auto_adjust=True, progress=False); Q.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in Q.columns]; Q = Q.dropna(); W = Q.resample('W-FRI').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
def regat(df, dates):
    regs = run_state_machine(df.reset_index(drop=True))[0]; k = df.index.searchsorted(pd.DatetimeIndex(dates).normalize().values, side='left') - 1; return np.where(k >= 0, regs[np.clip(k, 0, len(regs) - 1)], 0)
B0['wk'] = regat(W, B0.entry); B0['dy'] = regat(Q, B0.entry); tr = ((B0.entry >= '2019-01-01') & (B0.entry < '2022-01-01')).values; te = ~tr
def f(m): s = B0.ret.values[m]; return '   n/a' if m.sum() < 25 else f'{s.mean():+.3%} (t{s.mean() / (s.std() / np.sqrt(len(s))):+4.1f}, n{len(s)}, win {(s > 0).mean():.0%})'
ST = {'BULL (wk bull, dy bull)': (B0.wk == 1) & (B0.dy == 1), 'CORRECTION (wk bull, dy bear)': (B0.wk == 1) & (B0.dy == -1), 'BEAR (wk bear, dy bear)': (B0.wk == -1) & (B0.dy == -1), 'REBOUND (wk bear, dy bull)': (B0.wk == -1) & (B0.dy == 1)}
print('A) long flips by QQQ state at entry'); 
for k, m in ST.items(): m = m.values; print(f'   {k:30s} ({m.mean():4.0%}) train {f(tr & m)} | test {f(te & m)}')
s0 = sim(B0); s0 = s0[s0.index >= '2019-01-01']; print(f'   BOOK baseline {cagr(s0):.1%} / {mdd(s0):.1%} / {sharpe(s0.values):.2f}')
for k, m in ST.items(): z = sim(B0[~m.values]); z = z[z.index >= '2019-01-01']; print(f'   BOOK skip {k:30s}: CAGR {cagr(z):.1%}, DD {mdd(z):.1%}, Sharpe {sharpe(z.values):.2f}')
def atrs(df): c = df.close.values; tr_ = np.maximum(df.high.values - df.low.values, np.maximum(np.abs(df.high.values - np.r_[c[0], c[:-1]]), np.abs(df.low.values - np.r_[c[0], c[:-1]]))); return pd.Series(tr_).ewm(alpha=1 / 14, adjust=False).mean().values
def shorts(df, k=None, wk_filter=None, cost=2e-4):
    regs, lv, _, fl = run_state_machine(df.reset_index(drop=True)); o, c, l = df.open.values, df.close.values, df.low.values; A = atrs(df); n = len(df); rows = []; pos = np.zeros(n)
    for ii, (i, d) in enumerate(fl):
        if d != -1 or i + 1 >= n: continue
        nxt = [j for j, dd in fl if dd == 1 and j > i]; j = nxt[0] if nxt else n - 2; xi = min(j + 1, n - 1)
        if wk_filter is not None and wk_filter[i] != -1: continue
        if k is not None:
            ml = np.inf
            for t in range(i + 1, min(j, n - 2) + 1):
                ml = min(ml, l[t])
                if t < j and c[t] > ml + k * A[t]: xi = t + 1; break
        rows.append((df.index[i + 1], df.index[xi], o[i + 1], o[xi])); pos[i + 1:xi] = -1
    E = pd.DataFrame(rows, columns=['t0', 't1', 'p0', 'p1']); E['short'] = -(E.p1 / E.p0 - 1) - 2 * cost; return E, pos
def equity(pos_s, df, long_regs=None, cost=2e-4):
    r = (df.open.shift(-1) / df.open - 1).fillna(0).values; pl = np.r_[0, (long_regs[:-1] == 1).astype(float)] if long_regs is not None else np.zeros(len(df)); p = pl + pos_s
    c = np.abs(np.diff(np.r_[0, p])) * cost; return pd.Series(p * r - c, index=df.index)
def st(s): e = (1 + s).cumprod(); y = (s.index[-1] - s.index[0]).days / 365.25; return f'CAGR {e.iloc[-1] ** (1 / y) - 1:6.1%}, DD {(e / e.cummax() - 1).min():6.1%}'
regD = run_state_machine(Q.reset_index(drop=True))[0]; wkD = regat(W, Q.index)
print('\nB) QQQ short rules, daily bars 1999-2026 (long vector = regs==1 shown for the combined line)')
E0, p0 = shorts(Q); print(f'   vector short (reference): n {len(E0)}, mean {E0.short.mean():+.2%}, win {(E0.short > 0).mean():.0%}, {st(equity(p0, Q))}; long+short {st(equity(p0, Q, regD))}')
for k in (2, 3, 4):
    E1, p1 = shorts(Q, k=k); print(f'   trail k={k}: n {len(E1)}, mean {E1.short.mean():+.2%}, win {(E1.short > 0).mean():.0%}, short-only {st(equity(p1, Q))}; long+short {st(equity(p1, Q, regD))}')
E2, p2 = shorts(Q, k=3, wk_filter=wkD); print(f'   trail k=3 + weekly-bear filter (BEAR state only): n {len(E2)}, mean {E2.short.mean():+.2%}, win {(E2.short > 0).mean():.0%}, short-only {st(equity(p2, Q))}; long+short {st(equity(p2, Q, regD))}')
print('   episodes (k=3 + filter):'); 
for a, b in (('2000-01-01', '2003-12-31'), ('2007-06-01', '2009-12-31'), ('2022-01-01', '2023-12-31'), ('2026-01-01', '2026-12-31')): s = E2[(E2.t0 >= a) & (E2.t0 <= b)]; print(f'      {a[:4]}-{b[:4]}: n {len(s)}, sum {s.short.sum():+.1%}, mean {s.short.mean():+.2%}, win {(s.short > 0).mean():.0%}')
print(f'   long only (vector): {st(equity(np.zeros(len(Q)), Q, regD))}; buy&hold {st(Q.close.pct_change().fillna(0))}')
E3, p3 = shorts(W, k=3); print(f'   WEEKLY bars trail k=3: n {len(E3)}, mean {E3.short.mean():+.2%}, win {(E3.short > 0).mean():.0%}; episodes', {str(r.t0.date()): round(r.short * 100, 1) for r in E3.itertuples()})
rows = []
for fn in sorted(glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/*_15m_2018-01-01_*.parquet')):
    tk = fn.split('/')[-1].split('_15m')[0]; H = pd.read_parquet(fn); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index(); mm = H.index.hour * 60 + H.index.minute; H = H[(mm >= 570) & (mm < 960)]
    Dd = H.groupby(H.index.normalize()).agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).dropna(); E, _ = shorts(Dd, k=3); E['tk'] = tk; rows.append(E)
S = pd.concat(rows); a = S.t0 < '2019-01-01'; b = (S.t0 >= '2019-01-01') & (S.t0 < '2022-01-01'); c_ = S.t0 >= '2022-01-01'
g = lambda m: f'{S.short[m].mean():+.2%} (t{S.short[m].mean() / (S.short[m].std() / np.sqrt(m.sum())):+4.1f}, n{m.sum()}, win {(S.short[m] > 0).mean():.0%})'
print(f'\nC) 54 large caps, daily bear flip + trail k=3 (vs vector short -2.77%): all {g(S.t0 > "2000")} | 2018 {g(a)} | train {g(b)} | test {g(c_)}')
