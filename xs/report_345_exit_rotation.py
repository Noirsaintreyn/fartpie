"""Nov 22 2026 report 345. Rey: closing early has always hurt; rotate 15m exits and 1h exits by context?  Declared before running.
Same vector (xs/vector_pine.run_state_machine, same parameters) run on 1h bars aggregated from the RTH 15m bars (bins 09:30, 10:30, ... 15:30; the last bin is 30 min).  Entry unchanged (15m flip, open of the next bar).  1h exit = the first 1h BEAR flip after entry; the exit price is the open of the first 15m bar at/after that 1h bar's end (no look-ahead).  Context at entry = the 1h regime at the last COMPLETED 1h bar (bull = the 1h vector agrees).
THEORY (Rey): the 15m bear flip is noisy inside a larger 1h trend; if the 1h trend is still bull, a 15m bear flip is often a shakeout and the 1h exit lets the trade run; if the 1h is not bull the 15m exit is right.  Rules (fixed before running): A all 1h exits; B 1h exit when 1h-bull at entry else 15m exit; C later of the 15m exit and the 1h exit when 1h-bull else 15m exit (never earlier than the 15m exit).
Measured: per-trade return by rule (train 2019-21 | test 2022-26), share of trades with 1h-bull at entry, and the 6-slot book (CAGR / DD / Sharpe) with the exits swapped (2+2 bp, 8% breaker).  Why it should fail: a longer exit holds through real reversals (drawdown) and holds slots longer so fewer new flips are taken."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from stress_lib import *; from vector_pine import run_state_machine
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
n = int(B0.shape[0]); bull1 = np.zeros(n, bool); x1 = np.full(n, np.datetime64('NaT'), dtype='datetime64[ns]'); r1 = np.full(n, np.nan); r15 = np.full(n, np.nan); have = np.zeros(n, bool)
for tk in B0.tk.unique():
    f = sorted(glob.glob(f'/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')); H = pd.read_parquet(f[-1]); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index()
    mm = H.index.hour * 60 + H.index.minute; Hr = H[(mm >= 570) & (mm < 960)].copy(); mr = (Hr.index.hour * 60 + Hr.index.minute)
    Hr['bin'] = Hr.index.normalize() + pd.to_timedelta(570 + ((mr - 570) // 60) * 60, unit='m')
    G = Hr.groupby('bin').agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).dropna(); G['end'] = G.index + pd.Timedelta(minutes=60)
    regs, lv, atr, fl = run_state_machine(G); bear_ends = np.array([G.end.values[i] for i, d in fl if d == -1], dtype='datetime64[ns]'); ends = G.end.values
    o = H.open.values; idx = H.index.values
    m = np.where((B0.tk == tk).values)[0]; te = B0.entry.values[m]; tx = B0.exit.values[m]
    k = np.searchsorted(ends, te, side='right') - 1; bull1[m] = np.where(k >= 0, regs[np.clip(k, 0, len(regs) - 1)] == 1, False)
    for j, t0, t15 in zip(m, te, tx):
        q = np.searchsorted(bear_ends, t0, side='right')   # first 1h bear flip whose bar END is after entry
        ie = np.searchsorted(idx, t0); i15 = np.searchsorted(idx, t15)
        if q < len(bear_ends): ix = np.searchsorted(idx, bear_ends[q]); ix = min(ix, len(idx) - 1)
        else: ix = i15
        x1[j] = idx[ix]; r1[j] = o[ix] / o[ie] - 1 - .0002; r15[j] = o[i15] / o[ie] - 1 - .0002
print('sanity: recomputed 15m return vs b302 ret, corr', np.corrcoef(r15, B0.ret)[0, 1].round(4), '| share 1h-bull at entry', bull1.mean().round(3))
B0['bull1'] = bull1; B0['x1'] = pd.to_datetime(x1); B0['r1'] = r1; tr = ((B0.entry >= '2019-01-01') & (B0.entry < '2022-01-01')).values; te_ = (B0.entry >= '2022-01-01').values
def mk(rule):
    X = B0.copy()
    if rule == 'A': X['exit'] = X.x1; X['ret'] = X.r1
    elif rule == 'B': m = X.bull1; X.loc[m, 'exit'] = X.loc[m, 'x1']; X.loc[m, 'ret'] = X.loc[m, 'r1']
    elif rule == 'C': m = X.bull1 & (X.x1 > X.exit); X.loc[m, 'exit'] = X.loc[m, 'x1']; X.loc[m, 'ret'] = X.loc[m, 'r1']
    return X
def f(X, m): s = X.ret.values[m]; return f'{s.mean():+.3%} (t{s.mean() / (s.std() / np.sqrt(len(s))):+4.1f}, win {(s > 0).mean():.0%}, hold d {((X.exit - X.entry).values[m] / np.timedelta64(1, "D")).mean():.1f})'
for rule, lab in (('0', '15m exit (live)'), ('A', 'A all 1h exits'), ('B', 'B 1h if 1h-bull else 15m'), ('C', 'C later-of if 1h-bull')):
    X = B0 if rule == '0' else mk(rule); z = sim(X); z = z[z.index >= '2019-01-01']; print(f'{lab:26s} per trade train {f(X, tr)} | test {f(X, te_)} || BOOK CAGR {cagr(z):.1%}, DD {mdd(z):.1%}, Sharpe {sharpe(z.values):.2f}')
for lab, m in (('1h-bull at entry', bull1), ('1h-not-bull at entry', ~bull1)): print(f'\n{lab} (n {m.sum()}): 15m exit train {f(B0, tr & m)} | test {f(B0, te_ & m)}\n   1h exit  train {f(mk("A"), tr & m)} | test {f(mk("A"), te_ & m)}')
print('\nSLOT SENSITIVITY (CAGR / DD / Sharpe), the 1h exit holds ~18 days so it needs more slots:')
for N in (6, 8, 10, 12, 15):
    out = []
    for rule in ('0', 'A', 'B', 'C'):
        X = B0 if rule == '0' else mk(rule); z = sim(X, N=N); z = z[z.index >= '2019-01-01']; out.append(f'{rule}: {cagr(z):5.1%}/{mdd(z):6.1%}/{sharpe(z.values):4.2f}')
    print(f'   N={N:2d}  ' + ' | '.join(out))
