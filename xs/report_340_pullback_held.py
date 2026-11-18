"""Nov 18 2026 report 340. Rey: a pullback can mean the break FAILED; sometimes volume must hold after the open.  Split report 339's pullbacks (late entry at bar k open, price <= entry open, flip still alive) into HELD vs FAILED.
HELD (structure) = lowest low of bars entry..k-1... (all bars between flip bar and the late entry bar) stays >= the flip bar's low (bar before entry).  VOLUME held = mean volume of the bars from entry to k-1 >= 0.7 x the flip bar's volume (fixed before running; one cut).
Measured per candidate trade: late-entry return (to the vector's own exit) for held vs failed, vol-held vs vol-faded, and 2x2; train 2019-21 | test 2022-26.  Then the 6-slot book on held-pullbacks."""
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
O = {}
for tk in B0.tk.unique():
    f = sorted(glob.glob(f'/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))
    H = pd.read_parquet(f[-1]); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); O[tk] = H.sort_index()
print('columns', list(O[next(iter(O))].columns))
tr = ((B0.entry >= '2019-01-01') & (B0.entry < '2022-01-01')).values; te = (B0.entry >= '2022-01-01').values
for k in (1, 2):
    n = len(B0); e0 = np.full(n, np.nan); ek = np.full(n, np.nan); xk = np.full(n, np.nan); alive = np.zeros(n, bool); held = np.zeros(n, bool); vh = np.zeros(n, bool)
    for tk, H in O.items():
        m = np.where((B0.tk == tk).values)[0]; ie = H.index.searchsorted(B0.entry.values[m]); ix = H.index.searchsorted(B0.exit.values[m]); L = len(H); ov, lo, vo = H.open.values, H.low.values, H.volume.values
        for j, i, x in zip(m, ie, ix):
            if i < 1 or i + k >= L: continue
            e0[j] = ov[i]; ek[j] = ov[i + k]; xk[j] = ov[min(x, L - 1)]; alive[j] = i + k < x
            held[j] = lo[i:i + k].min() >= lo[i - 1]; vh[j] = vo[i:i + k].mean() >= .7 * max(vo[i - 1], 1)
    pb = alive & (ek <= e0); rk = xk / ek - 1 - .0002
    print(f'\n--- k={k} ({15 * k} min): pullback trades {pb.sum()} ({pb.mean():.0%}); of them structure held {held[pb].mean():.0%}, volume held {vh[pb].mean():.0%}')
    def f(c): 
        a, b = rk[c & tr], rk[c & te]; return f'train {a.mean():+.3%} (n{len(a)}, win {np.mean(a > 0):.0%}) | test {b.mean():+.3%} (n{len(b)}, win {np.mean(b > 0):.0%})'
    for lab, c in (('pullback ALL', pb), ('held structure', pb & held), ('FAILED structure', pb & ~held), ('volume held', pb & vh), ('volume faded', pb & ~vh), ('held & vol held', pb & held & vh), ('held & vol faded', pb & held & ~vh), ('failed & vol faded', pb & ~held & ~vh)): print(f'   {lab:20s} {f(c)}')
    for lab, c in (('held structure', pb & held), ('held & vol held', pb & held & vh)):
        Bs = B0[c].copy(); Bs['ret'] = rk[c]; z = sim(Bs); z = z[z.index >= '2019-01-01']; print(f'   BOOK only {lab:16s}: {c.sum()} trades, CAGR {cagr(z):.1%}, DD {mdd(z):.1%}, Sharpe {sharpe(z.values):.2f}')
