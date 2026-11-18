"""Nov 18 2026 report 321. Idea from Rey's L/S desk doc (D6): strip market beta and keep only the relative view. Our long-only flips: is there alpha after hedging beta x SPY over the hold (report 301 'rest')? THEORY (declared): the vector flip catches post-bear-leg reversals/trends in a name; if the persistence of the move is name-specific the hedged residual keeps most of the return; if it is the market's drift, the hedged book is near zero. Why it would fail: 6-slot long book is ~beta 1.1 on a rising market; 2019-26 drift is the return. Tests: (a) mean ret vs mean residual per flip (train|test, t), (b) slot book on residual returns (live order, equal size, 2bp cost kept as is): CAGR / DD / ann vol / return-per-vol vs the long-only book."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
B = pd.read_parquet(SP + 'b302.parquet'); B = B[B.exit < END].copy(); B['h1'] = False
for nm, m in (('train', B.tr), ('test', ~B.tr)):
    z = B[m]; print(f'(a) {nm}: ret {100 * z.ret.mean():+.2f}% (t{z.ret.mean() / (z.ret.std() / np.sqrt(len(z))):+.1f}) | residual vs beta x SPY {100 * z.rest.mean():+.2f}% (t{z.rest.mean() / (z.rest.std() / np.sqrt(len(z))):+.1f}) | share of mean from beta leg {1 - z.rest.mean() / z.ret.mean():.0%} | sd ret {100 * z.ret.std():.1f} resid {100 * z.rest.std():.1f}')
def sim(B, N=6):
    global T0
    X = B.assign(_k=B.brank.values + 1e-6 * B.ratio.rank(ascending=False).values).sort_values(['entry', '_k', 'tk'], kind='stable'); T0 = X; tk_, _ = taken(N)
    R = np.zeros(len(D)); Wt = np.zeros(len(D))
    for tk, e, x, r, ra in tk_:
        a, b = D.searchsorted(e.normalize()), D.searchsorted(x.normalize())
        if a >= len(D): continue
        b = min(b, len(D) - 1); n = b - a + 1; w = 1 / N; R[a:b + 1] += w * ((1 + r) * (1 - 2e-4)) ** (1 / n) - w; Wt[a:b + 1] += w
    return pd.Series(R * np.where(Wt > 1, 1 / Wt, 1.0), index=D)
def st(s):
    o = []
    for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')):
        z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25; c = e.iloc[-1] ** (1 / y) - 1; v = z.std() * np.sqrt(252); o.append(f'{c:6.1%} / DD {(e / e.cummax() - 1).min():6.1%} / vol {v:4.1%} / CAGR per vol {c / v:4.2f}')
    return ' | '.join(o)
print('\n(b) slot book [2019-22 | 2023-26]'); print(' long-only flips        ', st(sim(B)))
H = B.copy(); H['ret'] = H.rest; print(' beta-hedged (residual) ', st(sim(H)))
