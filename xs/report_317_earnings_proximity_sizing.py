"""Nov 18 2026 report 317. Follow-up to 316 H2 (flips whose next earnings report is <=5 business days away have ~1.5x the dispersion, mean not reliably higher: train +0.66% sd 6.1 | test +0.20% sd 5.1 vs far-from-report +0.25% sd 3.6 | +0.24% sd 3.7). Mechanism: an unhedged binary event inside the hold (gap risk); the screener edge is flat in sigma (report 303), so event proximity adds risk without return. Declared rules, one parameter each, set a priori: (R1) skip flips with next report <=5 bdays; (R2) half size on them. Slot book (6 slots, live ratio-rank order, 2bp cost), CAGR / maxDD / ann vol, 2019-22 | 2023-26, vs live."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
B = pd.read_parquet(SP + 'b302.parquet'); B = B[B.exit < END].copy(); B['h1'] = False; U = pd.read_parquet(SP + 'b316.parquet')[['until', 'since']]; B = B.join(U)
def sim(B, order_key, wfun=None, N=6):
    global T0
    X = B.assign(_k=order_key).sort_values(['entry', '_k', 'tk'], kind='stable'); T0 = X; tk_, _ = taken(N); X2 = X.set_index(['tk', 'entry'])
    R = np.zeros(len(D)); Wt = np.zeros(len(D))
    for tk, e, x, r, ra in tk_:
        a, b = D.searchsorted(e.normalize()), D.searchsorted(x.normalize())
        if a >= len(D): continue
        b = min(b, len(D) - 1); n = b - a + 1; w = (1 / N) * (wfun(X2.loc[(tk, e)]) if wfun else 1.0)
        R[a:b + 1] += w * ((1 + r) * (1 - 2e-4)) ** (1 / n) - w; Wt[a:b + 1] += w
    sc = np.where(Wt > 1, 1 / Wt, 1.0); return pd.Series(R * sc, index=D)
def st2(s):
    out = []
    for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')):
        z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25; out.append(f'{e.iloc[-1] ** (1 / y) - 1:6.1%} / DD {(e / e.cummax() - 1).min():6.1%} / vol {z.std() * np.sqrt(252):4.1%}')
    return ' | '.join(out)
brank = B.brank.values + 1e-6 * B.ratio.rank(ascending=False).values
print('slot book CAGR / DD / ann.vol  [2019-22 | 2023-26]'); print(' live                       ', st2(sim(B, brank)))
print(' R2 half size if report<=5d ', st2(sim(B, brank, lambda r: 0.5 if r.until <= 5 else 1.0)))
Bk = B[~(B.until <= 5)].copy(); bk = Bk.brank.values + 1e-6 * Bk.ratio.rank(ascending=False).values
print(' R1 skip if report<=5d      ', st2(sim(Bk, bk))); print(f' (R1 removes {int((B.until <= 5).sum())} of {len(B)} flips, {(B.until <= 5).mean():.1%})')
