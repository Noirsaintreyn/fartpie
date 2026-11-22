"""Nov 22 2026 report 349. Rey: (a) SLOT SWAP: when the slots are full and a fresh flip arrives, replace a weak open position instead of skipping the new flip; (b) SCALE-OUT: take part off at +g% and trail the rest on the vector.  Declared before running (3 swap rules, 3 scale-out rules, nothing else).
SWAP (6 slots, 8% breaker, 2+2 bp; an open position is marked at the OPEN of the 15m bar where the new flip enters, the victim exits there at that price, cost 2bp):
 S1 replace the open position with the LOWEST unrealized return, only if that is < 0.
 S2 replace the OLDEST open position if it has been held >= 2 sessions and its unrealized return < +0.5%.
 S3 replace the lowest-unrealized position whenever it is < +0.5%.
SCALE-OUT (slot stays occupied until the vector exit): sell fraction f at entry x (1+g) (filled at the target, or at the bar's open if it gaps through; cost 2bp extra on the sold part), the rest exits on the vector as live: (f=50%, g=1%), (50%, 2%), (50%, 4%).
THEORY: the book is capacity-limited (35% of flips taken, skipped flips average +0.26% vs +0.32% taken, report 345), so recycling the weakest slot into a fresh flip raises exposure to flips at their start, the highest-expectancy part; scale-out banks the common +1-4% follow-through and leaves the tail on.  Why they fail: a weak position today is often the pullback before the big move (report 339); the tail winners (alpha ~2) pay for everything, scale-out halves them.
Measured: 6-slot book CAGR / DD / Sharpe, trades taken, swaps made; train/test halves of the book (2019-21 | 2022-26); mean realized return of the victims vs the flips swapped in."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
sys.path.insert(0, 'xs'); from stress_lib import *
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; B0 = pd.read_parquet(SP + 'b302.parquet'); B0 = B0[B0.exit < END].copy(); B0['entry'] = pd.to_datetime(B0.entry); B0['exit'] = pd.to_datetime(B0.exit); B0['h1'] = False; B0 = B0.reset_index(drop=True)
PX = {}
for tk in B0.tk.unique():
    f = sorted(glob.glob(f'/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')); H = pd.read_parquet(f[-1]); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index(); PX[tk] = (H.index.values, H.open.values, H.high.values, H.close.values)
def px_at(tk, t): idx, o, h, c = PX[tk]; return o[min(np.searchsorted(idx, np.datetime64(t)), len(o) - 1)]
def spread(TK, N=6):
    R = np.zeros(len(D))
    for tk, e, x, r, ra in TK:
        a, b = D.searchsorted(e.normalize()), D.searchsorted(x.normalize())
        if a >= len(D): continue
        b = min(b, len(D) - 1); n = b - a + 1; w = 1 / N; R[a:b + 1] += w * ((1 + r) * (1 - 2e-4)) ** (1 / n) - w
    return pd.Series(R, index=D)
def cagr(s): e = (1 + s).cumprod(); y = (s.index[-1] - s.index[0]).days / 365.25; return e.iloc[-1] ** (1 / y) - 1
def mdd(s): e = (1 + s).cumprod(); return float((e / e.cummax() - 1).min())
def order(Bs):
    key = Bs.brank.values + 1e-6 * Bs.ratio.rank(ascending=False).values; return Bs.assign(_k=key).sort_values(['entry', '_k', 'tk'], kind='stable')
def sim_swap(Bs, rule=None, N=6, brk=.08):
    T = order(Bs); op = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; out = []; swaps = []
    for e, x, r, tk, ra in zip(T.entry.values, T.exit.values, T.ret.values, T.tk.values, T.ratio.values):
        done = sorted([p for p in op if p[0] <= e], key=lambda z: z[0]); op = [p for p in op if p[0] > e]
        for p in done: real += np.log1p(p[1] / N); hist.append((p[0], real)); win.append((p[0], real))
        if len(op) >= N:
            if rule is None: continue
            un = [px_at(p[2], e) / p[5] - 1 for p in op]; age = [(e - p[3]) / np.timedelta64(1, 'D') for p in op]; v = None
            if rule == 'S1': j = int(np.argmin(un)); v = j if un[j] < 0 else None
            elif rule == 'S2': j = int(np.argmax(age)); v = j if (age[j] >= 2 and un[j] < .005) else None
            elif rule == 'S3': j = int(np.argmin(un)); v = j if un[j] < .005 else None
            if v is None: continue
            p = op.pop(v); rr = un[v] - .0002; p[0] = pd.Timestamp(e); p[1] = rr; p[6][2] = pd.Timestamp(e); p[6][3] = rr; real += np.log1p(rr / N); hist.append((e, real)); win.append((e, real)); swaps.append((rr, r))
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [hh for _, hh in win]) < np.log1p(-brk): continue
        rec = [tk, pd.Timestamp(e), pd.Timestamp(x), r, ra]; op.append([x, r, tk, pd.Timestamp(e), ra, px_at(tk, e), rec]); out.append(rec)
    return [tuple(z) for z in out], swaps
def show(lab, TK, swaps=None):
    z = spread(TK); z = z[z.index >= '2019-01-01']; a, b = z[z.index < '2022-01-01'], z[z.index >= '2022-01-01']
    extra = f' | swaps {len(swaps)}: victims mean {np.mean([s[0] for s in swaps]):+.2%}, swapped-in flips (full vector ret) {np.mean([s[1] for s in swaps]):+.2%}' if swaps else ''
    print(f'{lab:22s} trades {len(TK)}, CAGR {cagr(z):5.1%}, DD {mdd(z):6.1%}, Sharpe {sharpe(z.values):.2f} | 2019-21 CAGR {cagr(a):5.1%} Sh {sharpe(a.values):.2f} | 2022-26 CAGR {cagr(b):5.1%} Sh {sharpe(b.values):.2f}{extra}')
TK, _ = sim_swap(B0, None); show('LIVE (no swap)', TK)
for rule in ('S1', 'S2', 'S3'): TK, sw = sim_swap(B0, rule); show(f'SWAP {rule}', TK, sw)
# scale-out on the path
n = len(B0); SO = {k: np.full(n, np.nan) for k in ('1', '2', '4')}
for tk in B0.tk.unique():
    idx, o, h, c = PX[tk]; m = np.where((B0.tk == tk).values)[0]; ie = np.searchsorted(idx, B0.entry.values[m]); ix = np.searchsorted(idx, B0.exit.values[m])
    for j, a, b in zip(m, ie, ix):
        e = o[a]; base = o[b] / e - 1
        for k, g in (('1', .01), ('2', .02), ('4', .04)):
            hit = np.where(h[a:b] >= e * (1 + g))[0] if b > a else []
            if len(hit): jj = hit[0]; fill = max(o[a + jj], e * (1 + g)) if o[a + jj] >= e * (1 + g) else e * (1 + g); SO[k][j] = .5 * (fill / e - 1) + .5 * base - .0002 - .5 * .0002
            else: SO[k][j] = base - .0002
for k in ('1', '2', '4'):
    X = B0.copy(); X['ret'] = SO[k]; TK, _ = sim_swap(X, None); show(f'SCALE-OUT 50% at +{k}%', TK)
    X2 = B0.copy(); X2['ret'] = SO[k]; print(f'      per trade mean (all flips): {SO[k].mean():+.3%} vs live {B0.ret.mean():+.3%}')
