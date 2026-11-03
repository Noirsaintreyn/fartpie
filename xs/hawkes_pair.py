"""Nov 3 2026: pair the Hawkes sell-pressure intensity with today's 1h-confirmed 15m stop (Ainf: hold through 15m breaks while the 1h vector is bullish).
Hawkes events (as in xs/hawkes_test.py): sell burst = 15m bar change < -0.5 ATR(14) with RVOL >= 1.5 (volume / same-time-of-day mean of the prior 20 sessions);
S_t = S_{t-1} exp(-1/tau) + burst_t over all 15m bars (premarket + RTH).  DECLARED BEFORE RESULTS.
 AH(tau, T): pre-sticky exit = (15m flip down AND 1h bearish) OR S >= T      tau in {4, 8} bars, T in {1.0, 1.5, 2.0}   [Hawkes replaces the hard stop as the danger signal]
 LH(tau, T): current exit (15m flip down) OR S >= T                           (the Oct 20 'early' test, re-run on the same harness as reference, tau=8, T=1.5)
Scored vs the CURRENT model and vs Ainf, 40 matched runs, both periods; plus 2026 YTD."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
def hawkes(H, tau):
    c, h, l = H.close.values, H.high.values, H.low.values; atr = compute_atr_series(h, l, c, 14); dc = np.r_[np.nan, np.diff(c)]
    tod = H.m.values; v = pd.Series(H.volume.values.astype(float)); rv = v / v.groupby(tod).transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1)).values
    burst = ((dc < -.5 * atr) & (rv.values >= 1.5)).astype(float); S = np.zeros(len(H)); k = np.exp(-1 / tau)
    for i in range(1, len(H)): S[i] = S[i - 1] * k + burst[i]
    return S
def run3(H, reg, s1, S, mode, T):
    o, c, hi, d, m = H.open.values, H.close.values, H.high.values, H.dt.values, H.m.values
    g = [j for j in range(1, len(H)) if 570 <= m[j] < 960]; out = []; pos = False
    KR = [reg[j - 1] for j in g]; KS = [s1[j - 1] for j in g]; KX = [S[j - 1] for j in g]
    for p in range(1, len(g)):
        j = g[p]
        if not pos:
            if KR[p] == 1 and KR[p - 1] != 1: pos, e, px, b1, peak, h1 = True, j, o[j], False, o[j], False
            continue
        if j == e: continue
        k = g[p - 1]
        if k >= e: peak = max(peak, hi[k])
        b1 = b1 or KS[p] == 1; cur = c[j - 1] / px - 1; h1 = h1 or (cur >= .02 and b1)
        if h1: ex = KS[p] == -1
        elif mode == 'AH': ex = (KR[p] == -1 and KS[p] == -1) or KX[p] >= T
        else: ex = KR[p] == -1 or KX[p] >= T
        pg = peak / px - 1
        if pg >= .05 and cur <= .5 * pg: ex = True
        if ex: out.append((d[e], d[j], o[j] / px - 1 - C2)); pos = False
    return out
if __name__ == '__main__':
    VAR = {'AH4_1.0': ('AH', 4, 1.0), 'AH4_1.5': ('AH', 4, 1.5), 'AH4_2.0': ('AH', 4, 2.0), 'AH8_1.0': ('AH', 8, 1.0), 'AH8_1.5': ('AH', 8, 1.5), 'AH8_2.0': ('AH', 8, 2.0), 'LH8_1.5': ('LH', 8, 1.5)}
    Xo = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet'); rows = []
    for tk in sorted(Xo.tk.unique()):
        f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
        if not f: continue
        H = load(f[0]); reg0, s10 = states(H, vec); SS = {t: hawkes(H, t) for t in (4, 8)}
        for nm, (mode, tau, T) in VAR.items(): rows += [(tk, nm, False, pd.Timestamp(a), pd.Timestamp(b), r) for a, b, r in run3(H, reg0, s10, SS[tau], mode, T)]
        print(tk, flush=True)
    V = pd.read_parquet('/tmp/claude-0/ml/stop1h_V.parquet'); V = V[V['var'].isin(['L', 'Ainf', 'H1'])]
    pd.concat([V, pd.DataFrame(rows, columns=V.columns)]).to_parquet('/tmp/claude-0/ml/hawkes_V.parquet')
