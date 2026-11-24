"""Volume-gamma proxy of dealer net gamma from theta_compact (no open interest in the file): per day sum over contracts of sign(call +, put -) x BS gamma x volume x 100 x S^2 x 1%. IV from mid price of OTM contracts (bisection, r=0). Built so SPY/IWM can carry a gamma state like QQQ's real OI-based file. Validated vs QQQ real net_gex in report 441."""
import numpy as np, pandas as pd
from scipy.stats import norm
U = '/mnt/user-data/uploads/fartpie/'
def bs(S, K, T, s, cp):
    d1 = (np.log(S / K) + .5 * s * s * T) / (s * np.sqrt(T)); d2 = d1 - s * np.sqrt(T)
    return np.where(cp == 1, S * norm.cdf(d1) - K * norm.cdf(d2), K * norm.cdf(-d2) - S * norm.cdf(-d1))
def proxy(tk):
    d = pd.read_parquet(U + f'data/theta_compact/{tk}.parquet'); d = d[(d.dte >= 1) & (d.dte <= 45) & (d.bid > 0) & (d.ask >= d.bid)]
    d = d[np.abs(d.K / d.S - 1) < .12]; d = d[((d.r == 1) & (d.K >= d.S)) | ((d.r == -1) & (d.K < d.S))]  # OTM only
    S, K, cp = d.S.values.astype(float), d.K.values.astype(float), d.r.values; T = d.dte.values / 365.; mid = ((d.bid + d.ask) / 2).values
    lo, hi = np.full(len(d), .01), np.full(len(d), 3.)
    for _ in range(40):
        m = (lo + hi) / 2; p = bs(S, K, T, m, cp); up = p < mid; lo = np.where(up, m, lo); hi = np.where(up, hi, m)
    s = (lo + hi) / 2; ok = (s > .02) & (s < 2.9); d1 = (np.log(S / K) + .5 * s * s * T) / (s * np.sqrt(T)); g = norm.pdf(d1) / (S * s * np.sqrt(T))
    d = d.assign(gex=np.where(ok, cp * g * d.vol.values * 100 * S * S * .01, 0.), iv=s)
    out = d.groupby('d').gex.sum().rename('net_gex_vol'); return out
if __name__ == '__main__':
    import sys
    for tk in sys.argv[1:]:
        o = proxy(tk); o.to_csv(f'/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/gexproxy_{tk}.csv'); print(tk, len(o), o.index.min(), o.index.max(), flush=True)
