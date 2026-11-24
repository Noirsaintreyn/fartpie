"""Nov 24 2026: per-name daily Greek aggregates from EOD option chains (data/theta_compact, 38 names, 2018-2026).  IV by vectorised BS bisection on the bid/ask mid (r=4%, q=0); rows kept: 2 <= dte <= 60, 0.8 <= K/S <= 1.2, vol > 0, mid >= .05.
Volume-weighted 'customer-flow' exposures (no open interest in this file, direction unknown): X_net = sum_calls vol*X - sum_puts vol*X for X in {gamma, vega, vanna, charm} (per 1% / per day scaling dropped; ranked within name later), plus act = total contracts, atm_iv (|K/S-1|<=.03, dte 7-45), skew = IV(K/S~.9 puts) - IV(K/S~1.1 calls) at dte 15-45, term = IV(dte<=14 ATM) - IV(dte 30-60 ATM).  Output: scratchpad greeks_daily.parquet (tk, d, ...)."""
import os, numpy as np, pandas as pd
from math import pi
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; P = '/mnt/user-data/uploads/fartpie/data/theta_compact/'
from scipy.special import ndtr
def bs(S, K, T, r, s, cp):
    d1 = (np.log(S / K) + (r + .5 * s * s) * T) / (s * np.sqrt(T)); d2 = d1 - s * np.sqrt(T); return cp * (S * ndtr(cp * d1) - K * np.exp(-r * T) * ndtr(cp * d2)) if False else (S * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2) if cp is None else None)
def price(S, K, T, r, s, cp):
    d1 = (np.log(S / K) + (r + .5 * s * s) * T) / (s * np.sqrt(T)); d2 = d1 - s * np.sqrt(T); c = S * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2); return np.where(cp > 0, c, c - S + K * np.exp(-r * T))
def iv(S, K, T, r, mid, cp):
    lo = np.full(len(S), 1e-3); hi = np.full(len(S), 5.0)
    for _ in range(40):
        m = (lo + hi) / 2; p = price(S, K, T, r, m, cp); up = p < mid; lo = np.where(up, m, lo); hi = np.where(up, hi, m)
    return (lo + hi) / 2
out = []
for f in sorted(os.listdir(P)):
    tk = f[:-8]; D = pd.read_parquet(P + f); D = D[(D.dte >= 2) & (D.dte <= 60) & (D.vol > 0)].copy(); D['m'] = D.K / D.S; D = D[(D.m >= .8) & (D.m <= 1.2)]; D['mid'] = (D.bid + D.ask) / 2; D = D[(D.mid >= .05) & (D.ask >= D.bid) & (D.bid > 0)]
    S = D.S.values.astype(float); K = D.K.values.astype(float); T = D.dte.values / 365.; r_ = .04; cp = D.r.values.astype(float); mid = D.mid.values.astype(float)
    ok = mid > np.maximum(0, np.where(cp > 0, S - K * np.exp(-r_ * T), K * np.exp(-r_ * T) - S)) + 1e-3; D = D[ok]; S, K, T, cp, mid = S[ok], K[ok], T[ok], cp[ok], mid[ok]
    s = iv(S, K, T, r_, mid, cp); D['iv'] = s; d1 = (np.log(S / K) + (r_ + .5 * s * s) * T) / (s * np.sqrt(T)); d2 = d1 - s * np.sqrt(T); pdf = np.exp(-.5 * d1 * d1) / np.sqrt(2 * pi)
    D['gamma'] = pdf / (S * s * np.sqrt(T)); D['vega'] = S * pdf * np.sqrt(T); D['vanna'] = -pdf * d2 / s; D['charm'] = -pdf * (2 * r_ * T - d2 * s * np.sqrt(T)) / (2 * T * s * np.sqrt(T))
    for x in ('gamma', 'vega', 'vanna', 'charm'): D[x + '_w'] = D[x] * D.vol * cp * (S if x != 'vanna' and x != 'charm' else 1)
    D = D[np.isfinite(D.iv) & (D.iv < 4.9) & (D.iv > .002)]
    g = D.groupby('d'); A = g[['gamma_w', 'vega_w', 'vanna_w', 'charm_w']].sum().rename(columns=lambda c: c[:-2] + '_net'); A['act'] = g.vol.sum()
    atm = D[(D.m.sub(1).abs() <= .03)]; A['atm_iv'] = atm[(atm.dte >= 7) & (atm.dte <= 45)].groupby('d').iv.mean()
    pt = D[(D.r < 0) & D.m.between(.87, .93) & D.dte.between(15, 45)].groupby('d').iv.mean(); ct = D[(D.r > 0) & D.m.between(1.07, 1.13) & D.dte.between(15, 45)].groupby('d').iv.mean(); A['skew'] = pt - ct
    A['term'] = atm[atm.dte <= 14].groupby('d').iv.mean() - atm[(atm.dte >= 30)].groupby('d').iv.mean(); A['tk'] = tk; out.append(A.reset_index()); print(tk, len(A), flush=True)
G = pd.concat(out); G.to_parquet(SP + 'greeks_daily.parquet'); print(G.groupby('tk').size().describe(), G[['gamma_net', 'vega_net', 'vanna_net', 'charm_net', 'atm_iv', 'skew', 'term']].describe().T)
