"""Nov 24 2026. Net dealer gamma from raw ThetaData (data/theta/<TK>/<TK>_YYYY-MM.parquet: kind 'oi' rows keyed by report_date, kind 'eod' rows keyed by created[:10] with bid/ask). Run from the repo root on the Mac: python3 xs/build_gex_theta.py QQQ NVDA ...
Per day d: OI(report_date d) x EOD(d). Spot S_d from put-call parity (K + C - P at the strike with smallest |C-P|, median over expiries <= 45 dte). IV from OTM mid (bisection, r=4%, q=0) applied to both sides of the strike. Gamma = phi(d1)/(S sig sqrt T). Net GEX = sum(call OI gamma) - sum(put OI gamma), x 100 x S^2 x 0.01 (dollar gamma per 1% move). Contracts: dte 1-70, strike within 70-130% of S, bid > 0, mid >= 0.05.
Writes data/theta_gex/<TK>.csv (date, spot, net_gex, n). Validate against xs/ext/gex/<TK>.csv with xs/validate_gex_theta.py."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.special import ndtr
os.makedirs('data/theta_gex', exist_ok=True); R = 0.04
def bs(S, K, T, s, call):
    d1 = (np.log(S / K) + (R + .5 * s * s) * T) / (s * np.sqrt(T)); d2 = d1 - s * np.sqrt(T)
    c = S * ndtr(d1) - K * np.exp(-R * T) * ndtr(d2); p = K * np.exp(-R * T) * ndtr(-d2) - S * ndtr(-d1); return np.where(call, c, p)
def iv(S, K, T, px, call):
    lo = np.full(len(S), 1e-3); hi = np.full(len(S), 5.0)
    for _ in range(40):
        m = (lo + hi) / 2; v = bs(S, K, T, m, call); up = v < px; lo = np.where(up, m, lo); hi = np.where(up, hi, m)
    s = (lo + hi) / 2; ok = (s > 2e-3) & (s < 4.9); return np.where(ok, s, np.nan)
def month(f):
    x = pd.read_parquet(f, columns=['expiration', 'strike', 'right', 'open_interest', 'report_date', 'kind', 'created', 'bid', 'ask', 'close'])
    oi = x[x.kind == 'oi'].copy(); oi['d'] = pd.to_datetime(oi.report_date); oi = oi[['d', 'expiration', 'strike', 'right', 'open_interest']]
    ed = x[x.kind == 'eod'].copy(); ed['d'] = pd.to_datetime(ed.created.astype(str).str[:10], errors='coerce'); ed = ed.dropna(subset=['d'])
    ed['mid'] = np.where((ed.bid > 0) & (ed.ask > 0), (ed.bid + ed.ask) / 2, np.nan); ed['call'] = ed.right.astype(str).str.upper().str[0] == 'C'
    ed['T'] = (pd.to_datetime(ed.expiration) - ed.d).dt.days / 365.0; ed['dte'] = (pd.to_datetime(ed.expiration) - ed.d).dt.days; return oi, ed
out = []
for tk in sys.argv[1:]:
    rows = []
    for f in sorted(glob.glob(f'data/theta/{tk}/{tk}_*.parquet')):
        try: oi, ed = month(f)
        except Exception as e: print(tk, f, 'ERR', str(e)[:80]); continue
        if not len(oi) or not len(ed): continue
        c = ed[ed.call & ed.mid.notna()][['d', 'expiration', 'strike', 'mid', 'dte']].rename(columns={'mid': 'c'}); p = ed[~ed.call & ed.mid.notna()][['d', 'expiration', 'strike', 'mid']].rename(columns={'mid': 'p'})
        cp = c.merge(p, on=['d', 'expiration', 'strike']); cp = cp[(cp.dte >= 1) & (cp.dte <= 45)]; cp['gap'] = (cp.c - cp.p).abs()
        best = cp.sort_values('gap').groupby(['d', 'expiration']).head(1); best['S'] = best.strike + best.c - best.p; spot = best.groupby('d').S.median()
        ed = ed.join(spot.rename('S'), on='d'); e = ed[ed.S.notna() & ed.mid.notna() & (ed.dte >= 1) & (ed.dte <= 70) & (ed.mid >= .05) & (ed.strike > .7 * ed.S) & (ed.strike < 1.3 * ed.S)].copy()
        e = e[(e.call & (e.strike >= e.S)) | (~e.call & (e.strike <= e.S))]
        if not len(e): continue
        e['iv'] = iv(e.S.values, e.strike.values, e['T'].values, e.mid.values, e.call.values); e = e.dropna(subset=['iv']); e['g'] = np.exp(-.5 * ((np.log(e.S / e.strike) + (R + .5 * e.iv ** 2) * e['T']) / (e.iv * np.sqrt(e['T']))) ** 2) / np.sqrt(2 * np.pi) / (e.S * e.iv * np.sqrt(e['T']))
        g = e[['d', 'expiration', 'strike', 'g', 'S']].drop_duplicates(['d', 'expiration', 'strike'])
        o = oi.merge(g, on=['d', 'expiration', 'strike']); o['sgn'] = np.where(o.right.astype(str).str.upper().str[0] == 'C', 1, -1); o = o.drop_duplicates(['d', 'expiration', 'strike', 'right'], keep='last')
        o['v'] = o.sgn * o.open_interest * o.g * 100 * o.S ** 2 * .01; r = o.groupby('d').agg(net_gex=('v', 'sum'), spot=('S', 'first'), n=('v', 'size')).reset_index(); rows.append(r)
    if not rows: print(tk, 'no rows'); continue
    X = pd.concat(rows).drop_duplicates('d', keep='last').sort_values('d').rename(columns={'d': 'date'}); X[['date', 'spot', 'net_gex', 'n']].to_csv(f'data/theta_gex/{tk}.csv', index=False); print(tk, len(X), X.date.min().date(), X.date.max().date(), flush=True)
