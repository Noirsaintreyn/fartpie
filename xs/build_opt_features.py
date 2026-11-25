"""Option-feature panel per day (Nov 24 2026): gex, short-dated (<=2 DTE) vs longer gamma and share, vanna, charm, vega, DEX, put/call OI and volume, ATM IV 30d, term slope, 25d-ish skew. Same OI(report_date d) x EOD(d) join as build_gex_theta2. Run from repo root on the Mac: python3 xs/build_opt_features.py QQQ TSLA ...
"""
"""V2 (Nov 24 2026): also writes the ZERO-GAMMA LEVEL per day (price where net dollar gamma crosses 0 when spot is shifted -12%..+12% with each contract IV/OI held fixed; the crossing nearest spot), net gamma at spot, and ATM IV (median IV within 3% of spot, dte<=30). Writes data/theta_feat2/<TK>.csv.
Original doc: Net dealer gamma from raw ThetaData (data/theta/<TK>/<TK>_YYYY-MM.parquet: kind 'oi' rows keyed by report_date, kind 'eod' rows keyed by created[:10] with bid/ask). Run from the repo root on the Mac: python3 xs/build_gex_theta.py QQQ NVDA ...
Per day d: OI(report_date d) x EOD(d). Spot S_d from put-call parity (K + C - P at the strike with smallest |C-P|, median over expiries <= 45 dte). IV from OTM mid (bisection, r=4%, q=0) applied to both sides of the strike. Gamma = phi(d1)/(S sig sqrt T). Net GEX = sum(call OI gamma) - sum(put OI gamma), x 100 x S^2 x 0.01 (dollar gamma per 1% move). Contracts: dte 1-70, strike within 70-130% of S, bid > 0, mid >= 0.05.
Writes data/theta_gex/<TK>.csv (date, spot, net_gex, n). Validate against xs/ext/gex/<TK>.csv with xs/validate_gex_theta.py."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.special import ndtr
os.makedirs('data/theta_feat2', exist_ok=True); R = 0.04
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
def feat(x, e, ed_d, vol_d):
    S = float(x.S.iloc[0]); T = x['T'].values; sg = x.iv.values; K = x.strike.values; w = x.open_interest.values.astype(float); sgn = x.sgn.values
    sq = np.sqrt(T); d1 = (np.log(S / K) + (R + .5 * sg ** 2) * T) / (sg * sq); d2 = d1 - sg * sq; ph = np.exp(-.5 * d1 ** 2) / np.sqrt(2 * np.pi)
    gam = ph / (S * sg * sq); vanna = -ph * d2 / sg; charm = -ph * (2 * R * T - d2 * sg * sq) / (2 * T * sg * sq) / 365.0; vega = S * ph * sq * .01
    delta = ndtr(d1) - (sgn < 0)
    dte = ((pd.to_datetime(x.expiration) - x.d).dt.days).values; sh = dte <= 2
    g = sgn * w * gam * 100 * S * S * .01
    r = dict(gex=g.sum(), gex_short=g[sh].sum(), gex_long=g[~sh].sum(), short_share=np.abs(g[sh]).sum() / max(np.abs(g).sum(), 1e-9),
             vanna=(sgn * w * vanna * 100 * S * .01).sum(), charm=(sgn * w * charm * 100 * S).sum(), vega=(sgn * w * vega * 100).sum(), dex=(w * delta * 100 * S).sum(),
             pc_oi=w[sgn < 0].sum() / max(w[sgn > 0].sum(), 1.0),
             charm_d7=(sgn * w * charm * 100 * S)[dte <= 7].sum(), charm_d30=(sgn * w * charm * 100 * S)[(dte > 7) & (dte <= 30)].sum(), charm_dl=(sgn * w * charm * 100 * S)[dte > 30].sum())
    return r
for tk in sys.argv[1:]:
    rows = []
    for f in sorted(glob.glob(f'data/theta/{tk}/{tk}_*.parquet')):
        try:
            x = pd.read_parquet(f, columns=['expiration', 'strike', 'right', 'open_interest', 'report_date', 'kind', 'created', 'bid', 'ask', 'close', 'volume'])
        except Exception as ex: print(tk, f, 'ERR', str(ex)[:80]); continue
        oi = x[x.kind == 'oi'].copy(); oi['d'] = pd.to_datetime(oi.report_date); oi = oi[['d', 'expiration', 'strike', 'right', 'open_interest']]
        ed = x[x.kind == 'eod'].copy(); ed['d'] = pd.to_datetime(ed.created.astype(str).str[:10], errors='coerce'); ed = ed.dropna(subset=['d'])
        ed['mid'] = np.where((ed.bid > 0) & (ed.ask > 0), (ed.bid + ed.ask) / 2, np.nan); ed['call'] = ed.right.astype(str).str.upper().str[0] == 'C'
        ed['T'] = (pd.to_datetime(ed.expiration) - ed.d).dt.days / 365.0; ed['dte'] = (pd.to_datetime(ed.expiration) - ed.d).dt.days
        if not len(oi) or not len(ed): continue
        c = ed[ed.call & ed.mid.notna()][['d', 'expiration', 'strike', 'mid', 'dte']].rename(columns={'mid': 'c'}); p = ed[~ed.call & ed.mid.notna()][['d', 'expiration', 'strike', 'mid']].rename(columns={'mid': 'p'})
        cp = c.merge(p, on=['d', 'expiration', 'strike']); cp = cp[(cp.dte >= 1) & (cp.dte <= 45)]; cp['gap'] = (cp.c - cp.p).abs()
        best = cp.sort_values('gap').groupby(['d', 'expiration']).head(1); best['S'] = best.strike + best.c - best.p; spot = best.groupby('d').S.median()
        ed = ed.join(spot.rename('S'), on='d')
        vol = ed[ed.S.notna() & (ed.dte >= 0) & (ed.dte <= 45)].groupby(['d', 'call']).volume.sum().unstack().rename(columns={True: 'cv', False: 'pv'}); vol['pc_vol'] = vol.pv / vol.cv.clip(lower=1); vol['tot_vol'] = vol.pv + vol.cv
        va = ed[ed.S.notna() & (ed.dte >= 0)].groupby(['d', 'call']).volume.sum().unstack().rename(columns={True: 'cva', False: 'pva'}); vol['pc_vol_all'] = va.pva / va.cva.clip(lower=1)
        e = ed[ed.S.notna() & ed.mid.notna() & (ed.dte >= 1) & (ed.dte <= 70) & (ed.mid >= .05) & (ed.strike > .7 * ed.S) & (ed.strike < 1.3 * ed.S)].copy()
        e = e[(e.call & (e.strike >= e.S)) | (~e.call & (e.strike <= e.S))]
        if not len(e): continue
        e['iv'] = iv(e.S.values, e.strike.values, e['T'].values, e.mid.values, e.call.values); e = e.dropna(subset=['iv']); e['m'] = e.strike / e.S
        atm = e[(e.m - 1).abs() < .03]; iv_s = atm[(atm.dte >= 3) & (atm.dte <= 10)].groupby('d').iv.median(); iv_l = atm[(atm.dte >= 25) & (atm.dte <= 45)].groupby('d').iv.median(); iv30 = atm[atm.dte <= 30].groupby('d').iv.median()
        pu = e[(~e.call) & (e.m > .93) & (e.m < .97) & (e.dte >= 14) & (e.dte <= 45)].groupby('d').iv.median(); ca = e[e.call & (e.m > 1.03) & (e.m < 1.07) & (e.dte >= 14) & (e.dte <= 45)].groupby('d').iv.median()
        g = e[['d', 'expiration', 'strike', 'iv', 'T', 'S']].drop_duplicates(['d', 'expiration', 'strike'])
        o = oi.merge(g, on=['d', 'expiration', 'strike']); o['sgn'] = np.where(o.right.astype(str).str.upper().str[0] == 'C', 1, -1); o = o.drop_duplicates(['d', 'expiration', 'strike', 'right'], keep='last')
        res = []
        for d_, xx in o.groupby('d'):
            if len(xx) < 20: continue
            r = feat(xx, e, None, None); r['date'] = d_; r['spot'] = float(xx.S.iloc[0]); r['n'] = len(xx); res.append(r)
        if not res: continue
        r = pd.DataFrame(res).set_index('date'); r['iv30'] = iv30; r['iv_short'] = iv_s; r['iv_long'] = iv_l; r['term'] = iv_l - iv_s; r['skew'] = pu - ca
        r = r.join(vol[['pc_vol', 'tot_vol', 'pc_vol_all']]); rows.append(r.reset_index())
    if not rows: print(tk, 'no rows'); continue
    X = pd.concat(rows).drop_duplicates('date', keep='last').sort_values('date'); X.to_csv(f'data/theta_feat2/{tk}.csv', index=False); print(tk, len(X), X.date.min().date(), X.date.max().date(), flush=True)
