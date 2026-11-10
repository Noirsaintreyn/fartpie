"""WEBULL CHAIN -> GREEKS for the site's Daily Analysis (Nov 10 2026). Read-only; no orders.

Takes the chain that webull_options_log.py saved (live/options/<SYM>/<date>[_oi].csv) and computes, per underlying:
  GEX (dealer convention: calls +, puts -, $ per 1% move), net DEX ($ delta), net vanna, net charm, total vega ($ per vol point),
  ATM IV (expiry nearest 30 days), 25-delta skew (put IV - call IV), gamma flip (spot where net GEX crosses zero), call / put walls.
Uses Webull's own delta / gamma / vega when the snapshot has them, else Black-Scholes from the contract IV.

RUN on the Mac after the logger (cron 16:15 ET and 08:35 ET):
  python3 webull_greeks.py                      -> writes live/greeks_snapshot.json and POSTs it to the site (needs BACKEND_URL and SITE_TOKEN in .env)
  python3 webull_greeks.py --symbols QQQ,SPY --no-push
FIRST RUN: python3 webull_options_log.py --probe  shows Webull's field names; if a column isn't picked up, add its name to FIELDS below.
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
from math import erf
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception: pass

ROOT = os.path.dirname(os.path.abspath(__file__)); OPT = f'{ROOT}/live/options'
R_, Q_ = 0.04, 0.007
FIELDS = dict(                      # normalized column-name candidates (lowercase, no underscores, no leading 'c')
    strike=['strikeprice', 'strike'], kind=['optiontype', 'callput', 'type', 'right', 'side'], expiry=['expirationdate', 'expiredate', 'expiry', 'expiration', 'expdate'],
    iv=['impliedvolatility', 'iv', 'impliedvol', 'impvol'], oi=['openinterest', 'oi'], delta=['delta'], gamma=['gamma'], vega=['vega'], vol=['volume'])


def norm(c):
    c = c.lower().replace('_', '').replace(' ', '')
    return c[1:] if c.startswith('c') and c[1:] in sum(FIELDS.values(), []) else c


def pick(df, key):
    cols = {norm(c): c for c in df.columns}
    for cand in FIELDS[key]:
        if cand in cols: return df[cols[cand]]
    return None


def N(x): return 0.5 * (1 + np.vectorize(erf)(x / np.sqrt(2)))
def phi(x): return np.exp(-x * x / 2) / np.sqrt(2 * np.pi)


def load(sym, date=None):
    fs = sorted(glob.glob(f'{OPT}/{sym}/*.csv'))
    fs = [f for f in fs if date is None or os.path.basename(f).startswith(date)]
    if not fs: raise SystemExit(f'no chain csv under {OPT}/{sym} - run webull_options_log.py first')
    day = os.path.basename(fs[-1])[:10]; same = [f for f in fs if os.path.basename(f).startswith(day)]
    f = next((x for x in same if x.endswith('_oi.csv')), same[-1])            # the next-morning open-interest refresh wins when it exists
    return pd.read_csv(f), f


def chain(df):
    k = pick(df, 'kind'); e = pick(df, 'expiry'); K = pick(df, 'strike'); iv = pick(df, 'iv'); oi = pick(df, 'oi')
    if any(x is None for x in (k, e, K, iv, oi)): raise SystemExit(f'missing columns; have {list(df.columns)[:40]}')
    d = pd.DataFrame({'K': pd.to_numeric(K, errors='coerce'), 'call': k.astype(str).str.upper().str.startswith('C'), 'exp': pd.to_datetime(e, errors='coerce'), 'iv': pd.to_numeric(iv, errors='coerce'), 'oi': pd.to_numeric(oi, errors='coerce')})
    for key in ('delta', 'gamma', 'vega'):
        v = pick(df, key); d['w_' + key] = pd.to_numeric(v, errors='coerce') if v is not None else np.nan
    d = d.dropna(subset=['K', 'exp', 'iv', 'oi']); d = d[(d.oi > 0) & (d.iv > 0)]
    if d.iv.median() > 3: d['iv'] = d.iv / 100                                 # percent -> decimal
    return d.reset_index(drop=True)


def greeks(d, S, now):
    T = np.maximum((d.exp - now).dt.total_seconds() / (365 * 86400), 1 / 365 / 2); s = d.iv.values; K = d.K.values; sq = np.sqrt(T)
    d1 = (np.log(S / K) + (R_ - Q_ + s * s / 2) * T) / (s * sq); d2 = d1 - s * sq
    gam = np.exp(-Q_ * T) * phi(d1) / (S * s * sq); veg = S * np.exp(-Q_ * T) * phi(d1) * sq / 100
    dl = np.where(d.call, np.exp(-Q_ * T) * N(d1), -np.exp(-Q_ * T) * N(-d1)); van = -np.exp(-Q_ * T) * phi(d1) * d2 / s / 100
    chm = -np.exp(-Q_ * T) * phi(d1) * (2 * (R_ - Q_) * T - d2 * s * sq) / (2 * T * s * sq) / 365
    out = d.copy(); out['T'] = T
    out['gamma'] = d.w_gamma.fillna(pd.Series(gam, index=d.index)); out['delta'] = d.w_delta.fillna(pd.Series(dl, index=d.index)); out['vega'] = d.w_vega.fillna(pd.Series(veg, index=d.index))
    out['vanna'] = van; out['charm'] = chm; out['sgn'] = np.where(d.call, 1.0, -1.0); return out


def gex_at(d, S, now):
    g = greeks(d.assign(), S, now); return float((g.sgn * g.gamma * g.oi * 100 * S * S * 0.01).sum())


def summarize(sym, df, now):
    d = chain(df); S = float(df['spot'].iloc[0])
    g = greeks(d, S, now); m = g.oi * 100
    gex_k = (g.sgn * g.gamma * m * S * S * 0.01).groupby(g.K).sum(); cg = (g.gamma * m * S * S * 0.01)[g.call].groupby(g.K[g.call]).sum(); pg = (g.gamma * m * S * S * 0.01)[~g.call].groupby(g.K[~g.call]).sum()
    grid = S * (1 + np.linspace(-.06, .06, 49)); prof = np.array([gex_at(d, x, now) for x in grid]); flip = None
    for i in range(len(grid) - 1):
        if prof[i] == 0 or prof[i] * prof[i + 1] < 0:
            flip = float(grid[i] - prof[i] * (grid[i + 1] - grid[i]) / (prof[i + 1] - prof[i])); break
    near = g.iloc[(g['T'] * 365 - 30).abs().argsort()]; e0 = near.exp.iloc[0]; ex = g[g.exp == e0]
    atm = ex.iloc[(ex.K - S).abs().argsort()[:2]]; iv = float(atm.iv.mean())
    c = ex[ex.call]; p = ex[~ex.call]; skew = None
    if len(c) and len(p):
        skew = float(p.iloc[(p.delta + .25).abs().argsort()[:1]].iv.iloc[0] - c.iloc[(c.delta - .25).abs().argsort()[:1]].iv.iloc[0])
    return dict(spot=round(S, 2), gex=float((g.sgn * g.gamma * m * S * S * 0.01).sum()), dex=float((g.delta * m * S).sum()), vanna=float((g.sgn * g.vanna * m * S).sum()), vega=float((g.vega * m).sum()),
                charm=float((g.sgn * g.charm * m * S).sum()), iv=iv, skew=skew, expiry=str(pd.Timestamp(e0).date()), gamma_flip=round(flip, 2) if flip else None,
                call_wall=float(cg.idxmax()) if len(cg) else None, put_wall=float(pg.idxmax()) if len(pg) else None, contracts=int(len(g)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--symbols', default='QQQ,SPY'); ap.add_argument('--date', default=None); ap.add_argument('--no-push', action='store_true'); a = ap.parse_args()
    now = pd.Timestamp.now(tz='America/New_York').tz_localize(None); res = {}; asof = None
    for sym in a.symbols.split(','):
        try:
            df, f = load(sym, a.date); asof = os.path.basename(f)[:10]; res[sym] = summarize(sym, df, now); print(sym, f, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[sym].items()})
        except SystemExit as e: print(sym, 'skipped:', e)
        except Exception as e: print(sym, 'failed:', e)
    if not res: sys.exit(1)
    payload = dict(asof=asof, built=now.strftime('%Y-%m-%d %H:%M'), source='webull', symbols=res)
    os.makedirs(f'{ROOT}/live', exist_ok=True); json.dump(payload, open(f'{ROOT}/live/greeks_snapshot.json', 'w'), indent=1)
    url, tok = os.getenv('BACKEND_URL', '').rstrip('/'), os.getenv('SITE_TOKEN', '')
    if a.no_push or not (url and tok): return print('saved live/greeks_snapshot.json (not pushed)')
    import requests
    r = requests.post(f'{url}/api/site/greeks', json=payload, headers={'X-Site-Token': tok}, timeout=30); print('push', r.status_code, r.text[:80])


if __name__ == '__main__':
    main()
