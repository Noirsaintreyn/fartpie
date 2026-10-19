"""GEX-by-strike chart to pair with the THOD/TLOD posts. Dealer gamma per strike from ThetaData EOD quotes + OI
(same validated math as xs/gex_from_theta.py: parity forward per expiry, Black-76 IV from OTM mids, BSM spot gamma,
calls +, puts - , $ gamma per 1% move = gamma*OI*100*S^2*0.01). Uses the newest EOD day and the OI reported the next
morning (= positions at that close), so at 09:30 it reflects yesterday's close. NQ=F uses QQQ options, strikes scaled to NQ.
  refresh(tk)  pulls the current month from Theta Terminal (must be running)          render(tk, ...) -> PNG bytes"""
import glob, io, os, subprocess, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import norm
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BG, FG, GRID, MUT = '#0f1116', '#e8eaed', '#1d222c', '#8a93a3'
POS, NEG, TEAL, GOLD, BLUE = '#2fbf71', '#e5484d', '#3fd0c9', '#e8c15a', '#3d93ff'
RATE = .038
def opt_ticker(t): return 'QQQ' if t.upper().replace('=F', '') in ('NQ', 'MNQ', 'NDX', 'QQQ') else ('SPY' if t.upper().replace('=F', '') in ('ES', 'MES', 'SPX', 'SPY') else t.upper())
def refresh(t, timeout=240):
    tk = opt_ticker(t); today = pd.Timestamp.now(tz='America/New_York').tz_localize(None).normalize(); ms = today.replace(day=1)
    f = f'{ROOT}/data/theta/{tk}/{tk}_{ms:%Y-%m}.parquet'
    if os.path.exists(f): os.replace(f, f + '.bak')
    env = dict(os.environ, START=f'{ms:%Y-%m-%d}', END=f'{today:%Y-%m-%d}', WORKERS='2')
    try: subprocess.run([sys.executable, 'fetch_theta_options.py', tk], cwd=ROOT, env=env, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired: pass
    if not os.path.exists(f) and os.path.exists(f + '.bak'): os.replace(f + '.bak', f)        # keep the old file if the pull failed
    elif os.path.exists(f + '.bak'): os.remove(f + '.bak')
def _b76(F, K, T, s, c):
    v = s * np.sqrt(T); d1 = (np.log(F / K) + .5 * v * v) / v; d2 = d1 - v
    return np.where(c, F * norm.cdf(d1) - K * norm.cdf(d2), K * norm.cdf(-d2) - F * norm.cdf(-d1))
def _iv(p, F, K, T, c):
    lo, hi = np.full(len(p), .005), np.full(len(p), 5.)
    for _ in range(50):
        m = .5 * (lo + hi); up = _b76(F, K, T, m, c) > p; hi = np.where(up, m, hi); lo = np.where(up, lo, m)
    s = .5 * (lo + hi); intr = np.where(c, np.maximum(F - K, 0), np.maximum(K - F, 0))
    return np.where((p > intr + 1e-6) & (s > .006) & (s < 4.9), s, np.nan)
def profile(t, max_dte=120):
    tk = opt_ticker(t); files = sorted(glob.glob(f'{ROOT}/data/theta/{tk}/{tk}_*.parquet'))[-2:]
    X = pd.concat([pd.read_parquet(f, columns=['kind', 'expiration', 'strike', 'right', 'open_interest', 'report_date', 'created', 'bid', 'ask']) for f in files])
    X['right'] = X.right.str.upper()
    E = X[X.kind == 'eod'].copy(); E['d'] = pd.to_datetime(E.created.astype(str).str[:10]); d = E.d.max(); E = E[E.d == d]
    O = X[X.kind == 'oi'].copy(); O['rd'] = pd.to_datetime(O.report_date).dt.normalize()
    rd = O.rd[O.rd > d].min() if (O.rd > d).any() else O.rd.max()           # OI reported the morning after the EOD day
    g = E.merge(O[O.rd == rd][['expiration', 'strike', 'right', 'open_interest']], on=['expiration', 'strike', 'right'], how='left', suffixes=('_x', ''))
    g['open_interest'] = g.open_interest.fillna(0); g['dte'] = (pd.to_datetime(g.expiration) - d).dt.days; g = g[(g.dte >= 1) & (g.dte <= max_dte)]
    g['mid'] = np.where((g.bid > 0) & (g.ask >= g.bid), .5 * (g.bid + g.ask), np.nan)
    spot = np.nan
    for e, h in sorted(g.groupby('expiration'), key=lambda kv: kv[1].dte.iat[0])[:4]:
        P0 = h.pivot_table(index='strike', columns='right', values='mid', aggfunc='first')
        if 'CALL' in P0 and 'PUT' in P0 and not P0.dropna().empty:
            b0 = P0.dropna(); k0 = (b0.CALL - b0.PUT).abs().idxmin(); spot = k0 * np.exp(-RATE * h.dte.iat[0] / 365) + b0.at[k0, 'CALL'] - b0.at[k0, 'PUT']; break
    rows = []; C = []
    for e, h in g.groupby('expiration'):
        T = h.dte.iat[0] / 365; df = np.exp(-RATE * T)
        P = h.pivot_table(index='strike', columns='right', values=['mid', 'open_interest'], aggfunc='first')
        if ('mid', 'CALL') not in P or ('mid', 'PUT') not in P or P['mid'].dropna().empty: continue
        b = P['mid'].dropna(); k0 = (b.CALL - b.PUT).abs().idxmin(); F = k0 + (b.at[k0, 'CALL'] - b.at[k0, 'PUT']) / df
        K = P.index.values; oc = K >= F
        s = pd.Series(_iv(np.where(oc, P[('mid', 'CALL')].values, P[('mid', 'PUT')].values) / df, F, K, T, oc), index=K).interpolate(limit_area='inside').values
        q = RATE - np.log(F / spot) / T; v = s * np.sqrt(T); d1 = (np.log(F / K) + .5 * v * v) / v
        gm = np.nan_to_num(np.exp(-q * T) * norm.pdf(d1) / (spot * v)); u = 100 * spot * spot * .01
        C.append((K, T, s, q, np.nan_to_num(P[('open_interest', 'CALL')].values) - np.nan_to_num(P[('open_interest', 'PUT')].values)))
        rows.append(pd.DataFrame({'strike': K, 'call': gm * np.nan_to_num(P[('open_interest', 'CALL')].values) * u,
                                  'put': -gm * np.nan_to_num(P[('open_interest', 'PUT')].values) * u}))
    G = pd.concat(rows).groupby('strike').sum(); G['net'] = G.call + G.put
    near = G[(G.index > spot * .9) & (G.index < spot * 1.1)]
    cw = near[near.index >= spot].net.idxmax() if (near.index >= spot).any() else np.nan
    pw = near[near.index <= spot].net.idxmin() if (near.index <= spot).any() else np.nan
    # gamma flip: re-price every contract's gamma at hypothetical spots and find where total dealer gamma changes sign
    grid = np.linspace(spot * .9, spot * 1.1, 161); tot = []
    for S_ in grid:
        z = 0.0
        for K, T, sv, q, net_oi in C:
            Fh = S_ * np.exp((RATE - q) * T); v = sv * np.sqrt(T); d1 = (np.log(Fh / K) + .5 * v * v) / v
            z += np.nansum(np.exp(-q * T) * norm.pdf(d1) / (S_ * v) * net_oi) * 100 * S_ * S_ * .01
        tot.append(z)
    tot = np.array(tot); cx = np.where(np.sign(tot[1:]) != np.sign(tot[:-1]))[0]
    zg = float(grid[cx[np.argmin(np.abs(grid[cx] - spot))]]) if len(cx) else np.nan
    return dict(tk=tk, asof=d, oi_date=rd, spot=float(spot), G=G, call_wall=cw, put_wall=pw, zero_gamma=zg, total=float(G.net.sum()))
def render(t, ref_price=None, hl=None, hl4=None, width_pct=None):
    """ref_price: the chart ticker's price (e.g. NQ) to scale QQQ strikes; hl / hl4: backend hod_lod dicts (1h / 4h) to overlay."""
    P = profile(t); k = (ref_price / P['spot']) if ref_price and opt_ticker(t) != t.upper() else 1.0
    G = P['G'].copy(); G.index = G.index * k; spot = P['spot'] * k; w = width_pct or (.04 if k != 1 else .08)
    G = G[(G.index > spot * (1 - w)) & (G.index < spot * (1 + w))]
    fig = plt.figure(figsize=(9, 10), facecolor=BG); ax = fig.add_axes([.12, .07, .62, .82], facecolor=BG)
    step = np.median(np.diff(G.index.values)) if len(G) > 2 else 1
    ax.barh(G.index, G.call / 1e9, height=step * .8, color=POS, alpha=.35, lw=0); ax.barh(G.index, G.put / 1e9, height=step * .8, color=NEG, alpha=.35, lw=0)
    ax.barh(G.index, G.net / 1e9, height=step * .45, color=np.where(G.net >= 0, POS, NEG), lw=0)
    ax.axvline(0, color=MUT, lw=.8)
    LBL = []
    def hline(y, col, lab, ls='-', lw=1.4):
        if y is None or not np.isfinite(y) or not (G.index.min() <= y <= G.index.max()): return
        ax.axhline(y, color=col, lw=lw, ls=ls); LBL.append([ax.transData.transform((0, y))[1] / fig.dpi / fig.get_figheight(), f'{lab}  {y:,.2f}', col])
    hline(spot, FG, 'last close', lw=1.6)
    hline(P['call_wall'] * k if np.isfinite(P['call_wall']) else None, POS, 'call wall', '--')
    hline(P['put_wall'] * k if np.isfinite(P['put_wall']) else None, NEG, 'put wall', '--')
    hline(P['zero_gamma'] * k if np.isfinite(P['zero_gamma']) else None, GOLD, 'gamma flip', ':')
    for h_, tag, ls in ((hl, '1h', '-'), (hl4, '4h', '--')):
        if h_:
            try: hline(float(h_['hod']['1std']), TEAL, f'THOD {tag}', ls, 1.1); hline(float(h_['lod']['1std']), TEAL, f'TLOD {tag}', ls, 1.1)
            except Exception: pass
    LBL.sort(key=lambda z: z[0])
    for i in range(1, len(LBL)):                                   # keep labels at least 2% of the figure apart
        if LBL[i][0] - LBL[i - 1][0] < .02: LBL[i][0] = LBL[i - 1][0] + .02
    for yy, txt, col in LBL: fig.text(.75, yy, txt, color=col, fontsize=9, va='center')
    ax.tick_params(colors=MUT, labelsize=8, length=0); [s.set_visible(False) for s in ax.spines.values()]; ax.grid(axis='x', color=GRID, lw=.6)
    ax.set_xlabel('dealer gamma, $bn per 1% move   (green = calls / long gamma, red = puts / short gamma)', color=MUT, fontsize=8)
    name = t.replace('=F', '')
    fig.text(.12, .955, f'{name} · GAMMA EXPOSURE BY STRIKE', color=FG, fontsize=15, fontweight='bold')
    regime = 'POSITIVE gamma (dealers dampen moves)' if P['total'] > 0 else 'NEGATIVE gamma (dealers amplify moves)'
    fig.text(.12, .925, f"{P['tk']} options · close {P['asof']:%b %d} · OI {P['oi_date']:%b %d}" + (f" · strikes scaled ×{k:.2f} to {name}" if k != 1 else '') + f"\nnet {P['total'] / 1e9:+.2f}bn → {regime}", color=MUT, fontsize=9, va='top')
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=110, facecolor=BG); plt.close(fig); return buf.getvalue()
if __name__ == '__main__':
    t = sys.argv[1] if len(sys.argv) > 1 else 'NQ=F'; ref = float(sys.argv[2]) if len(sys.argv) > 2 else None
    open(f'/tmp/gex_{t}.png', 'wb').write(render(t, ref)); print('ok')
