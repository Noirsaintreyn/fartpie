"""GEX HEATMAP + DAILY NET-GAMMA FILE (Nov 24 2026).  Read-only; never places an order.
Builds a strike x expiry heatmap of net dealer gamma exposure ($ per 1% move) from the option chain that webull_options_log.py saved, marks the call wall (CW), put wall (PW), zero-gamma level (VT) and spot, and writes live/neg_gamma.json ({ticker: net_gex}) that the live screener reads for the negative-gamma channel.
Convention (same as webull_greeks.py): calls +, puts -, gamma x open interest x 100 x spot^2 x 1%.  Open interest should be the next-morning refresh (webull_options_log.py --oi-refresh) for the most accurate picture.
  python3 gex_heatmap.py --symbols QQQ,SPY            heatmaps -> live/gex_<SYM>.png, writes live/neg_gamma.json
  python3 gex_heatmap.py --symbols QQQ --post         also posts the image to GEX_CHANNEL_ID (bot token from discord_bot/.env)
  python3 gex_heatmap.py --csv path/to/chain.csv      one chain file, symbol from --symbols
  python3 gex_heatmap.py --selftest                   synthetic chain -> /tmp/gex_selftest.png
Needs the options-quotes data (OPRA) on your Webull API app, plus webull_options_log.py having run first."""
import argparse, json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
try:
    from dotenv import load_dotenv; load_dotenv('.env'); load_dotenv('discord_bot/.env')
except Exception: pass
import webull_greeks as WG
ROOT = os.path.dirname(os.path.abspath(__file__)); LIVE = os.getenv('LIVE_DIR', f'{ROOT}/live')
BG, PANEL, T1, T2, MUT = '#0b0f17', '#0f1520', '#dfe7f1', '#b8c4d4', '#6b7a90'
POS, NEG = np.array([0x2a, 0xc0, 0x6e]) / 255, np.array([0xb4, 0x5c, 0xf0]) / 255

def fmt(v):
    a = abs(v); s = '+' if v >= 0 else '-'
    return f'{s}{a / 1e9:.1f}B' if a >= 1e9 else f'{s}{a / 1e6:.1f}M' if a >= 1e6 else f'{s}{a / 1e3:.1f}K' if a >= 1e3 else f'{s}{a:.0f}'

def matrix(df, now, n_exp=8, n_strikes=12):
    d = WG.chain(df); S = float(df['spot'].iloc[0]); g = WG.greeks(d, S, now); g['gex'] = g.sgn * g.gamma * g.oi * 100 * S * S * 0.01
    g = g[g.exp.dt.normalize() >= now.normalize()]; exps = sorted(g.exp.dt.normalize().unique())[:n_exp]; g = g[g.exp.dt.normalize().isin(exps)]
    M = g.pivot_table(index='K', columns=g.exp.dt.normalize(), values='gex', aggfunc='sum').reindex(columns=exps).fillna(0.0)
    near = M.index[(M.index >= S * .94) & (M.index <= S * 1.06)]; M = M.loc[near]
    k0 = int(np.argmin(np.abs(M.index.values - S))); lo, hi = max(0, k0 - n_strikes), min(len(M), k0 + n_strikes + 1)
    tot = g.groupby('K').gex.sum(); callg = g[g.call].groupby('K').gex.sum(); putg = g[~g.call].groupby('K').gex.sum()
    summ = WG.summarize('x', df, now)
    return M.iloc[lo:hi].sort_index(ascending=False), S, dict(cw=float(callg.idxmax()) if len(callg) else None, pw=float(putg.idxmin()) if len(putg) else None, vt=summ.get('gamma_flip'), net=float(g.gex.sum()))

def render(sym, M, S, lv, path, asof=''):
    nr, nc = M.shape; fig = plt.figure(figsize=(11.5, 0.31 * nr + 1.4), facecolor=BG); XL = -2.1; ax = fig.add_axes([0.012, 0.065, 0.976, 0.85]); ax.set_facecolor(BG); ax.set_xlim(XL, nc); ax.set_ylim(nr, 0); ax.axis('off')
    vmax = np.percentile(np.abs(M.values), 97) or 1; strikes = M.index.values
    for i, K in enumerate(strikes):
        for j in range(nc):
            v = M.iloc[i, j]; a = min(1.0, (abs(v) / vmax) ** 0.6) if v else 0; col = (POS if v >= 0 else NEG) * (0.18 + 0.82 * a) + np.array([0x12, 0x18, 0x22]) / 255 * (1 - (0.18 + 0.82 * a))
            hot = abs(v) >= vmax * 0.55; ax.add_patch(Rectangle((j + .03, i + .06), .94, .88, fc=col if a > .04 else '#0e141d', ec='none', zorder=2))
            ax.text(j + .5, i + .5, fmt(v), ha='center', va='center', color='#0b0f17' if (hot and v >= 0) else T1 if hot else T2 if a > .2 else MUT, fontsize=8.6 if hot else 8, fontweight='bold' if hot else 'normal', zorder=3, family='DejaVu Sans')
        ax.text(-.12, i + .5, f'{K:g}', ha='right', va='center', color=T2, fontsize=8.6)
    tags = {}
    for name, k, c in (('CW', lv['cw'], '#2ac06e'), ('PW', lv['pw'], '#f59e0b'), ('VT', lv['vt'], '#b45cf0')):
        if k is not None and len(strikes): tags.setdefault(int(np.argmin(np.abs(strikes - k))), []).append((name, c))
    srow = int(np.argmin(np.abs(strikes - S))); tags.setdefault(srow, []).append(('SPOT', '#ffffff'))
    for i, lst in tags.items():
        for t, (name, c) in enumerate(lst):
            x = -1.05 - .98 * t; ax.add_patch(FancyBboxPatch((x - .1, i + .1), .72 + (.12 if name == 'SPOT' else 0), .8, boxstyle='round,pad=0,rounding_size=.3', fc='none', ec=c, lw=1.2, zorder=4, clip_on=False)); ax.text(x + .26 + (.06 if name == 'SPOT' else 0), i + .5, name, ha='center', va='center', color=c, fontsize=7.4, fontweight='bold', zorder=5, clip_on=False)
    below = srow + 1 if strikes[srow] >= S else srow + 1; ax.plot([-.3, nc], [below, below], color='#3a4658', lw=.8, zorder=1)
    for j, c in enumerate(M.columns): fig.text(0.012 + 0.976 * (j + .5 - XL) / (nc - XL), 0.935, f'{c:%b %d}', ha='center', color=T1, fontsize=8.6, fontweight='bold')
    fig.text(0.02, 0.975, sym, color=T1, fontsize=15, fontweight='bold', va='center'); fig.text(0.02 + .017 * len(sym) + .02, 0.975, f'${S:,.2f}', color=T2, fontsize=10, va='center')
    fig.text(0.985, 0.975, f'NET GEX · {asof} · net {fmt(lv["net"])}/1%', color=MUT, fontsize=8, ha='right', va='center')
    fig.text(0.012, 0.025, 'CW call wall · PW put wall · VT zero gamma · net GEX $ per 1% move · green = calls (dealers long γ) · purple = puts (dealers short γ)', color=MUT, fontsize=7); fig.text(0.985, 0.025, 'not financial advice', color=MUT, fontsize=7, ha='right')
    fig.savefig(path, facecolor=BG, dpi=150); plt.close(fig)

def post_image(path, caption=''):
    import requests
    tok, ch = os.getenv('DISCORD_TOKEN'), os.getenv('GEX_CHANNEL_ID') or os.getenv('NEGGAMMA_CHANNEL_ID') or os.getenv('VECTOR_CHANNEL_ID')
    if not (tok and ch): return print('  (not posted: need DISCORD_TOKEN and GEX_CHANNEL_ID / NEGGAMMA_CHANNEL_ID)')
    with open(path, 'rb') as f: r = requests.post(f'https://discord.com/api/v10/channels/{ch}/messages', headers={'Authorization': f'Bot {tok}'}, data={'content': caption}, files={'files[0]': (os.path.basename(path), f, 'image/png')}, timeout=30)
    print('  posted' if r.ok else f'  post failed: HTTP {r.status_code} {r.text[:120]}')

def selftest():
    rng = np.random.default_rng(3); S = 675.0; now = pd.Timestamp('2026-11-24 10:00'); rows = []
    for e in pd.bdate_range('2026-11-24', periods=10)[:8]:
        for K in np.arange(655, 696, 1.0):
            for kind in ('call', 'put'):
                w = np.exp(-((K - S) / 9) ** 2) * (1 + 3 * (K == 680 and kind == 'put') + 2 * (K == 683 and kind == 'call')); rows.append(dict(strikePrice=K, optionType=kind, expirationDate=e, impliedVolatility=.2 + .002 * abs(K - S), openInterest=int(rng.integers(50, 3000) * w * 20), spot=S))
    df = pd.DataFrame(rows); M, S_, lv = matrix(df, now); render('QQQ', M, S_, lv, '/tmp/gex_selftest.png', 'selftest'); print('levels', lv, 'matrix', M.shape); print('saved /tmp/gex_selftest.png')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--symbols', default='QQQ,SPY'); ap.add_argument('--csv', default=None); ap.add_argument('--date', default=None); ap.add_argument('--post', action='store_true'); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    if a.selftest: return selftest()
    now = pd.Timestamp.now(tz='America/New_York').tz_localize(None); os.makedirs(LIVE, exist_ok=True); ng = {}
    try: ng = json.load(open(f'{LIVE}/neg_gamma.json'))
    except Exception: pass
    for sym in a.symbols.split(','):
        try:
            if a.csv: df, f = pd.read_csv(a.csv), a.csv
            else: df, f = WG.load(sym, a.date)
            M, S, lv = matrix(df, now); asof = os.path.basename(f)[:10]; p = f'{LIVE}/gex_{sym}.png'; render(sym, M, S, lv, p, asof); ng[sym] = lv['net']
            print(sym, f"net GEX {fmt(lv['net'])}/1%  CW {lv['cw']} PW {lv['pw']} VT {lv['vt']}  -> {p}")
            if a.post: post_image(p)
        except SystemExit as e: print(sym, 'skipped:', e)
        except Exception as e: print(sym, 'failed:', repr(e)[:200])
    json.dump(ng, open(f'{LIVE}/neg_gamma.json', 'w'), indent=1); print(f'wrote {LIVE}/neg_gamma.json')

if __name__ == '__main__': main()
