"""/gex command (Nov 24 2026): dealer greeks for one ticker from the latest Webull option chain the worker logged (live/options/<SYM>/).
Shows net GEX, gamma flip, call/put walls, DEX, vanna, charm, vega, ATM IV and 25-delta skew, plus the strike x expiry GEX heatmap.
Descriptive only: of these, only the daily net-gamma SIGN has been backtested (reports 441-455); vanna/charm/DEX/skew are context, not signals."""
import io, os, sys, tempfile
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

def fmt(v):
    if v is None: return 'n/a'
    a = abs(v); s = '+' if v >= 0 else '-'
    return f'{s}{a / 1e9:.2f}B' if a >= 1e9 else f'{s}{a / 1e6:.1f}M' if a >= 1e6 else f'{s}{a / 1e3:.1f}K' if a >= 1e3 else f'{s}{a:.0f}'

def available():
    import webull_greeks as WG
    try: return sorted(d for d in os.listdir(WG.OPT) if os.path.isdir(f'{WG.OPT}/{d}'))
    except Exception: return []

def build(sym):
    """-> (summary dict, heatmap PNG bytes, asof 'YYYY-MM-DD', file name). Raises SystemExit if no chain was logged for sym."""
    import webull_greeks as WG, gex_heatmap as GH
    sym = sym.strip().upper(); df, f = WG.load(sym); asof = os.path.basename(f)[:10]
    now = pd.Timestamp.now(tz='America/New_York').tz_localize(None)
    s = WG.summarize(sym, df, now); M, S, lv = GH.matrix(df, now)
    p = os.path.join(tempfile.gettempdir(), f'gex_{sym}.png'); GH.render(sym, M, S, lv, p, asof)
    return s, open(p, 'rb').read(), asof, os.path.basename(f)

def text(sym, s, asof):
    sp = s['spot']; flip = s.get('gamma_flip'); cw = s.get('call_wall'); pw = s.get('put_wall'); sk = s.get('skew')
    regime = 'negative gamma (dealers amplify moves)' if s['gex'] < 0 else 'positive gamma (dealers dampen moves)'
    rel = '' if not flip else (' · spot is ' + ('below' if sp < flip else 'above') + f' the flip by {abs(sp / flip - 1) * 100:.1f}%')
    num = lambda v: f'{v:,.2f}' if v else 'n/a'
    L = [f"{'Spot':<12}{sp:>12,.2f}", f"{'Net GEX':<12}{fmt(s['gex']):>12}  $ per 1% move", f"{'Gamma flip':<12}{num(flip):>12}", f"{'Call wall':<12}{num(cw):>12}", f"{'Put wall':<12}{num(pw):>12}",
         f"{'DEX':<12}{fmt(s['dex']):>12}  net delta $", f"{'Vanna':<12}{fmt(s['vanna']):>12}  per 1 vol pt", f"{'Charm':<12}{fmt(s['charm']):>12}  per day", f"{'Vega':<12}{fmt(s['vega']):>12}  per 1 vol pt",
         f"{'ATM IV':<12}{s['iv'] * 100:>11.1f}%  {s['expiry']}", f"{'25d skew':<12}{('n/a' if sk is None else f'{sk * 100:+.1f} pts'):>12}  put IV - call IV"]
    return f"**{sym}** · chain {asof}\n```\n" + '\n'.join(L) + f"\n```\n{regime}{rel}\n_Context only: just the daily net-gamma sign has been backtested; vanna, charm, DEX and skew are not signals._"
