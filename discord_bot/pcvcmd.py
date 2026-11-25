"""/pcv command (Nov 24 2026): put/call volume for a ticker - last 120 days, the top-third line and the z>1 line, today's rank and z, whether the P/C VOL / P/C VOL z>1 signal is on, and its backtest.
Uses the same prior-close history as the signal-model alerts (option_models.history), so the numbers match what an alert would show. Names: the option-model universe (option_models.NAMES)."""
import io, json, os, sys, tempfile
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
BG = '#05080f'

def build(sym):
    import option_models as OM
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    sym = sym.strip().upper()
    if sym not in OM.NAMES: raise KeyError(sym)
    h, _ = OM.history(sym); h = h.dropna(subset=['pc_vol']).reset_index(drop=True)
    if len(h) < 60: raise KeyError(sym)
    rr = h.pc_vol.rolling(252, min_periods=60); q = rr.quantile(2 / 3).shift(1); z1 = (rr.mean() + rr.std()).shift(1); v = h.tail(120); r = OM.ranks(sym) or {}
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(v.date, v.pc_vol, color='#60a5fa', lw=1.6); ax.plot(v.date, q.tail(120), color='#e5e7eb', lw=1, ls=(0, (5, 2))); ax.plot(v.date, z1.tail(120), color='#f59e0b', lw=1, ls=(0, (2, 2)))
    hi = v.pc_vol >= q.tail(120); ax.scatter(v.date[hi], v.pc_vol[hi], s=14, color='#f59e0b', zorder=4); ax.scatter([v.date.iloc[-1]], [v.pc_vol.iloc[-1]], s=60, color='#f59e0b', ec='white', zorder=5)
    ax.set_title(f'{sym}  put/call volume (<=45 DTE) · white dashed = top third · orange dotted = z>1 · orange dots = above top third', color='#b8c4d4', fontsize=10, loc='left')
    ax.tick_params(colors='#7d8ca0', labelsize=8)
    for sp in ax.spines.values(): sp.set_color('#374151')
    ax.grid(color='#1f2937', lw=.6); fig.tight_layout(); p = os.path.join(tempfile.gettempdir(), f'pcv_{sym}.png'); fig.savefig(p, dpi=110, facecolor=BG); plt.close(fig)
    try: st = json.load(open(os.path.join(ROOT, 'live_data', 'model_stats.json')))
    except Exception: st = {}
    return dict(sym=sym, date=str(pd.Timestamp(v.date.iloc[-1]).date()), pc=float(v.pc_vol.iloc[-1]), rank=r.get('pc_vol'), z=r.get('pc_z'), st=st), open(p, 'rb').read()

def text(d):
    sym, rk, z, st = d['sym'], d['rank'], d['z'], d['st']; on_z = z is not None and z > 1; on_r = rk is not None and rk >= 2 / 3
    def bt(k):
        a, b = st.get(f'{k}|15m|ALL'), st.get(f'{k}|15m|{sym}')
        return (f"all names n {a['n']} · {a['bp']:+.0f}bp · {a['win']}% win" if a else '') + (f" | {sym} n {b['n']} · {b['bp']:+.0f}bp · {b['win']}% win" if b else '')
    L = [f"**{sym}** put/call volume · as of {d['date']} close", '```', f"{'P/C volume':<12}{d['pc']:>8.2f}", f"{'1y rank':<12}{('n/a' if rk is None else f'{rk * 100:.0f} percentile'):>8}   (signal: 67+)", f"{'z vs 1y':<12}{('n/a' if z is None else f'{z:+.2f}'):>8}   (strong signal: > +1)", '```']
    L.append(f"Signal: **{'P/C VOL z>1 ON' if on_z else 'P/C VOL ON' if on_r else 'off'}**" + ('' if (on_z or on_r) else ' (flips on this name are not tagged right now)'))
    L.append('P/C VOL z>1: ' + bt('PCZ')); L.append('P/C VOL (top third): ' + bt('PCV'))
    L.append('_15m long flips only. High put/call volume = heavy hedging; the backtest says flips taken into it did better. Prior-close data._')
    return '\n'.join(L)

def available():
    import option_models as OM
    return sorted(OM.NAMES)
