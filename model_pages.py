"""Nov 26 2026 (Rey: 'send the flips for each model, a chart of whatever was paired with the flip, and the backtest for what happened last time it happened'): one page per signal model that tagged a large-cap flip.
model_page(tk, track, tag, path) -> path. Chart = the paired feature (NEG net gamma, PCV/PCZ put/call volume, SKW skew, CHM charm) over the last ~120 sessions with its trigger line and today's reading; right = the model's backtest on this stock (the last times it fired, dates and results) and on all names vs every other flip.
Data: live_data/model_flips.csv + live_data/model_stats.json (xs/build_model_stats.py), option_models.history() for the feature series. Returns are per trade before costs; a stock's own count is often small and is labelled that way."""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE = os.path.dirname(os.path.abspath(__file__))
BG, PANEL, GRID, T1, T2, MUT = '#080c16', '#0f1626', '#1b2438', '#f1f5f9', '#94a3b8', '#56627a'
BLUE, RED, GOOD, GOLD, AMB = '#60a5fa', '#fb7185', '#34d399', '#fcd34d', '#f59e0b'
INFO = {'NEG': ('NEGATIVE GAMMA', 'net dealer gamma at the prior close', 'Dealers short gamma amplify moves: the flip has room to run.'),
        'PCV': ('PUT/CALL VOLUME', 'put/call volume ratio (expiries <= 45 days)', 'Top third of its own year: crowded hedging, a contrarian read.'),
        'PCZ': ('PUT/CALL VOLUME  z > 1', 'put/call volume ratio', 'More than 1 standard deviation above its own 1-year average.'),
        'SKW': ('SKEW', 'OTM put IV minus OTM call IV (14-45 days)', 'Top third of its own year: fear is priced into the puts.'),
        'CHM': ('CHARM', 'net dealer charm', 'Top third of its own year: dealer delta decay favours buying into the flip.')}
COL = {'NEG': 'gex', 'PCV': 'pc_vol', 'PCZ': 'pc_vol', 'SKW': 'skew', 'CHM': 'charm'}
_C = {}
def _load():
    if not _C:
        try: _C['f'] = pd.read_csv(os.path.join(HERE, 'live_data', 'model_flips.csv'), parse_dates=['entry'])
        except Exception: _C['f'] = pd.DataFrame(columns=['tk', 'entry', 'ret', 'tf', 'tag'])
        try: _C['s'] = json.load(open(os.path.join(HERE, 'live_data', 'model_stats.json')))
        except Exception: _C['s'] = {}
    return _C['f'], _C['s']
def _panel(fig, x, y, w, h, title=None):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.006', transform=fig.transFigure, fc=PANEL, ec=GRID, lw=.8, zorder=-8))
    if title: fig.text(x + .016, y + h - .012, title.upper(), fontsize=8, color=MUT, fontweight='bold', va='top')
def model_page(tk, track, tag, path, asof=None):
    import option_models as OM
    fl, ms = _load(); ttl, what, why = INFO[tag]; col = COL[tag]
    h, _ = OM.history(tk); h = h.copy(); h[col] = pd.to_numeric(h[col], errors='coerce'); h = h.dropna(subset=[col]).tail(380).reset_index(drop=True)
    if len(h) < 40: return None
    v = h.tail(120); r_ = h[col].rolling(252, min_periods=60)
    thr = None if tag == 'NEG' else ((r_.mean() + r_.std()).shift(1) if tag == 'PCZ' else r_.quantile(2 / 3).shift(1)); th = None if thr is None else thr.tail(120)
    fig = plt.figure(figsize=(9.6, 7.6), facecolor=BG); fig.patches.append(plt.Rectangle((0, .987), 1, .013, transform=fig.transFigure, fc=GOOD, ec='none', zorder=-5))
    fig.text(.04, .965, f'SIGNAL MODEL  ·  {track}', fontsize=9, color=GOOD, fontweight='bold', va='top'); fig.text(.04, .94, f'{tk}', fontsize=26, color=T1, fontweight='bold', va='top'); fig.text(.22, .925, ttl, fontsize=15, color=GOLD, fontweight='bold', va='top')
    fig.text(.04, .875, why, fontsize=9.5, color=T2, va='top')
    _panel(fig, .04, .43, .92, .42, f'{what} · last 120 sessions' + (' · $ billions' if tag == 'NEG' else '')); ax = fig.add_axes([.075, .47, .85, .31]); ax.set_facecolor(PANEL); [s.set_visible(False) for s in ax.spines.values()]; ax.tick_params(colors=MUT, labelsize=8, length=0); ax.grid(color=GRID, lw=.6)
    y = pd.to_numeric(v[col], errors='coerce').values.astype(float); x = pd.to_datetime(v.date)
    if tag == 'NEG': y = y / 1e9
    if tag == 'NEG':
        ax.fill_between(x, 0, y, where=y < 0, color=RED, alpha=.35, step=None); ax.fill_between(x, 0, y, where=y >= 0, color=BLUE, alpha=.25); ax.plot(x, y, color=T1, lw=1.2); ax.axhline(0, color=GOLD, lw=1, ls=(0, (4, 3))); ax.text(x.iloc[0], 0, ' zero gamma', color=GOLD, fontsize=8, va='bottom')
    else:
        ax.plot(x, y, color=BLUE, lw=1.6); ax.plot(x, th, color=T1, lw=1, ls=(0, (5, 2))); hi = y >= th.values; ax.scatter(x[hi], y[hi], s=14, color=AMB, zorder=4)
    ax.scatter([x.iloc[-1]], [y[-1]], s=70, color=AMB, ec='white', zorder=5)
    fig.text(.93, .835, f'prior close: {y[-1]:.3g}' + (' B' if tag == 'NEG' else ''), fontsize=10, color=AMB, fontweight='bold', ha='right', va='top')
    ax.text(.995, .03, 'orange = trigger zone' if tag != 'NEG' else 'red = dealers short gamma', transform=ax.transAxes, ha='right', fontsize=8, color=MUT)
    me = fl[(fl.tk == tk) & (fl.tag == tag) & (fl.tf == track)].sort_values('entry'); allk = ms.get(f'{tag}|{track}|ALL'); base = ms.get(f'BASE|{track}|ALL'); mine = ms.get(f'{tag}|{track}|{tk}')
    _panel(fig, .04, .04, .45, .36, f'Last times it fired on {tk}')
    if len(me):
        last = me.iloc[-1]; fig.text(.056, .335, f'{last.entry:%b %d, %Y}', fontsize=16, color=T1, fontweight='bold', va='top'); fig.text(.056, .295, f'{last.ret * 1e4:+.0f} bp  ' + ('win' if last.ret > 0 else 'loss'), fontsize=11, color=GOOD if last.ret > 0 else RED, fontweight='bold', va='top')
        k = int((me.ret > 0).sum()); fig.text(.056, .255, f'{len(me)} flips tagged: {k} wins ({k / len(me):.0%}), mean {me.ret.mean() * 1e4:+.0f} bp' + ('   few cases: anecdote' if len(me) < 8 else ''), fontsize=8.6, color=T2, va='top')
        yy = .22
        for r in me.tail(7).iloc[::-1].itertuples(): fig.text(.056, yy, f'{r.entry:%b %d %Y}', fontsize=8.4, color=T2, va='top'); fig.text(.19, yy, f'{r.ret * 1e4:+.0f} bp', fontsize=8.4, color=GOOD if r.ret > 0 else RED, fontweight='bold', va='top'); yy -= .024
    else: fig.text(.056, .33, 'No earlier flips on this stock carried this tag.', fontsize=10, color=T2, va='top')
    _panel(fig, .51, .04, .45, .36, 'All names · same model'); 
    if allk and base:
        fig.text(.526, .335, f"{allk['bp']:+.0f} bp", fontsize=26, color=GOOD if allk['bp'] > base['bp'] else T1, fontweight='bold', va='top'); fig.text(.526, .275, f"per trade, {allk['n']:,} flips · {allk['win']}% win" + (f" · t {allk['t']}" if allk.get('t') else ''), fontsize=9, color=T2, va='top')
        fig.text(.526, .235, f"every other flip: {base['bp']:+.0f} bp · {base['win']}% win\nlift over them: {allk['bp'] - base['bp']:+.0f} bp", fontsize=9.2, color=T1, va='top', linespacing=1.6)
        if mine: fig.text(.526, .15, f"{tk} alone: {mine['bp']:+.0f} bp · {mine['win']}% win ({mine['n']} flips)", fontsize=9.2, color=GOLD, va='top')
    fig.text(.526, .085, '23 large-cap names, 2019-2026, per trade before costs.\nContext for the signal, not advice.', fontsize=7.6, color=MUT, va='top', linespacing=1.5)
    fig.text(.96, .012, '~Valiant', fontsize=13, color=T2, style='italic', family='DejaVu Serif', ha='right'); fig.savefig(path, dpi=130, facecolor=BG); plt.close(fig); return path
