"""Nov 26 2026: two info charts for the daily sub-$5 alert (mock-up for Rey): overhead_chart (volume-at-price above the breakout) and breadth_chart (flips per day). Stats dicts come from daily_pairing_stats.json (xs/build_pairing_charts.py)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BG, PANEL, GRID, T1, T2, MUT = '#080c16', '#0f1626', '#1b2438', '#f1f5f9', '#94a3b8', '#56627a'
BLUE, RED, GOOD, GOLD, AMB = '#3b82f6', '#fb7185', '#34d399', '#fcd34d', '#f59e0b'
def _fig(w, h, title, sub, tag):
    fig = plt.figure(figsize=(w, h), facecolor=BG); fig.patches.append(plt.Rectangle((0, .985), 1, .015, transform=fig.transFigure, fc=GOOD, ec='none', zorder=-5))
    fig.text(.04, .935, tag, fontsize=9, color=GOOD, fontweight='bold'); fig.text(.04, .885, title, fontsize=19, color=T1, fontweight='bold'); fig.text(.04, .845, sub, fontsize=9.5, color=T2); return fig
def _panel(fig, x, y, w, h):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.006', transform=fig.transFigure, fc=PANEL, ec=GRID, lw=.8, zorder=-8))
def _style(ax):
    ax.set_facecolor(PANEL); [s.set_visible(False) for s in ax.spines.values()]; ax.tick_params(colors=MUT, labelsize=8.5, length=0); ax.grid(color=GRID, lw=.6, alpha=.8); ax.set_axisbelow(True)
def _tiers(fig, x, y, w, h, title, rows, hi_idx, foot, hi_col=GOOD):
    """rows = [(label, reach, n)]; highlight the row of today's reading."""
    _panel(fig, x, y, w, h); fig.text(x + .02, y + h - .02, title.upper(), fontsize=8.5, color=MUT, fontweight='bold', va='top'); ax = fig.add_axes([x + .15, y + .09, w - .19, h - .2]); _style(ax); ax.grid(False)
    base = float(np.mean([r[1] for r in rows])); yy = np.arange(len(rows))[::-1]
    for k, (lab, rch, n) in enumerate(rows):
        col = hi_col if k == hi_idx else '#334155'; ax.barh(yy[k], rch * 100, height=.58, color=col, zorder=3); ax.text(rch * 100 + 1.5, yy[k], f'{rch:.0%}  ({n})', va='center', color=T1 if k == hi_idx else T2, fontsize=10, fontweight='bold' if k == hi_idx else 'normal')
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], color=T1, fontsize=9.5); ax.set_xlim(0, 100); ax.set_xticks([]); ax.axvline(base * 100, color=GOLD, lw=1, ls=(0, (4, 3)), zorder=4)
    ax.text(base * 100, len(rows) - .45, 'average', color=GOLD, fontsize=8, ha='center', va='bottom'); fig.text(x + .02, y + .03, foot, fontsize=8, color=MUT)
def overhead_chart(tk, name, d_df, st, path):
    """d_df: daily OHLCV (RTH) up to the signal day. st = dict(tiers=[(label, reach, n)]*3, cut_lo, cut_hi, note)."""
    b = d_df.tail(253); c = float(b.close.iloc[-1]); tp = ((b.high + b.low + b.close) / 3).values; vo = b.volume.values.astype(float); tot = vo.sum(); ov = vo[(tp > c) & (tp <= 1.5 * c)].sum() / tot
    tier = 0 if ov <= st['cut_lo'] else (2 if ov >= st['cut_hi'] else 1)
    fig = _fig(9.6, 6.1, f'{tk}  ·  Overhead supply', f"{name[:46]}  ·  volume traded above the breakout price over the last year", 'BREAKOUT PAIRING')
    _panel(fig, .04, .09, .5, .72); ax = fig.add_axes([.075, .19, .43, .55]); _style(ax); ax.grid(axis='x', visible=False)
    lo, hi = float(b.low.min()), float(b.high.max()); bins = np.linspace(lo, hi, 36); h_, e_ = np.histogram(tp, bins=bins, weights=vo); m = (e_[:-1] + e_[1:]) / 2; w = e_[1] - e_[0]
    col = [BLUE if x <= c else (RED if x <= 1.5 * c else '#475569') for x in m]; ax.barh(m, h_ / h_.max() * 100, height=w * .86, color=col, zorder=3, alpha=.92)
    ax.axhline(c, color=T1, lw=1.6, zorder=5); ax.text(2, c, f'breakout {c:.2f}', color=T1, fontsize=9.5, fontweight='bold', va='bottom', ha='left', zorder=6, bbox=dict(fc=PANEL, ec='none', pad=1.5))
    ax.axhspan(c, min(1.5 * c, hi), color=RED, alpha=.07, zorder=1); ax.set_xticks([]); ax.set_xlim(0, 108); ax.set_ylim(lo, hi * 1.02); ax.yaxis.tick_right(); ax.yaxis.set_label_position('right'); ax.tick_params(axis='y', labelcolor=T2)
    fig.text(.055, .775, 'VOLUME AT PRICE, LAST 252 DAYS', fontsize=8.5, color=MUT, fontweight='bold', va='top')
    fig.text(.075, .115, '■ ', color=BLUE, fontsize=11); fig.text(.097, .118, 'held below (support)', color=T2, fontsize=8.5); fig.text(.27, .115, '■ ', color=RED, fontsize=11); fig.text(.292, .118, 'overhead supply (0 to +50%)', color=T2, fontsize=8.5)
    big = _panel(fig, .57, .56, .39, .245); fig.text(.59, .79, 'OVERHEAD SUPPLY NOW', fontsize=8.5, color=MUT, fontweight='bold', va='top'); fig.text(.59, .735, f'{ov:.0%}', fontsize=34, color=GOOD if tier == 0 else (RED if tier == 2 else T1), fontweight='bold', va='top')
    fig.text(.59, .625, ['CLEAR AIR above', 'MODERATE overhead', 'HEAVY overhead'][tier], fontsize=11, color=T1, fontweight='bold'); fig.text(.59, .595, 'of the last year\'s volume traded within +50% above', fontsize=8.5, color=T2)
    _tiers(fig, .57, .13, .39, .40, 'Flips like this reached +10% before', [(f'Clear air  (≤{st["cut_lo"]:.0%})', st['tiers'][0][1], st['tiers'][0][2]), ('Moderate', st['tiers'][1][1], st['tiers'][1][2]), (f'Heavy  (≥{st["cut_hi"]:.0%})', st['tiers'][2][1], st['tiers'][2][2])], tier, st['note'], hi_col=[GOOD, MUT, RED][tier])
    fig.text(.04, .035, 'Info only, no grade points: a lead in the broad sub-$5 population that fades inside the high-volatility regime (report 502).   ~Valiant', fontsize=7.8, color=MUT, style='italic'); fig.savefig(path, dpi=200, facecolor=BG); plt.close(fig); return path, ov
def breadth_chart(tk, name, counts, today_n, st, path):
    """counts: Series flips per day (recent window); today_n: flips on the signal day."""
    n = int(today_n); tier = 0 if n <= st['cut_lo'] else (2 if n >= st['cut_hi'] else 1)
    fig = _fig(9.6, 6.1, f'{tk}  ·  Breakout breadth', f"{name[:46]}  ·  how many breakouts fired the same day across the universe", 'BREAKOUT PAIRING')
    _panel(fig, .04, .09, .5, .72); ax = fig.add_axes([.075, .17, .43, .55]); _style(ax); ax.grid(axis='x', visible=False)
    x = np.arange(len(counts)); v = counts.values; colr = [GOOD if k == len(v) - 1 else ('#334155' if q < st['cut_hi'] else '#4b6b9a') for k, q in enumerate(v)]; ax.bar(x, v, color=colr, width=.78, zorder=3)
    ax.axhline(st['cut_hi'], color=GOLD, lw=1, ls=(0, (4, 3))); ax.text(0, st['cut_hi'], f' busy day ≥{st["cut_hi"]:.0f} flips', color=GOLD, fontsize=8.5, va='bottom'); ax.axhline(st['cut_lo'], color=RED, lw=1, ls=(0, (4, 3))); ax.text(0, st['cut_lo'], f' quiet day ≤{st["cut_lo"]:.0f}', color=RED, fontsize=8.5, va='bottom')
    ax.annotate(f'this close: {n} flips', (len(v) - 1, v[-1]), xytext=(-95, 60), textcoords='offset points', color=T1, fontsize=9.5, fontweight='bold', arrowprops=dict(arrowstyle='-', color=T2))
    idx = counts.index; ax.set_xticks([0, len(v) // 2, len(v) - 1]); ax.set_xticklabels([f'{idx[0]:%b %d}', f'{idx[len(v) // 2]:%b %d}', f'{idx[-1]:%b %d}'])
    fig.text(.055, .775, 'FLIPS PER DAY, 60 SESSIONS · SAMPLE WINDOW FROM HISTORY', fontsize=8.5, color=MUT, fontweight='bold', va='top')
    _panel(fig, .57, .56, .39, .245); fig.text(.59, .79, 'BREADTH TODAY', fontsize=8.5, color=MUT, fontweight='bold', va='top'); fig.text(.59, .735, f'{n}', fontsize=34, color=GOOD if tier == 2 else (RED if tier == 0 else T1), fontweight='bold', va='top')
    fig.text(.59, .625, ['QUIET tape', 'NORMAL tape', 'BUSY tape  (risk-on)'][tier], fontsize=11, color=T1, fontweight='bold'); fig.text(.59, .595, 'flips fired across the universe on this close', fontsize=8.5, color=T2)
    _tiers(fig, .57, .13, .39, .40, 'Flips like this reached +10% before', [(f'Busy  (≥{st["cut_hi"]:.0f})', st['tiers'][2][1], st['tiers'][2][2]), ('Normal', st['tiers'][1][1], st['tiers'][1][2]), (f'Quiet  (≤{st["cut_lo"]:.0f})', st['tiers'][0][1], st['tiers'][0][2])], 2 - tier, st['note'], hi_col=[GOOD, MUT, RED][2 - tier] if False else [RED, MUT, GOOD][tier])
    fig.text(.04, .035, 'Info only, no grade points: busy tapes lift reach, quiet-tape flips do worse; mostly overlaps volatility inside the live regime (report 502).   ~Valiant', fontsize=7.8, color=MUT, style='italic'); fig.savefig(path, dpi=200, facecolor=BG); plt.close(fig); return path
