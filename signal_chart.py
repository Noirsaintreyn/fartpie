"""Nov 12 2026: shared signal chart for the Discord screeners (mid/small now, large-cap optional).
One OHLCV frame in (1h bars, DatetimeIndex, ET-naive), one PNG out: vector-coloured candles, the vector level as a glowing step line with a cushion gradient,
1h EMA50 / EMA200, quarterly anchored VWAP, flip marker, volume panel with the flip bar highlighted, a 1h / 4h / daily regime ribbon, and a scorecard of the entry gates.
Colour key (printed on the chart): BLUE candles = vector bull, ROSE = vector bear, LIME step = vector level (the floor), AMBER = EMA50, VIOLET = EMA200, WHITE dots = quarterly VWAP."""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import patheffects as pe
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xs'))
from vector_pine import run_state_machine

BG, PANEL, GRID, T1, T2, MUT = '#080c16', '#0f1626', '#1b2438', '#f1f5f9', '#94a3b8', '#56627a'
BULL, BULLE, BEAR, BEARE = '#3b82f6', '#93c5fd', '#fb7185', '#9f1239'
LVL, EMA50, EMA200, VWAP, GOOD, CRIT, GOLD = '#a3e635', '#fbbf24', '#c084fc', '#e2e8f0', '#34d399', '#fb7185', '#fcd34d'
GRADE_COL = {'A': GOOD, 'B': GOLD, 'C': CRIT}

def _htf_regime(Y, tf):
    """vector regime of the last COMPLETED 4h / daily bar, mapped back onto every 1h bar (no look-ahead)"""
    day = Y.index.normalize(); G = Y.groupby([day, (Y.index.hour >= 13).astype(int)] if tf == '4h' else [day])
    o = G.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), volume=('volume', 'sum'))
    first = G.apply(lambda x: x.index[0]); last = G.apply(lambda x: x.index[-1]); o.index = pd.DatetimeIndex(first.values)
    end = pd.DatetimeIndex(last.values) + pd.to_timedelta(np.where(pd.DatetimeIndex(last.values).hour == 15, 30, 60), unit='m')
    reg = pd.Series(run_state_machine(o.assign(datetime=o.index))[0], index=end).sort_index()
    bar_end = Y.index + pd.to_timedelta(np.where(Y.index.hour == 15, 30, 60), unit='m')
    pos = np.searchsorted(reg.index.values, bar_end.values, side='right') - 1                        # last HTF bar that has closed by this 1h bar's close
    return pd.Series(np.where(pos >= 0, reg.values[np.clip(pos, 0, None)], 0), index=Y.index)

def _glow(ax, x, y, color, lw, step=False):
    for w, a in ((lw * 4.2, .06), (lw * 2.4, .13), (lw, 1)):
        (ax.step(x, y, where='mid', color=color, lw=w, alpha=a, solid_capstyle='round', zorder=5) if step else ax.plot(x, y, color=color, lw=w, alpha=a, solid_capstyle='round', zorder=5))


def _kslope(c):
    """causal local-linear-trend Kalman slope on log close (same fixed noise ratios as midsmall_screener.kalman_up)"""
    l = np.log(c); r = np.diff(l); R = max(float(np.var(r[:500])) if len(r) > 3 else 1e-4, 1e-8); q1, q2 = .05 * R, .001 * R; x0, x1 = l[0], 0.; P00, P01, P11 = R, 0., R * .01
    for i in range(1, len(l)):
        x0p = x0 + x1; P00p = P00 + 2 * P01 + P11 + q1; P01p = P01 + P11; P11p = P11 + q2; S = P00p + R; K0, K1 = P00p / S, P01p / S; y = l[i] - x0p
        x0 = x0p + K0 * y; x1 = x1 + K1 * y; P00 = (1 - K0) * P00p; P01 = (1 - K0) * P01p; P11 = P11p - K1 * P01p
    return x1

def score_gates(Y, i=-1):
    """the 5-gate entry scorecard for the bar at position i (default: the last closed bar). xs/report_12_entrytool.py: G1 flip-bar volume >= 1.5x prior-20 mean | G2 4h AND daily vector bull (last completed) |
    G3 close above the 1h and 4h EMA200 | G4 Kalman slope > 0 | G5 not a 15:30 (last-session-bar) flip.  Grade A >= 4 points, B = 3, C <= 2.  Returns dict(gates, pts, grade, vr)."""
    Y = Y[['open', 'high', 'low', 'close', 'volume']].astype(float).dropna(); i = len(Y) + i if i < 0 else i; c = Y.close; v = Y.volume.values
    vm = float(v[max(0, i - 20):i].mean()) if i >= 1 else np.nan; vr = float(v[i] / vm) if vm and vm == vm and vm > 0 else np.nan
    r4, rd = _htf_regime(Y.iloc[:i + 1], '4h'), _htf_regime(Y.iloc[:i + 1], 'd'); b4, bd = bool(r4.iat[-1] == 1), bool(rd.iat[-1] == 1)
    day = Y.index[:i + 1].normalize(); G = Y.iloc[:i + 1].groupby([day, (Y.index[:i + 1].hour >= 13).astype(int)]); cl4 = G.close.last(); end4 = [x.index[-1] + pd.Timedelta(minutes=30 if x.index[-1].hour == 15 else 60) for _, x in G]
    e4 = pd.Series(cl4.ewm(span=200, adjust=False).mean().values, index=pd.DatetimeIndex(end4)); bar_end = Y.index[i] + pd.Timedelta(minutes=30 if Y.index[i].hour == 15 else 60); e4v = float(e4[e4.index <= bar_end].iat[-1]) if (e4.index <= bar_end).any() else np.nan
    e1 = float(c.ewm(span=200, adjust=False).mean().iat[i]); a1, a4 = bool(c.iat[i] > e1), bool(c.iat[i] > e4v); ku = bool(_kslope(c.values[:i + 1]) > 0); lastbar = bool(Y.index[i].hour == 15)
    gates = [('VOLUME \u22651.5\u00d7', bool(vr == vr and vr >= 1.5), (f'{vr:.1f}\u00d7 the 20-bar avg' if vr == vr else 'n/a')), ('4H + DAILY BULL', b4 and bd, f"4h {'bull' if b4 else 'bear'} \u00b7 daily {'bull' if bd else 'bear'}"),
             ('ABOVE EMA200 1h+4h', a1 and a4, f"1h {'yes' if a1 else 'no'} \u00b7 4h {'yes' if a4 else 'no'}"), ('KALMAN UP', ku, 'smoothed slope ' + ('> 0' if ku else '< 0')), ('NOT LAST BAR', not lastbar, 'flip on the 15:30 bar: gap risk' if lastbar else f'flip on the {Y.index[i]:%H:%M} bar')]
    pts = sum(1 for g in gates if g[1]); return dict(gates=gates, pts=pts, grade='A' if pts >= 4 else ('B' if pts == 3 else 'C'), vr=vr, a1=a1, a4=a4, b4=b4, bd=bd, kal=ku, e1=e1, e4=e4v)

def make_chart(tk, Y, path, kind='BUY', info=None, bars=110):
    """info (all optional): vr (flip-bar volume ratio), gates [(label, ok, detail)], pts, grade ('A'/'B'/'C'), entry (px), tag (e.g. 'mid/small screener'), note (one line under the title)"""
    info = dict(info or {}); Y = Y[['open', 'high', 'low', 'close', 'volume']].astype(float).dropna()
    if kind == 'BUY' and not info.get('gates'):
        try: sg = score_gates(Y); info.update({k: info.get(k, sg[k]) for k in ('gates', 'pts', 'grade', 'vr')})
        except Exception as e: print('  ! score_gates:', str(e)[:80])
    reg, lev, _, flips = run_state_machine(Y.assign(datetime=Y.index)); c = Y.close
    e50, e200 = c.ewm(span=50, adjust=False).mean(), c.ewm(span=200, adjust=False).mean(); q = Y.index.to_period('Q'); tp = (Y.high + Y.low + Y.close) / 3
    qv = (tp * Y.volume).groupby(q).cumsum() / (Y.volume.groupby(q).cumsum() + 1e-9); r4, rd = _htf_regime(Y, '4h'), _htf_regime(Y, 'd')
    n = min(bars, len(Y)); s = slice(len(Y) - n, len(Y)); X = Y.iloc[s]; t = np.arange(n); o, h, l, cl, v = X.open.values, X.high.values, X.low.values, X.close.values, X.volume.values
    rg, lv = reg[s], pd.Series(lev, index=Y.index).iloc[s].values; bull = rg == 1
    fl = [i - (len(Y) - n) for i, st in flips if st == (1 if kind == 'BUY' else -1) and i >= len(Y) - n]; fi = fl[-1] if fl else n - 1
    fig = plt.figure(figsize=(12.4, 7.2), facecolor=BG, dpi=150); gs = fig.add_gridspec(3, 2, width_ratios=[4.3, 1.25], height_ratios=[6.2, 1.5, .62], hspace=.04, wspace=.045, left=.035, right=.985, top=.885, bottom=.055)
    ax, av, ar = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0], sharex=None), fig.add_subplot(gs[2, 0]); pn = fig.add_subplot(gs[:, 1])
    for a_ in (ax, av, ar, pn): a_.set_facecolor(PANEL); [sp.set_visible(False) for sp in a_.spines.values()]
    fig.patches.append(FancyBboxPatch((.006, .01), .988, .98, boxstyle='round,pad=0,rounding_size=.012', transform=fig.transFigure, fc='none', ec=GRID, lw=1.2, zorder=-5))
    # ---- candles
    col = np.where(bull, BULL, np.where(rg == -1, BEAR, MUT)); ecol = np.where(bull, BULLE, np.where(rg == -1, '#fda4af', T2))
    ax.vlines(t, l, h, color=ecol, lw=1, zorder=3, alpha=.9)
    bt, bb = np.maximum(o, cl), np.minimum(o, cl); ax.bar(t, np.maximum(bt - bb, (h.max() - l.min()) * .0025), bottom=bb, width=.68, color=col, edgecolor=ecol, linewidth=.6, zorder=4, alpha=.95)
    # ---- level (floor) with glow + cushion gradient to price while bull
    lvl = np.where(bull, lv, np.nan); _glow(ax, t, lvl, LVL, 1.5, step=True)
    for k, a in enumerate(np.linspace(.17, .015, 7)):
        lo_ = lv + (cl - lv) * k / 7; hi_ = lv + (cl - lv) * (k + 1) / 7; ax.fill_between(t, lo_, hi_, where=bull & (cl > lv), color=BULL, alpha=a, lw=0, step='mid', zorder=1)
    # ---- MAs
    ax.plot(t, e50.iloc[s].values, color=EMA50, lw=1.25, alpha=.95, zorder=5); ax.plot(t, e200.iloc[s].values, color=EMA200, lw=1.6, alpha=.95, zorder=5)
    ax.plot(t, qv.iloc[s].values, color=VWAP, lw=1.1, ls=(0, (1, 2.2)), alpha=.85, zorder=5)
    # ---- flip marker
    ymin, ymax = np.nanmin([l.min(), np.nanmin(lvl) if np.isfinite(lvl).any() else l.min()]), h.max(); pad = (ymax - ymin) * .08; ax.set_ylim(ymin - pad, ymax + pad * 1.9); ax.set_xlim(-1, n + 11)
    yl = ax.get_ylim(); ax.axvline(fi, color=GOOD if kind == 'BUY' else CRIT, lw=.9, ls=(0, (3, 3)), alpha=.55, zorder=2)
    mk = '^' if kind == 'BUY' else 'v'; my = l[fi] - (yl[1] - yl[0]) * .035 if kind == 'BUY' else h[fi] + (yl[1] - yl[0]) * .035; mc = GOOD if kind == 'BUY' else CRIT
    for s_, a in ((520, .09), (240, .2)): ax.scatter([fi], [my], s=s_, color=mc, alpha=a, lw=0, zorder=6)
    ax.scatter([fi], [my], s=95, marker=mk, color=mc, edgecolor=PANEL, linewidth=1.5, zorder=7)
    vr = info.get('vr'); lab = ('FLIP' if kind == 'BUY' else 'EXIT') + (f"  {vr:.1f}× vol" if vr and vr == vr else '')
    ax.annotate(lab, (fi, my), xytext=(-46, -34 if kind == 'BUY' else 34), textcoords='offset points', ha='center', va='center', color=T1, fontsize=9, fontweight='bold', zorder=8,
                bbox=dict(boxstyle='round,pad=.35', fc=BG, ec=mc, lw=1), arrowprops=dict(arrowstyle='-', color=mc, lw=.8))
    # ---- last price tag + entry / level tags on the right
    last = cl[-1]; ax.axhline(last, color=T2, lw=.7, ls=(0, (4, 3)), alpha=.7, zorder=2)
    def tag(y, txt, fc, tc=BG): ax.text(n + .4, y, txt, color=tc, fontsize=8.6, fontweight='bold', va='center', ha='left', bbox=dict(boxstyle='round,pad=.25', fc=fc, ec='none'), zorder=9)
    tag(last, f'{last:.2f}', T1)
    if np.isfinite(lv[-1]) and bull[-1]: tag(lv[-1], f'{lv[-1]:.2f}', LVL)
    if info.get('entry'): ax.hlines(info['entry'], fi, n - 1, color=GOLD, lw=1, ls=(0, (2, 2)), zorder=4); ax.text(fi - .6, info['entry'], f"entry {info['entry']:.2f} ", color=GOLD, fontsize=8, va='bottom', ha='right', zorder=8)
    days = pd.Series(X.index.normalize()); ticks = [i for i in range(1, n) if days[i] != days[i - 1]]
    for i in ticks: ax.axvline(i - .5, color=GRID, lw=.7, zorder=0); av.axvline(i - .5, color=GRID, lw=.7, zorder=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.tick_params(length=0)
    # ---- legend (what the colours mean)
    items = [(BULL, 'vector bull'), (BEAR, 'vector bear'), (LVL, 'vector level'), (EMA50, 'EMA50'), (EMA200, 'EMA200'), (VWAP, 'qtr VWAP')]; x0 = .012
    for cc, tx in items:
        ax.text(x0, .965, '●', transform=ax.transAxes, color=cc, fontsize=9, va='center'); ax.text(x0 + .014, .965, tx, transform=ax.transAxes, color=T2, fontsize=7.6, va='center'); x0 += .014 + .0085 * len(tx) + .022
    # ---- volume panel
    avg = Y.volume.rolling(20).mean().shift().iloc[s].values; vc = np.where(bull, BULL, np.where(rg == -1, BEAR, MUT)); ec = np.where(np.arange(n) == fi, T1, 'none')
    av.bar(t, v, width=.72, color=vc, alpha=.8, edgecolor=ec, linewidth=1.3, zorder=3); av.plot(t, avg, color=GOLD, lw=1, alpha=.9, zorder=4); av.plot(t, avg * 1.5, color=GOLD, lw=.7, ls=(0, (3, 3)), alpha=.6, zorder=4)
    av.text(.005, .86, 'VOLUME  ·  gold = 20-bar avg, dashed = 1.5×', transform=av.transAxes, color=MUT, fontsize=7.2, va='center'); av.set_xlim(-1, n + 11); av.set_xticks([]); av.set_yticks([]); av.set_ylim(0, max(v.max(), np.nanmax(avg * 1.5)) * 1.12)
    # ---- regime ribbon 1h / 4h / D
    ar.set_xlim(-1, n + 11); ar.set_ylim(0, 3); ar.set_xticks([]); ar.set_yticks([])
    for row, (nm, rr) in enumerate((('D', rd), ('4h', r4), ('1h', pd.Series(reg, index=Y.index)))):
        vals = rr.iloc[s].values
        for i in range(n): ar.add_patch(Rectangle((i - .5, row + .08), 1, .84, fc=BULL if vals[i] == 1 else (BEAR if vals[i] == -1 else GRID), ec='none', alpha=.9 if vals[i] else .5))
        ar.text(n + .6, row + .5, nm, color=T2, fontsize=8, fontweight='bold', va='center')
    for i in ticks:
        ar.text(i - .5, -.25, f'{X.index[i]:%a %d}', color=MUT, fontsize=7, ha='left', va='top', clip_on=False)
    # ---- header
    px_ = cl[-1]; chg = px_ / o[0] - 1; g = info.get('grade'); pts = info.get('pts')
    fig.text(.045, .945, tk, color=T1, fontsize=25, fontweight='bold', va='center', path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
    fig.text(.045 + .021 * len(tk) + .03, .945, f"{px_:.2f}", color=T1, fontsize=17, va='center')
    fig.text(.045, .902, f"{'BUY' if kind == 'BUY' else 'SELL'} · 1h vector {'bull flip' if kind == 'BUY' else 'bear flip'} · {Y.index[-1]:%a %b %d %H:%M} bar" + (f"  ·  {info['tag']}" if info.get('tag') else ''), color=T2, fontsize=10, va='center')
    if g: fig.text(.70, .935, f"GRADE {g}", color=GRADE_COL.get(g, T1), fontsize=22, fontweight='bold', ha='right', va='center', path_effects=[pe.withStroke(linewidth=4, foreground=BG)]); fig.text(.70, .897, f"{pts}/5 gates" if pts is not None else '', color=T2, fontsize=10, ha='right', va='center')
    # ---- scorecard
    pn.set_xlim(0, 1); pn.set_ylim(0, 1); pn.set_xticks([]); pn.set_yticks([]); pn.text(.07, .965, 'ENTRY SCORECARD', color=MUT, fontsize=8.5, fontweight='bold', va='center')
    if not info.get('stats'):
        lv_ = lv[-1] if np.isfinite(lv[-1]) else np.nan; ent_ = info.get('entry') or cl[-1]
        info['stats'] = [('Entry', f"next open ~{ent_:.2f}"), ('Vector level', f'{lv_:.2f}'), ('Risk to level', f'{lv_ / ent_ - 1:+.1%}' if lv_ == lv_ else 'n/a'), ('Exit', '1h vector bear flip')] + list(info.get('stats_extra', []))
    y = .915
    for lab_, ok, det in info.get('gates', []):
        cc = GOOD if ok else CRIT; pn.add_patch(FancyBboxPatch((.05, y - .068), .9, .076, boxstyle='round,pad=0,rounding_size=.02', fc=BG, ec=cc if ok else GRID, lw=1, alpha=1))
        pn.text(.095, y - .03, '✔' if ok else '✖', color=cc, fontsize=12, fontweight='bold', va='center'); pn.text(.2, y - .016, lab_, color=T1, fontsize=8.6, fontweight='bold', va='center'); pn.text(.2, y - .051, det, color=T2, fontsize=7.5, va='center'); y -= .095
    y -= .02
    for k_, v_ in info.get('stats', []): pn.text(.07, y, k_, color=MUT, fontsize=8, va='center'); pn.text(.93, y, v_, color=T1, fontsize=8.6, fontweight='bold', va='center', ha='right'); y -= .052
    if info.get('note'):
        import textwrap; yy = max(y - .015, .13)
        for ln in textwrap.wrap(info['note'], 34)[:5]: pn.text(.07, yy, ln, color=T2, fontsize=7.4, va='center'); yy -= .036
    pn.text(.5, .025, 'Evidence: grade A beat C in train and test on $1–2;\nsmall samples, survivors only. Not advice.', color=MUT, fontsize=6.6, ha='center', va='center')
    fig.savefig(path, facecolor=BG); plt.close(fig); return path
