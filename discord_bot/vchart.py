"""Candles colored by the SVM-Fractals vector regime (blue = positive/support, red = negative/resistance, grey = not enough history yet). The vector line itself is NOT drawn.
Overlays come from one or two SOURCE timeframes (which can differ from the candle timeframe, e.g. 5m candles with 1h levels):
  view='thod'   -> that timeframe's THOD/TLOD 1/2/3 sigma bands
  view='levels' -> that timeframe's ML-filtered levels (open states exist behind SHOW_OPEN, off: no edge beyond distance/vol)."""
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pivot_ratchet_core import run_state_machine   # same state machine as the MotiveWave 'SVM - Fractals' port (light copy of backtest_pivot_ratchet's)

BG, FG, GRID = '#0f1116', '#e8eaed', '#1d222c'
POS, NEG, NEUT = '#4aa3ff', '#ff5a5f', '#6b7280'
TOUCH_MARKS = False     # dots / rings on the lines where price touched them (removed Oct 9 2026; the lines still start at their most recent touch)
SHOW_OPEN = False
REGULAR = True         # Oct 13 2026: plain green/red candles (close vs open) instead of vector-regime colors
FULL_LINES = True      # Oct 13 2026: THOD/TLOD lines run across the whole chart
UP, DN = '#26a69a', '#ef5350'
GEXP, GEXN, FLIP = '#2fbf71', '#e5484d', '#e8c15a'      # open states dropped (Oct 8 2026): no reach edge beyond distance/vol, with or without vector alignment
STYLE = [dict(thod='#2dd4a7', level='#9aa0a6', open='#ffa24a', filled=True),            # source 1
         dict(thod='#c4b5fd', level='#8da2c0', open='#ffd08a', filled=False),           # source 2 (hollow markers, cooler tint)
         dict(thod='#f6c177', level='#c9b18a', open='#ffd08a', filled=False)]           # source 3 (daily)
SHADE_THOD = False     # Oct 13 2026: no sigma-band shading; the only shaded areas are the options zones


def _frame(bars):
    df = pd.DataFrame(bars)
    df['dt'] = pd.to_datetime(df['date'])
    return df


def vector_regime(df):
    if len(df) < 80:
        return np.zeros(len(df), int)
    reg = run_state_machine(pd.DataFrame({'datetime': df['dt'].values, 'open': df['open'].values, 'high': df['high'].values,
                                          'low': df['low'].values, 'close': df['close'].values}))[0]
    return np.asarray(reg).astype(int)


def open_states(src_df, chart_df, price, ks=(8, 21), per_side=4):
    """untouched wave pivots of the SOURCE timeframe (fractal half-width k bars): highs above / lows below that no later source bar - and no
    later chart bar - has traded through. Returns price, pivot time, k, side."""
    H, L = src_df['high'].values, src_df['low'].values
    n = len(src_df)
    out = {}
    blen = src_df['dt'].diff().median() if n > 2 else pd.Timedelta(0)
    CH, CL, CT = chart_df['high'].values, chart_df['low'].values, chart_df['dt'].values
    for k in ks:
        if n < 2 * k + 3:
            continue
        sh = pd.Series(H).rolling(2 * k + 1, center=True).max().values
        sl = pd.Series(L).rolling(2 * k + 1, center=True).min().values
        for p in range(k, n - k):
            t_after = np.datetime64(src_df['dt'].iloc[p] + blen)
            j = int(np.searchsorted(CT, t_after))
            if H[p] == sh[p] and H[p] > price and H[p + 1:].max() < H[p] and (j >= len(CH) or CH[j:].max() < H[p]):
                out[(round(H[p], 2), 1)] = dict(price=float(H[p]), t=src_df['dt'].iloc[p], k=k, side=1)
            if L[p] == sl[p] and L[p] < price and L[p + 1:].min() > L[p] and (j >= len(CL) or CL[j:].min() > L[p]):
                out[(round(L[p], 2), -1)] = dict(price=float(L[p]), t=src_df['dt'].iloc[p], k=k, side=-1)
    ups = sorted([v for v in out.values() if v['side'] == 1], key=lambda v: v['price'])[:per_side]
    dns = sorted([v for v in out.values() if v['side'] == -1], key=lambda v: -v['price'])[:per_side]
    return ups + dns


def spread(ys, gap):
    """nudge label y-positions so neighbours are at least `gap` apart (keeps order)"""
    order = np.argsort(ys); pos = np.array(ys, float)[order]
    for i in range(1, len(pos)):
        if pos[i] - pos[i - 1] < gap: pos[i] = pos[i - 1] + gap
    pos += (np.array(ys, float)[order].mean() - pos.mean())
    out = np.empty_like(pos); out[order] = pos; return out


def _glow(ax, xs, ys, color, lw=1.0, ls='-', zorder=2, alpha=0.95):
    for w, a in ((lw * 5.0, 0.05), (lw * 3.0, 0.10), (lw * 1.8, 0.18)):
        ax.plot(xs, ys, color=color, lw=w, alpha=a, ls=ls, zorder=zorder - 0.2, solid_capstyle='round')
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, ls=ls, zorder=zorder, solid_capstyle='round')


def render(ticker, tf, view, resp, sources, nbars=110, issue_time=None, gex=None, bands=None, xlines=None):
    """bands: [(lo, hi, color, label)] drawn as filled bands; xlines: [(price, label, color)] thin solid lines"""
    bands = bands or []; xlines = xlines or []
    """sources: list (1-2) of dict(tf=..., resp=data response, hl=hod_lod response or None, levels=analysis.all_levels(resp))
    issue_time (naive NY Timestamp): REPLAY - the lines were issued then; candles after it are drawn and each line's touches after issue are marked (filled ring = first touch)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    MONO = 'DejaVu Sans Mono'
    df_all = _frame(resp['priceData'])
    reg_all = vector_regime(df_all)
    df = df_all.iloc[-nbars:].reset_index(drop=True)
    reg = reg_all[-nbars:]
    price = float(resp['currentPrice'])
    PD = 2 if price < 1000 else 1                                   # price decimals in labels: stocks in cents, index futures to a tenth
    n = len(df)
    fig = plt.figure(figsize=(12.6, 7.0), dpi=125); fig.patch.set_facecolor('#05070d')
    bgax = fig.add_axes([0, 0, 1, 1], zorder=-5); bgax.axis('off'); g = np.linspace(0, 1, 256).reshape(-1, 1); bgax.imshow(np.dstack([0.02 + 0.045 * g, 0.03 + 0.06 * g, 0.05 + 0.13 * g]), aspect='auto', extent=[0, 1, 0, 1], origin='lower')
    clo, chi = float(df['low'].min()), float(df['high'].max()); span = chi - clo; lo_ok, hi_ok = clo - 0.6 * span, chi + 0.6 * span
    for src in sources[:3]:                                         # always keep every source's 1 sigma pair on screen (daily lines sit far out on a 5m chart)
        if view == 'thod' and src.get('hl'): lo_ok, hi_ok = min(lo_ok, src['hl']['lod']['1std'] - 1e-9), max(hi_ok, src['hl']['hod']['1std'] + 1e-9)
    marks = []                                                      # every price to type out under the chart: (price, label, color)
    items, opens, zones, hidden = [], [], [], 0                                          # items: (price, label, color, linestyle, strength)
    for si, src in enumerate(sources[:3]):
        st = dict(STYLE[si]); st['thod'] = src.get('color', st['thod']); tag = f"{src['tf']} " if (src['tf'] != tf or len(sources) > 1) else ''
        if view == 'thod' and src.get('hl'):
            hl = src['hl']; vals = {}
            for kk, key in ((1, '1std'), (2, '2std'), (3, '3std')):
                for nm, v in (('THOD', hl['hod'][key]), ('TLOD', hl['lod'][key])):
                    kl = src['names'][kk - 1] if src.get('names') else f'{kk}σ'
                    vals[(nm, kk)] = v; inr = lo_ok <= v <= hi_ok; marks.append((v, f'{tag}{nm} {kl}' + ('' if inr else '*'), st['thod']))
                    if inr: items.append((v, f'{tag}{nm} {kl}  {v:,.{PD}f}', st['thod'], '-', 0))
                    else: hidden += 1
            for nm in ('THOD', 'TLOD'):
                for a_, b_, al in ((1, 2, 0.06), (2, 3, 0.11)): zones.append((vals[(nm, a_)], vals[(nm, b_)], st['thod'], al))
        elif view == 'levels':
            lv = [l for l in (src.get('levels') or []) if lo_ok <= l['price'] <= hi_ok]
            for l in sorted(lv, key=lambda l: abs(l['price'] - price))[:(10 if si == 0 else 7)]:
                items.append((l['price'], f"{tag}{l['price']:,.{PD}f}  {l['name'][:11]}", st['level'], '-', l['strength'])); marks.append((l['price'], f"{tag}{l['name'][:11]}", st['level']))
            sdf = _frame(src['resp']['priceData'])
            for o in (open_states(sdf, df_all, price) if SHOW_OPEN else []):
                if not (lo_ok <= o['price'] <= hi_ok): continue
                x0 = int(np.searchsorted(df['dt'].values, np.datetime64(o['t'])))
                x0, older = (0, True) if x0 <= 0 else (min(x0, n - 1), False)
                opens.append((o['price'], x0, older, st, f"{tag}open {o['price']:,.1f}")); marks.append((o['price'], f'{tag}open', st['open']))
    gzones = []                                                     # (lo, hi, color, label, strike)
    if gex is not None and view == 'thod':
        G = gex['G']; sp = gex['spot']; near = G[(G.index > sp * .92) & (G.index < sp * 1.08)].net
        near = near[(near.index >= lo_ok) & (near.index <= hi_ok)]
        if len(near) > 2:
            ks = np.sort(near.index.values); step_k = float(np.median(np.diff(ks))) if len(ks) > 1 else 1.0; big = near.abs().max()
            pick = pd.concat([near[near > 0].nlargest(3), near[near < 0].nsmallest(3)]); pick = pick[pick.abs() >= .15 * big]
            def fmt(x): return f"{x / 1e9:+.1f}B" if abs(x) >= 1e9 else f"{x / 1e6:+.0f}M"
            for k_, v_ in pick.sort_index().items():
                if gzones and np.sign(v_) == np.sign(gzones[-1][5]) and k_ - gzones[-1][4] <= step_k * 1.01:     # merge neighbouring strikes of the same sign
                    lo0, _, c0, _, _, v0 = gzones[-1]; gzones[-1] = (lo0, k_ + step_k / 2, c0, None, k_, v0 + v_); continue
                gzones.append((k_ - step_k / 2, k_ + step_k / 2, GEXP if v_ > 0 else GEXN, None, k_, v_))
            out = []
            for lo0, hi0, c0, _, k_, v_ in gzones:
                mid = (lo0 + hi0) / 2; nm = ('call wall' if k_ == gex.get('call_wall') else 'put wall' if k_ == gex.get('put_wall') else ('+γ' if v_ > 0 else '−γ'))
                out.append((lo0, hi0, c0, f"{nm} {mid:,.{PD}f}  {fmt(v_)}", mid))
            gzones = out
            # Oct 13 2026: options zones are unlabeled colored boxes only (no ladder labels, no flip line) to cut clutter
    for lo0, hi0, c0, lab in bands:
        if lo_ok <= (lo0 + hi0) / 2 <= hi_ok: items.append(((lo0 + hi0) / 2, f"{lab}  {lo0:,.{PD}f}-{hi0:,.{PD}f}", c0, 'none', 0)); marks.append(((lo0 + hi0) / 2, lab, c0))
    for p0, lab, c0 in xlines:
        if lo_ok <= p0 <= hi_ok: items.append((p0, f"{lab}  {p0:,.{PD}f}", c0, '-', 0)); marks.append((p0, lab, c0))
    TICK = 0.25 if price >= 1000 else 0.01                              # strip rounding: futures index points vs stock cents
    NCOL = 6; nrow = max(1, -(-len(marks) // NCOL)); ax_bottom = 0.075 + 0.05 + 0.024 * nrow                  # leave room under the chart for the typed-out prices
    ax = fig.add_axes([0.055, ax_bottom, 0.915, 0.82 - ax_bottom]); ax.set_facecolor('none')
    lo, hi = clo, chi
    for it in items: lo, hi = min(lo, it[0]), max(hi, it[0])
    for o in opens: lo, hi = min(lo, o[0]), max(hi, o[0])
    pad = (hi - lo) * 0.06; ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(-1, n + 27)
    ex = n - 1 if issue_time is None else int(np.searchsorted(df['dt'].values, np.datetime64(pd.Timestamp(issue_time)) - np.timedelta64(5, 'm'), side='right') - 1)      # the candle the lines are anchored to: they were issued with the latest data, so nothing to the left of it had them
    for a_, b_, col, al in (zones if SHADE_THOD else []): ax.axhspan(min(a_, b_), max(a_, b_), xmin=(0 if FULL_LINES else (ex + 1) / (n + 28)), xmax=(n + 1) / (n + 28), color=col, alpha=al, lw=0, zorder=0.5)
    for lo0, hi0, c0, lab in bands:
        ax.add_patch(plt.Rectangle((-1, lo0), n + 1.6, max(hi0 - lo0, (hi - lo) * .002), facecolor=c0, alpha=0.28, edgecolor=c0, lw=0.7, zorder=0.7))
    for lo0, hi0, c0, lab, mid in gzones:
        ax.add_patch(plt.Rectangle((-1, lo0), n + 1.6, hi0 - lo0, facecolor=c0, alpha=0.13, edgecolor=c0, lw=0.8, zorder=0.6))
    # candles with a soft glow
    for i, r in df.iterrows():
        col = (UP if r['close'] >= r['open'] else DN) if REGULAR else (POS if reg[i] == 1 else (NEG if reg[i] == -1 else NEUT)); top, bot = max(r['open'], r['close']), min(r['open'], r['close']); body = max(top - bot, 1e-9)
        ax.plot([i, i], [r['low'], r['high']], color=col, lw=3.2, alpha=0.10, zorder=2.5, solid_capstyle='round'); ax.plot([i, i], [r['low'], r['high']], color=col, lw=0.9, zorder=3, solid_capstyle='round')
        ax.add_patch(plt.Rectangle((i - 0.42, bot - 0.02 * span * 0.05), 0.84, body + 0.0, facecolor=col, edgecolor='none', alpha=0.16, zorder=3.4))
        ax.add_patch(plt.Rectangle((i - 0.30, bot), 0.60, body, facecolor=col, edgecolor=col, lw=0.6, alpha=0.95, zorder=4))
    lows_, highs_ = df['low'].values, df['high'].values
    for p, lab, col, ls, stg in items:
        if ls == 'none': continue
        if FULL_LINES and view == 'thod':
            ax.plot([-1, n + 0.2], [p, p], color=col, lw=0.7, ls='-', alpha=0.9, zorder=2); continue
        lw_ = 0.95 if view == 'thod' else 0.8 + 1.3 * min(stg / 100, 1)
        hit = np.where((lows_[:ex + 1] <= p) & (highs_[:ex + 1] >= p))[0]                    # candles (up to the issue candle) whose range contains the line
        if len(hit):                                                                         # anchor at the MOST RECENT touch; faint dots only between the first and the last touch in view
            last = int(hit[-1]); _glow(ax, [last, n + 0.2], [p, p], col, lw=lw_, ls=ls, zorder=2)
            starts = [int(hit[0])] + [int(h) for q, h in zip(hit[:-1], hit[1:]) if h - q > 1]
            if last > starts[0]: ax.plot([starts[0], last], [p, p], color=col, lw=0.75, ls=(0, (1, 3)), alpha=0.5, zorder=1.8)
            if TOUCH_MARKS and len(starts) > 1: ax.scatter(starts[:-1][-12:], [p] * len(starts[:-1][-12:]), s=11, facecolor='none', edgecolor=col, linewidths=0.8, alpha=0.85, zorder=4.9)
            if TOUCH_MARKS: ax.scatter([last], [p], s=16, color=col, zorder=5.0)
        else:                                                                                # never touched in view: starts at the candle the lines were issued
            _glow(ax, [ex, n + 0.2], [p, p], col, lw=lw_, ls=ls, zorder=2)
        if issue_time is not None and ex < n - 1:                                            # replay: touches AFTER the lines were issued
            post = np.where((lows_[ex + 1:] <= p) & (highs_[ex + 1:] >= p))[0] + ex + 1
            if len(post) and TOUCH_MARKS:
                ax.scatter([int(post[0])], [p], s=70, facecolor='none', edgecolor='#ffffff', linewidths=1.4, zorder=6); ax.scatter([int(post[0])], [p], s=20, color=col, zorder=6.1)
                later = [int(h) for q, h in zip(post[:-1], post[1:]) if h - q > 1]
                if later: ax.scatter(later[:12], [p] * len(later[:12]), s=11, facecolor='none', edgecolor=col, linewidths=0.8, alpha=0.85, zorder=4.9)
    ax.axvline(ex, color='#e8eaed', lw=0.7, alpha=0.35, ls=(0, (1, 3)), zorder=1.9)
    if issue_time is not None:
        nx = int(np.searchsorted(df['dt'].values, np.datetime64(pd.Timestamp(issue_time).normalize() + pd.Timedelta(hours=12))))
        if ex < nx < n: ax.axvline(nx, color='#8a94a6', lw=0.7, alpha=0.5, ls=(0, (4, 3)), zorder=1.9); ax.text(nx + 0.3, hi + pad * 0.1, '12:00', color='#8a94a6', fontsize=7.5, va='top', family=MONO)
        i0 = int(np.searchsorted(df['dt'].values, np.datetime64(pd.Timestamp(issue_time).normalize() + pd.Timedelta(hours=9, minutes=30))))        # the 09:30 open: where the day (and, for stocks, the first print after the levels) starts
        if 0 < i0 < n: ax.axvline(i0 - 0.5, color='#ffd166', lw=1.0, alpha=0.8, ls=(0, (5, 3)), zorder=2.0); ax.text(i0 - 0.1, hi + pad * 0.1, '09:30 open', color='#ffd166', fontsize=7.5, va='top', family=MONO, zorder=6)
    ax.text(ex - 0.6, lo - pad + (hi - lo + 2 * pad) * 0.012, f"lines issued {pd.Timestamp(df['dt'].iloc[ex]) + pd.Timedelta(minutes={'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240}.get(tf, 0)):%H:%M} ET (data time)", color='#8a94a6', fontsize=7.5, ha='right', va='bottom', family=MONO, zorder=6)
    for p, x0, older, st, lab in opens:
        _glow(ax, [x0, n - 1], [p, p], st['open'], lw=0.9, zorder=2)
        ax.scatter([x0], [p], s=150, color=st['open'], alpha=0.10, zorder=4.8); ax.scatter([x0], [p], s=60, color=st['open'], alpha=0.20, zorder=4.9)
        ax.scatter([x0], [p], marker='<' if older else 'D', s=30 if older else 26, facecolor=st['open'] if st['filled'] else '#05070d', edgecolor=st['open'], linewidths=1.3, zorder=5)
        items.append((p, lab, st['open'], 'none', 0))
    # right-hand ladder: rail, ticks, labels, strength bars
    rail = n + 0.6; ax.plot([rail, rail], [lo - pad, hi + pad], color='#2b3447', lw=1, zorder=1)
    py_ = price
    if items:
        allp = [it[0] for it in items] + [price]; ys = spread(allp, (hi - lo + 2 * pad) * 0.029); py_ = ys[-1]; ys = ys[:-1]    # price tag gets its own slot on the ladder
        for (p, lab, col, ls, stg), y in zip(items, ys):
            ax.plot([rail - 0.5, rail + 0.5], [p, p], color=col, lw=1.2, zorder=3); ax.plot([rail + 0.5, rail + 1.4, rail + 1.9], [p, y, y], color=col, lw=0.6, alpha=0.7, zorder=3)
            ax.text(rail + 2.1, y, lab, color=col, va='center', ha='left', fontsize=7.7, family=MONO, zorder=6)
            if view == 'levels' and stg > 0: ax.add_patch(plt.Rectangle((n + 22.0, y - (hi - lo) * 0.006), 4.0 * min(stg / 100, 1), (hi - lo) * 0.012, facecolor=col, alpha=0.55, edgecolor='none', zorder=5)); ax.add_patch(plt.Rectangle((n + 22.0, y - (hi - lo) * 0.006), 4.0, (hi - lo) * 0.012, facecolor='none', edgecolor=col, lw=0.4, alpha=0.4, zorder=5))
    # price tag + arrow on the rail
    ax.plot([-1, (n + 0.2) if issue_time is None else (ex + 0.2)], [price, price], color='#ffffff', lw=0.7, ls=(0, (2, 3)), alpha=0.8, zorder=2); ax.annotate(f'{price:,.2f}', xy=(rail, price), xytext=(rail + 2.1, py_), arrowprops=dict(arrowstyle='-', color='#e8eaed', lw=0.6, alpha=0.7), color='#05070d', fontsize=8.2, family=MONO, fontweight='bold', va='center', ha='left', zorder=8, bbox=dict(boxstyle='round,pad=0.25', facecolor='#e8eaed', edgecolor='none'))
    ax.scatter([rail], [price], marker='>', s=28, color='#e8eaed', zorder=8)
    ax.tick_params(colors='#7f8aa3', labelsize=8, length=3, color='#2b3447'); [t.set_family(MONO) for t in ax.get_yticklabels()]
    ax.grid(axis='y', color='#ffffff', alpha=0.045, lw=0.7); ax.grid(axis='x', color='#ffffff', alpha=0.03, lw=0.6)
    for k_, s_ in ax.spines.items(): s_.set_visible(k_ in ('bottom', 'left')); s_.set_color('#2b3447')
    step = max(1, n // 7); ax.set_xticks(range(0, n, step)); fmt = '%m-%d' if tf in ('1d', '4h') else '%m-%d %H:%M'
    ax.set_xticklabels([df['dt'].iloc[i].strftime(fmt) for i in range(0, n, step)], family=MONO)
    srcs = ' + '.join(s['tf'].upper() for s in sources[:3]); vtxt = {'thod': 'THOD / TLOD', 'levels': 'LEVELS + OPEN STATES'}[view]
    # header
    fig.text(0.058, 0.935, f"{ticker}", color='#e8eaed', fontsize=19, fontweight='bold', family=MONO); fig.text(0.058 + 0.012 * (len(ticker) + 1.2), 0.9385, f"{tf.upper()} CANDLES", color='#7f8aa3', fontsize=10.5, family=MONO)
    fig.text(0.058, 0.905, f"{vtxt}  ·  FROM {srcs}", color='#8ea2c9', fontsize=8.8, family=MONO)
    fig.text(0.968, 0.935, f"{price:,.2f}", color='#e8eaed', fontsize=19, fontweight='bold', family=MONO, ha='right')
    fig.add_artist(plt.Line2D([0.058, 0.968], [0.893, 0.893], color='#2b3447', lw=0.8)); fig.add_artist(plt.Line2D([0.058, 0.20], [0.893, 0.893], color=POS, lw=1.6)); fig.add_artist(plt.Line2D([0.20, 0.34], [0.893, 0.893], color=NEG, lw=1.6))
    handles = ([Line2D([0], [0], color=UP, lw=4, label='up candle'), Line2D([0], [0], color=DN, lw=4, label='down candle')] if REGULAR else
               [Line2D([0], [0], color=POS, lw=4, label='positive vector'), Line2D([0], [0], color=NEG, lw=4, label='negative vector')])
    if gzones:
        from matplotlib.patches import Patch
        handles += [Patch(facecolor=GEXP, alpha=.35, label='+GEX zone (dealers long γ: pin / slow)'), Patch(facecolor=GEXN, alpha=.35, label='−GEX zone (dealers short γ: fast)')]
    for si, src in enumerate(sources[:3]):
        st = STYLE[si]
        st = dict(st); st['thod'] = src.get('color', st['thod'])
        if view == 'thod': handles.append(Line2D([0], [0], color=st['thod'], lw=1.2, ls='-', label=f"{src['tf']} THOD/TLOD " + ('/'.join(src['names']) if src.get('names') else '1-3σ') + (' (shaded)' if SHADE_THOD else '')))
        else: handles += [Line2D([0], [0], color=st['level'], lw=1.4, label=f"{src['tf']} level (bar = strength)"), Line2D([0], [0], color=st['open'], lw=1.0, marker='D', markersize=5, markerfacecolor=st['open'] if st['filled'] else '#05070d', label=f"{src['tf']} open state")]
    leg = ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.005), ncol=5, frameon=False, fontsize=7.8, labelcolor='#c8d0e0', handlelength=1.8, columnspacing=1.4)
    if marks:
        fig.text(0.058, ax_bottom - 0.062, f'PRICES  (high → low, nearest {TICK:g})', color='#8a94a6', fontsize=8, fontweight='bold', family=MONO)
        for k, (p, lab, col) in enumerate(sorted(marks, key=lambda m: -m[0])):
            r_, c_ = k // NCOL, k % NCOL; fig.text(0.058 + c_ * 0.158, ax_bottom - 0.092 - r_ * 0.024, f'{round(p / TICK) * TICK:>10,.2f}  {lab[:15]}', color=col, fontsize=7.6, family=MONO)
    note = ('regular candles  ·  THOD/TLOD lines full width' if REGULAR else 'candle color = vector regime (line not drawn)  ·  lines start at their most recent touch (untouched ones at the issue candle)') + \
           (f"  ·  GEX zones = dealer gamma by strike from {pd.Timestamp(gex['asof']):%m-%d} close + next-morning OI" if gzones else '')
    note2 = ('REPLAY: lines as issued, candles since  ·  ' if issue_time is not None else '') + ('* = off-screen at this zoom  ·  ' if hidden else '') + 'references, not signals  ·  not financial advice'
    fig.text(0.058, 0.034, note, color='#52607a', fontsize=7, family=MONO); fig.text(0.058, 0.016, note2, color='#52607a', fontsize=7, family=MONO); fig.text(0.968, 0.018, 'DEGENCAP', color='#2b3447', fontsize=9, family=MONO, ha='right', fontweight='bold')
    buf = io.BytesIO(); fig.savefig(buf, format='png', facecolor=fig.get_facecolor()); plt.close(fig); buf.seek(0)
    return buf
