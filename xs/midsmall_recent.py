"""Last 5 mid/small-cap screener BUY signals (replay of midsmall_screener.py rules on Yahoo 1h bars): entry = next open after the flip bar, exit = next open after the bear flip, open trades marked at the last close. Charts: 1h candles coloured by vector state."""
import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import midsmall_screener as M
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BULL, BEAR, BG, FG = '#3b6cff', '#ff5d8f', '#0b0b0c', '#8a8d93'
data = M.fetch_all(M.universe()); Fs = {}; Ys = {}
for tk, Y in data.items(): Yc = M.closed_only(Y); Ys[tk] = Yc; Fs[tk] = M.features(Yc)
evs = []
for tk, F in Fs.items():
    for ts, kind, vr, px in M.events(F): evs.append((ts, tk, kind, vr))
evs.sort(key=lambda e: (e[0], -(e[3] if e[3] == e[3] else 0))); pos = {}; trades = []
for ts, tk, kind, vr in evs:
    Y = Ys[tk]; i = Y.index.get_loc(ts)
    if kind == 'SELL' and tk in pos:
        t = pos.pop(tk); j = i + 1; t['exit_i'] = i; t['xpx'] = float(Y.open.iat[j]) if j < len(Y) else float(Y.close.iat[i]); t['xt'] = Y.index[j] if j < len(Y) else Y.index[i]; t['open'] = False; trades.append(t)
    elif kind == 'BUY' and tk not in pos and len(pos) < M.SLOTS and i + 1 < len(Y):
        pos[tk] = dict(tk=tk, flip_i=i, entry_i=i + 1, epx=float(Y.open.iat[i + 1]), et=Y.index[i + 1], vr=vr, open=True)
    elif kind == 'BUY' and tk not in pos and i + 1 >= len(Y):                       # flip on the very last closed bar: entry is the next open (not printed yet)
        pos[tk] = dict(tk=tk, flip_i=i, entry_i=None, epx=float(Y.close.iat[i]), et=Y.index[i], vr=vr, open=True)
for t in pos.values(): trades.append(t)
for t in trades:
    Y = Ys[t['tk']]
    if t['open']: t['xpx'] = float(Y.close.iat[-1]); t['xt'] = Y.index[-1]
    t['ret'] = t['xpx'] / t['epx'] - 1
T = sorted(trades, key=lambda t: t['et'])[-5:]; files = []; rows = []
for k, t in enumerate(T):
    Y = Ys[t['tk']]; F = Fs[t['tk']]; fi = t['flip_i']; ei = t['entry_i'] if t['entry_i'] is not None else fi
    xi = Y.index.get_loc(t['xt']); a = max(0, fi - 45); b = min(len(Y), xi + 14); B = Y.iloc[a:b]; rg = F.reg.values[a:b]; n = len(B)
    fig = plt.figure(figsize=(11, 6), facecolor=BG); gs = fig.add_gridspec(2, 1, height_ratios=[5, 1.3], hspace=.06, left=.06, right=.93, top=.86, bottom=.07); ax = fig.add_subplot(gs[0]); vx = fig.add_subplot(gs[1], sharex=ax)
    rngv = B.high.max() - B.low.min()
    for q in range(n):
        c = BULL if rg[q] > 0 else BEAR; ax.plot([q, q], [B.low.iat[q], B.high.iat[q]], color=c, lw=.9); lo, hi = sorted([B.open.iat[q], B.close.iat[q]]); ax.add_patch(plt.Rectangle((q - .34, lo), .68, max(hi - lo, rngv * .002), color=c)); vx.bar(q, B.volume.iat[q], width=.7, color=c)
    qi = ei - a; qx = xi - a
    ax.scatter([qi], [t['epx']], marker='^', s=110, color='#ffffff', edgecolor=BULL, zorder=6, lw=1.4); ax.axhline(t['epx'], color='#d0a94a', lw=.7, ls=':')
    if not t['open']: ax.scatter([qx], [t['xpx']], marker='v', s=110, color='#ffffff', edgecolor=BEAR, zorder=6, lw=1.4)
    for a_ in (ax, vx): a_.set_facecolor(BG); [s_.set_color('#26282c') for s_ in a_.spines.values()]
    ax.yaxis.tick_right(); vx.yaxis.tick_right(); ax.tick_params(colors=FG, labelsize=8); vx.tick_params(colors=FG, labelsize=7); plt.setp(ax.get_xticklabels(), visible=False)
    dts = [f'{B.index[q]:%b %d}' for q in range(0, n, max(1, n // 8))]; vx.set_xticks(list(range(0, n, max(1, n // 8)))); vx.set_xticklabels(dts, color=FG, fontsize=8); ax.set_xlim(-1, n + 1)
    st_ = 'OPEN (marked at last close)' if t['open'] else f"closed {t['xt']:%b %d %H:%M}"
    fig.text(.06, .945, f"{t['tk']}  ·  BUY {t['et']:%a %b %d %H:%M}  ·  {t['ret']:+.1%}", color='#f1f2f4', fontsize=18, fontweight='bold')
    fig.text(.06, .905, f"entry {t['epx']:.2f} ({t['vr']:.1f}x volume, low-vol day)  ·  exit {t['xpx']:.2f}  ·  {st_}", color=FG, fontsize=9)
    p = f"xs/ms_recent_{k + 1}_{t['tk']}.png"; fig.savefig(p, dpi=125, facecolor=BG); plt.close(fig); files.append(p); rows.append(f"{k + 1} {t['tk']:5s} BUY {t['et']:%b %d %H:%M} @ {t['epx']:.2f} -> {t['xpx']:.2f} {t['ret']:+.1%} {'OPEN' if t['open'] else 'closed ' + format(t['xt'], '%b %d')}")
print('\n'.join(rows)); print('FILES', files); print('mean', np.mean([t['ret'] for t in T]))
