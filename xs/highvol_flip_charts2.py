"""Screenshots of 10 high-volume days (RTH volume >= 1.5x trailing 20d mean) that had a vector flip with the ENTRY at/after 08:00 ET (fill = next bar open >= 08:00), through Nov 10, for KGS LCID RIOT Z IREN. Ten picked evenly across the window, earliest -> latest.
Each chart: that day's 15m candles (pre+RTH, 04:00-16:00; before 08:00 dimmed), colour = 15m vector state, triangles = 15m flips (entries), strip underneath = 1h vector state, volume bars at the bottom."""
import sys, numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from vector_pine import run_state_machine
BULL, BEAR, BG, FG = '#3b6cff', '#ff5d8f', '#0b0b0c', '#8a8d93'
cands = []; DATA = {}
for tk in ('KGS', 'LCID', 'RIOT', 'Z', 'IREN'):
    dd = yf.download(tk, period='1y', interval='1d', auto_adjust=False, progress=False); dd.columns = [c[0] for c in dd.columns]; dd.index = dd.index.normalize()
    rel = dd.Volume / dd.Volume.rolling(20).mean().shift(); dret = dd.Close / dd.Open - 1
    d15 = yf.download(tk, period='60d', interval='15m', prepost=True, progress=False, auto_adjust=False); d15.columns = [c[0].lower() for c in d15.columns]
    d15.index = d15.index.tz_convert('America/New_York').tz_localize(None); d15 = d15[['open', 'high', 'low', 'close', 'volume']].dropna(subset=['open', 'high', 'low', 'close']); d15 = d15[~d15.index.duplicated()]
    m = d15.index.hour * 60 + d15.index.minute; d15 = d15[(m >= 240) & (m < 960)]
    reg15 = run_state_machine(d15.assign(datetime=d15.index))[0]
    rth = d15[(d15.index.hour * 60 + d15.index.minute) >= 570]; h = rth.resample('60min', origin='start_day', offset='30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    reg1 = run_state_machine(h.assign(datetime=h.index))[0]
    d5 = yf.download(tk, period='60d', interval='5m', prepost=True, progress=False, auto_adjust=False); d5.columns = [c[0].lower() for c in d5.columns]
    d5.index = d5.index.tz_convert('America/New_York').tz_localize(None); d5 = d5[['open', 'high', 'low', 'close', 'volume']].dropna(subset=['open', 'high', 'low', 'close']); d5 = d5[~d5.index.duplicated()]
    m5 = d5.index.hour * 60 + d5.index.minute; d5 = d5[(m5 >= 240) & (m5 < 960)]
    r5, l5, _, _ = run_state_machine(d5.assign(datetime=d5.index)); _, l15, _, _ = run_state_machine(d15.assign(datetime=d15.index))
    ema50 = d15.close.ewm(span=50, adjust=False).mean().values
    DATA[tk] = (d15, reg15, h, reg1, d5, r5, l5, l15, ema50)
    flips = [(d15.index[i], int(reg15[i]), i) for i in range(1, len(d15)) if reg15[i] != reg15[i - 1] and reg15[i] != 0]
    for t, v in rel.dropna().items():
        if v < 1.5 or t > pd.Timestamp('2026-11-10') or t < pd.Timestamp('2026-08-20'): continue
        f = [(ts, s, i) for ts, s, i in flips if ts.normalize() == t and i + 1 < len(d15) and (d15.index[i + 1].hour * 60 + d15.index[i + 1].minute) >= 480]
        if f: cands.append(dict(tk=tk, day=t, relvol=float(v), ret=float(dret[t]) * 100, flips=f))
C = pd.DataFrame(cands).sort_values('day').reset_index(drop=True); print('qualifying days', len(C), flush=True)
sel = C.iloc[np.unique(np.linspace(0, len(C) - 1, 10).round().astype(int))].reset_index(drop=True); print('chosen', len(sel), flush=True)
files = []; lines = []
for k, r in sel.iterrows():
    d15, reg15, h, reg1, d5, r5, l5, l15, ema50 = DATA[r.tk]; day = r.day; sel15 = (d15.index.normalize() == day); B = d15[sel15]; rg = reg15[sel15]
    fig = plt.figure(figsize=(11, 6.4), facecolor=BG); gs = fig.add_gridspec(5, 1, height_ratios=[6, .4, .4, .4, 1.4], hspace=.07, left=.05, right=.93, top=.86, bottom=.06)
    ax = fig.add_subplot(gs[0]); sx = fig.add_subplot(gs[1], sharex=ax); s5 = fig.add_subplot(gs[2], sharex=ax); ex = fig.add_subplot(gs[3], sharex=ax); vx = fig.add_subplot(gs[4], sharex=ax); n = len(B); mins = (B.index.hour * 60 + B.index.minute).values
    rngv = B.high.max() - B.low.min()
    for i in range(n):
        c = BULL if rg[i] > 0 else BEAR; a = .38 if mins[i] < 480 else 1.0
        ax.plot([i, i], [B.low.iloc[i], B.high.iloc[i]], color=c, lw=.9, alpha=a, zorder=2)
        lo, hi = sorted([B.open.iloc[i], B.close.iloc[i]]); ax.add_patch(plt.Rectangle((i - .34, lo), .68, max(hi - lo, rngv * .002), color=c, alpha=a, zorder=3)); vx.bar(i, B.volume.iloc[i], width=.7, color=c, alpha=a)
    e8 = int((mins >= 480).argmax()); ax.axvline(e8 - .5, color='#d0a94a', lw=.9, ls=':', zorder=1); ax.text(e8 - .3, B.high.max() + rngv * .03, 'entries from 08:00', color='#d0a94a', fontsize=8)
    ev = []
    for ts, s, i0 in r.flips:
        j = list(B.index).index(ts) if ts in B.index else None
        if j is None: continue
        fill_i = j + 1; ok = (mins[min(fill_i, n - 1)] >= 480)
        y = B.low.iloc[j] - rngv * .035 if s > 0 else B.high.iloc[j] + rngv * .035
        ax.scatter([j], [y], marker='^' if s > 0 else 'v', s=90 if ok else 40, color=BULL if s > 0 else BEAR, edgecolor='white' if ok else 'none', lw=.8, zorder=5, alpha=1 if ok else .35)
        if ok:
            fp = B.open.iloc[fill_i] if fill_i < n else float('nan'); ft = B.index[fill_i] if fill_i < n else None; ev.append(f"{'BULL' if s > 0 else 'BEAR'} entry {ft:%H:%M} @ {fp:.2f}" if ft is not None else f"{'BULL' if s > 0 else 'BEAR'} flip at close")
    # 1h state strip (RTH hours)
    hd = h[h.index.normalize() == day]; hr = reg1[h.index.normalize() == day]
    for t_, s_ in zip(hd.index, hr):
        x0 = ((t_.hour * 60 + t_.minute) - mins[0]) / 15 - .5; sx.add_patch(plt.Rectangle((x0, 0), 4, 1, color=BULL if s_ > 0 else BEAR, alpha=.85))
    m5d = (d5.index.normalize() == day); b5 = d5[m5d]; g5 = r5[m5d]; mm5 = (b5.index.hour * 60 + b5.index.minute).values; x5 = (mm5 - mins[0]) / 15
    xe5 = np.r_[x5[1:], n - .5]
    for x_, xe_, s_ in zip(x5, xe5, g5): s5.add_patch(plt.Rectangle((x_ - 1/6, 0), xe_ - x_, 1, color=BULL if s_ > 0 else BEAR, alpha=.85, lw=0))
    s5.set_ylim(0, 1); s5.set_yticks([]); s5.set_ylabel('5m', color=FG, fontsize=8, rotation=0, labelpad=12)
    lv5 = l5[m5d]; ax.step(x5, lv5, where='post', color='#ffffff', lw=.9, ls='--', alpha=.55, zorder=4, label='5m ratchet')
    lv15 = l15[sel15]; ax.step(np.arange(n), lv15, where='post', color='#ffd24a', lw=1.3, zorder=4, label='15m ratchet')
    em = ema50[sel15]; ax.plot(np.arange(n), em, color='#9be37a', lw=1.1, zorder=4, label='EMA50 (15m)')
    for i in range(n): ex.add_patch(plt.Rectangle((i - .5, 0), 1, 1, color=BULL if B.close.iloc[i] > em[i] else BEAR, alpha=.85, lw=0))
    ex.set_ylim(0, 1); ex.set_yticks([]); ex.set_ylabel('vs EMA50', color=FG, fontsize=8, rotation=0, labelpad=22)
    ax.set_ylim(B.low.min() - rngv * .08, B.high.max() + rngv * .10); ax.legend(loc='upper left', fontsize=7, facecolor=BG, edgecolor='#26282c', labelcolor=FG, framealpha=.8)
    sx.set_ylim(0, 1); sx.set_yticks([]); sx.set_ylabel('1h', color=FG, fontsize=8, rotation=0, labelpad=12)
    for a_ in (ax, sx, s5, ex, vx):
        a_.set_facecolor(BG); [s_.set_color('#26282c') for s_ in a_.spines.values()]
    ax.set_xlim(-1, n); ax.yaxis.tick_right(); ax.tick_params(colors=FG, labelsize=8); plt.setp(ax.get_xticklabels(), visible=False); plt.setp(sx.get_xticklabels(), visible=False); plt.setp(s5.get_xticklabels(), visible=False); plt.setp(ex.get_xticklabels(), visible=False)
    vx.yaxis.tick_right(); vx.tick_params(colors=FG, labelsize=7); tk_idx = [i for i in range(n) if mins[i] % 60 == 0]; vx.set_xticks(tk_idx); vx.set_xticklabels([f'{mins[i]//60:02d}:00' if mins[i]//60 % 2 == 0 else '' for i in tk_idx], color=FG, fontsize=8)
    fig.text(.05, .945, f'{r.tk}  ·  {day:%a %b %d}', color='#f1f2f4', fontsize=18, fontweight='bold')
    fig.text(.05, .905, f'volume {r.relvol:.1f}x its 20-day average  ·  day {r.ret:+.1f}% (open to close)  ·  ' + ('; '.join(ev) if ev else 'no entry'), color=FG, fontsize=9)
    p = f'xs/hvflip2_{k+1:02d}_{r.tk}_{day:%m%d}.png'; fig.savefig(p, dpi=125, facecolor=BG); plt.close(fig); files.append(p); lines.append(f'{k+1:2d} {r.tk:5s} {day:%b %d} vol {r.relvol:.1f}x move {r.ret:+.1f}% | ' + '; '.join(ev))
print('\n'.join(lines)); print('FILES', files)
