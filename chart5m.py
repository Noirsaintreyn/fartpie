"""Nov 24 2026: 5-minute chart for a DAILY-vector signal (Rey: keep the daily vector, but show price on 5m so entries/gaps are readable).
make_chart_5m(tk, path, flip_day=None): 5m bars incl. pre/post-market (Yahoo, last ~30 sessions), the DAILY vector state machine (same run_state_machine as the screeners) run on daily bars, drawn as the daily floor/ceiling level on every session,
session VWAP (RTH), prior-close line (gap), flip-day marker and the entry (next session's open). Sessions shown: the one before the flip day through the latest (max 6)."""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xs'))
from vector_pine import run_state_machine
BG, PANEL, GRID, T1, T2, MUT = '#080c16', '#0f1626', '#1b2438', '#f1f5f9', '#94a3b8', '#56627a'
UP, DN, LVL, VWAP, GOLD = '#3b82f6', '#fb7185', '#a3e635', '#e2e8f0', '#fcd34d'
ET = 'America/New_York'
def _dl(tk, interval, period, prepost):
    import yfinance as yf
    D = yf.download(tk, period=period, interval=interval, prepost=prepost, progress=False, auto_adjust=False, threads=False)
    if isinstance(D.columns, pd.MultiIndex): D.columns = D.columns.get_level_values(0)
    D = D.dropna(subset=['Close']).rename(columns=str.lower)[['open', 'high', 'low', 'close', 'volume']]
    if D.index.tz is not None: D.index = D.index.tz_convert(ET).tz_localize(None)
    return D[~D.index.duplicated()]
def daily_state(tk):
    d = _dl(tk, '1d', '3y', False); d.index = pd.DatetimeIndex(d.index.date); reg, lev, _, flips = run_state_machine(d.assign(datetime=d.index)); return d, pd.Series(reg, index=d.index), pd.Series(lev, index=d.index)
def make_chart_5m(tk, path, flip_day=None, info=None, sessions=14):
    info = info or {}; d, reg, lev = daily_state(tk); b = _dl(tk, '5m', '30d', True)
    days = sorted(set(b.index.normalize())); fd = pd.Timestamp(flip_day) if flip_day is not None else None
    if fd is None:
        fl = reg[(reg == 1) & (reg.shift() != 1)]; fd = fl.index[-1] if len(fl) else days[-2]
    keep = [x for x in days if x >= (max([y for y in days if y < fd] or [fd]))][:sessions]; keep = keep[:1] + keep[1:][-(sessions - 1):]; X = b[b.index.normalize().isin(keep)].copy(); n = len(X); t = np.arange(n)
    rth = ((X.index.hour * 60 + X.index.minute) >= 570) & ((X.index.hour * 60 + X.index.minute) < 960); dayv = X.index.normalize()
    tp = (X.high + X.low + X.close) / 3; vw = pd.Series(np.nan, index=X.index)
    for dd in keep:
        m = (dayv == dd) & rth
        if m.any(): vw[m] = (tp[m] * X.volume[m]).cumsum() / (X.volume[m].cumsum() + 1e-9)
    fig = plt.figure(figsize=(12.4, 7.2), facecolor=BG, dpi=140); gs = fig.add_gridspec(2, 1, height_ratios=[5.2, 1.3], hspace=.04, left=.05, right=.985, top=.88, bottom=.07); ax = fig.add_subplot(gs[0]); av = fig.add_subplot(gs[1], sharex=ax)
    for a in (ax, av): a.set_facecolor(PANEL); [s.set_color(GRID) for s in a.spines.values()]; a.tick_params(colors=MUT, labelsize=7.5); a.grid(color=GRID, lw=.5, alpha=.7)
    BULL, BULLE, BEAR, BEARE = '#3b82f6', '#93c5fd', '#fb7185', '#9f1239'
    rg_bar = np.zeros(n)                                                       # DAILY vector regime in force for each 5m bar = the last COMPLETED daily bar before that session
    for dd in keep:
        prev = d.index[d.index < dd]; rg_bar[dayv == dd] = reg.reindex(prev[-1:]).iat[0] if len(prev) else 0
    upc = X.close.values >= X.open.values; al = np.where(rth, 1., .38)
    for i in range(n):
        bull = rg_bar[i] == 1; main, dark = (BULL, BULLE) if bull else (BEAR, BEARE)
        ax.plot([t[i], t[i]], [X.low.iat[i], X.high.iat[i]], color=main, lw=.8, alpha=al[i], zorder=3)
        ax.add_patch(plt.Rectangle((t[i] - .34, min(X.open.iat[i], X.close.iat[i])), .68, max(abs(X.close.iat[i] - X.open.iat[i]), 1e-4), facecolor=main if upc[i] else PANEL, edgecolor=main, lw=.6, alpha=al[i], zorder=4))
    col = np.where(rg_bar == 1, BULL, BEAR)
    for n_, cl_ in ((100, '#fbbf24'), (200, '#c084fc')):                                                         # Nov 25 2026: session VWAP dropped (report 485); slow daily EMAs as context only (as of the prior close, constant per session)
        ema = d.close.ewm(span=n_, adjust=False).mean(); ev_ = np.array([ema.reindex(d.index[d.index < dd][-1:]).iat[0] if (d.index < dd).any() else np.nan for dd in dayv])
        ev_ = np.full(n, np.nan)
        for dd in keep:
            prev = d.index[d.index < dd]
            if len(prev): ev_[dayv == dd] = ema.reindex(prev[-1:]).iat[0]
        lo_, hi_ = X.low.min(), X.high.max()
        if np.nanmin(ev_) < hi_ * 1.25 and np.nanmax(ev_) > lo_ * .8: ax.plot(t, np.where((ev_ > lo_ * .85) & (ev_ < hi_ * 1.15), ev_, np.nan), color=cl_, lw=1.0, alpha=.8, zorder=5, label=f'daily EMA{n_}')
    firsts = []; lv_ = []
    for k, dd in enumerate(keep):
        idx = np.where(dayv == dd)[0]; i0, i1 = idx[0], idx[-1]; firsts.append(i0)
        prev = d.index[d.index < dd]; lvl_day = lev.reindex(prev[-1:]).iat[0] if len(prev) else np.nan; rg = reg.reindex(prev[-1:]).iat[0] if len(prev) else 0
        lv_.append((i0, i1, lvl_day, rg))
        if len(prev): ax.hlines(d.close.reindex(prev[-1:]).iat[0], i0, i1, color=GOLD, lw=.8, ls=(0, (4, 3)), alpha=.7, zorder=2, label='prior close (gap ref)' if k == 0 else None)
        ax.axvline(i0 - .5, color=GRID, lw=1); ax.text(i0 + 1, X.high.max() + (X.high.max() - X.low.min()) * .03, f'{dd:%a %m-%d}' + ('  <- DAILY FLIP' if dd == fd else ''), color=GOLD if dd == fd else T2, fontsize=8, va='bottom', fontweight='bold' if dd == fd else 'normal')
        # entry marker = first RTH bar after the flip day
        if dd > fd and not any(isinstance(c_, str) and c_ == 'entry' for c_ in []):
            pass
    nxt = [dd for dd in keep if dd > fd]
    if nxt:
        m = (dayv == nxt[0]) & rth; i = np.where(m)[0][0]; ax.annotate(f'ENTRY (next open) {X.open.iat[i]:.2f}', (i, X.open.iat[i]), xytext=(i + 6, X.open.iat[i] - (X.high.max() - X.low.min()) * .12), color=T1, fontsize=9, fontweight='bold', arrowprops=dict(arrowstyle='->', color=T1, lw=1.2), zorder=9); ax.scatter([i], [X.open.iat[i]], s=60, color=T1, zorder=10)
    lo_, hi_ = X.low.min(), X.high.max(); pad = (hi_ - lo_) * .08; ax.set_ylim(lo_ - pad, hi_ + pad * 2.2)
    for i0, i1, lvl_day, rg in lv_:
        if np.isfinite(lvl_day) and lo_ - pad * 3 <= lvl_day <= hi_ + pad * 3: ax.hlines(lvl_day, i0, i1, color=LVL if rg == 1 else DN, lw=2, alpha=.95, zorder=6, label='daily vector level' if i0 == lv_[0][0] else None)
        elif np.isfinite(lvl_day) and i0 == lv_[0][0]: ax.text(i0 + 1, lo_ - pad * .6, f'daily vector level {lvl_day:.2f} (below the view)', color=LVL, fontsize=8)
    av.bar(t, X.volume.values, color=col, alpha=.55, width=.8); av.set_ylabel('vol', color=MUT, fontsize=7.5)
    for a in (ax, av): a.set_xlim(-1, n + 1)
    av.set_xticks(firsts); av.set_xticklabels([f'{dd:%m-%d}' for dd in keep])
    fig.text(.985, .915, 'BLUE = daily vector BULL  ·  ROSE = daily vector BEAR  ·  filled = up bar, hollow = down bar  ·  green/rose line = vector level', color=T2, fontsize=8, ha='right')
    ttl = f'{tk}  ·  5m chart of the DAILY vector signal  ·  last {X.close.iat[-1]:.2f}'; fig.text(.05, .945, ttl, color=T1, fontsize=14, fontweight='bold')
    sub = '  ·  '.join([f'{k}: {v}' for k, v in info.items()]) or 'faint candles = pre/post-market, bright = regular hours'; fig.text(.05, .915, sub, color=T2, fontsize=9)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.93), facecolor=PANEL, edgecolor=GRID, labelcolor=T2, fontsize=8, framealpha=.9); fig.savefig(path, facecolor=BG); plt.close(fig); return path
