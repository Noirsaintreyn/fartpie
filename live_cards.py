"""Discord cards for live_screener.py: board_card (09:00 open positions) and signal_dash (BUY display)."""
import os, numpy as np, pandas as pd, matplotlib, matplotlib.ticker
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
ASSETS = 'xs/ext/live_assets'; QQQ_DAILY = 'etf_daily/QQQ.parquet'
START = pd.Timestamp(os.getenv('SCREENER_START', '2026-10-13 09:30'))
NAVY, NAVY2, PANEL2, CYAN, CYAN2, WHITE, MUT2, RED2, GRN2, GRIDC = '#0b1626', '#0f2035', '#12263d', '#4fd1e8', '#2a7fa8', '#eef6fb', '#7f93a8', '#ff5c6c', '#57e3a0', '#1b3048'
T2 = '#c3c2b7'
def _bot_sigma(o, h, l, c, window=60):
    """bot THOD/TLOD sigma (xs/thod_core.py): Garman-Klass session sigma over 60 bars + Hurst fBm adjustment; returns (anchor, sigma)"""
    oo, hh, ll, cc = o[-window:], h[-window:], l[-window:], c[-window:]
    oo = np.maximum(oo, 1e-9); hh = np.maximum(hh, oo * .99); ll = np.maximum(ll, oo * .99); cc = np.maximum(cc, ll)
    s = np.sqrt(np.maximum(.5 * np.log(hh / (ll + 1e-9)) ** 2 - (2 * np.log(2) - 1) * np.log(cc / (oo + 1e-9)) ** 2, 1e-9).mean()) * c[-1]
    lags = range(2, min(20, len(c) // 2)); tau = [np.std(c[k:] - c[:-k]) for k in lags]; Hh = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    m = 1 + .3 * (Hh - .5) / .5 if Hh > .6 else (1 - .2 * (.5 - Hh) / .5 if Hh < .4 else 1)
    return c[-1], (s * m if abs(2 * s * m - 2 * s) > s * .1 else s)
def daily_bands(H):
    H = H[H.dt < H.dt.iloc[-1].normalize()]                                   # completed sessions only (anchor = prior close)
    D = H.groupby(H.dt.dt.normalize()).agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last')).iloc[-102:]
    return _bot_sigma(D.o.values, D.h.values, D.l.values, D.c.values)
def read_log():
    f = 'live/signal_log.csv'
    return pd.read_csv(f) if os.path.exists(f) else None
def _glow(ax, x, y, col, lws=(9, 5, 2.2), alphas=(.06, .14, 1), z=6):
    for lw, a in zip(lws, alphas): ax.plot(x, y, color=col, lw=lw, alpha=a, solid_capstyle='round', zorder=z)
def slot_wheel(ax, rows, meta, ring_bg=PANEL2):
    P = sorted([r for r in rows if r['action'] in ('HOLD', 'SELL')], key=lambda r: -r['pnl']); n = meta['slots']
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2; w = 2 * np.pi / n * .86
    ax.set_theta_direction(-1); ax.set_axis_off(); ax.set_ylim(0, 1.0)
    for k in range(n):
        if k < len(P):
            pnl = P[k]['pnl']; cc = GRN2 if pnl >= 0 else RED2; a = .35 + .65 * min(1, abs(pnl) / .06)
            ax.bar(th[k], .30, width=w, bottom=.62, color=cc, alpha=a, edgecolor=ring_bg, linewidth=2)
            ax.text(th[k], .77, P[k]['ticker'], ha='center', va='center', fontsize=7.5, fontweight='bold', color=WHITE)
        else: ax.bar(th[k], .30, width=w, bottom=.62, color=GRIDC, alpha=.9, edgecolor=ring_bg, linewidth=2)
    ax.text(.5, .53, f'{len(P) / n:.0%}', transform=ax.transAxes, ha='center', va='center', fontsize=20, fontweight='bold', color=WHITE)
    ax.text(.5, .38, 'invested', transform=ax.transAxes, ha='center', va='center', fontsize=8, color=MUT2)

BK, LINE, TXT, MUTD, BLUE1, BLUE2, RED1, GRN1 = '#05080f', '#1a2638', '#dfe7f1', '#7d8ca0', '#2f8cff', '#7cc4ff', '#ff4d5e', '#4ade80'
CMAP = LinearSegmentedColormap.from_list('b', ['#0b1424', '#123058', '#1d5fae', '#3d93ff', '#a9d6ff'])
def _kde(v, g, bw): d = (g[:, None] - v[None, :]) / bw; return np.exp(-.5 * d * d).sum(1) / (len(v) * bw * np.sqrt(2 * np.pi))
def _hdr(fig, x, y, t, sub=None):
    fig.text(x, y, t, fontsize=10.5, color=BLUE2, fontweight='bold')
    if sub: fig.text(x, y - .022, sub, fontsize=8, color=MUTD)
def _row(fig, x, y, k, v, w=.16, col=TXT, bold=False):
    fig.text(x, y, k, fontsize=10, color=MUTD); fig.text(x + w, y, v, fontsize=10.5, color=col, ha='right', fontweight='bold' if bold else 'normal')
def _ax(fig, rect):
    a = fig.add_axes(rect); a.set_facecolor('none'); [s.set_color(LINE) for s in a.spines.values()]; a.spines[['top', 'right']].set_visible(False)
    a.tick_params(colors=MUTD, labelsize=8, length=0); a.grid(color=LINE, lw=.5, alpha=.7); return a

def signal_dash(r, meta, path, trades=None, n_paths=400, horizon=52, seed=7):
    """BUY display. Every panel states what it is and where the number comes from."""
    if trades is None and os.path.exists(f'{ASSETS}/rule_trades_chop.parquet'): trades = pd.read_parquet(f'{ASSETS}/rule_trades_chop.parquet')
    H = r['H']; c = H.close.values; last = r['last']; stop = r['stop']; tk = r['ticker']; buy = r['action'] == 'BUY'
    entry = last if buy else r['px']
    dd = H.dt.dt.normalize().values[-26 * 60:]; lr = np.diff(np.log(c[-26 * 60:]))[dd[1:] == dd[:-1]]
    rng = np.random.default_rng(seed); B = 4
    st_ = rng.integers(0, len(lr) - B, size=(n_paths, horizon // B + 1))
    steps = np.concatenate([lr[s[:, None] + np.arange(B)] for s in st_.T], axis=1)[:, :horizon]
    P = np.hstack([np.full((n_paths, 1), last), last * np.exp(np.cumsum(steps, 1))])
    anc, sd = daily_bands(H); s2 = sd * np.sqrt(2)                                 # bot daily sigma; 2-day = sigma x sqrt(2)
    atr = (pd.concat([H.high - H.low, (H.high - H.close.shift()).abs(), (H.low - H.close.shift()).abs()], axis=1).max(axis=1)).rolling(14).mean().iloc[-1]
    D = H.groupby(H.dt.dt.normalize()).close.last(); rv20 = np.log(D).diff().tail(20).std() * np.sqrt(252); rv60 = np.log(D).diff().tail(60).std() * np.sqrt(252)
    rv5 = np.log(D).diff().tail(5).std() * np.sqrt(252); regime = 'Expanding' if rv5 > rv60 * 1.15 else ('Compressing' if rv5 < rv60 * .85 else 'Normal')
    sim_sd = np.std(np.log(P[:, 26] / last)) * last                              # simulated 1-day spread, to compare with the bot's sigma
    fig = plt.figure(figsize=(15.36, 10.24), facecolor=BK)
    fig.text(.018, .955, tk, fontsize=26, fontweight='bold', color=TXT)
    fig.text(.018 + .019 * len(tk) + .012, .962, '|', fontsize=16, color=LINE)
    fig.text(.018 + .019 * len(tk) + .028, .962, 'NEW LONG' if buy else 'OPEN LONG', fontsize=12, color=GRN1 if buy else TXT, fontweight='bold')
    fig.text(.33, .962, 'US equities   |   RTH', fontsize=13, color=MUTD)
    t_ = r['bar'] + pd.Timedelta('15min'); fig.text(.982, .965, f'{t_:%Y-%m-%d  %H:%M} ET', fontsize=13, color=TXT, ha='right')
    fig.text(.982, .945, f'Price {last:,.2f}   |   prior close {anc:,.2f}   |   bot daily σ {sd:,.2f} ({sd / anc:.2%})', fontsize=9, color=MUTD, ha='right')
    for y in (.925, .335): fig.lines.append(plt.Line2D([.012, .988], [y, y], transform=fig.transFigure, color=LINE, lw=1))
    for x, y0, y1 in ((.735, .345, .915), (.335, .02, .325), (.665, .02, .325)):
        fig.lines.append(plt.Line2D([x, x], [y0, y1], transform=fig.transFigure, color=LINE, lw=1))
    # ---------- center: volatility cone (3D)
    ax = fig.add_axes([.0, .33, .74, .58], projection='3d'); ax.set_facecolor(BK)
    ts = np.arange(2, horizon + 1, 2); lo, hi = np.percentile(P, 1), np.percentile(P, 99); lo = min(lo, stop - (hi - lo) * .05)
    g = np.linspace(lo, hi, 120); Z = np.array([_kde(P[:, t], g, (hi - lo) * .025) for t in ts])
    TT, GG = np.meshgrid(ts * .25, g, indexing='ij'); zmax = Z.max()
    fc = CMAP(np.clip(Z / zmax, 0, 1) ** .8); redc = np.array(matplotlib.colors.to_rgba(RED1)); below = GG < stop
    fc[below] = fc[below] * .35 + redc * .65 * np.clip(Z[below] / zmax * 1.6 + .25, 0, 1)[:, None]; fc[..., 3] = .97
    ax.plot_surface(TT, GG, Z, facecolors=fc, linewidth=.2, edgecolor=(.55, .75, 1, .18), rstride=1, cstride=2, antialiased=True, shade=False)
    tE = ts[-1] * .25
    ax.plot([0, tE], [stop, stop], [0, 0], color=RED1, lw=2.2); ax.plot([0, tE], [entry, entry], [0, 0], color=TXT, lw=1, ls=':')
    for k in (1, 2):                                                               # bot daily THOD/TLOD on the floor (1-day bands)
        for lv in (anc + k * sd, anc - k * sd):
            if lo < lv < hi: ax.plot([0, tE], [lv, lv], [0, 0], color='#5eead4', lw=1, ls='--', alpha=.8)
    if lo < anc + sd < hi: ax.text(0, anc + sd, 0, 'THOD1 (bot)  ', color='#5eead4', fontsize=8, ha='right')
    if lo < anc - sd < hi: ax.text(0, anc - sd, 0, 'TLOD1 (bot)  ', color='#5eead4', fontsize=8, ha='right')
    ax.plot([0, 0], [last, last], [0, zmax * 1.05], color=TXT, lw=1.1); ax.scatter([0], [last], [zmax * 1.05], color='white', s=36, depthshade=False)
    ax.text(0, last, zmax * 1.15, f'Entry {last:,.2f}', color=TXT, fontsize=10)
    ridge = [g[np.argmax(z)] for z in Z]; ax.plot(ts * .25, ridge, Z.max(1) * 1.01, color='#cfe8ff', lw=1.6)
    ax.text(tE, stop, 0, f'  STOP {stop:,.2f}', color=RED1, fontsize=10, fontweight='bold')
    for a_ in (ax.xaxis, ax.yaxis, ax.zaxis): a_.set_pane_color((0, 0, 0, 0)); a_._axinfo['grid']['color'] = (0.15, 0.22, 0.32, .6); a_.label.set_color(MUTD)
    ax.tick_params(colors=MUTD, labelsize=8); ax.set_xlabel('Trading hours ahead', fontsize=9, labelpad=6); ax.set_ylabel('Price', fontsize=9, labelpad=10); ax.set_zlabel('Likelihood', fontsize=9)
    ax.set_zticks([]); ax.view_init(elev=28, azim=-62); ax.set_box_aspect((1.4, 1.6, .7))
    fig.text(.018, .895, 'VOLATILITY CONE · PRICE × TIME × LIKELIHOOD', fontsize=10.5, color=BLUE2, fontweight='bold')
    fig.text(.018, .876, f'How far {tk} typically travels in the next 13 trading hours, built from its own intraday moves of the last 60 days.', fontsize=8, color=MUTD)
    fig.text(.018, .860, 'It shows SPREAD, not direction. Red = prices below the stop. Teal = your bot\'s daily THOD/TLOD.', fontsize=8, color=MUTD)
    # ---------- right top: where price ends up vs bot bands
    _hdr(fig, .75, .895, 'WHERE PRICE ENDS UP · +2 DAYS', 'simulated closes vs your bot\'s σ bands (daily σ × √2)')
    a2 = _ax(fig, [.765, .66, .21, .17]); gg = np.linspace(np.percentile(P[:, -1], .5), np.percentile(P[:, -1], 99.5), 300); dd_ = _kde(P[:, -1], gg, (gg[-1] - gg[0]) * .03)
    a2.fill_between(gg, dd_, where=gg >= stop, color=BLUE1, alpha=.35, lw=0); a2.fill_between(gg, dd_, where=gg < stop, color=RED1, alpha=.45, lw=0); a2.plot(gg, dd_, color=BLUE2, lw=1.4)
    for k in (-2, -1, 1, 2):
        lv = anc + k * s2
        if gg[0] < lv < gg[-1]: a2.axvline(lv, color='#5eead4', lw=.9, ls='--', alpha=.8); a2.text(lv, dd_.max() * 1.02, f'{k:+d}σ', color='#5eead4', fontsize=7, ha='center')
    a2.axvline(entry, color=TXT, lw=.9, ls=':'); a2.axvline(stop, color=RED1, lw=1); a2.set_yticks([])
    fig.text(.765, .618, f'simulated 1-day σ {sim_sd:,.2f} vs bot σ {sd:,.2f} → ' + ('agree' if abs(sim_sd / sd - 1) < .25 else 'differ'), fontsize=8, color=MUTD)
    # ---------- right mid: path fan
    _hdr(fig, .75, .585, 'PATH FAN', '1 line = 1 possible path from recent moves · red = hits stop')
    a3 = _ax(fig, [.765, .375, .21, .165]); tt = np.arange(horizon + 1) * .25
    for p in P[rng.choice(n_paths, 80, replace=False)]: a3.plot(tt, p, lw=.6, alpha=.22 if p.min() < stop else .45, color=RED1 if p.min() < stop else BLUE2)
    a3.axhline(stop, color=RED1, lw=1, ls='--'); a3.axhline(entry, color=TXT, lw=.7, ls=':'); a3.scatter([0], [last], color='white', s=18, zorder=5)
    a3.set_xlabel('Trading hours ahead', fontsize=8, color=MUTD); a3.set_ylim(np.percentile(P, 1), np.percentile(P, 99))
    # ---------- bottom 1: trend + trailing stop (candles colored by the 15m trend state)
    _hdr(fig, .018, .30, 'TREND & STOP · LAST 3 DAYS', 'blue = trend up, red = trend down · red step = stop')
    a4 = _ax(fig, [.035, .045, .285, .2]); x = H.iloc[-78:].reset_index(drop=True); tx = np.arange(len(x)); bull = x.reg.values == 1
    a4.vlines(tx, x.low, x.high, color=np.where(bull, '#2a5d9c', '#7a2e36'), lw=.8)
    a4.bar(tx, np.maximum(abs(x.close - x.open), (x.high.max() - x.low.min()) * .002), bottom=np.minimum(x.open, x.close), width=.62, color=np.where(bull, BLUE1, '#c2414f'))
    s_ = np.where(bull, x['stop'], np.nan); a4.step(tx, s_, where='mid', color=RED1, lw=1.6); a4.set_xticks([])
    # ---------- bottom 2: volume profile
    _hdr(fig, .35, .30, 'VOLUME PROFILE · 20 DAYS', 'shares traded at each price · teal = volume between stop and entry (support context)')
    a5 = _ax(fig, [.365, .045, .285, .2]); y20 = H.iloc[-26 * 20:]; bins = np.linspace(y20.low.min(), y20.high.max(), 28)
    vp, _ = np.histogram(y20.close, bins, weights=y20.volume); mids = (bins[:-1] + bins[1:]) / 2
    a5.barh(mids, vp, height=(bins[1] - bins[0]) * .8, color=np.where((mids > stop) & (mids < entry * 1.0001 + (bins[1] - bins[0])), '#5eead4', BLUE1), alpha=.5)
    a5.axhline(entry, color=TXT, lw=1, ls=':'); a5.axhline(stop, color=RED1, lw=1.2); a5.set_xticks([])
    a5.text(vp.max(), entry, f' entry {entry:,.2f}', color=TXT, fontsize=8, va='bottom', ha='right'); a5.text(vp.max(), stop, f' stop {stop:,.2f}', color=RED1, fontsize=8, va='top', ha='right')
    # ---------- bottom 3: this name under the rule
    _hdr(fig, .68, .30, f'{tk} TRACK RECORD', 'model backtest 2018-2025')
    if trades is not None and (trades.tk == tk).any():
        z = trades[trades.tk == tk]; w = z.ret[z.ret > 0]; l = z.ret[z.ret <= 0]; y = .235
        for k, v, col in (('Trades', f'{len(z):,}', TXT), ('Win rate', f'{(z.ret > 0).mean():.0%}', TXT), ('Avg win', f'{w.mean():+.2%}', GRN1), ('Avg loss', f'{l.mean():+.2%}', RED1),
                          ('Avg trade', f'{z.ret.mean():+.2%}', GRN1 if z.ret.mean() > 0 else RED1)):
            _row(fig, .68, y, k, v, w=.29, col=col); y -= .036
        fig.text(.68, .035, 'Low win rate is normal: small losses, few large winners.', fontsize=8, color=MUTD)
    fig.text(.018, .006, f'{tk} | simulations = resampled recent volatility, not a forecast | not financial advice', fontsize=8, color=MUTD)
    fig.text(.982, .006, 'SCREENER', fontsize=8, color=MUTD, ha='right')
    fig.savefig(path, dpi=100, facecolor=BK); plt.close(fig)

def board_card(rows, meta, path):
    _G = globals(); _saved = {k: _G[k] for k in ('NAVY', 'NAVY2', 'PANEL2', 'CYAN', 'CYAN2', 'GRN2', 'GRIDC', 'MUT2')}
    _G.update(NAVY='#020409', NAVY2='#06101f', PANEL2='#07111f', CYAN='#3d93ff', CYAN2='#1d5fae', GRN2='#4da3ff', GRIDC='#132238', MUT2='#7d8ca0')
    try: _board(rows, meta, path)
    finally: _G.update(_saved)

def _board(rows, meta, path):
    P = sorted([r for r in rows if r['action'] in ('HOLD', 'SELL')], key=lambda r: -r['pnl'])
    def panel(fig, x, y, w, h, title=None):
        fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.006', transform=fig.transFigure, fc=PANEL2, ec='#15315a', lw=1, zorder=-5, alpha=.9))
        if title: fig.text(x + .012, y + h - .028, title, fontsize=10, color=CYAN, fontweight='bold')
    fig = plt.figure(figsize=(19.2, 11), facecolor=NAVY)
    bg = fig.add_axes([0, 0, 1, 1], zorder=-20); bg.axis('off')
    bg.imshow(np.linspace(0, 1, 256)[:, None], cmap=LinearSegmentedColormap.from_list('n', [NAVY2, NAVY]), aspect='auto', extent=[0, 1, 0, 1])
    asof = pd.Series([r['bar'] for r in rows if r['ticker'] != 'QQQ']).max() + pd.Timedelta('15min')
    fig.text(.018, .955, 'OPEN POSITIONS', fontsize=24, fontweight='bold', color=WHITE)
    fig.text(.215, .958, 'long only · equal weight · cash in T-bills when flat', fontsize=12, color=MUT2)
    fig.text(.982, .958, f'{asof:%a %b %d %Y  %H:%M} ET', fontsize=12, color=WHITE, ha='right')
    fig.text(.982, .938, f"live since {START:%b %d, %Y}", fontsize=8.5, color=MUT2, ha='right')
    # ---------------- left column: KPIs + wheel + rule
    inv = len(P) / meta['slots']; avg = np.mean([r['pnl'] for r in P]) if P else 0
    kp = [('SLOTS', f"{len(P)}/{meta['slots']}", f"{meta['w']:.1%} each"), ('INVESTED', f'{inv:.0%}', 'rest in T-bills'),
          ('AVG OPEN P&L', f'{avg:+.1%}', f"{sum(r['pnl'] > 0 for r in P)} green / {sum(r['pnl'] <= 0 for r in P)} red"), ('WATCHLIST', f"{meta['slots']}", 'names on watch')]
    for k, (h, v, s_) in enumerate(kp):
        x = .018 + (k % 2) * .118; y = .80 - (k // 2) * .105
        panel(fig, x, y, .11, .095); fig.text(x + .01, y + .068, h, fontsize=8, color=MUT2, fontweight='bold')
        fig.text(x + .01, y + .03, v, fontsize=20, fontweight='bold', color=(GRN2 if (h == 'AVG OPEN P&L' and avg >= 0) else RED2 if h == 'AVG OPEN P&L' else WHITE)); fig.text(x + .01, y + .012, s_, fontsize=7.5, color=MUT2)
    panel(fig, .018, .325, .228, .36, 'SLOT WHEEL')
    wax = fig.add_axes([.032, .335, .20, .32], projection='polar'); wax.set_facecolor('none')
    slot_wheel(wax, rows, meta, ring_bg=PANEL2)
    panel(fig, .018, .035, .228, .275, 'TRACK RECORD')
    _D = pd.read_parquet(f'{ASSETS}/rule_rawratio_daily.parquet')['ALL 26_1.5_C'].loc[:'2025-12-31']; _O = pd.read_parquet(f'{ASSETS}/oos2026_daily.parquet')['stocks only 15m exit C (1/13)']
    _e = (1 + _D).cumprod(); _eo = (1 + _O).cumprod()
    rec = [('BACKTEST', '2018-2025'), ('CAGR', f"{_e.iloc[-1] ** (252 / len(_e)) - 1:+.1%} / yr"), ('MAX DD', f"{(_e / _e.cummax() - 1).min():.0%}  (QQQ -35%)"),
           ('2022', f"{(1 + _D.loc['2022']).prod() - 1:+.0%}  (QQQ -33%)"), ('2026 OOS', f"{_eo.iloc[-1] - 1:+.1%}  Jan-Sep"), ('POSITIONS', 'long only, equal size')]
    for k, (a_, b_) in enumerate(rec):
        fig.text(.03, .255 - k * .036, a_, fontsize=8.5, color=CYAN, fontweight='bold'); fig.text(.10, .255 - k * .036, b_, fontsize=9, color=WHITE)
    # ---------------- center/right: positions board (no chart)
    panel(fig, .256, .335, .726, .585, None)
    fig.text(.268, .895, 'POSITIONS', fontsize=11, color=CYAN, fontweight='bold')
    cols = [('TICKER', .275), ('ENTRY', .42), ('STOP', .57), ('NOW', .72), ('P&L', .87)]
    for h, x in cols: fig.text(x, .862, h, fontsize=10, color=MUT2, fontweight='bold')
    fig.lines.append(plt.Line2D([.268, .970], [.852, .852], transform=fig.transFigure, color='#15315a', lw=1))
    n = len(P); step = .036 if n > 12 else .041
    for k, r in enumerate(P[:13]):
        y = .815 - k * step; col = GRN2 if r['pnl'] >= 0 else RED2
        if k % 2 == 0: fig.patches.append(FancyBboxPatch((.262, y - .012), .714, step * .85, boxstyle='round,pad=0,rounding_size=0.004', transform=fig.transFigure, fc='#0c1a30', ec='none', zorder=-4, alpha=.6))
        fig.text(.275, y, r['ticker'], fontsize=15, color=WHITE, fontweight='bold')
        fig.text(.42, y, f"{r['px']:,.2f}", fontsize=14, color=WHITE)
        fig.text(.57, y, f"{r['stop']:,.2f}", fontsize=14, color=RED2, fontweight='bold')
        fig.text(.72, y, f"{r['last']:,.2f}", fontsize=14, color=T2)
        fig.text(.87, y, 'SELL' if r['action'] == 'SELL' else f"{r['pnl']:+.1%}", fontsize=15, color=RED2 if r['action'] == 'SELL' else col, fontweight='bold')
    if not P: fig.text(.62, .62, 'No open positions - all cash / T-bills', ha='center', fontsize=15, color=T2)
    # ---------------- bottom: equity (2018-25 backtest + 2026 OOS), R distribution, recent signals
    D = pd.read_parquet(f'{ASSETS}/rule_rawratio_daily.parquet')['ALL 26_1.5_C'].loc[:'2025-12-31']
    O = pd.read_parquet(f'{ASSETS}/oos2026_daily.parquet')['stocks only 15m exit C (1/13)']
    Q = pd.read_parquet(QQQ_DAILY); Q['d'] = pd.to_datetime(Q.datetime).dt.tz_localize(None).dt.normalize()
    R = pd.concat([D, O]); q = Q.drop_duplicates('d').set_index('d').adjclose.pct_change().reindex(R.index).fillna(0)
    eq, eqq = (1 + R).cumprod(), (1 + q).cumprod()
    panel(fig, .256, .035, .30, .285, 'EQUITY · 2018-25 BACKTEST + 2026 OUT-OF-SAMPLE (log)')
    ea = fig.add_axes([.268, .075, .28, .195]); ea.set_facecolor('none')
    ea.plot(eqq.index, eqq, color=MUT2, lw=1.2, alpha=.8); _glow(ea, eq.index, eq.values, CYAN, lws=(7, 4, 1.8))
    ea.axvspan(pd.Timestamp('2026-01-01'), eq.index[-1], color=GRN2, alpha=.08); ea.text(pd.Timestamp('2026-01-15'), 1.05, '2026\nOOS', color=GRN2, fontsize=8, va='bottom')
    ea.set_yscale('log'); ea.tick_params(colors=MUT2, labelsize=8, length=0); [s.set_visible(False) for s in ea.spines.values()]; ea.grid(color=GRIDC, lw=.5)
    ea.set_yticks([1, 2, 4, 8]); ea.set_yticklabels(['1x', '2x', '4x', '8x']); ea.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cg, cq = eq.iloc[-1] ** (252 / len(eq)) - 1, eqq.iloc[-1] ** (252 / len(eqq)) - 1
    dd, ddq = (eq / eq.cummax() - 1).min(), (eqq / eqq.cummax() - 1).min()
    ea.text(.01, .95, f'model  {cg:+.1%}/yr · max DD {dd:.0%}', transform=ea.transAxes, color=CYAN, fontsize=9, fontweight='bold', va='top')
    ea.text(.01, .82, f'QQQ  {cq:+.1%}/yr · max DD {ddq:.0%}', transform=ea.transAxes, color=MUT2, fontsize=9, va='top')
    X = pd.read_parquet(f'{ASSETS}/rating_inputs.parquet')
    panel(fig, .566, .035, .24, .285, 'WHERE THE EDGE COMES FROM · 4,310 TRADES')
    ra = fig.add_axes([.578, .075, .22, .19]); ra.set_facecolor('none')
    Rm = X.R.clip(-3, 12); bins = np.arange(-3, 12.5, .5); hcnt, _ = np.histogram(Rm, bins)
    cols = [RED2 if b < 0 else (GRN2 if b >= 2 else CYAN2) for b in bins[:-1]]
    ra.bar(bins[:-1] + .25, hcnt, width=.42, color=cols, alpha=.85); ra.set_yscale('log')
    ra.tick_params(colors=MUT2, labelsize=8, length=0); [s.set_visible(False) for s in ra.spines.values()]; ra.set_yticks([]); ra.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ra.set_xlabel('trade result in R (stop distance)', color=MUT2, fontsize=8)
    top = X.ret.nlargest(int(len(X) * .1)).sum() / X.ret.sum()
    ra.text(.98, .95, f'win rate {np.mean(X.ret > 0):.0%}\navg win {X.ret[X.ret > 0].mean():+.1%} · avg loss {X.ret[X.ret <= 0].mean():+.1%}\nbest 10% of trades = {top:.0%} of profit', transform=ra.transAxes, ha='right', va='top', color=WHITE, fontsize=8.5)
    panel(fig, .816, .035, .166, .285, 'LIVE LOG')
    log = read_log()
    if log is None or log.empty: fig.text(.826, .25, f"tracking signals from\n{START:%a %b %d %H:%M} ET", fontsize=10, color=MUTD, va='top')
    else:
        for k, (_, z) in enumerate(log.sort_values('bar', ascending=False).head(9).iterrows()):
            y = .26 - k * .025; fig.text(.826, y, f"{pd.Timestamp(z.bar):%m-%d %H:%M}", fontsize=8.5, color=MUT2, family='DejaVu Sans Mono')
            fig.text(.893, y, z.ticker, fontsize=9, color=WHITE, fontweight='bold'); a_ = z.action
            fig.text(.935, y, a_ if a_ == 'BUY' else f"SELL {z.pnl:+.1%}" if pd.notna(z.pnl) else 'SELL', fontsize=9, color=GRN2 if a_ == 'BUY' else RED2, fontweight='bold')
    fig.text(.5, .01, 'Equity & edge panels: backtest 2018-2025 + 2026 out-of-sample (Jan-Sep). Positions & log: live, from the start date. Not financial advice.', ha='center', fontsize=8, color=MUT2)

    fig.savefig(path, dpi=110, facecolor=NAVY); plt.close(fig)
