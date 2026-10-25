"""Morning report (Oct 25 2026): a written, chart-based report of the book for Discord (posted 09:00 ET, and on /brief).
Cover: the book at a glance, sector ETFs with their daily + weekly vector, SPY / QQQ, today's US calendar and top market news.
One page per position: 15m chart with the stop, the model read (15m / 1h / daily vector, HAR expected range, earnings), why we hold,
what we want to see, what would say the trade failed (rule exit vs early warnings), and the news on the name and its sector.
Uses the screener's own state machine (vector) on every timeframe and the HAR range regression; news from Helious."""
import os, re, textwrap, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ETFS = ['XLK', 'XLC', 'XLY', 'XLF', 'XLV', 'XLP', 'XLI', 'XLE', 'XLU', 'XLB', 'XLRE']
ETF_NAME = {'XLK': 'Technology', 'XLC': 'Communication', 'XLY': 'Discretionary', 'XLF': 'Financials', 'XLV': 'Health Care', 'XLP': 'Staples', 'XLI': 'Industrials',
            'XLE': 'Energy', 'XLU': 'Utilities', 'XLB': 'Materials', 'XLRE': 'Real Estate', 'SPY': 'S&P 500', 'QQQ': 'Nasdaq 100'}

def _daily_states(LS, closes_only=False):
    """daily + weekly vector state, 1d / 5d / 1m returns for the sector ETFs, SPY, QQQ (completed sessions only)"""
    import yfinance as yf
    tk = ETFS + ['SPY', 'QQQ']
    Y = yf.download(tk, period='2y', interval='1d', progress=False, auto_adjust=True, group_by='ticker')
    now = LS.now_et(); out = {}
    for t in tk:
        try: x = Y[t].dropna().copy()
        except Exception: continue
        x.index = x.index.tz_localize(None) if x.index.tz is not None else x.index
        if now.hour * 60 + now.minute < LS.RTH1: x = x[x.index < now.normalize()]
        if len(x) < 120: continue
        d = x.rename(columns=str.lower).reset_index().rename(columns={'Date': 'dt', 'index': 'dt'})[['dt', 'open', 'high', 'low', 'close']]
        dv = LS.vec(d); w = x.rename(columns=str.lower).resample('W-FRI').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna().reset_index().rename(columns={'Date': 'dt', 'index': 'dt'})
        wv = LS.vec(w); c = x.Close if 'Close' in x else x['close']
        out[t] = dict(d=int(dv.reg.iloc[-1]), w=int(wv.reg.iloc[-1]), r1=float(c.iloc[-1] / c.iloc[-2] - 1), r5=float(c.iloc[-1] / c.iloc[-6] - 1), r21=float(c.iloc[-1] / c.iloc[-22] - 1))
    return out

def _stock_daily_state(LS, H):
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; R = H[(m >= LS.RTH0) & (m < LS.RTH1)]
    D = R.groupby(R.dt.dt.normalize()).agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).reset_index().rename(columns={'dt': 'dt'})
    D.columns = ['dt', 'open', 'high', 'low', 'close']
    if len(D) < 60: return None
    return int(LS.vec(D).reg.iloc[-1])

def _news(LS, held):
    """Helious feed once: items per held ticker, per held sector, top market items, US calendar"""
    try: items = LS.helious('/v1/feed', dict(limit=100)).get('data', {}).get('items', [])
    except Exception: items = []
    def txt(it): return f"{it.get('title', '')} {it.get('text', '')}"
    per = {}
    for t in held:
        pat = rf'\b{t}\b' + (f"|{LS.HEL_NAMES[t]}" if t in LS.HEL_NAMES else '')
        per[t] = [it for it in items if re.search(pat, txt(it), re.I)][:3]
    sec = {}
    for t in held:
        sc = LS.sector_of(t)
        if sc in LS.SECTOR_WORDS:
            sec[t] = [it for it in sorted(items, key=lambda it: -abs(it.get('score', 0) or 0)) if abs(it.get('score', 0) or 0) >= 1 and re.search(LS.SECTOR_WORDS[sc], txt(it), re.I)
                      and it not in per[t]][:2]
    top = sorted(items, key=lambda it: -abs(it.get('score', 0) or 0))[:4]
    cal = []
    try:
        ev = LS.helious('/v1/calendar', dict(days_back=0, days_ahead=1, country='US', limit=40), ttl=3600).get('data', {}).get('events', [])
        cal = [e for e in ev if e.get('impact') == 'High' or (e.get('impact') == 'Medium' and e.get('type') != 'auction')][:7]
    except Exception: pass
    return per, sec, top, cal, bool(items)

def _when(it):
    try: return pd.Timestamp(it['created_at']).tz_convert('America/New_York').strftime('%a %H:%M')
    except Exception: return ''

def _wrap(ax, x, y, text, width, size=9.5, color=None, weight='normal', lh=1.45, **kw):
    """write wrapped text at (x, y) in axes coords, return the y below it"""
    lines = []
    for para in text.split('\n'): lines += textwrap.wrap(para, width) or ['']
    for ln in lines:
        ax.text(x, y, ln, fontsize=size, color=color, fontweight=weight, va='top', transform=ax.transAxes, **kw); y -= lh * size / 520
    return y

def _state_word(s): return 'bull' if s == 1 else 'bear' if s == -1 else 'n/a'

def build(LS, rows, meta, outdir, fetch=True):
    os.makedirs(outdir, exist_ok=True)
    BG, PANEL, T1, T2, MUT, GOOD, CRIT, GOLD = LS.BG, LS.PANEL, LS.T1, LS.T2, LS.MUT, LS.GOOD, LS.CRIT, '#c9a227'
    plt.rcParams.update({'text.color': T1, 'font.family': 'DejaVu Sans'})
    held = sorted([r for r in rows if r['action'] in ('HOLD', 'SELL') and r['px']], key=lambda r: -r['pnl'])
    seen = set(); held = [r for r in held if not (r['base'] in seen or seen.add(r['base']))]
    names = [r['base'] for r in held]; now = LS.now_et()
    try: E = _daily_states(LS) if fetch else {}
    except Exception: E = {}
    per, sec, top, cal, have_news = _news(LS, names) if fetch else ({t: [] for t in names}, {}, [], [], False)
    X = {r['base']: (LS.extreme_info(r['H']) or {}) for r in held}
    files = []
    # ---------------------------------------------------------------- cover
    fig = plt.figure(figsize=(11, 8.5), facecolor=BG); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_facecolor(BG)
    ax.add_patch(plt.Rectangle((0.03, 0.03), 0.004, 0.94, color=GOLD, transform=ax.transAxes))
    ax.text(0.06, 0.95, 'Morning report', fontsize=24, fontweight='bold', va='top', color=T1)
    ax.text(0.06, 0.905, f"{now:%A %B %d, %Y} · {now:%H:%M} ET · vector screener · {len(held)} of {meta['slots']} slots", fontsize=10.5, color=T2, va='top')
    # book table
    y = 0.85; ax.text(0.06, y, 'THE BOOK', fontsize=10, color=GOLD, fontweight='bold', va='top'); y -= 0.035
    hdr = ['Ticker', 'P&L', 'Price', 'Entry', 'Stop', 'Stop basis', 'Vol regime', 'Note']; xs = [0.06, 0.15, 0.23, 0.32, 0.41, 0.50, 0.60, 0.71]
    xs = [0.06, 0.14, 0.22, 0.31, 0.40, 0.49, 0.59, 0.70]
    for h_, x_ in zip(hdr, xs): ax.text(x_, y, h_, fontsize=8.5, color=MUT, va='top')
    y -= 0.03
    for r in held:
        x = X[r['base']]; room = r['last'] / r['stop'] - 1
        note = 'BELOW STOP: sells at the open' if room < 0 or r['action'] == 'SELL' else ('profit lock on' if r.get('peak', 0) >= LS.LOCK_K else '')
        vals = [r['base'], f"{r['pnl']:+.1%}", f"{r['last']:.2f}", f"{r['px']:.2f}", f"{r['stop']:.2f}", '1h close' if r['h1'] else '15m close', LS.vol_word(x.get('pct', .5)), note]
        for k, (v, x_) in enumerate(zip(vals, xs)):
            ax.text(x_, y, v, fontsize=10, va='top', fontweight='bold' if k == 0 else 'normal', color=(GOOD if r['pnl'] >= 0 else CRIT) if k == 1 else (CRIT if k == 7 and 'BELOW' in v else T1))
        y -= 0.032
    if not held: ax.text(0.06, y, 'No open positions.', fontsize=10, color=T2, va='top'); y -= 0.032
    # sector ETFs
    y -= 0.02; ytab = y; ax.text(0.06, y, 'SECTORS AND MARKET  (daily + weekly vector)', fontsize=10, color=GOLD, fontweight='bold', va='top'); y -= 0.035
    for h_, x_ in zip(['ETF', 'Sector', 'Daily', 'Weekly', '1 day', '5 days', '1 month'], [0.06, 0.12, 0.26, 0.33, 0.41, 0.48, 0.56]): ax.text(x_, y, h_, fontsize=8.5, color=MUT, va='top')
    y -= 0.028; mine = {LS.SECTOR_ETF.get(LS.sector_of(t)) for t in names}
    order = ['SPY', 'QQQ'] + sorted([t for t in ETFS if t in E], key=lambda t: -E[t]['r5'])
    for t in order:
        if t not in E: continue
        e = E[t]
        row = [t, ETF_NAME[t] + (' ●' if t in mine else ''), _state_word(e['d']), _state_word(e['w']), f"{e['r1']:+.1%}", f"{e['r5']:+.1%}", f"{e['r21']:+.1%}"]
        for k, (v, x_) in enumerate(zip(row, [0.06, 0.12, 0.26, 0.33, 0.41, 0.48, 0.56])):
            col = T1 if k < 2 else (GOOD if v == 'bull' else CRIT if v == 'bear' else (GOOD if v.startswith('+') else CRIT))
            ax.text(x_, y, v, fontsize=9, va='top', color=col, fontweight='bold' if k == 0 else 'normal')
        y -= 0.024
        if t == 'QQQ': y -= 0.008
    _wrap(ax, 0.06, y - 0.005, '● = a sector you hold. Vector states use completed sessions. The sector vector is context only: skipping or ranking flips by it did not help (tested Oct 25).', 100, size=7.5, color=MUT)
    # right column: calendar + top news
    yy = ytab; ax.text(0.68, yy, "TODAY'S CALENDAR (US)", fontsize=10, color=GOLD, fontweight='bold', va='top'); yy -= 0.035
    if cal:
        for e in cal:
            t = pd.Timestamp(e['datetime_utc']).tz_convert('America/New_York')
            yy = _wrap(ax, 0.68, yy, f"{t:%a %H:%M}  {e.get('title', '')}" + (f"  (prev {e['previous']})" if e.get('previous') else ''), 46, size=8.5, color=CRIT if e.get('impact') == 'High' else T2)
    else: yy = _wrap(ax, 0.68, yy, 'No high-impact US releases today.' if have_news else 'Calendar unavailable.', 46, size=8.5, color=T2)
    yy -= 0.02; ax.text(0.68, yy, 'TOP MARKET NEWS', fontsize=10, color=GOLD, fontweight='bold', va='top'); yy -= 0.035
    for it in top[:3]:
        if yy < 0.16: break
        yy = _wrap(ax, 0.68, yy, f"{_when(it)} · {it.get('title', '')}", 46, size=8.5, color=T1)
        if it.get('helious_read'): yy = _wrap(ax, 0.69, yy, it['helious_read'], 46, size=7.8, color=MUT)
        yy -= 0.008
    if not top: yy = _wrap(ax, 0.68, yy, 'No headlines (add HELIOUS_API_KEY; the free feed runs 60 minutes behind).', 46, size=8.5, color=MUT)
    ax.text(0.06, 0.04, 'Rules decide every buy and sell; this report is context. Sources: screener bars and state machine, HAR range model, Yahoo daily ETFs, Helious news.', fontsize=7.5, color=MUT)
    f0 = os.path.join(outdir, '0_cover.png'); fig.savefig(f0, dpi=110, facecolor=BG); plt.close(fig); files.append(f0)
    # ---------------------------------------------------------------- one page per position
    for k, r in enumerate(held):
        x = X[r['base']]; H = r['H']; t = r['base']; sig = x.get('sig', np.nan)
        fig = plt.figure(figsize=(11, 8.5), facecolor=BG); bg = fig.add_axes([0, 0, 1, 1]); bg.axis('off')
        bg.add_patch(plt.Rectangle((0.03, 0.03), 0.004, 0.94, color=GOOD if r['pnl'] >= 0 else CRIT, transform=bg.transAxes))
        bg.text(0.06, 0.95, f"{t}", fontsize=24, fontweight='bold', va='top'); bg.text(0.16, 0.945, f"{r['pnl']:+.1%}", fontsize=20, va='top', color=GOOD if r['pnl'] >= 0 else CRIT)
        et = H.dt.iloc[r['e']] if r['e'] is not None else None
        bg.text(0.06, 0.905, f"{LS.sector_of(t)} · price {r['last']:.2f} · entry {r['px']:.2f}" + (f" ({et:%a %b %d %H:%M})" if et is not None else '') +
                f" · stop {r['stop']:.2f} on the {'1h' if r['h1'] else '15m'} close · peak {r.get('peak', 0):+.1%}", fontsize=10, color=T2, va='top')
        ax = fig.add_axes([0.06, 0.56, 0.56, 0.31]); LS._chart(ax, H, r['e'], r['px'], r['stop'], 130, big=True, h1=r['h1']); ax.tick_params(colors=MUT, labelsize=8)
        dstate = _stock_daily_state(LS, H); etf = LS.SECTOR_ETF.get(LS.sector_of(t)); es = E.get(etf, {})
        try:
            ev = [e for e in LS.earn_events(t) if not e[2] and e[0] > now] if fetch else []
        except Exception: ev = []
        nxt = min(ev, key=lambda e: e[0])[0] if ev else None
        # model read (right column)
        y = 0.87; bg.text(0.66, y, 'MODEL READ', fontsize=10, color=GOLD, fontweight='bold', va='top'); y -= 0.035
        lines = [f"15m vector: {_state_word(int(H.reg.iloc[-1]))}", f"1h vector: {_state_word(int(H.s1.iloc[-1]))}", f"Daily vector: {_state_word(dstate)}",
                 f"Sector {etf}: daily {_state_word(es.get('d'))}, weekly {_state_word(es.get('w'))}" if etf else 'Sector ETF: n/a',
                 f"Expected day range: ~{1.665 * sig:.1%} ({LS.vol_word(x.get('pct', .5))} vol, {x.get('pct', .5):.0%} pct)" if sig == sig else 'Expected range: n/a',
                 f"Next earnings: {nxt:%a %b %d %H:%M}" if nxt is not None else 'Next earnings: none in the calendar']
        for ln in lines: y = _wrap(bg, 0.66, y, ln, 44, size=9.5, color=T1)
        if x.get('live') and (x.get('low') or x.get('high')):
            for side in ('low', 'high'):
                e_ = x.get(side)
                if e_: y = _wrap(bg, 0.66, y, f"Today's {side} looks in ({e_['p']:.0%} it holds): other side ~{e_['tgt'][0]:.2f} typical, 80% band {e_['tgt'][1]:.2f}", 44, size=9, color=GOOD)
        # the write-up (bottom, full width)
        room = r['last'] / r['stop'] - 1
        why = (f"Bought on a {'1h' if r['track'] == '1h' else '15m'} vector flip" + (f" on {et:%a %b %d} at {r['px']:.2f}" if et is not None else '') +
               f". Since then the trade has peaked at {r.get('peak', 0):+.1%} and sits at {r['pnl']:+.1%}. " +
               ("It moved past +2% with a bullish 1h vector, so it now trails the 1h stop (sticky), which gives a trend room to breathe. " if r['h1'] else "It trails the 15m stop until it is up 2% with a bullish 1h vector. ") +
               (f"The profit lock is on: if it gives back half of the peak gain it exits at the next open. " if r.get('peak', 0) >= LS.LOCK_K else ''))
        want = ((f"Closes holding above {r['stop']:.2f} ({room:.1%} below the last price). " if room >= 0 else f"Price ({r['last']:.2f}) is already under the stop ({r['stop']:.2f}); only a reclaim before the open would keep it. ")
                + f"The {'1h' if r['h1'] else '15m'} vector staying bull. "
                + (f"A normal day for {t} spans about {1.665 * sig:.1%}; swings inside that are noise, not information. " if sig == sig else '')
                + (f"{etf} holding its daily trend helps the backdrop. " if etf and es.get('d') == 1 else f"{etf} is in a daily downtrend, so this is a stock-specific move: it needs to keep outperforming its sector. " if etf else ''))
        fail = (f"RULE: a {'1h' if r['h1'] else '15m'} close below {r['stop']:.2f} sells at the next open" + (" (the stop is already broken: it sells at the 09:30 open)." if room < 0 or r['action'] == 'SELL' else '.') +
                " EARLY WARNINGS (information, not rules): the 15m vector flipping bear while the 1h holds; the daily vector turning bear; "
                + (f"{etf}'s daily vector flipping bear; " if etf else '') + "a negative headline on the name (Helious score -3 or worse)"
                + ('; a high-impact US release today (' + ', '.join(e.get('title', '')[:30] for e in cal if e.get('impact') == 'High')[:120] + ') can gap the whole tape' if any(e.get('impact') == 'High' for e in cal) else '')
                + (f"; earnings on {nxt:%b %d}: the earnings guard sells at the close before it if the trade is up 3%+ but less than its typical earnings move" if nxt is not None and (nxt - now).days <= 10 else '') + '.')
        yb = 0.515
        for head, body in (('WHY WE HOLD', why), ('WHAT WE WANT TO SEE', want), ('WHAT WOULD SAY IT FAILED', fail)):
            bg.text(0.06, yb, head, fontsize=9.5, color=GOLD, fontweight='bold', va='top'); yb = _wrap(bg, 0.06, yb - 0.026, body, 100, size=8.6, color=T1, lh=1.32) - 0.010
        # news column
        yn = y - 0.02; bg.text(0.66, yn, 'NEWS', fontsize=10, color=GOLD, fontweight='bold', va='top'); yn -= 0.035
        its = per.get(t, []) + sec.get(t, [])
        for it in its[:4]:
            tag = t if it in per.get(t, []) else 'sector'
            d_ = it.get('direction', '');
            yn = _wrap(bg, 0.66, yn, f"[{tag}] {_when(it)} · {it.get('title', '')}" + (f" ({d_.lower()})" if d_ and d_ != 'NEUTRAL' else ''), 44, size=8.3,
                       color=CRIT if (it.get('score') or 0) <= -3 else GOOD if (it.get('score') or 0) >= 3 else T1) - 0.006
        if not its: yn = _wrap(bg, 0.66, yn, 'No headlines on the name or its sector in the feed window.' if have_news else 'News unavailable.', 44, size=8.5, color=MUT)
        fp = os.path.join(outdir, f'{k + 1}_{t}.png'); fig.savefig(fp, dpi=110, facecolor=BG); plt.close(fig); files.append(fp)
    return files
