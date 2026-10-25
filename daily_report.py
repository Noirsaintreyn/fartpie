"""Morning report (Oct 25 2026): a written report of the book, built once at 09:00 ET.
  analysis()  -> a dict (market overview + one write-up per position); saved to live/analysis.json, served on the website
  build()     -> the same report as PNG pages for Discord (09:00 post and /brief)
Cover: the book, sector ETFs with their daily + weekly trend, SPY / QQQ, today's US calendar and top market news.
One section per position: the model read (15m / 1h / daily trend, HAR range, earnings), why we hold, what we want to see,
why it would fail, and the news on the name (Massive, with sentiment) and its sector (Helious).
The expected day range is not frozen here: the website recomputes it on every scan from today's extremes so far."""
import os, re, json, textwrap, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ETFS = ['XLK', 'XLC', 'XLY', 'XLF', 'XLV', 'XLP', 'XLI', 'XLE', 'XLU', 'XLB', 'XLRE']
ETF_NAME = {'XLK': 'Technology', 'XLC': 'Communication', 'XLY': 'Discretionary', 'XLF': 'Financials', 'XLV': 'Health Care', 'XLP': 'Staples', 'XLI': 'Industrials',
            'XLE': 'Energy', 'XLU': 'Utilities', 'XLB': 'Materials', 'XLRE': 'Real Estate', 'SPY': 'S&P 500', 'QQQ': 'Nasdaq 100'}


# ================================================================ data
def _daily_states(LS):
    """daily + weekly trend state, 1d / 5d / 1m returns for the sector ETFs, SPY, QQQ (completed sessions only)"""
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
    D = R.groupby(R.dt.dt.normalize()).agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).reset_index()
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

def _state_word(s): return 'up' if s == 1 else 'down' if s == -1 else 'n/a'
def _trend(s): return 'rising' if s == 1 else 'falling' if s == -1 else 'unclear'
def _pc(v): return f"{v:+.1%}".replace('-', '−')


# ================================================================ the written analysis
def _market_text(E, mine, cal, top, have_news):
    if not E: return 'Market data was unavailable this morning.'
    s, q = E.get('SPY'), E.get('QQQ'); p = []
    if s and q:
        both = s['d'] == q['d'] == s['w'] == q['w']
        if both: p.append(f"The broad market is {'trending higher' if s['d'] == 1 else 'under pressure'} on both the daily and weekly charts: the S&P 500 is {_pc(s['r5'])} over five sessions and the Nasdaq 100 {_pc(q['r5'])}.")
        else: p.append(f"The S&P 500 is {_trend(s['d'])} on the daily chart and {_trend(s['w'])} on the weekly ({_pc(s['r5'])} over five sessions); the Nasdaq 100 is {_trend(q['d'])} daily and {_trend(q['w'])} weekly ({_pc(q['r5'])}).")
    sec = sorted([t for t in ETFS if t in E], key=lambda t: -E[t]['r5'])
    if len(sec) >= 4:
        lead = ' and '.join(f"{ETF_NAME[t].lower()} ({_pc(E[t]['r5'])})" for t in sec[:2]); lag = ' and '.join(f"{ETF_NAME[t].lower()} ({_pc(E[t]['r5'])})" for t in sec[-2:][::-1])
        up = sum(E[t]['d'] == 1 for t in sec)
        p.append(f"Over the past week leadership has come from {lead}, while {lag} have lagged; {up} of {len(sec)} sectors are in a daily uptrend.")
    held = [t for t in mine if t in E]
    if held:
        p.append('Among the sectors we are exposed to, ' + '; '.join(f"{ETF_NAME[t].lower()} is {_trend(E[t]['d'])} on the daily and {_trend(E[t]['w'])} on the weekly" for t in held) + '.')
    hi = [e for e in cal if e.get('impact') == 'High']
    if hi: p.append('The calendar carries ' + ', '.join(f"{e.get('title', '')} at {pd.Timestamp(e['datetime_utc']).tz_convert('America/New_York'):%H:%M}" for e in hi[:4]) + ', any of which can move the whole tape.')
    elif have_news: p.append('There are no high-impact US releases on the calendar today.')
    return ' '.join(p)

def _position(LS, r, x, E, per, sec, cal, now, fetch):
    H = r['H']; t = r['base']; sig = x.get('sig', np.nan)
    et = H.dt.iloc[r['e']] if r['e'] is not None else None
    dstate = _stock_daily_state(LS, H); etf = LS.SECTOR_ETF.get(LS.sector_of(t)); es = E.get(etf, {})
    since = min(now - pd.Timedelta(days=5), (et - pd.Timedelta(days=2)) if et is not None else now)
    allnews = [dict(a_, ts=pd.Timestamp(a_['t']).tz_convert('America/New_York').tz_localize(None)) for a_ in (LS.stock_news(t, since, limit=100) if fetch else []) if a_.get('n', 9) <= 4]
    sn = [a_ for a_ in allnews if now - pd.Timedelta(days=5) <= a_['ts'] <= now]                  # one call per name (free plan: 5 a minute)
    sn = sorted(sn, key=lambda a_: (a_['sent'] in (None, 'neutral'), a_.get('n', 9), -a_['ts'].value))      # opinionated, on-name, recent first
    npos, nneg = sum(a_['sent'] == 'positive' for a_ in sn), sum(a_['sent'] == 'negative' for a_ in sn); nneu = len(sn) - npos - nneg
    ent = [a_ for a_ in allnews if et is not None and et - pd.Timedelta(days=2) <= a_['ts'] <= et + pd.Timedelta(hours=6)]
    try: ev = [e for e in LS.earn_events(t) if not e[2] and e[0] > now] if fetch else []
    except Exception: ev = []
    nxt = min(ev, key=lambda e: e[0])[0] if ev else None
    room = r['last'] / r['stop'] - 1; basis = 'hourly' if r['h1'] else '15-minute'; pk = r.get('peak', 0) or 0
    below = room < 0 or r['action'] == 'SELL'
    # why we hold
    why = (f"We bought {t} on {et:%A, %B %d} at {r['px']:.2f}, when its short-term trend turned higher." if et is not None else f"We bought {t} at {r['px']:.2f}, when its short-term trend turned higher.")
    why += f" The position has reached {_pc(pk)} at its best and stands at {_pc(r['pnl'])} this morning."
    why += (" Having cleared +2% with the hourly trend confirming, the stop now trails on hourly closes, which gives an established trend room to develop."
            if r['h1'] else " Until the trade clears +2% with the hourly trend confirming, the stop trails on 15-minute closes.")
    if pk >= LS.LOCK_K: why += f" Because it ran more than 5% at its best, a profit lock is in force: surrendering half of the peak gain closes the position at the next open."
    if ent: why += f" Around the entry, {ent[0]['pub'] or 'the press'} ran “{ent[0]['title'][:120]}”" + (f", read as {ent[0]['sent']} for the stock." if ent[0]['sent'] else '.')
    # what we want to see
    want = (f"We want closes to hold above {r['stop']:.2f}, {room:.1%} beneath the last price, and a pattern of higher lows on the {basis} chart." if not below
            else f"The price ({r['last']:.2f}) is already beneath the stop ({r['stop']:.2f}); only a recovery above it before the open would keep the trade alive.")
    if sig == sig: want += f" A normal session for {t} spans about {1.665 * sig:.1%}, so moves inside that band are noise rather than signal."
    if etf: want += (f" A rising {ETF_NAME.get(etf, etf).lower()} sector ({etf}) is a tailwind." if es.get('d') == 1 else f" Its sector ({etf}) is falling on the daily chart, so {t} must keep outperforming its peers.")
    if npos > nneg: want += f" Coverage this week has leaned positive ({npos} favourable articles against {nneg} unfavourable)."
    elif nneg > npos: want += f" Coverage this week has leaned negative ({nneg} against {npos}), so price is carrying the trade against the headlines."
    # why it would fail
    fail = f"The trade ends on {'an hourly' if r['h1'] else 'a 15-minute'} close below {r['stop']:.2f}, which sells at the next open" + (" — it is already below that level, so it sells at 09:30." if below else '.')
    warn = ["the 15-minute trend rolling over while the hourly holds", "the daily trend turning lower"]
    if etf: warn.append(f"{etf} turning down on the daily chart")
    warn.append('negative news on the name')
    hi = [e.get('title', '')[:40] for e in cal if e.get('impact') == 'High']
    if hi: warn.append(f"today's high-impact releases ({', '.join(hi[:3])}), which can gap the whole market")
    if nxt is not None and (nxt - now).days <= 10: warn.append(f"earnings on {nxt:%B %d}; if the stock is up 3% or more but by less than its usual earnings move, it is sold at the prior close")
    fail += ' Earlier warning signs would be ' + ', '.join(warn[:-1]) + (', and ' if len(warn) > 1 else '') + warn[-1] + '.'
    news = [dict(t=f"{pd.Timestamp(a_['t']).tz_convert('America/New_York'):%a %H:%M}", title=a_['title'], pub=a_.get('pub', ''), sent=a_['sent'], why=(a_.get('why') or '')[:220],
                 url=a_.get('url', '')) for a_ in sn[:4]]
    news += [dict(t=_when(it), title=it.get('title', ''), pub='sector' if it in sec.get(t, []) else 'name', sent='negative' if (it.get('score') or 0) <= -3 else None, why=it.get('helious_read', '') or '', url='')
             for it in (per.get(t, []) + sec.get(t, []))[:2]]
    return dict(tk=t, pnl=round(float(r['pnl']) * 100, 2), last=round(float(r['last']), 2), entry=round(float(r['px']), 2), stop=round(float(r['stop']), 2),
                basis=basis, peak=round(pk * 100, 2), sector=LS.sector_of(t), etf=etf, entered=f'{et:%a %b %d %H:%M}' if et is not None else '',
                selling=bool(below), lock=bool(pk >= LS.LOCK_K),
                read=dict(t15=_state_word(int(H.reg.iloc[-1])), t1h=_state_word(int(H.s1.iloc[-1])), td=_state_word(dstate),
                          sec_d=_state_word(es.get('d')), sec_w=_state_word(es.get('w')), vol=LS.vol_word(x.get('pct', .5)), vol_pct=round(float(x.get('pct', .5)) * 100),
                          day_range=round(1.665 * sig * 100, 2) if sig == sig else None, earnings=f'{nxt:%a %b %d}' if nxt is not None else None),
                why=why, want=want, fail=fail, news=news, counts=dict(pos=npos, neg=nneg, neu=nneu), _sn=sn, _ent=ent, _sig=sig)

def analysis(LS, rows, meta, fetch=True):
    held = sorted([r for r in rows if r['action'] in ('HOLD', 'SELL') and r['px']], key=lambda r: -r['pnl'])
    seen = set(); held = [r for r in held if not (r['base'] in seen or seen.add(r['base']))]
    names = [r['base'] for r in held]; now = LS.now_et()
    try: E = _daily_states(LS) if fetch else {}
    except Exception: E = {}
    per, sec, top, cal, have_news = _news(LS, names) if fetch else ({t: [] for t in names}, {}, [], [], False)
    X = {r['base']: (LS.extreme_info(r['H']) or {}) for r in held}
    mine = []
    for t in names:
        e = LS.SECTOR_ETF.get(LS.sector_of(t))
        if e and e not in mine: mine.append(e)
    P = [_position(LS, r, X[r['base']], E, per, sec, cal, now, fetch) for r in held]
    A = dict(asof=now.isoformat(), date=f'{now:%A, %B %d, %Y}', time=f'{now:%H:%M}', slots=meta['slots'], n=len(held),
             market=_market_text(E, mine, cal, top, have_news),
             sectors=[dict(etf=t, name=ETF_NAME[t], d=_state_word(E[t]['d']), w=_state_word(E[t]['w']), r1=round(E[t]['r1'] * 100, 2), r5=round(E[t]['r5'] * 100, 2),
                           r21=round(E[t]['r21'] * 100, 2), mine=t in mine) for t in ['SPY', 'QQQ'] + sorted([t for t in ETFS if t in E], key=lambda t: -E[t]['r5']) if t in E],
             calendar=[dict(t=f"{pd.Timestamp(e['datetime_utc']).tz_convert('America/New_York'):%H:%M}", title=e.get('title', ''), impact=e.get('impact', ''), prev=e.get('previous') or '') for e in cal],
             headlines=[dict(t=_when(it), title=it.get('title', ''), read=it.get('helious_read', '') or '') for it in top[:3]],
             positions=P)
    return A, held, X, E, cal, top, have_news

def save(A, path):
    clean = dict(A, positions=[{k: v for k, v in p.items() if not k.startswith('_')} for p in A['positions']])
    tmp = path + '.tmp'; json.dump(clean, open(tmp, 'w'), default=str); os.replace(tmp, path)


# ================================================================ PNG pages (Discord)
def _wrap(ax, x, y, text, width, size=9.5, color=None, weight='normal', lh=1.45, **kw):
    """write wrapped text at (x, y) in axes coords, return the y below it"""
    lines = []
    for para in text.split('\n'): lines += textwrap.wrap(para, width) or ['']
    for ln in lines:
        ax.text(x, y, ln, fontsize=size, color=color, fontweight=weight, va='top', transform=ax.transAxes, **kw); y -= lh * size / 520
    return y

def build(LS, rows, meta, outdir, fetch=True):
    os.makedirs(outdir, exist_ok=True)
    BG, PANEL, T1, T2, MUT, GOOD, CRIT, GOLD = LS.BG, LS.PANEL, LS.T1, LS.T2, LS.MUT, LS.GOOD, LS.CRIT, '#c9a227'
    plt.rcParams.update({'text.color': T1, 'font.family': 'DejaVu Sans'})
    A, held, X, E, cal, top, have_news = analysis(LS, rows, meta, fetch)
    save(A, os.path.join(outdir, 'analysis.json'))
    now = LS.now_et(); files = []
    # ---------------------------------------------------------------- cover
    fig = plt.figure(figsize=(11, 8.5), facecolor=BG); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_facecolor(BG)
    ax.add_patch(plt.Rectangle((0.03, 0.03), 0.004, 0.94, color=GOLD, transform=ax.transAxes))
    ax.text(0.06, 0.95, 'Morning report', fontsize=24, fontweight='bold', va='top', color=T1)
    ax.text(0.06, 0.905, f"{now:%A %B %d, %Y} · {now:%H:%M} ET · screener · {len(held)} of {meta['slots']} slots", fontsize=10.5, color=T2, va='top')
    y = 0.85; ax.text(0.06, y, 'THE BOOK', fontsize=10, color=GOLD, fontweight='bold', va='top'); y -= 0.035
    hdr = ['Ticker', 'P&L', 'Price', 'Entry', 'Stop', 'Stop basis', 'Vol regime', 'Note']; xs = [0.06, 0.14, 0.22, 0.31, 0.40, 0.49, 0.59, 0.70]
    for h_, x_ in zip(hdr, xs): ax.text(x_, y, h_, fontsize=8.5, color=MUT, va='top')
    y -= 0.03
    for r in held:
        x = X[r['base']]; room = r['last'] / r['stop'] - 1
        note = 'BELOW STOP: sells at the open' if room < 0 or r['action'] == 'SELL' else ('profit lock on' if r.get('peak', 0) >= LS.LOCK_K else '')
        vals = [r['base'], f"{r['pnl']:+.1%}", f"{r['last']:.2f}", f"{r['px']:.2f}", f"{r['stop']:.2f}", 'hourly close' if r['h1'] else '15-min close', LS.vol_word(x.get('pct', .5)), note]
        for k, (v, x_) in enumerate(zip(vals, xs)):
            ax.text(x_, y, v, fontsize=10, va='top', fontweight='bold' if k == 0 else 'normal', color=(GOOD if r['pnl'] >= 0 else CRIT) if k == 1 else (CRIT if k == 7 and 'BELOW' in v else T1))
        y -= 0.032
    if not held: ax.text(0.06, y, 'No open positions.', fontsize=10, color=T2, va='top'); y -= 0.032
    y -= 0.02; ytab = y; ax.text(0.06, y, 'SECTORS AND MARKET  (daily + weekly trend)', fontsize=10, color=GOLD, fontweight='bold', va='top'); y -= 0.035
    for h_, x_ in zip(['ETF', 'Sector', 'Daily', 'Weekly', '1 day', '5 days', '1 month'], [0.06, 0.12, 0.26, 0.33, 0.41, 0.48, 0.56]): ax.text(x_, y, h_, fontsize=8.5, color=MUT, va='top')
    y -= 0.028
    for s in A['sectors']:
        row = [s['etf'], s['name'] + (' ●' if s['mine'] else ''), s['d'], s['w'], f"{s['r1']:+.1f}%", f"{s['r5']:+.1f}%", f"{s['r21']:+.1f}%"]
        for k, (v, x_) in enumerate(zip(row, [0.06, 0.12, 0.26, 0.33, 0.41, 0.48, 0.56])):
            col = T1 if k < 2 else (GOOD if v == 'up' else CRIT if v == 'down' else (GOOD if v.startswith('+') else CRIT))
            ax.text(x_, y, v, fontsize=9, va='top', color=col, fontweight='bold' if k == 0 else 'normal')
        y -= 0.024
        if s['etf'] == 'QQQ': y -= 0.008
    _wrap(ax, 0.06, y - 0.005, '● = a sector you hold. Trends use completed sessions. The sector trend is context only: filtering trades by it did not help (tested Oct 25).', 100, size=7.5, color=MUT)
    yy = ytab; ax.text(0.68, yy, "TODAY'S CALENDAR (US)", fontsize=10, color=GOLD, fontweight='bold', va='top'); yy -= 0.035
    if cal:
        for e in A['calendar']:
            yy = _wrap(ax, 0.68, yy, f"{e['t']}  {e['title']}" + (f"  (prev {e['prev']})" if e['prev'] else ''), 46, size=8.5, color=CRIT if e['impact'] == 'High' else T2)
    else: yy = _wrap(ax, 0.68, yy, 'No high-impact US releases today.' if have_news else 'Calendar unavailable.', 46, size=8.5, color=T2)
    yy -= 0.02; ax.text(0.68, yy, 'TOP MARKET NEWS', fontsize=10, color=GOLD, fontweight='bold', va='top'); yy -= 0.035
    for it in A['headlines']:
        if yy < 0.16: break
        yy = _wrap(ax, 0.68, yy, f"{it['t']} · {it['title']}", 46, size=8.5, color=T1)
        if it['read']: yy = _wrap(ax, 0.69, yy, it['read'], 46, size=7.8, color=MUT)
        yy -= 0.008
    if not A['headlines']: yy = _wrap(ax, 0.68, yy, 'No headlines (add HELIOUS_API_KEY; the free feed runs 60 minutes behind).', 46, size=8.5, color=MUT)
    ax.text(0.06, 0.04, 'Rules decide every buy and sell; this report is context. Sources: screener price data, HAR range model, Yahoo daily ETFs, Massive + Helious news.', fontsize=7.5, color=MUT)
    f0 = os.path.join(outdir, '0_cover.png'); fig.savefig(f0, dpi=110, facecolor=BG); plt.close(fig); files.append(f0)
    # ---------------------------------------------------------------- one page per position
    for k, (r, p) in enumerate(zip(held, A['positions'])):
        x = X[r['base']]; H = r['H']; t = r['base']; rd = p['read']; sn = p['_sn']
        fig = plt.figure(figsize=(11, 8.5), facecolor=BG); bg = fig.add_axes([0, 0, 1, 1]); bg.axis('off')
        bg.add_patch(plt.Rectangle((0.03, 0.03), 0.004, 0.94, color=GOOD if r['pnl'] >= 0 else CRIT, transform=bg.transAxes))
        bg.text(0.06, 0.95, t, fontsize=24, fontweight='bold', va='top'); bg.text(0.16, 0.945, f"{r['pnl']:+.1%}", fontsize=20, va='top', color=GOOD if r['pnl'] >= 0 else CRIT)
        bg.text(0.06, 0.905, f"{p['sector']} · price {r['last']:.2f} · entry {r['px']:.2f}" + (f" ({p['entered']})" if p['entered'] else '') +
                f" · stop {r['stop']:.2f} on {'an hourly' if r['h1'] else 'a 15-minute'} close · best {r.get('peak', 0):+.1%}", fontsize=10, color=T2, va='top')
        ax = fig.add_axes([0.06, 0.56, 0.56, 0.31]); LS._chart(ax, H, r['e'], r['px'], r['stop'], 130, big=True, h1=r['h1']); ax.tick_params(colors=MUT, labelsize=8)
        y = 0.87; bg.text(0.66, y, 'MODEL READ', fontsize=10, color=GOLD, fontweight='bold', va='top'); y -= 0.035
        lines = [f"Short-term trend (15-min): {rd['t15']}", f"Hourly trend: {rd['t1h']}", f"Daily trend: {rd['td']}",
                 f"Sector {p['etf']}: daily {rd['sec_d']}, weekly {rd['sec_w']}" if p['etf'] else 'Sector ETF: n/a',
                 f"Volatility: {rd['vol']} ({rd['vol_pct']}% percentile of the last year)", f"Next earnings: {rd['earnings']}" if rd['earnings'] else 'Next earnings: none in the calendar']
        for ln in lines: y = _wrap(bg, 0.66, y, ln, 44, size=9.5, color=T1)
        rl = LS.range_levels(H)
        if rl:
            y = _wrap(bg, 0.66, y, f"Range budget today: ~{rl['budget']:.1%} (wide {2.4 * rl['sig']:.1%}), {'premarket' if rl['pre'] else 'today'} so far {rl['lo']:.2f} - {rl['hi']:.2f}", 44, size=9, color=T1)
            y = _wrap(bg, 0.66, y, f"From the low so far: expected high ~{rl['hi_typ']:.2f} (wide {rl['hi_wide']:.2f})", 44, size=9, color=GOOD)
            y = _wrap(bg, 0.66, y, f"From the high so far: expected low ~{rl['lo_typ']:.2f} (wide {rl['lo_wide']:.2f})", 44, size=9, color=CRIT)
            y = _wrap(bg, 0.66, y, "(the range stays the same size: a lower low moves the expected high down by the same amount)", 50, size=7.5, color=MUT)
        if x.get('live') and (x.get('low') or x.get('high')):
            for side in ('low', 'high'):
                e_ = x.get(side)
                if e_: y = _wrap(bg, 0.66, y, f"Today's {side} looks in ({e_['p']:.0%} it holds): other side ~{e_['tgt'][0]:.2f} typical, 80% band {e_['tgt'][1]:.2f}", 44, size=9, color=GOOD)
        yb = 0.515
        for head, body in (('WHY WE HOLD', p['why']), ('WHAT WE WANT TO SEE', p['want']), ('WHY IT WOULD FAIL', p['fail'])):
            bg.text(0.06, yb, head, fontsize=9.5, color=GOLD, fontweight='bold', va='top'); yb = _wrap(bg, 0.06, yb - 0.026, body, 100, size=8.6, color=T1, lh=1.32) - 0.010
        c = p['counts']; yn = y - 0.02; bg.text(0.66, yn, f"NEWS (5 days): {c['pos']} positive · {c['neg']} negative · {c['neu']} neutral", fontsize=9, color=GOLD, fontweight='bold', va='top'); yn -= 0.032
        for a_ in p['news']:
            if yn < 0.12: break
            c_ = GOOD if a_['sent'] == 'positive' else CRIT if a_['sent'] == 'negative' else T1
            tag = f"[{a_['pub']}] " if a_['pub'] in ('name', 'sector') else ''
            yn = _wrap(bg, 0.66, yn, f"{tag}{a_['t']} · {a_['title']}", 46, size=8.2, color=c_)
            if a_['why'] and not tag: yn = _wrap(bg, 0.67, yn, a_['why'][:150] + ('…' if len(a_['why']) > 150 else ''), 50, size=7.4, color=MUT)
            yn -= 0.006
        if not p['news']: _wrap(bg, 0.66, yn, 'No recent articles on the name or its sector.', 46, size=8.5, color=MUT)
        fp = os.path.join(outdir, f'{k + 1}_{t}.png'); fig.savefig(fp, dpi=110, facecolor=BG); plt.close(fig); files.append(fp)
    return files
