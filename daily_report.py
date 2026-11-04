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
def _clip(t, n): return t if len(t) <= n else t[:n].rsplit(' ', 1)[0].rstrip(',;:') + '…'
def _pc(v): return f"{v:+.1%}".replace('-', '−')


# ================================================================ the written analysis
def _market_text(E, mine, cal, top, have_news):
    if not E: return 'Market data was unavailable this morning.'
    s, q = E.get('SPY'), E.get('QQQ'); p = []
    if s and q:
        both = s['d'] == q['d'] == s['w'] == q['w']
        if both:
            soft = (s['d'] == 1 and s['r5'] < 0) or (s['d'] == -1 and s['r5'] > 0)
            p.append(f"The broad market is {'trending higher' if s['d'] == 1 else 'under pressure'} on both the daily and weekly charts" + (", though the past five sessions ran the other way" if soft else '')
                     + f": the S&P 500 is {_pc(s['r5'])} over that stretch and the Nasdaq 100 {_pc(q['r5'])}.")
        else: p.append(f"The S&P 500 is {_trend(s['d'])} on the daily chart and {_trend(s['w'])} on the weekly ({_pc(s['r5'])} over five sessions); the Nasdaq 100 is {_trend(q['d'])} daily and {_trend(q['w'])} weekly ({_pc(q['r5'])}).")
    sec = sorted([t for t in ETFS if t in E], key=lambda t: -E[t]['r5'])
    if len(sec) >= 4:
        lead = ' and '.join(f"{ETF_NAME[t].lower()} ({_pc(E[t]['r5'])})" for t in sec[:2]); lag = ' and '.join(f"{ETF_NAME[t].lower()} ({_pc(E[t]['r5'])})" for t in sec[-2:][::-1])
        up = sum(E[t]['d'] == 1 for t in sec)
        p.append(f"Over the past week leadership has come from {lead}, while {lag} have lagged; {up} of {len(sec)} sectors are in a daily uptrend.")
    held = [t for t in mine if t in E]
    if held:
        grp = {}
        for t in held: grp.setdefault((E[t]['d'], E[t]['w']), []).append(ETF_NAME[t].lower())
        def names(v): return v[0] if len(v) == 1 else ', '.join(v[:-1]) + ' and ' + v[-1]
        parts = []
        for (d, w), v in grp.items():
            verb = 'is' if len(v) == 1 else 'are'
            parts.append(f"{names(v)} {verb} {_trend(d)} on both timeframes" if d == w else f"{names(v)} {verb} {_trend(d)} on the daily chart but {_trend(w)} on the weekly")
        txt = '; '.join(parts); p.append('Of the sectors we hold, ' + txt + '.')
    hi = [e for e in cal if e.get('impact') == 'High' and e.get('type') != 'auction']
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
    hi = [e.get('title', '')[:40] for e in cal if e.get('impact') == 'High' and e.get('type') != 'auction']
    if hi: warn.append(f"today's high-impact releases ({', '.join(hi[:3])}), which can gap the whole market")
    if nxt is not None and (nxt - now).days <= 10: warn.append(f"earnings on {nxt:%B %d}; if the stock is up 3% or more but by less than its usual earnings move, it is sold at the prior close")
    fail += ' Earlier warning signs would be ' + ', '.join(warn[:-1]) + (', and ' if len(warn) > 1 else '') + warn[-1] + '.'
    news = [dict(t=f"{pd.Timestamp(a_['t']).tz_convert('America/New_York'):%a %H:%M}", title=a_['title'], pub=a_.get('pub', ''), sent=a_['sent'], why=_clip(a_.get('why') or '', 230),
                 url=a_.get('url', '')) for a_ in sn[:4]]
    news += [dict(t=_when(it), title=it.get('title', ''), pub='sector' if it in sec.get(t, []) else 'name', sent='negative' if (it.get('score') or 0) <= -3 else None, why=it.get('helious_read', '') or '', url='')
             for it in (per.get(t, []) + sec.get(t, []))[:2]]
    return dict(tk=t, pnl=round(float(r['pnl']) * 100, 2), last=round(float(r['last']), 2), entry=round(float(r['px']), 2), stop=round(float(r['stop']), 2),
                basis=basis, peak=round(pk * 100, 2), sector=LS.sector_of(t), etf=etf, entered=f'{et:%a %b %d %H:%M}' if et is not None else '', entered_ts=str(et) if et is not None else '', track='1h' if r['h1'] else '15m',
                selling=bool(below), lock=bool(pk >= LS.LOCK_K),
                read=dict(t15=_state_word(int(H.reg.iloc[-1])), t1h=_state_word(int(H.s1.iloc[-1])), td=_state_word(dstate),
                          sec_d=_state_word(es.get('d')), sec_w=_state_word(es.get('w')), vol=LS.vol_word(x.get('pct', .5)), vol_pct=round(float(x.get('pct', .5)) * 100),
                          day_range=round(1.665 * sig * 100, 2) if sig == sig else None, earnings=f'{nxt:%a %b %d}' if nxt is not None else None),
                why=why, want=want, fail=fail, news=news, counts=dict(pos=npos, neg=nneg, neu=nneu), _sn=sn, _ent=ent, _sig=sig)

# ================================================================ expectations (Nov 4 2026): what we are looking for, from what the backtest taught us
HERE = os.path.dirname(os.path.abspath(__file__))
SEMI = {'NVDA', 'AMD', 'AVGO', 'MU', 'LRCX', 'QCOM', 'SMCI', 'AMAT', 'KLAC', 'MRVL', 'ON', 'TXN', 'INTC', 'ARM', 'ADI', 'NXPI', 'MCHP'}
def _tables():
    try: return json.load(open(os.path.join(HERE, 'live_data', 'expect_tables.json')))
    except Exception: return {}

def _regime_now(fetch=True):
    """the market regime the backtest used: SPY above its 200d (prior close) x VIX / VIX3M > 1 (stress); plus SPY 20d direction"""
    if not fetch: return None
    try:
        import yfinance as yf
        d = yf.download(['SPY', '^VIX', '^VIX3M'], period='14mo', interval='1d', progress=False, auto_adjust=True)['Close'].dropna(how='all').ffill()
        d = d.iloc[:-1] if pd.Timestamp.now(tz='America/New_York').hour < 16 and len(d) and d.index[-1].date() == pd.Timestamp.now(tz='America/New_York').date() else d   # completed sessions only
        spy = d['SPY']; up = bool(spy.iloc[-1] > spy.rolling(200).mean().iloc[-1]); vts = float(d['^VIX'].iloc[-1] / d['^VIX3M'].iloc[-1]); m20 = float(spy.iloc[-1] / spy.iloc[-21] - 1)
        name = ('calm bull' if vts <= 1 else 'bull+stress') if up else ('bear calm' if vts <= 1 else 'bear+stress')
        return dict(name=name, spy_above_200d=up, vix_vix3m=round(vts, 3), spy_20d=round(m20 * 100, 2), asof=f'{d.index[-1]:%Y-%m-%d}')
    except Exception: return None

def _ledger_check(LS, held):
    """the book on this report must be the ledger's open set. Two hosts (Mac / Render) each keep a ledger.json; if they differ the books diverge."""
    out = dict(ok=None, note='ledger not readable')
    try:
        LG = {k: v for k, v in LS.ledger_load().items() if not k.startswith('__') and isinstance(v, dict)}
        opn = {(v['tk'], v.get('track', '15m')) for v in LG.values() if v.get('status') == 'open'}
        mine = {(r['base'], '1h' if r['h1'] else '15m') for r in held}
        import hashlib
        raw = open(LS.LEDGER, 'rb').read() if os.path.exists(LS.LEDGER) else b''
        seed = open(os.path.join(HERE, 'live_data', 'ledger_seed.json'), 'rb').read() if os.path.exists(os.path.join(HERE, 'live_data', 'ledger_seed.json')) else b'x'
        t0s = sorted(str(v.get('t0', '')) for v in LG.values())
        miss, extra = sorted(opn - mine), sorted(mine - opn)
        out = dict(ok=not miss and not extra, n=len(LG), n_open=len(opn), first=t0s[0] if t0s else '', last=t0s[-1] if t0s else '', sha=hashlib.sha1(raw).hexdigest()[:8], is_seed=raw == seed,
                   in_ledger_not_book=[f'{a} {b}' for a, b in miss], in_book_not_ledger=[f'{a} {b}' for a, b in extra], path=LS.LEDGER)
        out['note'] = ('ledger and book agree' if out['ok'] else 'BOOK DOES NOT MATCH THE LEDGER') + (' · ledger is the Oct 21 seed: the disk was reset' if out['is_seed'] else '')
    except Exception as ex: out['note'] = f'ledger check failed: {str(ex)[:60]}'
    return out

def _cellstr(c, base):
    if not c: return 'n/a'
    return f"{c['avg']:+.2f}% avg trade, {c['win']:.0f}% win, {c['big']:.1f}% runners (n {c['n']:,})"

def _age_cell(T, track, days):
    best = None
    for d in (15, 7, 3, 1, 0):
        if days >= d: best = d; break
    c = (T.get('age_track', {}).get(track, {}) or {}).get(str(best)) or (T.get('age', {}) or {}).get(str(best)); return best, c

def _expect(LS, P, held, meta, E, now, regime, rows=None, A_market=None):
    """forward-looking section: what each position needs, what the history says about positions like it, what would change the book"""
    T = _tables(); slots = meta['slots']; ratios = {k: v for k, v in (meta.get('ratios') or {}).items() if v == v}
    rank = {k: i + 1 for i, k in enumerate(sorted(ratios, key=lambda k: -ratios[k]))}
    pos = []
    for p in P:
        et = pd.Timestamp(p['entered_ts']) if p.get('entered_ts') else None; age = (now - et).total_seconds() / 86400 if et is not None else 0.0
        b, c = _age_cell(T, p['track'], age)
        stop_gap = p['last'] / p['stop'] - 1; dr = (p['read'].get('day_range') or 0) / 100
        locked = p['stop'] / p['entry'] - 1; pk = p['peak'] / 100; px = p['entry']
        nxt = []
        if p['track'] == '15m': nxt.append(f"+2% ({px * 1.02:.2f}) with the hourly trend bull moves the stop to hourly closes" if p['pnl'] < 2 else "stop moves to hourly closes once the hourly trend confirms bull")
        if pk < LS.LOCK_K: nxt.append(f"+5% ({px * 1.05:.2f}) arms the profit lock")
        else: nxt.append(f"profit lock is on: it closes the trade if price falls to {px * (1 + LS.LOCK_G * pk):.2f} (half the peak gain)")
        e = p['read'].get('earnings'); ed = None
        if e:
            try: ed = (pd.Timestamp(e + f' {now.year}') - now.normalize()).days
            except Exception: ed = None
        ec = None if ed is None else ('<=3d to report' if ed <= 3 else '4-10d' if ed <= 10 else None)
        pos.append(dict(tk=p['tk'], track=p['track'], age_days=round(age, 1), pnl=p['pnl'], peak=p['peak'], stop=p['stop'], last=p['last'], stop_gap_pct=round(stop_gap * 100, 2),
                        stop_gap_ranges=round(stop_gap / dr, 2) if dr else None, locked_pct=round(locked * 100, 2), ratio=round(ratios.get(p['tk'], float('nan')), 2) if p['tk'] in ratios else None,
                        ratio_rank=rank.get(p['tk']), age_bucket=b, age_stats=c, next_steps=nxt, earn_days=ed, earn_cell=(T.get('earn', {}) or {}).get(ec) if ec else None,
                        sector_cell=(T.get('sector', {}) or {}).get('semi') if p['tk'] in SEMI else None, selling=p['selling']))
    w = 1 / slots; invested = len(P) * w
    book = dict(slots=slots, n=len(P), invested=round(invested * 100), open_pnl=round(sum(x['pnl'] for x in pos) * w, 2), locked_if_all_stop=round(sum(x['locked_pct'] for x in pos) * w, 2),
                giveback=round((sum(x['pnl'] for x in pos) - sum(x['locked_pct'] for x in pos)) * w, 2), near_stop=[x['tk'] for x in pos if x['stop_gap_ranges'] is not None and x['stop_gap_ranges'] < 1.0],
                aged=[x['tk'] for x in pos if x['age_days'] >= 7])
    reg = None
    if regime:
        c = (T.get('regime', {}) or {}).get(regime['name']); reg = dict(regime, cell=c, base=T.get('all'))
    # candidates the book would take if a slot opened: flips today + names near a flip, best ratio first
    cand = []
    for r in (rows or []):
        if r.get('action') == 'BUY' and r['base'] not in {x['tk'] for x in pos}: cand.append(dict(tk=r['base'], kind='flip today', ratio=round(ratios.get(r['base'], float('nan')), 2), track=r.get('track', '15m')))
    for r in (rows or []):
        if r.get('track', '15m') == '15m' and r.get('action') == 'FLAT' and r.get('state15') == -1 and r.get('trig') == r.get('trig') and 'no new entries' not in str(r.get('note', '')) and r['base'] in ratios and r['base'] not in {x['tk'] for x in pos}:
            d = r['trig'] / r['last'] - 1 if r.get('last') else None
            if d is not None and 0 < d <= .02: cand.append(dict(tk=r['base'], kind=f'{d:.1%} below its trigger ({r["trig"]:.2f})', ratio=round(ratios[r['base']], 2), track='15m'))
    cand = sorted(cand, key=lambda z: -(z['ratio'] if z['ratio'] == z['ratio'] else 0))[:8]
    return dict(regime=reg, book=book, positions=pos, candidates=cand, full=len(P) >= slots, tables=dict(all=T.get('all'), runners=T.get('runners'), age=T.get('age'), tracks=T.get('tracks'), sector=T.get('sector'), earn=T.get('earn')))

def _expect_lines(X):
    """plain-language expectations for the page / site; every number comes from the backtest tables"""
    T = X['tables']; a = T.get('all') or {}; R = T.get('runners') or {}; L = []
    reg = X.get('regime')
    if reg and reg.get('cell'):
        c = reg['cell']; b = reg['base']; d = c['avg'] - b['avg']
        L.append(('REGIME', f"{reg['name']} (SPY {'above' if reg['spy_above_200d'] else 'below'} its 200d, VIX/VIX3M {reg['vix_vix3m']:.2f}). Entries taken in this regime since 2019: {_cellstr(c, b)} against {b['avg']:+.2f}% overall."
                  + (' Stress inside a bull trend is the one market state that paid in both halves of the sample (2019-21 and 2022-26).' if reg['name'] == 'bull+stress' else ' This is the ordinary state; no market filter has helped, so the rules run unchanged.')))
    bk = X['book']
    L.append(('BOOK', f"{bk['n']} of {bk['slots']} slots, {bk['invested']}% invested, open P&L {bk['open_pnl']:+.2f}% of equity. If every stop were hit today the book would keep {bk['locked_if_all_stop']:+.2f}% (gives back {bk['giveback']:.2f}%)."
              + (f" Closest to a stop (inside one normal day's range): {', '.join(bk['near_stop'])}." if bk['near_stop'] else ' No stop is inside one normal day\'s range.')))
    ag = T.get('age') or {}
    if ag.get('3') and ag.get('7'):
        L.append(('HOW IT PAYS', f"The edge is in holding. A typical trade lasts {R.get('med_days_all', 1):.1f} days and loses ({a.get('med', 0):+.2f}% median); {a.get('win', 0):.0f}% win. Trades still open after 3 days finish green {ag['3']['win']:.0f}% of the time (avg {ag['3']['avg']:+.1f}%); after 7 days {ag['7']['win']:.0f}% (avg {ag['7']['avg']:+.1f}%)."
                  f" Only {R.get('share_ge15', 1.3):.1f}% of trades run +15% or more, they hold about {R.get('med_days_runner', 23):.0f} days, and they carry about {R.get('share_of_pnl', 24):.0f}% of the gross profit. Nothing at entry identifies them, so the stops give every trade room."))
    tk = T.get('tracks') or {}
    if tk.get('1h') and tk.get('15m'): L.append(('TRACKS', f"Entries from the hourly track average {tk['1h']['avg']:+.2f}% with {tk['1h']['big']:.1f}% runners; 15-minute entries {tk['15m']['avg']:+.2f}% with {tk['15m']['big']:.1f}%. Both run together; each beats either alone."))
    if X['full'] and X['candidates']:
        L.append(('IF A SLOT OPENS', 'Slots are full, so new flips are tracked, not bought. If one opens, the first buy is the highest options ratio among confirmed flips: ' + ', '.join(f"{c['tk']} ({c['ratio']:.1f}, {c['kind']})" for c in X['candidates'][:5]) + f". A new entry averages {a.get('avg', 0):+.2f}% and loses most of the time; its value is the chance of a runner, not the first day."))
    elif X['candidates']:
        L.append(('NEXT BUYS', 'Open slots go to the highest options ratio among confirmed flips: ' + ', '.join(f"{c['tk']} ({c['ratio']:.1f}, {c['kind']})" for c in X['candidates'][:5]) + '.'))
    return L

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
    A = dict(asof=now.isoformat(), date=f'{now:%A, %B %d, %Y}', time=f'{now:%H:%M}', slots=meta['slots'], n=len(held), ledger=_ledger_check(LS, held),
             expect=None,
             market=_market_text(E, mine, cal, top, have_news),
             sectors=[dict(etf=t, name=ETF_NAME[t], d=_state_word(E[t]['d']), w=_state_word(E[t]['w']), r1=round(E[t]['r1'] * 100, 2), r5=round(E[t]['r5'] * 100, 2),
                           r21=round(E[t]['r21'] * 100, 2), mine=t in mine) for t in ['SPY', 'QQQ'] + sorted([t for t in ETFS if t in E], key=lambda t: -E[t]['r5']) if t in E],
             calendar=[dict(t=f"{pd.Timestamp(e['datetime_utc']).tz_convert('America/New_York'):%H:%M}", title=e.get('title', ''), impact=e.get('impact', ''), prev=e.get('previous') or '') for e in cal],
             headlines=[dict(t=_when(it), title=it.get('title', ''), read=it.get('helious_read', '') or '') for it in top[:3]],
             positions=P)
    try: A['expect'] = _expect(LS, P, held, meta, E, now, _regime_now(fetch), rows); A['expect']['lines'] = _expect_lines(A['expect'])
    except Exception as ex: A['expect'] = dict(error=str(ex)[:120])
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

def _page_expect(LS, A, outdir, colors):
    BG, T1, T2, MUT, GOOD, CRIT, GOLD = colors; X = A.get('expect') or {}
    if not X or 'positions' not in X: return None
    fig = plt.figure(figsize=(11, 8.5), facecolor=BG); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
    ax.add_patch(plt.Rectangle((0.03, 0.03), 0.004, 0.94, color=GOLD, transform=ax.transAxes))
    ax.text(0.06, 0.95, 'What to expect today', fontsize=22, fontweight='bold', va='top', color=T1)
    ax.text(0.06, 0.893, 'Every figure is from the 17,447-trade backtest of these exact rules (S&P, options ratio >= 1.5, 2019-26). Context, not a signal: the rules still decide every buy and sell.', fontsize=8.3, color=T2, va='top')
    y = 0.855
    for head, body in X.get('lines', []):
        ax.text(0.06, y, head, fontsize=8.8, color=GOLD, fontweight='bold', va='top'); y = _wrap(ax, 0.17, y, body, 112, size=8.6, color=T1, lh=1.34) - 0.011
    y -= 0.008; ax.text(0.06, y, 'EACH POSITION', fontsize=10, color=GOLD, fontweight='bold', va='top'); y -= 0.03
    hdr = ['Position', 'Age', 'P&L', 'Stop gap', 'If stopped', 'Ratio (rank)', 'Green at this age', 'Earnings']; xs = [0.06, 0.14, 0.20, 0.26, 0.42, 0.51, 0.60, 0.85]
    for h_, x_ in zip(hdr, xs): ax.text(x_, y, h_, fontsize=8, color=MUT, va='top')
    y -= 0.026
    for p in X['positions']:
        c = p.get('age_stats'); g = f"{c['win']:.0f}% (avg {c['avg']:+.1f}%, n {c['n']:,})" if c else 'n/a'
        gap = f"{p['stop_gap_pct']:.1f}%" + (f" ({p['stop_gap_ranges']:.1f} day-ranges)" if p.get('stop_gap_ranges') is not None else '')
        vals = [f"{p['tk']} {p['track']}", f"{p['age_days']:.1f}d", f"{p['pnl']:+.1f}%", gap, f"{p['locked_pct']:+.1f}%", f"{p['ratio']:.1f} (#{p['ratio_rank']})" if p.get('ratio') is not None else 'n/a', g,
                (f"in {p['earn_days']}d" if p.get('earn_days') is not None and p['earn_days'] <= 10 else 'none <=10d')]
        for k, (v, x_) in enumerate(zip(vals, xs)):
            ax.text(x_, y, v, fontsize=8.4, va='top', fontweight='bold' if k == 0 else 'normal', color=(GOOD if p['pnl'] >= 0 else CRIT) if k == 2 else (CRIT if (k == 3 and p['selling']) else T1))
        y -= 0.026
        yy = y; ax.text(0.07, yy, 'next: ' + '; '.join(p['next_steps']), fontsize=7.5, color=MUT, va='top'); y -= 0.024
    y -= 0.005
    _wrap(ax, 0.06, y, f"Ledger: {(A.get('ledger') or {}).get('note', 'n/a')}" + (f" · {(A.get('ledger') or {}).get('n_open')} open of {(A.get('ledger') or {}).get('n')} recorded · sha {(A.get('ledger') or {}).get('sha')}" if (A.get('ledger') or {}).get('n') else ''), 150, size=7.5, color=GOOD if (A.get('ledger') or {}).get('ok') else CRIT)
    fp = os.path.join(outdir, '0b_expect.png'); fig.savefig(fp, dpi=110, facecolor=BG); plt.close(fig); return fp

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
    lg = A.get('ledger') or {}
    if lg.get('ok') is False or lg.get('is_seed'): ax.text(0.06, 0.885, '! ' + lg.get('note', '') + (' · in ledger, not in this book: ' + ', '.join(lg.get('in_ledger_not_book', [])) if lg.get('in_ledger_not_book') else '') + (' · in this book, not in ledger: ' + ', '.join(lg.get('in_book_not_ledger', [])) if lg.get('in_book_not_ledger') else ''), fontsize=8.6, color=CRIT, va='top', fontweight='bold')
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
    try:
        fx = _page_expect(LS, A, outdir, (BG, T1, T2, MUT, GOOD, CRIT, GOLD))
        if fx: files.append(fx)
    except Exception as ex: print(f'  ! expectations page: {str(ex)[:80]}')
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
