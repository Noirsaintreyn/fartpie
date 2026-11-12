"""MID/SMALL-CAP GROWTH SCREENER (second Discord screener, separate from live_screener.py = the original mega-cap book).
Rules (found + tested in xs/report_lowvol_combo.py / report_volcheck.py / report_smallcap_book.py, variant C):
  Bars      1h regular-hours bars from Yahoo (no premarket), vector = the same pivot-ratchet state machine as the original screener (xs/vector_pine.py)
  Entry     1h bar closes with a BULL FLIP, AND that bar's volume >= VOL_X (1.5) x the mean of the prior 20 bars, AND the previous day's range is in the
            bottom third of the stock's own history (low-vol day, expanding rank). Entry = next open (alert fires at the bar close).
  Nov 12 2026 additions (all tested in xs/, see the logs named there):
    PRICE   entry price < MAX_PX ($5; MIDSMALL_MAX_PX=0 turns it off). Same flips on >= $5 names lose after costs (xs/report_post_earnings.log control, xs/report_sub5_earn_macro.log).
    VWAP    skip flips whose close is more than 1 ATR(14, 1h) above the session VWAP (extended entries) - xs/report_sub5_stack.py
    EARN    skip entries <= 3 sessions before an earnings reaction (Yahoo earnings dates) - xs/report_sub5_stack.py
    SHORT   FINRA Reg SHO short-sale share of volume (prior 20 sessions) above 53% -> HALF-size slot (cut/size tests: xs/report_sub5_svr_rank.py). Half-size, not a skip: cutting those costs the winners.
    CONTEXT every post carries market context, recent news, upcoming/last earnings, the setup flags (signal_context.py)
  Exit      the vector's own bear flip on a 1h close (sell next open). No other stop / target (Rey's rule for flip strategies).
  Slots     SLOTS (8), equal weight, first come first served, higher flip-volume ratio first when several fire on the same bar; skipped = honorable mention (logged).
  Long only. Universe: midsmall_universe.txt (one ticker per line; `--refresh-universe` rebuilds it from Yahoo's screener: US stocks < $5, > $150M cap, > 1M avg volume).
  Backtest, sub-$5 universe, 8 slots, 40 bp round trip, Dec 2023-Nov 2026 (xs/report_sub5_stack.py, report_sub5_svr_rank.py): about 14 BUYs a month (13/month taken with 8 slots), ~28% win rate,
  median trade -3%, mean +4%, a few big winners carry it (13-19% of trades > +10%). Universe = today's survivors and the thresholds were chosen on the same data: IN-SAMPLE. Paper-trade first.
Usage:
  python3 midsmall_screener.py --once --dry       scan the latest closed 1h bar, print what would be posted
  python3 midsmall_screener.py --replay 60        replay the last 60 days of signals (sanity check, nothing posted)
  python3 midsmall_screener.py --loop             weekday service: scans seconds after every 1h close (10:30 ... 16:00 ET) and posts to Discord
Discord (.env): DISCORD_TOKEN + MIDSMALL_CHANNEL_ID (bot), or MIDSMALL_WEBHOOK_URL (webhook)."""
import argparse, json, os, sys, time, warnings
import numpy as np, pandas as pd, requests
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xs'))
try:
    from dotenv import load_dotenv; load_dotenv('.env'); load_dotenv('discord_bot/.env')
except Exception: pass
from vector_pine import run_state_machine
ET = 'America/New_York'; LIVE = 'live'; os.makedirs(LIVE, exist_ok=True)
STATE_F = f'{LIVE}/midsmall_state.json'; SLOTS = int(os.getenv('MIDSMALL_SLOTS', '8')); VOL_X = float(os.getenv('MIDSMALL_VOL_X', '1.5'))
MAX_PX = float(os.getenv('MIDSMALL_MAX_PX', '2')); MIN_PX = float(os.getenv('MIDSMALL_MIN_PX', '1'));  # Nov 12 2026: $1-2 only (most consistent price band, xs/report_12_tf.log); MIDSMALL_MIN_PX=0 MIDSMALL_MAX_PX=5 restores the sub-$5 book
GRADE_TXT = {'TOP': 'Grade TOP: Kalman trend up + above the 1h and 4h EMA200. Backtest ($1-2, vol>=1.5x flips): +12.1% avg per trade, 39% win, ~46 signals/yr, -12% book drawdown (in-sample, small sample).', 'STRONG': 'Grade STRONG: Kalman trend up + above the 1h EMA200. Backtest ($1-2, vol>=1.5x flips): +7.8% avg per trade, 36% win, ~94 signals/yr, -19% book drawdown (in-sample).', 'WEAK': ''}
DAILY_TXT = {True: 'Daily trend: BULL - 1h flip inside a daily uptrend. Backtest $1-2 names, vol>=1.5x: +8.0% avg per trade, 36% win (p=0.02) vs +2.1% when the daily vector was not bull.', False: 'Daily trend: NOT bull - lone 1h flip against the daily vector; weaker in the backtest (+2.1% avg per trade vs +8.0% with daily bull, $1-2 names).', None: ''}
REQ_LV = os.getenv('MIDSMALL_REQ_LOWVOL', '0') == '1'; MIN_GRADE = os.getenv('MIDSMALL_MIN_GRADE', 'STRONG'); FH = {}; LV = {}
DAILY_GATE = os.getenv('MIDSMALL_DAILY_GATE', '0') == '1'; DBULL = {}; MA = {}; EMA200_GATE = os.getenv('MIDSMALL_EMA200_GATE', '0') == '1'; KAL_GATE = os.getenv('MIDSMALL_KALMAN_GATE', '0') == '1'; VWAP_X = float(os.getenv('MIDSMALL_VWAP_X', '1.0')); EARN_SKIP = int(os.getenv('MIDSMALL_EARN_SKIP', '3'))
try: import signal_context as SC
except Exception as _e: SC = None; print(f'  ! signal_context unavailable ({_e}); posts will have no context, no earnings/short rules')
BACKTEST_NOTE = 'Backtest (in-sample, sub-$5): setups with >=3 flags averaged about +4-6% per trade vs about +1% with <=2; win rate ~29% either way (median trade -3%), so size small.'
DEFAULT = ['KGS', 'LCID', 'RIOT', 'Z', 'IREN', 'PLUG', 'UPST', 'CLOV', 'TLRY', 'BB', 'SOFI', 'FUBO', 'OPEN', 'ETSY', 'DKNG', 'ROKU', 'CROX', 'FIVE', 'CHWY', 'MTCH', 'LYFT', 'BBWI',
           'RIVN', 'SNAP', 'PINS', 'HOOD', 'AFRM', 'RKLB', 'SOUN', 'MARA', 'CLSK', 'HIMS', 'CELH', 'JOBY', 'ACHR', 'PATH', 'NIO', 'LI', 'W', 'CHPT', 'RUN', 'ENPH', 'SEDG', 'LUMN']
def universe():
    f = 'midsmall_universe.txt'
    return [l.strip().upper() for l in open(f) if l.strip() and not l.startswith('#')] if os.path.exists(f) else DEFAULT
def now_et(): return pd.Timestamp.now(tz=ET).tz_localize(None)

# ---------------------------------------------------------------- data (one batched Yahoo call)
def fetch_all(tks, period='400d', min_rows=300):
    """1h bars in chunks of 100 names, 4 chunks in parallel (one batched Yahoo call per chunk; 400d is enough warm-up for the vector state, the 20-bar volume mean and the 60-day low-vol rank)"""
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time(); out = {}
    def one(ch):
        D = None
        for _ in range(2):
            try: D = yf.download(ch, period=period, interval='1h', prepost=False, group_by='ticker', threads=False, progress=False, auto_adjust=False); break
            except Exception: time.sleep(3)
        res = {}
        if D is None: return res
        for tk in ch:
            try:
                Y = (D[tk] if len(ch) > 1 else D).dropna(subset=['Close']).rename(columns=str.lower)[['open', 'high', 'low', 'close', 'volume']]
                Y.index = Y.index.tz_convert(ET).tz_localize(None); Y = Y[~Y.index.duplicated()]
                if len(Y) >= min_rows: res[tk] = Y
            except Exception: pass
        return res
    chunks = [tks[i:i + 100] for i in range(0, len(tks), 100)]
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(one, chunks): out.update(r)
    print(f'  fetched {len(out)}/{len(tks)} tickers in {time.time() - t0:.1f}s', flush=True); return out
BARS_F = f'{LIVE}/midsmall_bars.pkl'
def get_data(tks):
    """bars cache: first run (or cache older than 5 days) = 400d fetch; afterwards each scan only fetches the last 5 days and splices them in (keeps a scan to ~20s instead of ~90s)"""
    import pickle
    try: C = pickle.load(open(BARS_F, 'rb'))
    except Exception: C = None
    if not C or time.time() - C['t'] > 5 * 86400 or set(tks) - set(C['d']): data = fetch_all(tks, '400d'); C = {'t': time.time(), 'd': data}
    else:
        new = fetch_all(tks, '5d', min_rows=1); data = C['d']
        for tk, Y in new.items():
            if tk in data: Y0 = data[tk]; Y0 = Y0[Y0.index < Y.index.min()]; data[tk] = pd.concat([Y0, Y])
            else: data[tk] = Y
        C['d'] = data
    try: pickle.dump(C, open(BARS_F, 'wb'))
    except Exception as e: print(f'  ! bars cache not saved: {e}')
    return C['d']
def refresh_universe():
    """rebuild midsmall_universe.txt from Yahoo's screener: US NMS/NYQ stocks under $5, market cap > $150M, 3-month avg volume > 1M (add to what is already in the file, never drops names)"""
    import yfinance as yf
    from yfinance import EquityQuery as EQ
    q = EQ('and', [EQ('eq', ['region', 'us']), EQ('is-in', ['exchange', 'NMS', 'NYQ']), EQ('lt', ['intradayprice', 5]), EQ('gt', ['avgdailyvol3m', 1e6]), EQ('gt', ['intradaymarketcap', 1.5e8])]); new = []
    for off in range(0, 750, 250):
        r = yf.screen(q, size=250, offset=off, sortField='avgdailyvol3m', sortAsc=False); new += [x['symbol'] for x in r['quotes']]
        if len(r['quotes']) < 250: break
    cur = universe(); allt = sorted(set(cur) | {t for t in new if t.isalpha() and len(t) <= 5}); open('midsmall_universe.txt', 'w').write('\n'.join(allt) + '\n'); print(f'universe: {len(cur)} -> {len(allt)} names ({len(allt) - len(cur)} new)')
def closed_only(Y, t=None):
    """drop the bar still forming. 1h RTH bars start at :30 (09:30, 10:30 ... 15:30; the last is a 30-minute bar closing 16:00)."""
    t = now_et() if t is None else t; end = Y.index + pd.Timedelta('60min'); last_open = (Y.index.hour == 15) & (Y.index.minute == 30); end = end.where(~last_open, Y.index + pd.Timedelta('30min'))
    return Y[end <= t + pd.Timedelta('1s')]

# ---------------------------------------------------------------- signals
def features(Y):
    """vector state + volume ratio + low-vol-day flag for every bar (identical definitions to xs/report_lowvol_combo.py)"""
    reg = run_state_machine(Y.assign(datetime=Y.index))[0]; v = Y.volume.values.astype(float)
    vrel = v / pd.Series(v).rolling(20).mean().shift().values; day = Y.index.normalize()
    d = pd.DataFrame({'h': Y.high.groupby(day).max(), 'l': Y.low.groupby(day).min()}); rg = d.h - d.l
    ratio = rg.shift() / rg.shift().rolling(20).mean().shift(); rk = ratio.expanding(min_periods=60).apply(lambda z: (z.iloc[:-1] < z.iloc[-1]).mean(), raw=False)
    lv = (rk.reindex(day) <= 1 / 3).fillna(False).values
    h, l, c = Y.high.values.astype(float), Y.low.values.astype(float), Y.close.values.astype(float); pc = np.r_[c[0], c[:-1]]; tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))); atr = pd.Series(tr).rolling(14).mean().values
    vw = (pd.Series((h + l + c) / 3 * v).groupby(day).cumsum() / (pd.Series(v).groupby(day).cumsum() + 1e-9)).values; vwx = (c - vw) / atr      # flip-bar close vs session VWAP in 1h ATRs (same definition as xs/report_sub5.py)
    return pd.DataFrame({'reg': reg, 'vrel': vrel, 'lv': lv, 'close': Y.close.values, 'vwx': vwx}, index=Y.index)
def events(F):
    """bull entries (flip to +1 passing the filters) and exits (flip to -1) as (bar_open_time, kind, vrel, close)"""
    ev = []; r = F.reg.values
    for i in range(1, len(F)):
        if r[i] == r[i - 1] or r[i] == 0: continue
        px_ok = (MAX_PX <= 0 or F.close.iat[i] < MAX_PX) and F.close.iat[i] >= MIN_PX; vw_ok = not (F.vwx.iat[i] > VWAP_X)      # NaN vwx (warm-up) passes
        if r[i] == 1 and F.vrel.iat[i] >= VOL_X and (F.lv.iat[i] or not REQ_LV) and not np.isnan(F.vrel.iat[i]) and px_ok and vw_ok: ev.append((F.index[i], 'BUY', float(F.vrel.iat[i]), float(F.close.iat[i])))
        elif r[i] == 1: ev.append((F.index[i], 'FLIP_FILTERED', float(F.vrel.iat[i]) if not np.isnan(F.vrel.iat[i]) else np.nan, float(F.close.iat[i])))
        elif r[i] == -1: ev.append((F.index[i], 'SELL', float(F.vrel.iat[i]) if not np.isnan(F.vrel.iat[i]) else np.nan, float(F.close.iat[i])))
    return ev
def bar_close(ts):
    ts = pd.Timestamp(ts); return ts + (pd.Timedelta('30min') if (ts.hour == 15 and ts.minute == 30) else pd.Timedelta('60min'))

def daily_bull(Yc):
    """Nov 12 2026: is the DAILY vector regime bull on the last completed daily bar (daily = the 1h bars resampled by session)? xs/report_12_mtf.py"""
    try:
        d = Yc.index.normalize(); G = Yc.groupby(d); o = G.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), volume=('volume', 'sum')); o.index = pd.DatetimeIndex(G.apply(lambda x: x.index[0]).values)
        if Yc.index[-1].hour < 15: o = o.iloc[:-1]                                                              # today's daily bar is not complete yet
        return bool(run_state_machine(o.assign(datetime=o.index))[0][-1] == 1)
    except Exception: return None
def kalman_up(Yc):
    """Nov 12 2026: causal local-linear-trend Kalman filter on log close (fixed noise ratios, nothing fitted): is the filtered slope > 0 at the last closed bar? xs/report_12_smooth.py"""
    try:
        y = np.log(Yc.close.values.astype(float)); R = max(np.nanvar(np.diff(y)[:500]), 1e-8); x = np.array([y[0], 0.]); P = np.diag([R, R * .01]); F = np.array([[1, 1], [0, 1]]); Q = np.diag([.05 * R, .001 * R])
        for i in range(1, len(y)):
            x = F @ x; P = F @ P @ F.T + Q; K = P[:, 0] / (P[0, 0] + R); x = x + K * (y[i] - x[0]); P = P - np.outer(K, P[0])
        return bool(x[1] > 0)
    except Exception: return None
def ma_context(Yc):
    """Nov 12 2026: moving-average context at the flip bar (1h EMA50 / EMA200 and the quarterly anchored VWAP), OHLCV only. xs/report_12_ma_exit.py"""
    try:
        c = Yc.close; e50 = float(c.ewm(span=50, adjust=False).mean().iat[-1]); e200 = float(c.ewm(span=200, adjust=False).mean().iat[-1]); q = Yc.index.to_period('Q'); tp = (Yc.high + Yc.low + Yc.close) / 3
        qv = float(((tp * Yc.volume).groupby(q).cumsum() / (Yc.volume.groupby(q).cumsum() + 1e-9)).iat[-1]); px = float(c.iat[-1])
        def htf200(tf):                                                                                                     # EMA200 of the last COMPLETED 4h / daily bar built from the 1h bars
            day = Yc.index.normalize(); G = Yc.groupby([day, (Yc.index.hour >= 13).astype(int)] if tf == '4h' else [day]); cl = G.close.last()
            if Yc.index[-1].hour < 15 and tf == 'd' or (tf == '4h' and Yc.index[-1].hour not in (12, 15)): cl = cl.iloc[:-1]                  # drop the still-forming bar
            return float(cl.ewm(span=200, adjust=False).mean().iat[-1]) if len(cl) >= 200 else None
        h4, hd = htf200('4h'), htf200('d')
        return dict(e4=h4, ed=hd, kal=kalman_up(Yc), px=px, e50=e50, e200=e200, qv=qv, a50=px > e50, a200=px > e200, avw=px > qv, x5020=e50 > e200)
    except Exception: return None
def ma_text(m):
    if not m: return ''
    t = lambda ok: '\u2705' if ok else '\u274c'
    return (f"MA context (1h): close {m['px']:.2f} | EMA50 {m['e50']:.2f} {t(m['a50'])} | EMA200 {m['e200']:.2f} {t(m['a200'])} | quarterly VWAP {m['qv']:.2f} {t(m['avw'])} | EMA50 {'above' if m['x5020'] else 'below'} EMA200 | 4h EMA200 {m['e4']:.2f} {t(m['px'] > m['e4']) if m.get('e4') else 'n/a'} | daily EMA200 {m['ed']:.2f} {t(m['px'] > m['ed']) if m.get('ed') else 'n/a'} (flip above {sum(1 for ok in (m['a200'], bool(m.get('e4') and m['px'] > m['e4']), bool(m.get('ed') and m['px'] > m['ed'])) if ok)} of 3 EMA200s: backtest 0-1 of 3 ~ +1% avg, 3 of 3 ~ +9.9%) | Kalman-smoothed trend {'UP' if m['kal'] else 'DOWN' if m['kal'] is False else 'n/a'} {t(m['kal'])}. "
            + ("Backtest ($1-2, vol>=1.5x): above the 200 EMA +5.7% avg per trade, below it +1.2% (p=0.03); Kalman trend up +5.0%, down -0.4% (p=0.03); daily bull + Kalman up +10.7% avg, 39% win, 122 trades (in-sample)." if True else ''))
# ---------------------------------------------------------------- state + Discord
def load_state():
    try: return json.load(open(STATE_F))
    except Exception: return {'pos': {}, 'seen': []}
def save_state(st): st['seen'] = st['seen'][-800:]; json.dump(st, open(STATE_F, 'w'), default=str)
def post(text):
    ch, tok, hook = os.getenv('MIDSMALL_CHANNEL_ID') or '1570332394714562681', os.getenv('DISCORD_TOKEN'), os.getenv('MIDSMALL_WEBHOOK_URL')
    try:
        if ch and tok: r = requests.post(f'https://discord.com/api/v10/channels/{ch}/messages', headers={'Authorization': f'Bot {tok}'}, json={'content': text}, timeout=20)
        elif hook: r = requests.post(hook, json={'content': text}, timeout=20)
        else: print('  ! no Discord target (MIDSMALL_CHANNEL_ID + DISCORD_TOKEN, or MIDSMALL_WEBHOOK_URL)'); return False
        r.raise_for_status(); return True
    except Exception as e: print(f'  ! discord post failed: {e}'); return False

def scan_once(dry, data=None, asof=None):
    tks = universe(); data = data or get_data(tks); st = load_state(); fired = []; ctxs = []; t = asof
    cand = []; conf_msgs = []
    for tk, Y in data.items():
        Yc = closed_only(Y, t)
        if len(Yc) < 300: continue
        p0 = st['pos'].get(tk)
        if p0 and p0.get('fh') and p0.get('conf') is None:                                                          # follow-through check: did the entry bar close above the flip bar's high?
            try:
                ix = Yc.index.get_loc(pd.Timestamp(p0['t']))
                if ix + 1 < len(Yc):
                    cb = float(Yc.close.iat[ix + 1]); ok = cb > p0['fh']; p0['conf'] = bool(ok)
                    conf_msgs.append(f"\U0001f50e FOLLOW-THROUGH {tk}: entry bar closed {cb:.2f}, " + (f"ABOVE the flip-bar high {p0['fh']:.2f} - CONFIRMED. Backtest ($1-2, Kalman + 1h EMA200): confirmed trades 44% win, +12.0% avg per trade (63 trades, in-sample)." if ok else f"below the flip-bar high {p0['fh']:.2f} - NOT confirmed. These still averaged about +6% in the backtest, so use it to size down, not as a stop (exiting early on weak starts lost money in every test)."))
            except Exception: pass
        if tk not in st['pos']:                                                                                      # cheap prefilter: a BUY needs this bar's volume >= VOL_X x prior-20 mean and price < MAX_PX
            v = Yc.volume.values.astype(float); m20 = v[-21:-1].mean()
            if not (m20 > 0 and v[-1] / m20 >= VOL_X and (MAX_PX <= 0 or Yc.close.iat[-1] < MAX_PX) and Yc.close.iat[-1] >= MIN_PX): continue
        F = features(Yc); last = F.index[-1]
        if (now_et() if t is None else t) - bar_close(last) > pd.Timedelta('3h') and t is None: continue          # stale feed / market closed overnight: ignore
        for (ts, kind, vr, px) in events(F):
            if ts != last: continue                                                                              # only the newest closed bar can fire
            if kind == 'BUY': DBULL[tk] = daily_bull(Yc); MA[tk] = ma_context(Yc); FH[tk] = float(Yc.high.iat[-1]); LV[tk] = bool(F.lv.iat[-1])
            cand.append((tk, ts, kind, vr, px, float(F.vwx.iat[-1]) if F.vwx.iat[-1] == F.vwx.iat[-1] else 0.0))
    # exits first (free slots), then entries by flip-volume ratio
    for tk, ts, kind, vr, px, vwx in cand:
        if kind != 'SELL' or tk not in st['pos']: continue
        key = f'{tk}|SELL|{ts}'
        if key in st['seen']: continue
        p = st['pos'].pop(tk); fired.append(f"\u26a1 SELL {tk}: 1h vector turned bearish on the {ts:%H:%M} bar (last {px:.2f}). Exit at the next open. Entry was {p['px']:.2f} ({px / p['px'] - 1:+.1%}), held since {p['t'][:16]}.")
        ctxs.append(SC.block(tk, 'SELL') if SC else None); st['seen'].append(key)
    buys = sorted([c for c in cand if c[2] == 'BUY'], key=lambda c: -c[3]); iwm_up = SC.market_line()[1] if (SC and buys) else None
    for tk, ts, kind, vr, px, vwx in buys:
        key = f'{tk}|BUY|{ts}'
        if key in st['seen'] or tk in st['pos']: continue
        e = SC.earnings_info(tk) if SC else {}; ses = e.get('sessions')
        if ses is not None and ses <= EARN_SKIP:                                                                   # EARN rule: binary gap risk right before a report
            print(f"  skipped {tk}: bull flip but earnings reaction in {ses} sessions ({e['next']:%b %d})"); st['seen'].append(key); continue
        if DAILY_GATE and DBULL.get(tk) is False: print(f'  skipped {tk}: daily vector not bull (MIDSMALL_DAILY_GATE=1)'); st['seen'].append(key); continue
        if EMA200_GATE and MA.get(tk) and not MA[tk]['a200']: print(f'  skipped {tk}: below the 1h 200 EMA (MIDSMALL_EMA200_GATE=1)'); st['seen'].append(key); continue
        m_ = MA.get(tk); a4_ = bool(m_ and m_.get('e4') and m_['px'] > m_['e4'])
        grade = 'TOP' if (m_ and m_.get('kal') and m_['a200'] and a4_) else ('STRONG' if (m_ and m_.get('kal') and m_['a200']) else 'WEAK')
        if MIN_GRADE == 'TOP' and grade != 'TOP' or MIN_GRADE == 'STRONG' and grade == 'WEAK': print(f'  skipped {tk}: grade {grade} (needs Kalman up + above the 1h EMA200{" + 4h EMA200" if MIN_GRADE == "TOP" else ""}; MIDSMALL_MIN_GRADE={MIN_GRADE})'); st['seen'].append(key); continue
        if KAL_GATE and MA.get(tk) and MA[tk].get('kal') is False: print(f'  skipped {tk}: Kalman trend down (MIDSMALL_KALMAN_GATE=1)'); st['seen'].append(key); continue
        svr = SC.short_share(tk) if SC else None; half = svr is not None and svr > SC.SVR_CUT
        size_txt = (f"Size: HALF slot - short sellers are {svr:.0%} of its volume (20-day), the persistent-short-pressure group that underperformed in the backtest." if half else
                    f"Size: full slot (1/{SLOTS} of the book)" + (f" - short share {svr:.0%} of volume." if svr is not None else '.'))
        flags = SC.midsmall_flags(vr, svr, ses, vwx, iwm_up) + (BACKTEST_NOTE,) if SC else None
        if len(st['pos']) >= SLOTS:
            fired.append(f"\u2b50 HONORABLE MENTION {tk}: low-vol day + high-volume bull flip on the {ts:%H:%M} bar ({vr:.1f}x volume, last {px:.2f}), but all {SLOTS} slots are full."); ctxs.append(None); st['seen'].append(key); continue
        st['pos'][tk] = {'px': px, 't': str(ts), 'vr': vr, 'size': 0.5 if half else 1.0, 'fh': FH.get(tk), 'conf': None, 'grade': grade}; st['seen'].append(key)
        fired.append(f"\u26a1 BUY {tk}: [{grade}] 1h vector bull flip on the {ts:%H:%M} bar with {vr:.1f}x volume{' on a low-volatility day' if LV.get(tk) else ''}, {vwx:+.1f} ATR vs VWAP. Entry next open (~{px:.2f}). Exit = the vector's bear flip (no other stop). Slot {len(st['pos'])}/{SLOTS}." + (' HALF SIZE.' if half else '') + (' Below 1h 200 EMA.' if MA.get(tk) and not MA[tk]['a200'] else '') + (' A+ GRADE: daily vector bull AND Kalman trend up.' if DBULL.get(tk) and MA.get(tk) and MA[tk].get('kal') else ' A-GRADE: daily vector is also bull.' if DBULL.get(tk) else (' Daily vector NOT bull (lone 1h flip).' if DBULL.get(tk) is False else '')))
        ctxs.append(SC.block(tk, 'BUY', extra=[x for x in (size_txt, GRADE_TXT.get(grade, ''), DAILY_TXT.get(DBULL.get(tk), ''), ma_text(MA.get(tk))) if x], flags=flags) if SC else None)
    for m in conf_msgs: fired.append(m); ctxs.append(None)
    for m in fired: print(m); (None if dry else post(m))                                                           # alerts first (fast) ...
    for m, c in zip(fired, ctxs):                                                                                   # ... then the context for each
        if c: print(c); (None if dry else post(c))
    if not dry: save_state(st)
    if not fired: print(f'  no new signals ({len(data)} names scanned, {len(st["pos"])} open: {", ".join(sorted(st["pos"])) or "-"})')
    return fired

def replay(days):
    """walk the last `days` days bar by bar with the same slot logic to see what the live service would have posted"""
    tks = universe(); data = fetch_all(tks); FS = {tk: features(closed_only(Y)) for tk, Y in data.items()}; t0 = now_et().normalize() - pd.Timedelta(days=days)
    evs = []
    for tk, F in FS.items():
        for ts, kind, vr, px in events(F):
            if ts >= t0: evs.append((ts, tk, kind, vr, px))
    evs.sort(key=lambda e: (e[0], -(e[3] if e[3] == e[3] else 0))); pos = {}; nb = ns = nh = 0
    for ts, tk, kind, vr, px in evs:
        if kind == 'SELL' and tk in pos: p = pos.pop(tk); ns += 1; print(f'{ts:%m-%d %H:%M} SELL {tk:5s} {px:8.2f}  ({px / p[0] - 1:+.1%} in {(ts - p[1]).days}d)')
        elif kind == 'BUY' and tk not in pos:
            if len(pos) < SLOTS: pos[tk] = (px, ts); nb += 1; print(f'{ts:%m-%d %H:%M} BUY  {tk:5s} {px:8.2f}  vol {vr:.1f}x  slot {len(pos)}/{SLOTS}')
            else: nh += 1; print(f'{ts:%m-%d %H:%M} HM   {tk:5s} {px:8.2f}  vol {vr:.1f}x  (slots full)')
    print(f'\nreplay {days}d: {nb} buys, {ns} sells, {nh} honorable mentions, {len(pos)} still open: {", ".join(sorted(pos)) or "-"}')

def wait_for_bar(target, timeout=120):
    """poll Yahoo (SPY 1h) until the bar closing at `target` is published"""
    import yfinance as yf; t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            y = yf.Ticker('SPY').history(period='5d', interval='1h', prepost=False)
            if len(y):
                ix = y.index[-1]; last = ix.tz_convert(ET).tz_localize(None) if ix.tzinfo else ix
                if bar_close(last) >= target and now_et() >= target + pd.Timedelta('3s'): return True
        except Exception: pass
        time.sleep(3)
    return False
def watchlist(data, n=12):
    """Nov 12 2026: JUMP WATCHLIST - the most volatile names in the price band (14-day ATR%), NOT signals. xs/report_12_jump.py: top-decile past volatility had a 22% chance of a +20% move within 5 sessions (base 9%) but a 7% chance of a -20% move (base 3.8%): it finds big-move candidates, not direction."""
    rows = []
    for tk, Y in data.items():
        try:
            Yc = closed_only(Y)
            if len(Yc) < 300: continue
            D = Yc.groupby(Yc.index.normalize()).agg(h=('high', 'max'), l=('low', 'min'), c=('close', 'last'), v=('volume', 'sum')); c = D.c; px = float(c.iat[-1])
            if not (MIN_PX <= px < MAX_PX if MAX_PX > 0 else px >= MIN_PX): continue
            if float((c * D.v).rolling(20).mean().iat[-1]) < 2e5: continue
            pc = c.shift(); tr = pd.concat([D.h - D.l, (D.h - pc).abs(), (D.l - pc).abs()], axis=1).max(axis=1); atr = float(tr.rolling(14).mean().iat[-1] / px)
            rows.append(dict(tk=tk, px=px, atr=atr, maxr=float(c.pct_change().rolling(21).max().iat[-1]), dist=float(1 - px / D.h.rolling(252, min_periods=120).max().iat[-1]), rv=float(D.v.iat[-1] / D.v.rolling(20).mean().iat[-1])))
        except Exception: continue
    if not rows: return None
    W = pd.DataFrame(rows).sort_values('atr', ascending=False).head(n); L = []
    for r in W.itertuples():
        Yc = closed_only(data[r.tk]); m = ma_context(Yc) or {}; reg = int(run_state_machine(Yc.assign(datetime=Yc.index))[0][-1]); a4 = bool(m.get('e4') and m['px'] > m['e4'])
        L.append(f"{r.tk:<6}{r.px:>5.2f}  ATR {r.atr:>4.0%}  MAX21 {r.maxr:>+4.0%}  {('-' + format(r.dist, '.0%')) if r.dist == r.dist else 'n/a':>4} from 52wH  vol {r.rv:>3.1f}x | 1h vector {'BULL' if reg == 1 else 'BEAR'} | Kalman {'up' if m.get('kal') else 'dn'}  1hEMA200 {'Y' if m.get('a200') else 'n'}  4hEMA200 {'Y' if a4 else 'n'}")
    return (f"\U0001f52d JUMP WATCHLIST ${MIN_PX:g}-{MAX_PX:g} (volatile names, NOT signals) - ranked by 14-day ATR%. Backtest: the top decile by past volatility had a 22% chance of a +20% move within 5 sessions (base 9%), but also a 7% chance of -20% (base 3.8%): it finds where big moves are likely, not which way. Trade only when a BUY alert fires (grade TOP/STRONG).\n```\n" + '\n'.join(L) + "\n```")
def loop():
    print(f'midsmall screener up: {len(universe())} names, {SLOTS} slots, volume >= {VOL_X}x, price ${MIN_PX:g}-{MAX_PX:g}'); post(f'\U0001f7e2 Mid/small-cap screener online: {len(universe())} names, {SLOTS} slots. 1h vector bull flips on ${MIN_PX:g}-{MAX_PX:g} names with >= {VOL_X}x volume, graded TOP (Kalman trend up + above 1h and 4h EMA200) / STRONG (Kalman up + above 1h EMA200); weaker flips are skipped (MIDSMALL_MIN_GRADE). Not extended above VWAP, not within {EARN_SKIP} sessions of earnings; half size when short selling is heavy. A follow-through note posts after the entry bar closes. Exit = bear flip. Each alert comes with market context, MA context, news and setup flags.')
    try:
        st0 = load_state(); today = str(now_et().date())
        if st0.get('wl_date') != today and now_et().hour < 16:
            w = watchlist(get_data(universe()))
            if w: print(w); post(w); st0['wl_date'] = today; save_state(st0)
    except Exception as e: print(f'  ! watchlist failed: {e}')
    while True:
        t = now_et(); mins = t.hour * 60 + t.minute
        if t.dayofweek >= 5 or mins > 16 * 60 + 5: print('market closed - exiting'); return
        marks = [10 * 60 + 30, 11 * 60 + 30, 12 * 60 + 30, 13 * 60 + 30, 14 * 60 + 30, 15 * 60 + 30, 16 * 60]
        nxt = next((m for m in marks if m > mins - 1), None)
        if nxt is None: return
        tg = t.normalize() + pd.Timedelta(minutes=nxt); time.sleep(max(0, (tg - now_et()).total_seconds()))
        wait_for_bar(tg)
        try: scan_once(False)
        except Exception as e: print(f'  ! scan failed: {e}')
        time.sleep(5)
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--once', action='store_true'); ap.add_argument('--dry', action='store_true'); ap.add_argument('--loop', action='store_true'); ap.add_argument('--replay', type=int, default=0); ap.add_argument('--refresh-universe', action='store_true'); ap.add_argument('--watchlist', action='store_true')
    a = ap.parse_args()
    if a.refresh_universe: refresh_universe()
    elif a.watchlist: w = watchlist(get_data(universe())); print(w); (None if a.dry else post(w))
    elif a.replay: replay(a.replay)
    elif a.loop: loop()
    else: scan_once(a.dry)
