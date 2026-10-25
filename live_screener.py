"""
LIVE SCREENER + DISCORD - 15m vector flip, exit C, on high options-activity names (+ QQQ in negative gamma).
Rule = the version that held up best in the 2026 out-of-sample test (see PREREG_stock_gex_vector_screener.md).

  Universe   every ticker in xs/ext/optvol_ratio.csv (stocks) + QQQ
  Eligible   stock: trailing-252-day median options ratio >= 1.5 (contracts x 100 / RTH share volume, raw)
             QQQ:   always one slot, only takes bull flips when QQQ net GEX at the prior close < 0
  Entry      next 15m open after a 15m vector bull flip (name eligible at that bar)
  Exit (C)   next 15m open after a 15m close below the 15m vector stop; once the trade is up >= 2% and the 1h vector
             has been bull since entry, the exit moves to a 1h close below the 1h vector stop
  Stop       vector support level - 0.35 x ATR (a CLOSE below it is what flips the vector)
  Size       1 / (eligible names incl. QQQ) per position, long only
  --- from MODEL2_START (default 2026-10-19 09:30): the 15m + 1h + lock book (xs/best_combo.py: 30.5%/yr, -16.5% DD 2019-26)
  1h track   also enter at the next RTH open after a 1h vector bull flip; exit on the 1h bear flip (shown as 'TICKER 1h')
  Lock       both tracks: once a trade's peak (RTH highs) is >= +5%, a close at or below half the peak gain exits at the next open
  Slots      SLOTS (default 8) shared by both tracks, first come first served; a stock can hold one 15m and one 1h slot

Run from ~/fartpie:
  python3 live_screener.py                 print current signals / positions
  python3 live_screener.py --capital 25000 also $ sizes
  python3 live_screener.py --post-positions post the open-positions card to Discord now (test)
  python3 live_screener.py --loop          weekday service: 09:00 ET posts open positions, then after every 15m close
                                           during RTH posts a card for each new BUY / SELL; exits at 16:10 ET
                                           Mon 09:00 also posts the mid/small book card; Fri 16:45 builds it (mid_book.py)
                                           ~16:01 posts the closing open-positions board (after the last bar's scan)
                                           RTH: posts once per new whole-percent high on an open position (+1%, +2%, ...), live price, with the lock level
Live ledger starts at SCREENER_START (default 2026-10-13 09:30 ET): only signals from then on become positions.
Discord (either one, in ~/fartpie/.env):
  VECTOR_CHANNEL_ID=<channel id>   posts as your bot, using DISCORD_TOKEN from discord_bot/.env
  DISCORD_WEBHOOK_URL=<url>        or posts through a channel webhook
"""
import argparse, glob, json, os, shutil, socket, subprocess, sys, time
sys.stdout.reconfigure(line_buffering=True)   # log lines appear in live/screener.log immediately
import numpy as np, pandas as pd, requests
from dotenv import load_dotenv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
load_dotenv('.env'); load_dotenv('discord_bot/.env')
sys.path.insert(0, 'xs')
if os.getenv('VECTOR_ENGINE', 'pine') == 'pine':
    from vector_pine import run_state_machine        # exact port of the TradingView 'SVM - Fractals' indicator
else:
    from vector_core import run_state_machine
import live_cards as LC
START = LC.START       # positions are only counted from signals at/after this time (SCREENER_START in .env)

CUT, UP_SWITCH, BREAK_ATR, RTH0, RTH1 = float(os.getenv('OPT_CUT', '2.5')), 0.02, 0.35, 570, 960   # OPT_CUT: options-ratio bar
OPTVOL_FILE = os.getenv('OPTVOL_FILE', 'live_data/optvol_ratio_live.csv')   # Oct 20 2026: S&P-wide pull, refreshed weekly (tools/weekly_ratio.sh)
TIE_RATIO = os.getenv('TIE_RATIO', '1') == '1'
CUT2 = float(os.getenv('OPT_CUT2', '1.5'))                                       # Oct 22 2026: ratio bar lowered to 1.5 (full S&P options data)
CUT2_SWITCH = pd.Timestamp(os.getenv('CUT2_SWITCH', '2026-10-22 09:30'))
_SPM = json.load(open('sp500_pit_membership.json')); _SPK = sorted(_SPM); _SPS = {k: {'META' if t == 'FB' else t.replace('.', '') if t == 'BRK.B' else t for t in v} for k, v in _SPM.items()}
def in_sp(tk, t):
    """S&P 500 member in that month (point-in-time list; months after the list's last month use the last month)"""
    k = f'{pd.Timestamp(t):%Y-%m}'; k = k if k in _SPS else (_SPK[-1] if k > _SPK[-1] else _SPK[0]); return tk in _SPS[k]
OPT_SWITCH = pd.Timestamp(os.getenv('OPT_SWITCH', '2026-10-20 09:30'))      # from here: S&P universe at ratio >= 2.5; before: the old 8 names at >= 1.5
OLD_UNIVERSE = set(os.getenv('OLD_UNIVERSE', 'AAPL,ADBE,AMD,COST,NVDA,ORCL,TSLA,UNH').split(','))
NO_ETF = {'QQQ', 'SPY', 'IWM', 'DIA', 'SMH', 'XLE', 'XLF', 'XLK', 'GLD', 'SLV', 'TLT', 'IEF'}                      # 1: when signals tie, the highest options ratio gets the slot
INCLUDE_QQQ = os.getenv('INCLUDE_QQQ', '0') == '1'
MID_BOOK = os.getenv('MID_BOOK', '1') == '1'                 # 0 on Render: the mid/small book needs the xs data pipeline that only lives on the home machine
SESSION = os.getenv('SCREENER_SESSION', 'pre')        # 'pre' = vector on premarket+RTH bars (from 2026-10-14), 'rth' = RTH only
PRE0 = 240                                              # 04:00 ET
CUTOVER = pd.Timestamp(os.getenv('PM_START', '2026-10-14 09:30'))
MODEL2 = pd.Timestamp(os.getenv('MODEL2_START', '2026-10-19 09:30'))          # 1h-flip track + profit lock start here
SLOTS, LOCK_K, LOCK_G = int(os.getenv('SLOTS', '6')), 0.05, 0.5
SLOT_SWITCH = pd.Timestamp(os.getenv('SLOT_SWITCH', '2026-10-20 09:30'))     # 8 slots (12.5% each) before this, SLOTS after
STICKY_START = np.datetime64(pd.Timestamp(os.getenv('STICKY_START', '2026-10-20 09:30')))   # sticky 1h mode: once >= +2% with a 1h bull, keep the 1h stop
slots_at = lambda t0: 8 if pd.Timestamp(t0) < SLOT_SWITCH else SLOTS               # lock: after +5% peak keep >= 50% of the peak gain
VIRT, INF = np.datetime64('2262-01-01'), np.datetime64('2262-03-01')        # the next (virtual) RTH open / still open   # premarket rule applies to signals from here; earlier positions carried      # QQQ (neg-gamma) slot off by default: no daily options pull needed
LIVE = 'live'; os.makedirs(LIVE, exist_ok=True)
ET = 'America/New_York'
BG, EMB, PANEL, T1, T2, MUT, GRID = '#05080f', '#05080f', '#0a1322', '#dfe7f1', '#b8c4d4', '#7d8ca0', '#1a2638'
BLUE, GOOD, CRIT = '#3d93ff', '#4da3ff', '#ff4d5e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': T1, 'xtick.color': MUT, 'ytick.color': MUT, 'font.size': 9})

def now_et(): return pd.Timestamp.now(tz=ET).tz_localize(None)

# ================================================================ data
LSE_SYM = {'BRKB': 'BRK.B'}; YF_SYM = {'BRKB': 'BRK-B'}                   # feed symbols for tickers stored without the class dot
def load_bars(tk, fetch, first=None):
    first = RTH0 if first is None else first
    live_f = f'{LIVE}/{tk}_15m.parquet'
    parts = [pd.read_parquet(f) for f in sorted(glob.glob(f'lse_ohlcv_cache/{tk}_15m_*.parquet'))[-1:]]
    if os.path.exists(live_f): parts.append(pd.read_parquet(live_f))
    if not parts and fetch:                                              # fresh machine (Render): pull ~14 months of 15m bars from LSE once
        try:
            from fetch_lse_ohlcv import _fetch_range
            end = pd.Timestamp.utcnow().tz_localize(None).normalize() + pd.Timedelta(days=1)
            rows = _fetch_range(LSE_SYM.get(tk, tk), '15m', end - pd.Timedelta(days=430), end)
            if rows:
                N = pd.DataFrame(rows); N['datetime'] = pd.to_datetime(N['ts'], utc=True); parts.append(N[['datetime', 'open', 'high', 'low', 'close', 'volume']])
                print(f'  {tk}: bootstrapped {len(N):,} 15m bars from LSE')
        except Exception as e: print(f'  ! {tk}: bootstrap failed ({str(e)[:60]})')
    if not parts: return None
    H = pd.concat(parts); H['datetime'] = pd.to_datetime(H.datetime, utc=True)
    H = H.drop_duplicates('datetime', keep='last').sort_values('datetime')
    if fetch:
        try:
            from fetch_lse_ohlcv import _fetch_range
            start = (H.datetime.max() - pd.Timedelta(days=4)).tz_localize(None)
            end = pd.Timestamp.utcnow().tz_localize(None).normalize() + pd.Timedelta(days=1)
            rows = _fetch_range(LSE_SYM.get(tk, tk), '15m', start, end)
            if rows:
                N = pd.DataFrame(rows); N['datetime'] = pd.to_datetime(N['ts'], utc=True)
                H = pd.concat([H, N[['datetime', 'open', 'high', 'low', 'close', 'volume']]]).drop_duplicates('datetime', keep='last').sort_values('datetime')
        except Exception as e:
            print(f'  ! {tk}: LSE fetch failed ({str(e)[:40]}); trying yfinance')
            try:                                                           # fallback feed so a rate-limited LSE never blinds the screener
                import yfinance as yf
                Y = yf.Ticker(YF_SYM.get(tk, tk)).history(period='5d', interval='15m', prepost=True)
                if len(Y):
                    Y = Y.reset_index(); t = pd.to_datetime(Y.pop(Y.columns[0]), utc=True)
                    Y = Y.rename(columns=str.lower)[['open', 'high', 'low', 'close', 'volume']]; Y.insert(0, 'datetime', t.values)
                    Y['datetime'] = pd.to_datetime(Y.datetime, utc=True); Y.loc[Y.volume <= 0, 'volume'] = 1
                    H = pd.concat([H, Y]).drop_duplicates('datetime', keep='last').sort_values('datetime')
            except Exception as e2: print(f'  ! {tk}: yfinance fallback failed too ({str(e2)[:60]}); using bars on disk')
    H[['datetime', 'open', 'high', 'low', 'close', 'volume']].to_parquet(live_f, index=False)
    H['dt'] = H.datetime.dt.tz_convert(ET).dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute
    H = H[(m >= first) & (m < RTH1) & (H.volume > 0) & (H.dt.dt.dayofweek < 5)]
    H = H[H.dt + pd.Timedelta('15min') <= now_et()]                      # completed bars only
    H = H[H.dt >= H.dt.max() - pd.Timedelta(days=400)]                    # enough warm-up, keeps it fast
    return H.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)[['dt', 'open', 'high', 'low', 'close', 'volume']]

def vec(H):
    H = H.reset_index(drop=True).copy()
    reg, lv, atr, _ = run_state_machine(H[['open', 'high', 'low', 'close']].assign(datetime=H.dt))
    H['reg'], H['stop'], H['trig'] = reg, lv - BREAK_ATR * atr, lv + BREAK_ATR * atr   # bull: stop below support; bear: flips up on a close above trig
    return H

def resample_1h(H):
    x = H.set_index('dt')[['open', 'high', 'low', 'close', 'volume']]
    g = x.groupby([x.index.normalize(), pd.Grouper(freq='1h', origin='start_day', offset='9h30min')])
    r = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), volume=('volume', 'sum')).dropna()
    r.index = r.index.get_level_values(1); r = r.reset_index(); r.columns = ['dt'] + list(r.columns[1:])
    return r

def add_1h(H):
    H1 = vec(resample_1h(H))
    s = pd.DataFrame({'known': H1.dt + pd.Timedelta('1h'), 'r': H1.reg.values, 'st': H1['stop'].values}).sort_values('known')
    m = pd.merge_asof(pd.DataFrame({'t': H.dt + pd.Timedelta('15min')}), s, left_on='t', right_on='known', direction='backward')
    H['s1'], H['stop1'] = m.r.fillna(0).values, m.st.values
    return H

def options_ratio():
    f = OPTVOL_FILE
    if not os.path.exists(f): sys.exit('missing xs/ext/optvol_ratio.csv - run: python3 xs/options_volume_ratio.py')
    O = pd.read_csv(f, parse_dates=['date']); O = O[O.sh > 0]; O['r'] = O.opt_contracts * 100 / O.sh
    R = O.pivot_table(index='date', columns='tk', values='r').sort_index()
    last = O.groupby('tk').date.max(); thru = O.date.max()
    R = R.rolling(252, min_periods=60).median().shift(1)
    R[list(last[last < thru - pd.Timedelta(days=int(os.getenv('OPT_STALE_DAYS', '45')))].index)] = np.nan       # options history not current -> not eligible
    ff = os.getenv('RATIO_FROZEN', 'live_data/optvol_ratio_frozen_20261021.csv')        # the ratios the bot actually used before the Oct 22
    if os.path.exists(ff):                                                             # data refresh: past eligibility (and the ledger) stays as traded
        F = pd.read_csv(ff, parse_dates=['date']); F = F[F.sh > 0]; F['r'] = F.opt_contracts * 100 / F.sh
        RF = F.pivot_table(index='date', columns='tk', values='r').sort_index().rolling(252, min_periods=60).median().shift(1)
        cut = CUT2_SWITCH.normalize(); R = R.reindex(columns=R.columns.union(RF.columns)); NEW = R.ffill().iloc[-1].copy()
        old_rows = R.index < cut; R.loc[old_rows] = RF.reindex(index=R.index[old_rows], columns=R.columns).ffill().values
        if R.index.max() < cut: R.loc[cut] = NEW                                       # from the switch on, the rebuilt (full S&P) ratios apply
        R.attrs['now'] = NEW                                                           # today's ratios from the rebuilt file (universe list, tie-break)
    return R, thru

def theta_up():
    try: socket.create_connection(('127.0.0.1', 25503), timeout=2).close(); return True
    except OSError: return False

def update_gamma():
    today = now_et().normalize(); ms = today.replace(day=1)
    cur = f'data/theta/QQQ/QQQ_{ms:%Y-%m}.parquet'
    if os.path.exists(cur): os.remove(cur)
    env = dict(os.environ, START=f'{ms:%Y-%m-%d}', END=f'{today:%Y-%m-%d}', WORKERS='1')
    print('  updating QQQ gamma from ThetaData ...')
    subprocess.run([sys.executable, 'fetch_theta_options.py', 'QQQ'], env=env, capture_output=True)
    for _ in range(10):
        if 'time limit' not in subprocess.run([sys.executable, 'xs/gex_from_theta.py', 'QQQ'], capture_output=True, text=True).stdout: break

# ================================================================ rule replay
def replay(H, elig):
    """Entry / exit-C over history. Returns action for the next bar, entry index, entry price, exit mode."""
    reg, s1, o, c = H.reg.values, H.s1.values, H.open.values, H.close.values; n = len(H)
    inpos = pend = False; e = px = None; b1 = False
    for i in range(1, n):
        if pend: inpos, pend, e, px, b1 = True, False, i, o[i], False
        if not inpos:
            if reg[i] == 1 and reg[i - 1] != 1 and elig[i]: pend = True
            continue
        b1 = b1 or s1[i] == 1; h1 = c[i] / px - 1 >= UP_SWITCH and b1
        if (s1[i] == -1) if h1 else (reg[i] == -1):
            inpos = False
            if i == n - 1: return 'SELL', e, px, h1
    if pend: return 'BUY', None, None, False
    if inpos: return 'HOLD', e, px, (c[-1] / px - 1 >= UP_SWITCH and b1)
    return 'FLAT', None, None, False

def replay_pm(H, elig, carry=None):
    """Premarket+RTH vector, trades only at RTH opens (matches xs/eth_vector2.py). State known at an RTH bar's open = state at
    the close of the bar before it (premarket bars included). The last grid point is the NEXT RTH open (virtual).
    carry = (entry_dt, entry_px) for a position opened under the old RTH rule before CUTOVER."""
    reg, s1, o, c = H.reg.values, H.s1.values, H.open.values, H.close.values; n = len(H)
    m = (H.dt.dt.hour * 60 + H.dt.dt.minute).values; dts = H.dt.values
    g = [j for j in range(1, n) if m[j] >= RTH0] + [n]
    KR = [reg[j - 1] for j in g]; KS = [s1[j - 1] for j in g]; EL = [elig[j] if j < n else elig[-1] for j in g]
    PC = [c[g[p - 1]] if p > 0 else np.nan for p in range(len(g))]   # close of the previous RTH grid bar
    inpos, e, px, b1, p0 = False, None, None, False, 1
    if carry is not None:
        hit = [p for p, j in enumerate(g) if j < n and dts[j] == np.datetime64(carry[0])]
        if hit: inpos, e, px, p0 = True, g[hit[0]], carry[1], hit[0]
    for p in range(max(1, p0), len(g)):
        j = g[p]; virt = p == len(g) - 1
        if not inpos:
            if KR[p] == 1 and KR[p - 1] != 1 and EL[p]:
                if virt: return 'BUY', None, None, False
                inpos, e, px, b1, pe = True, j, o[j], False, p
            continue
        if j == e: continue
        up = PC[p] / px - 1; b1 = b1 or KS[p] == 1; h1 = up >= UP_SWITCH and b1
        if j < n and dts[j] < np.datetime64(CUTOVER): continue          # carried trade: old rule managed it until the cutover
        if (KS[p] == -1) if h1 else (KR[p] == -1):
            if virt: return 'SELL', e, px, h1
            inpos = False
    if inpos: return 'HOLD', e, px, (c[-1] / px - 1 >= UP_SWITCH and b1)
    return 'FLAT', None, None, False

NO_PM = pd.Timestamp(os.getenv('NO_PM_START', '2026-10-20 12:45'))            # Oct 20 2026: no new entries at RTH opens from 14:00 to the close
PM_CUT = int(os.getenv('PM_CUT', '840'))                                            # (afternoon flips: weakest in 2019-26, mostly reverse overnight)
def pm_blocked(t):
    t = pd.Timestamp(t); m = t.hour * 60 + t.minute
    return t >= NO_PM and PM_CUT <= m < RTH1

def replay_track(H, elig, track='15m', carry=None):
    """All trades of one track (15m flips / 1h flips) on premarket+RTH bars, entries/exits at RTH opens only.
    Returns trade dicts: status CLOSED / HOLD / BUY (enter at the next open) / SELL (exit at the next open)."""
    reg, s1, o, c, hi = H.reg.values, H.s1.values, H.open.values, H.close.values, H.high.values; n = len(H)
    m = (H.dt.dt.hour * 60 + H.dt.dt.minute).values; dts = H.dt.values; lk0 = np.datetime64(MODEL2)
    g = [j for j in range(1, n) if m[j] >= RTH0] + [n]
    KR = [reg[j - 1] for j in g]; KS = [s1[j - 1] for j in g]; EL = [elig[j] if j < n else elig[-1] for j in g]
    PC = [c[g[p - 1]] if p > 0 else np.nan for p in range(len(g))]
    T = []; inpos, e, px, b1, peak, p0, hs = False, None, None, False, None, 1, False
    if carry is not None:
        hit = [p for p, j in enumerate(g) if j < n and dts[j] == np.datetime64(carry[0])]
        if hit: inpos, e, px, p0, peak = True, g[hit[0]], carry[1], hit[0], carry[1]
    for p in range(max(1, p0), len(g)):
        j = g[p]; virt = p == len(g) - 1
        if not inpos:
            sig = (KS[p] == 1 and KS[p - 1] != 1) if track == '1h' else (KR[p] == 1 and KR[p - 1] != 1)
            if sig and EL[p] and pm_blocked(dts[j] if not virt else pd.Timestamp(dts[-1]) + pd.Timedelta('15min')): continue
            if sig and EL[p]:
                if virt: T.append(dict(status='BUY', e=None, px=None, t0=VIRT, t1=INF, h1=track == '1h', pg=0.0, lock=False)); break
                inpos, e, px, b1, peak, hs = True, j, o[j], False, o[j], False
            continue
        if j == e: continue
        if g[p - 1] >= e: peak = max(peak, hi[g[p - 1]])
        up = PC[p] / px - 1; b1 = b1 or KS[p] == 1; now1 = up >= UP_SWITCH and b1; hs = hs or now1
        h1 = track == '1h' or now1 or (hs and (virt or dts[j] >= STICKY_START))
        if j < n and dts[j] < np.datetime64(CUTOVER): continue          # carried trade: old rule managed it until the cutover
        ex = (KS[p] == -1) if h1 else (KR[p] == -1)
        pg = peak / px - 1
        lock = (virt or dts[j] >= lk0) and pg >= LOCK_K and c[j - 1] / px - 1 <= (1 - LOCK_G) * pg
        if ex or lock:
            T.append(dict(status='SELL' if virt else 'CLOSED', e=e, px=px, t0=dts[e], t1=VIRT if virt else dts[j], x=None if virt else j, h1=h1, pg=pg, lock=lock and not ex))
            inpos = False
            if virt: break
    if inpos:
        b1 = b1 or s1[-1] == 1
        T.append(dict(status='HOLD', e=e, px=px, t0=dts[e], t1=INF, h1=track == '1h' or (c[-1] / px - 1 >= UP_SWITCH and b1) or hs, pg=peak / px - 1, lock=False))
    for t in T: t['track'] = track
    return T

BREAKER = float(os.getenv('BREAKER', '0.08'))                                  # Oct 20 2026: no new entries while the book is > 8% below its 20-day high
BREAKER_START = pd.Timestamp(os.getenv('BREAKER_START', '2026-10-13 09:30'))   # (realized P&L of closed trades, 28 calendar days; existing positions keep their stops)
def breaker_on(out, t0):
    """book drawdown from the trailing 28-day high of realized equity (closed trades only, same as the backtest) is worse than -BREAKER"""
    if not BREAKER: return False
    t0 = np.datetime64(pd.Timestamp(t0))
    cl = sorted((x['t1'], x['r'] / slots_at(x['t0'])) for x in out if x['status'] == 'CLOSED' and 'r' in x and x['t1'] <= t0)
    if not cl: return False
    ts = np.array([a for a, _ in cl]); eq = np.cumsum(np.log1p([r for _, r in cl]))
    k = ts >= t0 - np.timedelta64(28, 'D'); lvl0 = eq[~k][-1] if (~k).any() else 0.0     # equity level at the start of the window counts too
    return eq[-1] - max(lvl0, eq[k].max() if k.any() else lvl0) < np.log1p(-BREAKER)

# ---- earnings guard (Oct 25 2026): the day before a report, if a position is up >= +3% but by less than the stock's typical earnings move
#      (mean |gap| over its last 8 reports), sell at that day's close - one normal gap could erase the gain. Backtest (xs/batch_oct25.py):
#      2019-21 +59.3/-11, 2022-26 +28.9/-16, 2019-26 +40.2/-16 vs live +55.1/-11, +28.5/-16, +38.5/-16.
EG_ON, EG_MIN, EG_K, EG_START = os.getenv('EARN_GUARD', '1') == '1', 0.03, 1.0, pd.Timestamp(os.getenv('EG_START', '2026-10-26 09:30'))
EARN_CACHE = f'{LIVE}/earn_cache.json'
def earn_events(tk):
    """[(report time ET, after close?, reported?)] from Yahoo, cached per name per day"""
    try: C = json.load(open(EARN_CACHE))
    except Exception: C = {}
    today = f'{now_et():%Y-%m-%d}'
    if C.get(tk, {}).get('day') != today:
        try:
            import yfinance as yf
            e = yf.Ticker(YF_SYM.get(tk, tk)).get_earnings_dates(limit=16)
            ev = [] if e is None else [[str(pd.Timestamp(t).tz_convert('America/New_York').tz_localize(None)), bool(r == r)] for t, r in zip(e.index, e['Reported EPS'])]
            C[tk] = dict(day=today, ev=ev)
            tmp = EARN_CACHE + '.tmp'; json.dump(C, open(tmp, 'w')); os.replace(tmp, EARN_CACHE)
        except Exception as ex: print(f'  ! earnings dates {tk}: {str(ex)[:60]}'); C.setdefault(tk, dict(day='', ev=[]))
    return [(pd.Timestamp(t), pd.Timestamp(t).hour >= 12, rep) for t, rep in C[tk]['ev']]
def earn_guard_check(tk, H, px):
    """-> None, or dict(up, move, gap_day) when tonight / tomorrow morning's report triggers the guard for a position entered at px"""
    ev = earn_events(tk)
    if not ev: return None
    rth = H[(H.dt.dt.hour * 60 + H.dt.dt.minute >= RTH0) & (H.dt.dt.hour * 60 + H.dt.dt.minute < RTH1)]
    days = pd.DatetimeIndex(sorted(rth.dt.dt.normalize().unique()))
    today = now_et().normalize()
    if not len(days) or days[-1] != today: return None
    nxt = pd.bdate_range(today + pd.Timedelta(days=1), periods=1)[0]
    up_next = [t for t, ac, rep in ev if not rep and ((ac and t.normalize() == today) or (not ac and t.normalize() == nxt))]
    if not up_next: return None
    O = rth.groupby(rth.dt.dt.normalize()).open.first(); C_ = rth.groupby(rth.dt.dt.normalize()).close.last(); gaps = []
    for t, ac, rep in sorted(ev):
        if not rep: continue
        g = t.normalize() if not ac else (days[days > t.normalize()][0] if (days > t.normalize()).any() else None)
        if g is None or g not in O.index: continue
        k = C_.index.get_loc(g)
        if k > 0: gaps.append(abs(O[g] / C_.iloc[k - 1] - 1))
    gaps = gaps[-8:]
    if len(gaps) < 4: return None
    up = float(rth.close.iloc[-1]) / px - 1; mv = float(np.mean(gaps))
    return dict(up=up, move=mv, n=len(gaps), report=up_next[0], fire=EG_MIN <= up < EG_K * mv)

LEDGER = os.getenv('LEDGER', f'{LIVE}/ledger.json')                                                   # Oct 21 2026: every trade recorded when taken, never recomputed
def _lkey(tk, track, t0): return f"{tk}|{track}|{pd.Timestamp(t0):%Y-%m-%d %H:%M}"
def ledger_load():
    if not os.path.exists(LEDGER) and os.path.exists('live_data/ledger_seed.json'):
        import shutil; shutil.copy('live_data/ledger_seed.json', LEDGER)
    try: return json.load(open(LEDGER))
    except Exception: return {}
def ledger_save(L):
    tmp = LEDGER + '.tmp'; json.dump(L, open(tmp, 'w'), indent=0, default=str); os.replace(tmp, LEDGER)

def allocate(trades):
    """SLOTS shared by every ticker and both tracks, first come first served (same as the backtest sim, deterministic ties)."""
    trades = sorted(trades, key=lambda t: (t['t0'], -t.get('ratio', 0) if TIE_RATIO else 0, t['track'] != '15m', t['tk'])); op, out = [], []
    for t in trades:
        op = [x for x in op if x > t['t0']]
        if t.get('forced'): op.append(t['t1']); out.append(t); continue           # open position already in the ledger: it holds its slot
        if len(op) < slots_at(t['t0']):
            if pd.Timestamp(t['t0']) >= BREAKER_START and breaker_on(out, t['t0']): t['blocked'] = True; continue
            op.append(t['t1']); out.append(t)
    return out

def scan(args):
    if args.update_gamma and theta_up(): update_gamma()
    RAT, rat_thru = options_ratio()
    last_ratio = RAT.attrs.get('now', RAT.ffill().iloc[-1])
    cut_now = CUT2 if now_et() >= CUT2_SWITCH else CUT
    elig_now = sorted(t for t in RAT.columns if t not in NO_ETF and ((last_ratio.get(t, 0) >= min(cut_now, CUT2) and in_sp(t, now_et())) or (t in OLD_UNIVERSE and last_ratio.get(t, 0) >= 1.5)))   # scan the >= 1.5 list now; entries are gated per bar
    # names that dropped below the cut still get scanned while they could hold a position from the live period (exits keep working;
    # new entries are gated per day by the ratio below), i.e. anything eligible on some day since the start
    since = RAT.loc[RAT.index >= max(START.normalize(), RAT.index.max() - pd.Timedelta(days=120)) - pd.Timedelta(days=1)]
    was = {t for t in since.columns if t not in NO_ETF and (((since[t] >= min(CUT, CUT2)).any() and in_sp(t, now_et())) or (t in OLD_UNIVERSE and (since[t] >= 1.5).any()))}
    LG = ledger_load(); lg_open = {v['tk'] for k, v in LG.items() if not k.startswith('__') and v.get('status') == 'open'}
    elig_now = sorted(set(elig_now) | was | (lg_open & set(RAT.columns)))
    gf = 'xs/ext/gex/QQQ.csv'                                             # only needed for the (off by default) QQQ slot
    G = pd.read_csv(gf, parse_dates=['date']).sort_values('date').set_index('date').net_gex if os.path.exists(gf) else pd.Series([0.0], index=[pd.Timestamp('2000-01-01')])
    slots = SLOTS; out = []; TR = []; HS = {}
    for tk in elig_now + (['QQQ'] if INCLUDE_QQQ else []):
        pre = SESSION == 'pre' and tk != 'QQQ'
        H = load_bars(tk, not args.no_fetch, PRE0 if pre else RTH0)
        if H is None or len(H) < 500: print(f'  ! {tk}: no price history'); continue
        H = add_1h(vec(H)); day = H.dt.dt.normalize()
        if tk == 'QQQ':
            gv = G.reindex(pd.DatetimeIndex(sorted(set(day))).union(G.index)).ffill().shift(1); elig = (day.map(gv) < 0).values
        else:
            rv = RAT[tk].reindex(pd.DatetimeIndex(sorted(set(day))).union(RAT.index)).ffill(); rr = day.map(rv); sp_ok = day.map(lambda x: in_sp(tk, x)).values.astype(bool)
            elig = np.where((H.dt < OPT_SWITCH).values, (rr >= 1.5) & (tk in OLD_UNIVERSE),
                            np.where((H.dt < CUT2_SWITCH).values, (rr >= CUT) & sp_ok, (rr >= CUT2) & sp_ok))
        elig = elig & (H.dt >= START).values                                   # live ledger: nothing before the start date
        carry = None
        if pre:
            mm = H.dt.dt.hour * 60 + H.dt.dt.minute; R0 = H[(mm >= RTH0).values].reset_index(drop=True)     # old RTH rule, for positions opened before the cutover
            R0 = add_1h(vec(R0)); k0 = (R0.dt < CUTOVER).values; el0 = (R0.dt.dt.normalize().map(rv) >= CUT).values & (R0.dt >= START).values
            a0, e0, px0, _ = replay(R0[k0].reset_index(drop=True), el0[k0]) if k0.sum() > 1 else ('FLAT', None, None, False)
            carry = (R0[k0].reset_index(drop=True).dt.iloc[e0], px0) if a0 == 'HOLD' else None
        el15 = elig & (H.dt >= CUTOVER).values if pre else elig
        tr = replay_track(H, el15, '15m', carry)
        if tk != 'QQQ': tr += replay_track(H, elig & (H.dt >= MODEL2).values, '1h')
        for t in tr: t['tk'] = tk; t['ratio'] = float(last_ratio.get(tk, 0) or 0)
        TR += tr; HS[tk] = H
    for t in TR:                                                      # realized return of closed trades (for the drawdown breaker)
        if t['status'] == 'CLOSED': t['r'] = float(HS[t['tk']].open.values[t['x']]) / t['px'] - 1
    # ---- trade ledger (Oct 21 2026): a trade is written down when it is taken and never recomputed after that.
    #      bar revisions / ratio rebuilds can't erase a real position (LLY 10-21) or invent a past one (AVGO 10-13).
    asof = pd.Timestamp(LG.get('__asof__', '2000-01-01')); LT = {k: v for k, v in LG.items() if not k.startswith('__')}
    have = {_lkey(t['tk'], t['track'], t['t0']): t for t in TR if t['px']}
    for k, v in LT.items():
        if k in have:                                                          # recorded trade: it keeps its slot whatever the re-sort says
            t = have[k]; t['forced'] = True
            if v['status'] == 'closed' and t['status'] != 'CLOSED':             # replay revived a trade the ledger closed: close it before the slot count
                d = HS[t['tk']].dt.values; t1 = np.datetime64(pd.Timestamp(v['t1']))
                t.update(status='CLOSED', t1=t1, x=int(min(np.searchsorted(d, t1), len(d) - 1)), lock=bool(v.get('lock')))
            continue
        if v['tk'] not in HS: continue
        H = HS[v['tk']]; d = H.dt.values; t0 = np.datetime64(pd.Timestamp(v['t0']))
        if v['status'] == 'open':                                              # replay lost it: keep managing it from the recorded entry
            o = [o for o in replay_track(H, np.ones(len(H), bool), v['track'], carry=(pd.Timestamp(v['t0']), float(v['px']))) if o['px'] and o['t0'] == t0][:1]
            if not o: continue
            o = o[0]
        else:                                                                  # closed and gone from the replay: keep it as recorded
            t1 = np.datetime64(pd.Timestamp(v['t1']))
            o = dict(status='CLOSED', px=float(v['px']), t0=t0, t1=t1, e=int(min(np.searchsorted(d, t0), len(d) - 1)),
                     x=int(min(np.searchsorted(d, t1), len(d) - 1)), h1=v['track'] == '1h', pg=0.0, lock=bool(v.get('lock')), ledger_only=True)
        o.update(tk=v['tk'], track=v['track'], ratio=99.0, forced=True)
        TR = [t for t in TR if not (t['tk'] == o['tk'] and t['track'] == o['track'] and t['t0'] >= o['t0'] and t['t0'] < o['t1'])] + [o]
    cut_old = np.datetime64(asof - pd.Timedelta('30min'))                     # trades that start before the last scan but were never recorded = phantoms
    TR = [t for t in TR if t.get('forced') or not t['px'] or t['t0'] >= cut_old or pd.Timestamp(t['t0']) < START]
    for t in TR:
        if t['status'] == 'CLOSED':
            k = _lkey(t['tk'], t['track'], t['t0']); v = LT.get(k)
            t['r'] = (float(v['xpx']) / float(v['px']) - 1) if v and v['status'] == 'closed' else float(HS[t['tk']].open.values[t['x']]) / t['px'] - 1
    AL = allocate(TR); live = {(t['tk'], t['track']): t for t in AL if t['status'] != 'CLOSED'}
    for t in AL:                                                      # on-time rule prices: entry = open of the entry bar, exit = open of the exit bar
        if t.get('ledger_only'): continue
        t['xpx'] = float(HS[t['tk']].open.values[t['x']]) if t['status'] == 'CLOSED' else np.nan
    for t in AL:                                                      # ledger sync: record new trades / exits once; recorded values win
        if not t['px'] or t['status'] not in ('HOLD', 'SELL', 'CLOSED') or pd.Timestamp(t['t0']) < START: continue
        k = _lkey(t['tk'], t['track'], t['t0']); v = LG.get(k)
        if v is None: v = LG[k] = dict(tk=t['tk'], track=t['track'], t0=str(pd.Timestamp(t['t0'])), px=float(t['px']), status='open')
        if v['status'] == 'open' and t['status'] == 'CLOSED':
            v.update(status='closed', t1=str(pd.Timestamp(t['t1'])), xpx=float(t['xpx']), lock=bool(t.get('lock')))
        t['px'] = float(v['px'])
        if v['status'] == 'closed':
            t['xpx'] = float(v['xpx']); t['t1'] = np.datetime64(pd.Timestamp(v['t1'])); t['lock'] = bool(v.get('lock'))
            if t['status'] != 'CLOSED':                                  # replay revived a trade the ledger already closed (bar revision): keep it closed
                d = HS[t['tk']].dt.values; t['status'] = 'CLOSED'; t['x'] = int(min(np.searchsorted(d, t['t1']), len(d) - 1))
                live.pop((t['tk'], t['track']), None)
    EG = []                                                               # earnings guard: warn from 15:30, sell at the 15:45 bar's close
    if EG_ON and now_et() >= EG_START:
        for t in AL:
            if t['status'] != 'HOLD' or not t['px'] or t['tk'] == 'QQQ': continue
            H = HS[t['tk']]; lb = H.dt.iloc[-1]
            if lb.normalize() != now_et().normalize() or lb.hour * 60 + lb.minute < 15 * 60 + 15: continue
            try: g = earn_guard_check(t['tk'], H, float(t['px']))
            except Exception as ex: print(f"  ! earnings guard {t['tk']}: {str(ex)[:60]}"); continue
            if not g: continue
            g.update(tk=t['tk'], track=t['track'], px=float(t['px']), last=float(H.close.iloc[-1]), bar=lb, final=lb.hour * 60 + lb.minute >= 15 * 60 + 45)
            EG.append(g)
            k = _lkey(t['tk'], t['track'], t['t0']); v = LG.get(k)
            if g['fire'] and g['final'] and v is not None and v['status'] == 'open':      # close at the 15:45 bar's close (= the 16:00 close)
                v.update(status='closed', t1=str(lb), xpx=float(H.close.iloc[-1]), lock=False, note='earnings guard')
                t.update(status='CLOSED', t1=np.datetime64(lb), xpx=float(H.close.iloc[-1]), x=len(H) - 1, eg=True); live.pop((t['tk'], t['track']), None)
    LG['__asof__'] = str(pd.Timestamp(max(H.dt.iloc[-1] for H in HS.values())))
    if not getattr(args, 'no_ledger', False): ledger_save(LG)
    def mk_row(tk, H, track, t):
        L = H.iloc[-1]; act = t['status'] if t else 'FLAT'
        e, px, h1, pg = (t['e'], t['px'], t['h1'], t['pg']) if t else (None, None, track == '1h', 0.0)
        stop = float(L.stop1 if h1 else L['stop'])
        if px and pg >= LOCK_K: stop = max(stop, px * (1 + (1 - LOCK_G) * pg))       # profit-lock floor
        note = '' if tk != 'QQQ' else ('QQQ neg gamma' if G.iloc[-1] < 0 else 'QQQ pos gamma: no new entries')
        if t and t['lock']: note = f"profit lock: gave back half of the +{pg:.1%} peak"
        elif track == '1h' and act == 'BUY': note = '1h vector flip'
        return dict(ticker=tk if track == '15m' else f'{tk} 1h', base=tk, track=track, action=act, H=H, e=e, px=px, h1=h1, last=float(L.close), bar=L['dt'],
                    stop=stop, state15=int(L.reg), state1h=int(L.s1), trig=float(L.trig) if L.reg == -1 else np.nan,
                    ratio=float(last_ratio.get(tk, np.nan)) if tk != 'QQQ' else np.nan, peak=pg,
                    pnl=(L.close / px - 1) if px else np.nan, note=note)
    for tk, H in HS.items():
        for track in (('15m',) if tk == 'QQQ' else ('15m', '1h')): out.append(mk_row(tk, H, track, live.get((tk, track))))
    # honorable mentions (Oct 25 2026): flips the book skipped because every slot was full - alerted and tracked, never traded
    taken = {id(t) for t in AL}
    hm = {}
    for t in TR:
        if id(t) in taken or t.get('blocked') or t['tk'] == 'QQQ' or t['status'] not in ('BUY', 'HOLD', 'SELL'): continue
        if t['status'] != 'BUY' and (not t['px'] or pd.Timestamp(t['t0']) < max(START, now_et() - pd.Timedelta(days=30))): continue
        hm[(t['tk'], t['track'])] = t
    hm_rows = [mk_row(tk, HS[tk], track, t) for (tk, track), t in hm.items()]
    brk = breaker_on(AL, now_et())
    if brk: print(f'  !! BREAKER: book is more than {BREAKER:.0%} below its 20-day high - no new entries (open positions keep their stops)')
    meta = dict(breaker=brk, trades=AL, HS=HS, hm=hm_rows, eg=EG, slots=slots, w=1 / slots, rat_thru=rat_thru, g_thru=G.index.max(), capital=args.capital,
                ratios={t: float(last_ratio.get(t, np.nan)) for t in elig_now})
    if not args.no_fetch and os.getenv('SITE_SNAPSHOT', '1') == '1':
        try: write_site_snapshot(out, meta)
        except Exception as e: print(f'  ! site snapshot: {str(e)[:80]}')
    return out, meta


def site_ranges(rows):
    """adaptive expected high / low per held name for the website, recomputed on every scan from today's extremes so far"""
    out = {}
    for r in rows:
        if r['action'] not in ('HOLD', 'SELL') or not r['px'] or r['base'] in out: continue
        try:
            rl = range_levels(r['H']); x = extreme_info(r['H']) or {}
            if not rl: continue
            f2 = lambda v: round(float(v), 2)
            z = dict(budget=round(rl['budget'] * 100, 2), wide=round(2.4 * rl['sig'] * 100, 2), lo=f2(rl['lo']), hi=f2(rl['hi']), last=f2(r['last']), pre=bool(rl['pre']),
                     used=round(rl['used'] / 1.7 * 100), hi_typ=f2(rl['hi_typ']), hi_wide=f2(rl['hi_wide']), lo_typ=f2(rl['lo_typ']), lo_wide=f2(rl['lo_wide']),
                     bar=f"{pd.Timestamp(r['H'].dt.iloc[-1]) + pd.Timedelta('15min'):%a %H:%M}")
            for side in ('low', 'high'):
                e = x.get(side) if x.get('live') else None
                z[side + '_in'] = dict(p=round(e['p'] * 100), typ=f2(e['tgt'][0]), b80=f2(e['tgt'][1])) if e else None
            out[r['base']] = z
        except Exception as e: print(f'  ! site range {r["base"]}: {str(e)[:60]}')
    return out

def write_site_snapshot(rows, meta):
    """live/site_snapshot.json for the website (screener_service pushes it to BACKEND_URL/api/site/push)."""
    t = now_et(); m = t.hour * 60 + t.minute
    held = {r['base'] for r in rows if r['action'] in ('HOLD', 'SELL') and r['px']}
    pos = [dict(name=r['ticker'], base=r['base'], entry=f"{r['px']:.2f}", stop=f"{r['stop']:.2f}", last=f"{r['last']:.2f}",
                pnl=round(r['pnl'] * 100, 2), peak=round(max(r.get('peak') or 0, r['pnl']) * 100, 2), selling=r['action'] == 'SELL')
           for r in rows if r['action'] in ('HOLD', 'SELL') and r['px']]
    sig = []
    f = f'{LIVE}/signal_log.csv'
    if os.path.exists(f):
        L = pd.read_csv(f, parse_dates=['run_at']); L = L[L.run_at.dt.normalize() == t.normalize()]
        for x in L.itertuples():
            sig.append(dict(time=f'{x.run_at:%H:%M}', action=x.action, ticker=x.ticker, price=f'{x.price:.2f}',
                            pnl=round(x.pnl * 100, 2) if (x.action == 'SELL' and x.pnl == x.pnl) else None))
        seen = set(); sig = [z for z in sig[::-1] if not ((z['time'], z['action'], z['ticker']) in seen or seen.add((z['time'], z['action'], z['ticker'])))]
    watch = [dict(tk=k, ratio=round(v, 2), held=k in held) for k, v in sorted(meta.get('ratios', {}).items(), key=lambda kv: -kv[1]) if v == v and (v >= min(CUT, CUT2) or k in held)]
    hist = {}
    try:
        T, g, book, spy = history(rows, meta, fetch=True); w = meta['w']
        fmt = lambda d: f'{pd.Timestamp(d):%a %m-%d %H:%M}'
        tr = [dict(name=z['name'], entry=f"{fmt(z['t0'])}  {z['p0']:.2f}", exit=('open  ' + (f"{z['p1']:.2f}" if z['p1'] == z['p1'] else '')) if z['open'] else f"{fmt(z['t1'])}  {z['p1']:.2f}",
                   ret=round(z['ret'] * 100, 2), peak=round(z['peak'] * 100, 2), book=round(z['ret'] * z['w'] * 100, 2), open=bool(z['open']), lock=bool(z['lock'])) for z in T]
        key = [pd.Timestamp(z['t1']) if not z['open'] else pd.Timestamp(z['t0']) for z in T]
        order = sorted(range(len(tr)), key=lambda i: (not tr[i]['open'], -key[i].value))
        tr = [tr[i] for i in order]
        Q = spy_15m(True, 'QQQ'); qs = pd.Series(Q.close.values, index=Q.dt.values); qb = qs[qs.index < np.datetime64(START)]
        q0 = qb.iloc[-1] if len(qb) else (qs.iloc[0] if len(qs) else np.nan)
        qq = np.r_[qs.reindex(g[:-1], method='ffill').values / q0 - 1, ((live_px('QQQ') or (qs.iloc[-1] if len(qs) else np.nan)) / q0 - 1)]
        k = np.unique(np.r_[np.linspace(0, len(g) - 1, min(len(g), 400)).astype(int), len(g) - 1])
        nz = lambda v: None if v != v else round(float(v) * 100, 3)
        curve = dict(t=[int(pd.Timestamp(g[i]).timestamp() * 1000) for i in k], book=[nz(book[i]) for i in k], spy=[nz(spy[i]) for i in k], qqq=[nz(qq[i]) for i in k])
        hist = dict(curve=curve, qqq=nz(qq[-1]), trades=tr, book=round(float(book[-1]) * 100, 2), spy=round(float(spy[-1]) * 100, 2) if spy[-1] == spy[-1] else None,
                    realized=round(sum(z['book'] for z in tr if not z['open']), 2), open=round(sum(z['book'] for z in tr if z['open']), 2), since=f'{pd.Timestamp(START):%a %b %d}')
    except Exception as e: print(f'  ! site history: {str(e)[:80]}')
    snap = dict(asof=t.isoformat(), asof_ms=int(pd.Timestamp(t, tz=ET).timestamp() * 1000), market_open=t.dayofweek < 5 and RTH0 <= m < RTH1,
                slots=meta['slots'], cutoff=CUT2, positions=pos, signals=sig, watch=watch, history=hist, ranges=site_ranges(rows))
    snap['analysis'] = None                                                         # the 09:00 write-up (frozen); ranges above refresh every scan
    for fa in (f'{LIVE}/analysis.json', 'live_data/analysis.json'):                # live copy, else the seed shipped in git
        try: snap['analysis'] = json.load(open(fa)); break
        except Exception: pass
    tmp = f'{LIVE}/site_snapshot.json.tmp'; json.dump(snap, open(tmp, 'w')); os.replace(tmp, f'{LIVE}/site_snapshot.json')

# ================================================================ cards
def _frame(fig, accent):
    fig.patches.append(FancyBboxPatch((0.015, 0.02), 0.97, 0.96, boxstyle='round,pad=0,rounding_size=0.01', transform=fig.transFigure, fc=EMB, ec='none', zorder=-10))
    fig.patches.append(FancyBboxPatch((0.015, 0.02), 0.006, 0.96, boxstyle='square,pad=0', transform=fig.transFigure, fc=accent, ec='none', zorder=-9))

def _chart(ax, H, e, px, stop, bars, big=False, mark_last=None, h1=False):
    """Vector-colored candles + the trailing stop as a glowing step line + a cushion gradient between price and stop."""
    x = H.iloc[-bars:].reset_index(drop=True); t = np.arange(len(x)); o, h, l, c = x.open.values, x.high.values, x.low.values, x.close.values
    ax.set_facecolor(PANEL); [s.set_visible(False) for s in ax.spines.values()]
    bull = x.reg.values == 1
    col = np.where(bull, BLUE, '#8a3a3d'); wick = np.where(bull, '#2a5d9c', '#5e2b2d')
    ax.vlines(t, l, h, color=wick, lw=1 if big else .8, zorder=2)
    body_lo, body_hi = np.minimum(o, c), np.maximum(o, c)
    ax.bar(t, np.maximum(body_hi - body_lo, (h.max() - l.min()) * .002), bottom=body_lo, width=.62, color=col, edgecolor='none', zorder=3)
    # trailing stop history (what the stop was on each bar while the vector was bull)
    sh = (x.stop1 if h1 else x['stop']).values.astype(float); sh = np.where(bull | h1, sh, np.nan)
    for lw, a in ((7, .06), (4, .12), (1.8, 1)):                                    # glow
        ax.step(t, sh, where='mid', color=CRIT, lw=lw if big else lw * .8, alpha=a, zorder=4, solid_capstyle='round')
    # cushion: gradient band from stop up to price, only while in the trade
    if px and e is not None:
        ei = max(0, len(x) - (len(H) - e)); band = slice(ei, len(x))
        for k, a in enumerate(np.linspace(.16, .02, 6)):
            lo_ = sh[band] + (c[band] - sh[band]) * k / 6; hi_ = sh[band] + (c[band] - sh[band]) * (k + 1) / 6
            ax.fill_between(t[band], lo_, hi_, where=c[band] > sh[band], color=GOOD, alpha=a, lw=0, zorder=1, step='mid')
    lo, hi = np.nanmin([l.min(), stop, px or l.min()]), np.nanmax([h.max(), px or h.max()])
    pad = (hi - lo) * .10 or 1; ax.set_ylim(lo - pad, hi + pad * 2); ax.set_xlim(-1, len(x) + (12 if big else 17))
    yl = ax.get_ylim(); gap = (yl[1] - yl[0]) * .07; xe = len(x) + .5
    if px:
        ax.hlines(px, 0, len(x) - 1, color=T2, lw=.9, ls=(0, (4, 3)), zorder=5, alpha=.8)
        ei = len(x) - (len(H) - e) if e is not None else None
        if ei is not None and 0 <= ei < len(x): ax.scatter([ei], [px], s=70 if big else 40, marker='^', color=GOOD, edgecolor=PANEL, linewidth=1.5, zorder=6)
    ye, ys = px, stop
    if px and abs(px - stop) < gap: ye, ys = (px + gap / 2, stop - gap / 2) if px >= stop else (px - gap / 2, stop + gap / 2)
    if px: ax.text(xe, ye, f' entry {px:.2f}', color=T2, fontsize=9.5 if big else 8, va='center')
    ax.text(xe, ys, f' stop {stop:.2f}', color=CRIT, fontsize=9.5 if big else 8, va='center', fontweight='bold')
    if mark_last:
        for s_, a in ((420, .10), (200, .22)): ax.scatter([t[-1]], [c[-1]], s=s_, color=mark_last[1], alpha=a, lw=0, zorder=6)
        ax.scatter([t[-1]], [c[-1]], s=90, marker=mark_last[0], color=mark_last[1], edgecolor=PANEL, linewidth=1.5, zorder=7)
    days = x.dt.dt.normalize(); ticks = [i for i in range(1, len(x)) if days[i] != days[i - 1]]
    ax.set_xticks(ticks); ax.set_xticklabels([f'{x.dt[i]:%a}' for i in ticks], fontsize=7); ax.set_yticks([]); ax.tick_params(length=0)
    [ax.axvline(i - .5, color=GRID, lw=.6, zorder=0) for i in ticks]

def _room(r):
    """how far price is above the active stop, in % (how much room before an exit)."""
    return r['last'] / r['stop'] - 1

def slot_wheel(ax, rows, meta):
    """Ring of all slots: filled = position (color = P&L), empty = cash. Center = invested %."""
    P = sorted([r for r in rows if r['action'] in ('HOLD', 'SELL')], key=lambda r: -r['pnl']); n = meta['slots']
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2; w = 2 * np.pi / n * .86
    ax.set_theta_direction(-1); ax.set_axis_off(); ax.set_ylim(0, 1.0)
    for k in range(n):
        if k < len(P):
            pnl = P[k]['pnl']; cc = GOOD if pnl >= 0 else CRIT; a = .35 + .65 * min(1, abs(pnl) / .06)
            ax.bar(th[k], .30, width=w, bottom=.62, color=cc, alpha=a, edgecolor=EMB, linewidth=2)
            ax.text(th[k], .77, P[k]['ticker'], ha='center', va='center', fontsize=7.5, fontweight='bold', color=T1, rotation=0)
        else:
            ax.bar(th[k], .30, width=w, bottom=.62, color=GRID, alpha=.55, edgecolor=EMB, linewidth=2)
    inv = len(P) / n
    ax.text(.5, .53, f'{inv:.0%}', transform=ax.transAxes, ha='center', va='center', fontsize=20, fontweight='bold', color=T1)
    ax.text(.5, .38, 'invested', transform=ax.transAxes, ha='center', va='center', fontsize=8, color=MUT)

def positions_card(rows, meta, path, title='Open positions', hm=False):
    P = sorted([r for r in rows if r['action'] in ('HOLD', 'SELL')], key=lambda r: -r['pnl'])
    ncol = 3; nrow = max(1, int(np.ceil(len(P) / ncol))); HDR = 2.6
    H_in = HDR + 3.0 * nrow; fig = plt.figure(figsize=(10, H_in), facecolor=BG); _frame(fig, BLUE)
    asof = pd.Series([r['bar'] for r in rows if r['ticker'] != 'QQQ'] or [r['bar'] for r in rows]).max() + pd.Timedelta('15min')
    fig.text(0.045, 1 - 0.55 / H_in, title, fontsize=19, fontweight='bold')
    cap = f" · ${meta['capital'] * meta['w']:,.0f} each" if meta['capital'] else ''
    sub = (f"{len(P)} flips skipped because all {meta['slots']} slots were full · tracked, not traded" if hm else f"{len(P)} of {meta['slots']} slots · {meta['w']:.1%} each{cap}")
    fig.text(0.045, 1 - 0.90 / H_in, sub, fontsize=10.5, color=T2)
    fig.text(0.045, 1 - 1.18 / H_in, f"as of {asof:%a %b %d %H:%M} ET · Vector screener", fontsize=9, color=MUT)
    if P:
        tot = np.mean([r['pnl'] for r in P]); best = P[0]; worst = P[-1]
        for k, (h, v, cc) in enumerate((('AVG OPEN P&L', f'{tot:+.1%}', GOOD if tot >= 0 else CRIT), ('BEST', f"{best['ticker']} {best['pnl']:+.1%}", GOOD if best['pnl'] >= 0 else CRIT),
                                        ('WEAKEST', f"{worst['ticker']} {worst['pnl']:+.1%}", GOOD if worst['pnl'] >= 0 else CRIT))):
            x0 = 0.045 + k * 0.205
            fig.patches.append(FancyBboxPatch((x0, 1 - 2.25 / H_in), 0.19, 0.85 / H_in, boxstyle='round,pad=0,rounding_size=0.008', transform=fig.transFigure, fc=PANEL, ec='none', zorder=-8))
            fig.text(x0 + 0.012, 1 - 1.62 / H_in, h, fontsize=7.5, color=MUT, fontweight='bold')
            fig.text(x0 + 0.012, 1 - 2.05 / H_in, v, fontsize=14, fontweight='bold', color=cc)
    wax = fig.add_axes([0.70, 1 - 2.45 / H_in, 0.24, 2.25 / H_in], projection='polar'); slot_wheel(wax, rows, meta)
    if not P:
        fig.text(0.5, 0.25, 'No open positions - all cash / T-bills', ha='center', fontsize=13, color=T2)
    else:
        gs = fig.add_gridspec(nrow, ncol, left=0.045, right=0.90, top=1 - (HDR + .55) / H_in, bottom=0.85 / H_in, wspace=0.42, hspace=0.85)
        for k, r in enumerate(P):
            ax = fig.add_subplot(gs[k // ncol, k % ncol]); _chart(ax, r['H'], r['e'], r['px'], r['stop'], 78, h1=r['h1'])
            ax.text(0, 1.24, r['ticker'], transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom')
            ax.text(1, 1.24, f"{r['pnl']:+.1%}", transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom', ha='right', color=GOOD if r['pnl'] >= 0 else CRIT)
            cu = _room(r); how = 'SELL at next open' if r['action'] == 'SELL' else f"{'1h' if r['h1'] else '15m'} stop · {cu:.1%} room"
            ax.text(0, 1.06, f"{r['last']:.2f} now  ·  {how}", transform=ax.transAxes, fontsize=8, va='bottom', color=CRIT if r['action'] == 'SELL' else MUT)
            # room meter: 0..6% above the stop
            ax.add_patch(plt.Rectangle((0, -0.17), 1, 0.035, transform=ax.transAxes, color=GRID, clip_on=False, lw=0))
            fr = float(np.clip(cu / .06, 0, 1)) if np.isfinite(cu) else 0
            ax.add_patch(plt.Rectangle((0, -0.17), fr, 0.035, transform=ax.transAxes, color=GOOD if cu > .01 else '#fab219', clip_on=False, lw=0))
    fig.text(0.045, 0.18 / H_in, 'candles: blue = vector bull, red = bear   ·   glowing red step = trailing stop (exit if a bar CLOSES below)   ·   green band = room above stop   ·   ▲ entry', fontsize=8, color=MUT)
    fig.savefig(path, dpi=140, facecolor=BG); plt.close(fig)

def mark_live(rows, fetch):
    """replace the last completed 15m close with the live trade price (open positions only) -> P&L is not 15 minutes stale"""
    for r in rows:
        if r['action'] not in ('HOLD', 'SELL') or not r['px']: continue
        px = live_px(r['base']) if fetch else None
        if px: r['last'], r['live'] = px, True
        r['pnl'] = r['last'] / r['px'] - 1
    return rows

def spy_15m(fetch=True, tk='SPY'):
    """SPY (or tk) 15m RTH closes since the live start: cached in live/<tk>_15m.parquet, topped up from yfinance (60-day 15m window) so it keeps growing"""
    f = f'{LIVE}/{tk}_15m.parquet'; C = pd.read_parquet(f) if os.path.exists(f) else pd.DataFrame(columns=['dt', 'close'])
    if fetch:
        try:
            import yfinance as yf
            y = yf.Ticker(tk).history(period='60d', interval='15m')
            if len(y):
                N = pd.DataFrame({'dt': y.index.tz_convert(ET).tz_localize(None), 'close': y.Close.values})
                C = pd.concat([C, N]).drop_duplicates('dt', keep='last').sort_values('dt'); C.to_parquet(f, index=False)
        except Exception as e: print(f'  ! {tk} fetch failed: {e}')
    C['dt'] = pd.to_datetime(C.dt); return C[C.dt + pd.Timedelta('15min') <= now_et()]

def history(rows, meta, fetch=True):
    """every live trade (closed at the rule's on-time fills = open of the exit bar; open marked live) + the book P&L path on the 15m grid"""
    w = meta['w']; HS = meta['HS']; R = {(r['base'], r['track']): r for r in rows}
    T = [t for t in meta['trades'] if t['status'] in ('CLOSED', 'HOLD', 'SELL') and t['px']]
    ref = max(HS.values(), key=len); g = ref.dt[(ref.dt >= START) & ((ref.dt.dt.hour * 60 + ref.dt.dt.minute) >= RTH0)].values
    now = np.datetime64(now_et().floor('min')); book = np.zeros(len(g)); out = []
    for t in T:
        H = HS[t['tk']]; c = pd.Series(H.close.values, index=H.dt.values).reindex(g, method='ffill').values
        r_ = np.where(g < t['t0'], 0.0, c / t['px'] - 1); lr = R.get((t['tk'], t['track']))
        if t['status'] == 'CLOSED': rx = t['xpx'] / t['px'] - 1; r_ = np.where(g >= t['t1'], rx, r_); xp, xt = t['xpx'], pd.Timestamp(t['t1'])
        else: rx = lr['pnl'] if lr else 0.0; xp, xt = (lr['last'] if lr else np.nan), None
        wt = 1 / slots_at(t['t0']); book += wt * r_
        rth = ((H.dt.dt.hour * 60 + H.dt.dt.minute) >= RTH0).values; hi_ = H.high.values
        seg = slice(t['e'], t['x'] if t['status'] == 'CLOSED' else len(H))      # RTH highs from the entry bar until the exit open
        pk = np.nanmax(np.r_[hi_[seg][rth[seg]], xp if xp == xp else -np.inf, t['px']]) / t['px'] - 1
        out.append(dict(w=wt, peak=pk, name=t['tk'] + ('' if t['track'] == '15m' else ' 1h'), t0=pd.Timestamp(t['t0']), p0=t['px'], t1=xt, p1=xp, ret=rx,
                        open=t['status'] != 'CLOSED', sell=t['status'] == 'SELL', lock=bool(t.get('lock'))))
    tot = sum(z['w'] * z['ret'] for z in out)
    S = spy_15m(fetch); sp = pd.Series(S.close.values, index=S.dt.values)
    base = sp[sp.index < np.datetime64(START)]; b0 = base.iloc[-1] if len(base) else (sp.iloc[0] if len(sp) else np.nan)
    spy = sp.reindex(g, method='ffill').values / b0 - 1 if len(sp) else np.full(len(g), np.nan)
    spy_now = (live_px('SPY') if fetch else None) or (sp.iloc[-1] if len(sp) else np.nan)
    return out, np.r_[g, now], np.r_[book, tot], np.r_[spy, spy_now / b0 - 1]

def history_card(rows, meta, path, fetch=True):
    """Discord /history: every trade since the start with its return, the whole book P&L, and the book vs SPY."""
    out, gx, by, sy = history(rows, meta, fetch); w = meta['w']
    C = sorted([z for z in out if not z['open']], key=lambda z: z['t1'], reverse=True); O = sorted([z for z in out if z['open']], key=lambda z: -z['ret'])
    L = O + C; MAXR = 28; show = L[:MAXR]; nr = len(show)
    H_in = 6.3 + .27 * max(nr, 4); fig = plt.figure(figsize=(10, H_in), facecolor=BG); _frame(fig, BLUE); Y = lambda inch: 1 - inch / H_in
    tot, real, op = by[-1], sum(z['w'] * z['ret'] for z in C), sum(z['w'] * z['ret'] for z in O); spy = sy[-1]
    wins = [z['ret'] for z in C if z['ret'] > 0]; loss = [z['ret'] for z in C if z['ret'] <= 0]
    fig.text(.045, Y(.55), 'Trade history', fontsize=19, fontweight='bold')
    fig.text(.045, Y(.85), f"since {START:%a %b %d} · {len(C)} closed, {len(O)} open · each trade = 1 slot ({w:.1%}) · on-time rule fills, open marked "
             + ('live' if any(r.get('live') for r in rows) else 'at the last close') + f" · {pd.Timestamp(gx[-1]):%a %b %d %H:%M} ET", fontsize=8.5, color=MUT)
    kp = [('BOOK P&L', f'{tot:+.2%}', f'realized {real:+.2%} · open {op:+.2%}', tot), ('SPY', f'{spy:+.2%}' if spy == spy else '—', 'same period', spy),
          ('VS SPY', f'{tot - spy:+.2%}' if spy == spy else '—', 'book minus SPY', tot - spy if spy == spy else 0),
          ('WIN RATE', f'{len(wins) / max(1, len(C)):.0%}', (f'avg win {np.mean(wins):+.1%} · loss {np.mean(loss):+.1%}' if wins and loss else f'{len(C)} closed'), None)]
    for k, (h, v, sub, val) in enumerate(kp):
        x0 = .045 + k * .228
        fig.patches.append(FancyBboxPatch((x0, Y(2.05)), .215, 1.0 / H_in, boxstyle='round,pad=0,rounding_size=0.008', transform=fig.transFigure, fc=PANEL, ec='none', zorder=-8))
        fig.text(x0 + .012, Y(1.32), h, fontsize=7.5, color=MUT, fontweight='bold')
        fig.text(x0 + .012, Y(1.72), v, fontsize=16, fontweight='bold', color=T1 if val is None or val != val else (GOOD if val >= 0 else CRIT))
        fig.text(x0 + .012, Y(1.95), sub, fontsize=7, color=MUT)
    ax = fig.add_axes([.08, Y(4.75), .87, 2.45 / H_in]); ax.set_facecolor(PANEL); [s_.set_visible(False) for s_ in ax.spines.values()]
    xs = pd.DatetimeIndex(gx); daily = (xs[-1] - xs[0]).days > 30
    if daily:                                                                    # long history: one point per day
        k_ = pd.Series(range(len(xs)), index=xs).groupby(xs.normalize()).max().values; xs, by_, sy_ = xs[k_], by[k_], sy[k_]
    else: by_, sy_ = by, sy
    xi = np.arange(len(xs)); ax.axhline(0, color=MUT, lw=.7)
    ax.plot(xi, sy_ * 100, color='#9aa6b8', lw=1.4, alpha=.9, label='SPY')
    for lw_, a in ((6, .07), (3.5, .14), (1.8, 1)): ax.plot(xi, by_ * 100, color=BLUE, lw=lw_, alpha=a, label='book' if a == 1 else None)
    ax.scatter([xi[-1]], [by_[-1] * 100], s=36, color=BLUE, zorder=5); ax.scatter([xi[-1]], [sy_[-1] * 100], s=24, color='#9aa6b8', zorder=5)
    ax.text(xi[-1] + 1, by_[-1] * 100, f' book {by_[-1]:+.2%}', color=BLUE, fontsize=8.5, va='center', fontweight='bold')
    ax.text(xi[-1] + 1, sy_[-1] * 100, f' SPY {sy_[-1]:+.2%}', color='#9aa6b8', fontsize=8.5, va='center')
    dd = xs.normalize(); tk_ = [i for i in range(1, len(xs)) if dd[i] != dd[i - 1]]
    if daily: tk_ = tk_[::max(1, len(tk_) // 8)]
    ax.set_xticks(tk_); ax.set_xticklabels([f'{xs[i]:%b %d}' if daily else f'{xs[i]:%a %d}' for i in tk_], fontsize=7.5)
    if not daily: [ax.axvline(i - .5, color=GRID, lw=.6) for i in tk_]
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:+.1f}%')); ax.tick_params(length=0, labelsize=8); ax.set_xlim(-1, len(xs) * 1.13 + 2)
    # trade table
    y0 = Y(5.25); cols = (('TRADE', .045), ('ENTRY', .19), ('EXIT', .42), ('RETURN', .69), ('PEAK', .795), ('BOOK', .9))
    for h, x in cols: fig.text(x, y0, h, fontsize=8, color=MUT, fontweight='bold')
    for k, z in enumerate(show):
        y = y0 - (.3 + .27 * k) / H_in
        if k % 2 == 0: fig.patches.append(FancyBboxPatch((.04, y - .07 / H_in), .925, .24 / H_in, boxstyle='square,pad=0', transform=fig.transFigure, fc=PANEL, ec='none', zorder=-8, alpha=.7))
        col = GOOD if z['ret'] >= 0 else CRIT
        fig.text(.045, y, z['name'], fontsize=9, fontweight='bold', color=T1)
        fig.text(.19, y, f"{z['t0']:%a %m-%d %H:%M}  {z['p0']:,.2f}", fontsize=8.5, color=T2, family='DejaVu Sans Mono')
        ex = (f"{z['t1']:%a %m-%d %H:%M}  {z['p1']:,.2f}" + ('  lock' if z['lock'] else '')) if not z['open'] else (f"OPEN  {z['p1']:,.2f}" + ('  SELL next open' if z['sell'] else ''))
        fig.text(.42, y, ex, fontsize=8.5, color=(GOOD if z['open'] and not z['sell'] else CRIT if z['sell'] else T2), family='DejaVu Sans Mono')
        fig.text(.69, y, f"{z['ret']:+.2%}", fontsize=9, fontweight='bold', color=col)
        fig.text(.795, y, f"{z['peak']:+.2%}", fontsize=8.5, color=GOOD if z['peak'] > 0 else MUT); fig.text(.9, y, f"{z['w'] * z['ret']:+.2%}", fontsize=8.5, color=col)
    if len(L) > MAXR: fig.text(.045, y0 - (.3 + .27 * nr) / H_in, f'showing the latest {MAXR} of {len(L)} trades', fontsize=8, color=MUT)
    fig.text(.045, .1 / H_in, 'RETURN = trade return; PEAK = highest return while held (RTH highs); BOOK = its share of the book (1 slot). Book P&L = sum of slot returns (not compounded), before costs.', fontsize=7.5, color=MUT)
    fig.savefig(path, dpi=140, facecolor=BG); plt.close(fig)

def signal_card(r, meta, path):
    buy = r['action'] == 'BUY'; accent = GOOD if buy else CRIT; H = r['H']
    fig = plt.figure(figsize=(10, 6.2), facecolor=BG); _frame(fig, accent)
    fig.text(0.045, 0.92, f"{'BUY' if buy else 'SELL'}  {r['ticker']}", fontsize=22, fontweight='bold')
    why = 'new long' if buy else 'exit signal'
    fig.text(0.045, 0.875, f"{why} · {r['bar'] + pd.Timedelta('15min'):%a %b %d %H:%M} ET" + (f" · {r['note']}" if r['note'] else ''), fontsize=10, color=T2)
    ax = fig.add_axes([0.045, 0.30, 0.80, 0.53])
    _chart(ax, H, r['e'], r['px'], r['stop'], 104, big=True, mark_last=('^', GOOD) if buy else ('v', CRIT), h1=r['h1'])
    w = meta['w']; cap = meta['capital']; last = r['last']
    if buy:
        risk = (last - r['stop']) / last
        items = [('ENTRY', 'now', f'next open · ~{last:.2f}'), ('STOP', f"{r['stop']:.2f}", f'15m close below · -{risk:.1%}'),
                 ('SIZE', f'${cap * w:,.0f}' if cap else f'{w:.1%}', f"1/{meta['slots']}" + (f' · ~{cap * w / last:.0f} sh' if cap else '')),
                 ('TRAIL', '+2%', 'then 1h close < 1h stop')]
    else:
        items = [('EXIT', 'now', f'next open · ~{last:.2f}'), ('ENTRY', f"{r['px']:.2f}", f"{H.dt[r['e']]:%a %b %d %H:%M}"),
                 ('P&L', f"{r['pnl']:+.1%}", 'before costs'), ('HELD', (lambda d: f'{d.days}d' if d.days else f'{d.seconds // 3600}h')(H.dt.iloc[-1] + pd.Timedelta('15min') - H.dt[r['e']]), 'since entry')]
    for k, (h, v, s) in enumerate(items):
        x0 = 0.045 + k * 0.235
        fig.patches.append(FancyBboxPatch((x0, 0.07), 0.215, 0.17, boxstyle='round,pad=0,rounding_size=0.008', transform=fig.transFigure, fc=PANEL, ec='none', zorder=-8))
        fig.text(x0 + 0.015, 0.205, h, fontsize=8.5, color=MUT, fontweight='bold')
        col = CRIT if h == 'STOP' else GOOD if (h in ('ENTRY',) and buy) else (GOOD if h == 'P&L' and r['pnl'] >= 0 else CRIT if h in ('P&L', 'EXIT') else T1)
        fig.text(x0 + 0.015, 0.135, v, fontsize=17, fontweight='bold', color=col)
        fig.text(x0 + 0.015, 0.095, s, fontsize=8, color=T2)
    fig.savefig(path, dpi=140, facecolor=BG); plt.close(fig)

# ================================================================ discord
def post_long(text, channel=None):
    """post text over several messages (Discord's 2000-character limit), split on line breaks"""
    chunk, ok = '', True
    for line in text.splitlines():
        if len(chunk) + len(line) > 1900: ok = post(chunk, [], channel=channel) and ok; chunk = ''
        chunk += line + '\n'
    return (post(chunk, [], channel=channel) if chunk.strip() else True) and ok

def post(text, files, channel=None):
    ch, tok, hook = channel or os.getenv('VECTOR_CHANNEL_ID'), os.getenv('DISCORD_TOKEN'), os.getenv('DISCORD_WEBHOOK_URL')
    if ch and tok: url, hdr = f'https://discord.com/api/v10/channels/{ch}/messages', {'Authorization': f'Bot {tok}'}
    elif hook: url, hdr = hook, {}
    else: print('  ! no Discord target: set VECTOR_CHANNEL_ID (bot) or DISCORD_WEBHOOK_URL in .env'); return False
    fh = [open(f, 'rb') for f in files]
    try:
        if not files: r = requests.post(url, headers=hdr, json={'content': text}, timeout=30); r.raise_for_status(); return True
        r = requests.post(url, headers=hdr, data={'payload_json': json.dumps({'content': text})},
                          files={f'files[{i}]': (os.path.basename(f), h, 'image/png') for i, (f, h) in enumerate(zip(files, fh))}, timeout=30)
        r.raise_for_status(); return True
    except Exception as e:
        print(f'  ! discord post failed: {e}'); return False
    finally:
        [h.close() for h in fh]

STATE = f'{LIVE}/posted.json'
def _state():
    try: return json.load(open(STATE))
    except Exception: return {'morning': '', 'signals': []}
def _save(st): st['signals'] = st['signals'][-500:]; json.dump(st, open(STATE, 'w'))

def post_positions(rows, meta):
    p = f'{LIVE}/card_positions.png'
    try: LC.board_card(rows, meta, p)
    except Exception as e:                                                # board needs research assets that only live on the Mac (not on Render)
        print(f'  ! board card failed ({str(e)[:80]}); simple positions card'); positions_card(rows, meta, p)
    return post('', [p])

# ---- day-extreme alerts (Oct 25 2026, xs/lod_hod.py + xs/range_models.py): once a held name's low (or high) of the day looks in - set >= 30 min
#      earlier and price >= 0.5 daily sigma off it - post how far the other side should go. Vol = HAR on the daily high/low range (pooled fit),
#      bands fit 2019-21, checked 2022-26 (80% band held 80-83%, 95% band 95.5-96.3%). From 12:00 the extreme holds ~87-89%, from 14:00 ~95-96%.
XQ = {('low', 630): (.774, 1.725, 2.395, 3.440), ('low', 720): (.870, 1.756, 2.409, 3.448), ('low', 840): (.948, 1.778, 2.446, 3.500),
      ('high', 630): (.780, 1.740, 2.507, 3.473), ('high', 720): (.891, 1.827, 2.550, 3.524), ('high', 840): (.958, 1.846, 2.556, 3.531)}
def har_sigma(D):
    """next-day sigma from the HAR range model (pooled fit) for a daily frame with h/l, using rows up to the last one"""
    x = np.log(np.log(D.h / D.l).clip(lower=1e-5))
    return float(np.exp(-0.6186 + 0.2939 * x.iloc[-1] + 0.3661 * x.iloc[-5:].mean() + 0.1783 * x.iloc[-22:].mean()) / 1.665)
def range_levels(H, now=None):
    """adaptive day range: a constant range budget (HAR forecast) anchored on today's extremes so far - if the low drops $5, the expected
    high drops $5. Typical = 1.7 daily sigma (median low-to-high), wide = 2.4 sigma (80% of days stay inside), from xs/range_models.py."""
    now = now or now_et(); m = H.dt.dt.hour * 60 + H.dt.dt.minute; R_ = H[(m >= RTH0) & (m < RTH1)]
    D = R_.groupby(R_.dt.dt.normalize()).agg(h=('high', 'max'), l=('low', 'min'), c=('close', 'last'))
    if len(D) < 30: return None
    today = H[H.dt.dt.normalize() == H.dt.iloc[-1].normalize()]; tm = today.dt.dt.hour * 60 + today.dt.dt.minute
    rth = today[tm >= RTH0]; src = rth if len(rth) else today                           # before the open: premarket so far
    hist = D[D.index < H.dt.iloc[-1].normalize()]
    sig = har_sigma(hist); lo, hi, last = float(src.low.min()), float(src.high.max()), float(src.close.iloc[-1])
    return dict(sig=sig, pct=sig, lo=lo, hi=hi, last=last, pre=not len(rth), budget=1.7 * sig, hi_typ=lo * (1 + 1.7 * sig), hi_wide=lo * (1 + 2.4 * sig),
                lo_typ=hi * (1 - 1.7 * sig), lo_wide=hi * (1 - 2.4 * sig), used=(hi / lo - 1) / sig)

def extreme_info(H, now=None):
    """today's (or the last session's) high/low status for 15m bars H: is the low / high 'in', odds it holds, projected other side"""
    now = now or now_et(); m = H.dt.dt.hour * 60 + H.dt.dt.minute; R_ = H[(m >= RTH0) & (m < RTH1)]
    D = R_.groupby(R_.dt.dt.normalize()).agg(h=('high', 'max'), l=('low', 'min'), c=('close', 'last'))
    if len(D) < 30: return None
    day = D.index[-1]; live = day == now.normalize() and now.hour * 60 + now.minute < RTH1 + 5
    P_ = D[D.index < min(day, now.normalize())] if live else D[D.index < now.normalize()]     # completed sessions before today
    P_ = P_ if len(P_) >= 25 else D.iloc[:-1]
    sig = har_sigma(P_); s20 = float(np.log(P_.c).diff().iloc[-20:].std())
    hist = [har_sigma(D.iloc[:k]) for k in range(max(30, len(D) - 252), len(D) + 1)]; pct = float(np.mean(np.array(hist) <= sig))
    g = R_[R_.dt.dt.normalize() == day]; gm = g.dt.dt.hour * 60 + g.dt.dt.minute
    mins = (now.hour * 60 + now.minute) if live else RTH1
    lo, hi, last = float(g.low.min()), float(g.high.max()), float(g.close.iloc[-1]); tl, th = int(gm[g.low.idxmin()]), int(gm[g.high.idxmax()])
    tier = 840 if mins >= 840 else 720 if mins >= 720 else 630 if mins >= 630 else None
    out = dict(day=day, live=live, sig=sig, pct=pct, lo=lo, hi=hi, last=last, tl=tl, th=th, tier=tier, bar=g.dt.iloc[-1])
    for side, ok, ext in (('low', mins - (tl + 15) >= 30 and last >= lo * (1 + .5 * s20), lo), ('high', mins - (th + 15) >= 30 and last <= hi * (1 - .5 * s20), hi)):
        if live and ok and tier:
            p, q50, q80, q95 = XQ[(side, tier)]
            out[side] = dict(p=p, tgt=[ext * (1 + q * sig) if side == 'low' else ext * (1 - q * sig) for q in (q50, q80, q95)])
        else: out[side] = None
    return out
def day_extremes(rows, st):
    now = now_et(); mins = now.hour * 60 + now.minute
    if not (12 * 60 <= mins < RTH1) or os.getenv('EXTREME_ALERTS', '1') != '1': return
    today = f'{now:%Y-%m-%d}'; done = set()
    for r in rows:
        if r['action'] not in ('HOLD', 'SELL') or r['base'] in done: continue
        done.add(r['base']); x = extreme_info(r['H'], now)
        if not x or not x['live']: continue
        for side in ('low', 'high'):
            e = x[side]; key = f"{r['base']}|XT|{side}|{today}"
            if not e or key in st['signals']: continue
            t_ = x['tl'] if side == 'low' else x['th']; ext = x['lo'] if side == 'low' else x['hi']; other = 'high' if side == 'low' else 'low'
            word = 'stay under' if side == 'low' else 'stay above'
            msg = (f"\U0001f4cd {side.upper()} LIKELY IN - {r['base']}: today's {side} {ext:.2f} ({t_ // 60}:{t_ % 60:02d}) holds into the close ~{e['p']:.0%} of the time from here. "
                   f"Projected {other}: ~{e['tgt'][0]:.2f} typical, 80% {word} {e['tgt'][1]:.2f}, 95% {word} {e['tgt'][2]:.2f}. Now {x['last']:.2f}, {other} so far {x['hi'] if side == 'low' else x['lo']:.2f}.")
            if post(msg, [], channel=os.getenv('EXTREME_CHANNEL_ID')): st['signals'].append(key); _save(st)

def post_new_signals(rows, meta, st):
    try: day_extremes(rows, st)
    except Exception as ex: print(f'  ! day extremes: {str(ex)[:80]}')
    held = st.setdefault('held', {}); today = now_et().strftime('%Y-%m-%d')
    bk = f'breaker|{today}'
    if meta.get('breaker') and bk not in st['signals']:
        if post(f"\U0001f6d1 BREAKER ON: the book is more than {BREAKER:.0%} below its 20-day high. No new buys until it recovers; open positions keep their stops.", []):
            st['signals'].append(bk); _save(st)
    for g in meta.get('eg', []):                                       # earnings guard alerts
        tk = g['tk'] if g['track'] == '15m' else f"{g['tk']} 1h"; day = f"{g['bar']:%Y-%m-%d}"
        when = f"{g['report']:%a %H:%M}" if g['report'].hour else f"{g['report']:%a}"
        if not g['final'] and g['fire'] and f'{tk}|EGW|{day}' not in st['signals']:
            if post(f"\u26a0\ufe0f EARNINGS GUARD: {tk} reports {when}. Up {g['up']:+.1%} from {g['px']:.2f}, under its typical earnings move ({g['move']:.1%}, last {g['n']} reports). "
                    f"The rule SELLS AT TODAY'S CLOSE - use a market-on-close order (last {g['last']:.2f}).", []): st['signals'].append(f'{tk}|EGW|{day}'); _save(st)
        if g['final'] and g['fire']: st.setdefault('held', {}).pop(tk, None); _save(st)          # the guard's own SOLD post replaces the late-SELL post
        if g['final'] and f'{tk}|EGF|{day}' not in st['signals']:
            msg = (f"\U0001f6aa SOLD {tk} at the close {g['last']:.2f} (earnings guard): P&L {g['last'] / g['px'] - 1:+.1%} before {when}'s report." if g['fire'] else
                   f"\u2705 {tk} holds through earnings: closed {g['up']:+.1%}, " + ("above its typical move." if g['up'] >= g['move'] else "under the +3% guard line."))
            if (g['fire'] or f'{tk}|EGW|{day}' in st['signals']) and post(msg, []): st['signals'].append(f'{tk}|EGF|{day}'); _save(st)
            elif not g['fire']: st['signals'].append(f'{tk}|EGF|{day}'); _save(st)
    for r in meta.get('hm', []):                                       # honorable mention: flipped, but every slot is full -> alert to the HM channel, no trade
        if r['action'] != 'BUY': continue
        tk = r['ticker']; key = f"{tk}|HM|{r['bar']:%Y-%m-%d %H:%M}" if now_et().hour * 60 + now_et().minute >= RTH0 else f"{tk}|HM|pre|{today}"
        if key in st['signals']: continue
        p = f"{LIVE}/card_{tk.replace(' ', '_')}_HM.png"
        try: signal_card(r, meta, p); files = [p]
        except Exception as e: print(f'  ! HM card failed ({str(e)[:60]})'); files = []
        msg = (f"\u2b50 HONORABLE MENTION - {tk} flipped bullish ({r['bar']:%H:%M} bar, last {r['last']:.2f}, stop {r['stop']:.2f}) "
               f"but all {meta['slots']} slots are full, so the book is NOT buying it. Tracked in /hm.")
        if post(msg, files, channel=os.getenv('HM_CHANNEL_ID')): st['signals'].append(key); _save(st)
    for r in rows:
        tk = r['ticker']; late = ''
        if r['action'] == 'HOLD' and r['e'] is not None:            # entry the scan never caught on its flip bar (late bar, skipped scan)
            fb = r['H'].dt.iloc[r['e'] - 1]
            key = f"{tk}|BUY|{fb:%Y-%m-%d %H:%M}"
            if key in st['signals'] or tk in held or fb.strftime('%Y-%m-%d') != today:
                held[tk] = [r['px'], f"{fb:%Y-%m-%d %H:%M}"]; _save(st); continue
            held[tk] = [r['px'], f"{fb:%Y-%m-%d %H:%M}"]
            r = dict(r, action='BUY', bar=fb); late = f"late post - entered {r['H'].dt.iloc[r['e']]:%H:%M} at {r['px']:.2f}"
        elif r['action'] == 'FLAT' and tk in held:                  # exit the scan never caught
            px, fb = held.pop(tk); _save(st)
            key = f"{tk}|SELL|late|{fb}"
            if key not in st['signals']:
                if post(f"SELL {tk} (late post) - position from {fb} closed, last {r['last']:.2f}, entry {px:.2f}, P&L {r['last'] / px - 1:+.1%}", []):
                    st['signals'].append(key); _save(st)
            continue
        elif r['action'] not in ('BUY', 'SELL'): continue
        key = f"{tk}|{r['action']}|{r['bar']:%Y-%m-%d %H:%M}"
        if r['action'] == 'SELL' and r['e'] is not None:                 # Oct 25 2026: one SELL per position (premarket re-scans used to repost it every bar)
            key = f"{tk}|SELL|trade|{r['H'].dt.iloc[r['e']]:%Y-%m-%d %H:%M}"
        elif r['action'] == 'BUY' and now_et().hour * 60 + now_et().minute < RTH0:   # one premarket BUY per name per day
            key = f"{tk}|BUY|pre|{today}"
        if r['action'] == 'SELL': held.pop(tk, None); _save(st)
        if key in st['signals']: continue
        if r['action'] == 'SELL' and now_et().hour * 60 + now_et().minute < RTH0:
            late = f"\u26a0\ufe0f PREMARKET FLIP: {tk} vector turned bearish premarket ({r['bar']:%H:%M} bar, last {r['last']:.2f}). Rule exits at the 09:30 open."
        p = f"{LIVE}/card_{tk.replace(' ', '_')}_{r['action']}.png"
        try: LC.signal_dash(r, meta, p) if r['action'] == 'BUY' else signal_card(r, meta, p)
        except Exception as e: print(f'  ! signal dash failed ({str(e)[:80]}); simple card'); signal_card(r, meta, p)
        if not post(late, [p]): continue
        st['signals'].append(key); _save(st)
        log = f'{LIVE}/signal_log.csv'
        pd.DataFrame([dict(run_at=now_et(), ticker=r['ticker'], action=r['action'], bar=r['bar'], price=r['last'], stop=r['stop'], entry=r['px'], pnl=r['pnl'])]).to_csv(log, mode='a', header=not os.path.exists(log), index=False)

# ================================================================ /pro and /brief (Oct 25 2026): text analysis for Discord
SECTOR_OF = {'GOOG': 'Communication Services', 'GOOGL': 'Communication Services', 'META': 'Communication Services', 'NFLX': 'Communication Services',
             'TSLA': 'Consumer Discretionary', 'AMZN': 'Consumer Discretionary', 'CVNA': 'Consumer Discretionary', 'BRKB': 'Financials', 'COIN': 'Financials',
             'HOOD': 'Financials', 'PLTR': 'Information Technology', 'CRWD': 'Information Technology', 'APP': 'Information Technology'}
SECTOR_ETF = {'Information Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV', 'Consumer Discretionary': 'XLY', 'Consumer Staples': 'XLP',
              'Industrials': 'XLI', 'Utilities': 'XLU', 'Real Estate': 'XLRE', 'Materials': 'XLB', 'Energy': 'XLE', 'Communication Services': 'XLC'}
def sector_of(tk):
    try: S = json.load(open('sp500_sectors.json'))
    except Exception: S = {}
    return SECTOR_OF.get(tk) or S.get(tk, 'Other')
def vol_word(p): return 'high' if p >= .8 else 'elevated' if p >= .6 else 'low' if p <= .2 else 'normal'
def vec_line(L):
    a = f"15m {'bull' if L.reg == 1 else 'bear'}" + (f" (stop {L['stop']:.2f})" if L.reg == 1 else f" (flips bull on a close > {L.trig:.2f})")
    b = f"1h {'bull' if L.s1 == 1 else 'bear'}" + (f" (stop {L.stop1:.2f})" if L.s1 == 1 and L.stop1 == L.stop1 else '')
    return a + ' · ' + b
def pro_text(tk, fetch=True):
    tk = tk.upper().replace('.', '').replace('-', '')
    H = load_bars(tk, fetch, PRE0 if SESSION == 'pre' else RTH0)
    if H is None or len(H) < 500: return f'{tk}: no price data.'
    H = add_1h(vec(H)); L = H.iloc[-1]; x = extreme_info(H)
    if not x: return f'{tk}: not enough daily history.'
    pre = L['dt'].normalize() == now_et().normalize() and L['dt'].hour * 60 + L['dt'].minute < RTH0
    out = [f"**{tk}** {L.close:.2f} · last bar {L['dt']:%a %H:%M} ET" + (' (before the open)' if pre else '' if x['live'] else ' (market closed: last session)'),
           f"Vector: {vec_line(L)}",
           f"Volatility: expected day range ~{1.665 * x['sig']:.1%} ({vol_word(x['pct'])}, {x['pct']:.0%} percentile of the last year)"]
    rl = range_levels(H)
    if not x['live']:
        out.append(f"Last session: high {x['hi']:.2f} ({x['th'] // 60}:{x['th'] % 60:02d}), low {x['lo']:.2f} ({x['tl'] // 60}:{x['tl'] % 60:02d}), close {x['last']:.2f}.")
        if not (pre and rl): return '\n'.join(out)
    if rl:
        out.append(f"Range budget today: ~{rl['budget']:.1%} (wide {2.4 * rl['sig']:.1%}); {'premarket' if rl['pre'] else 'session'} so far {rl['lo']:.2f}-{rl['hi']:.2f} uses {rl['used'] / 1.7:.0%} of it." + (" Already past the typical range: odds favor the high and low so far holding." if rl['used'] > 1.7 and not rl['pre'] else ''))
        out.append(f"From the low so far {rl['lo']:.2f}: expected high ~{rl['hi_typ']:.2f} (wide {rl['hi_wide']:.2f}). From the high so far {rl['hi']:.2f}: expected low ~{rl['lo_typ']:.2f} (wide {rl['lo_wide']:.2f}). Levels move with each new extreme.")
    if pre: return '\n'.join(out)
    if x['tier'] is None: out.append(f"Too early to call (before 10:30). High so far {x['hi']:.2f}, low so far {x['lo']:.2f}.")
    for side in ('low', 'high'):
        e = x[side]; ext = x['lo'] if side == 'low' else x['hi']; t_ = x['tl'] if side == 'low' else x['th']; other = 'high' if side == 'low' else 'low'
        if e:
            word = 'under' if side == 'low' else 'above'
            out.append(f"\U0001f4cd {side.upper()} IS IN: {ext:.2f} ({t_ // 60}:{t_ % 60:02d}) holds ~{e['p']:.0%} of the time from here. Projected {other}: ~{e['tgt'][0]:.2f} typical, "
                       f"80% {word} {e['tgt'][1]:.2f}, 95% {word} {e['tgt'][2]:.2f}.")
        elif x['tier']: out.append(f"{side.capitalize()} not confirmed yet ({side} so far {ext:.2f} at {t_ // 60}:{t_ % 60:02d}).")
    return '\n'.join(out)
HEL_NAMES = {'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'Nvidia', 'AMD': 'AMD', 'AMZN': 'Amazon', 'META': 'Meta', 'GOOGL': 'Alphabet|Google', 'TSLA': 'Tesla',
             'AVGO': 'Broadcom', 'MU': 'Micron', 'ORCL': 'Oracle', 'ADBE': 'Adobe', 'COST': 'Costco', 'UNH': 'UnitedHealth', 'LLY': 'Lilly', 'GS': 'Goldman',
             'JPM': 'JPMorgan|JP Morgan', 'BA': 'Boeing', 'CAT': 'Caterpillar', 'NFLX': 'Netflix', 'PLTR': 'Palantir', 'CRWD': 'CrowdStrike', 'COIN': 'Coinbase',
             'HOOD': 'Robinhood', 'APP': 'AppLovin', 'CVNA': 'Carvana', 'BRKB': 'Berkshire'}
SECTOR_WORDS = {'Information Technology': r'chip|semiconductor|software|\bAI\b|cloud|data cent', 'Health Care': r'health|drug|pharma|FDA|insur',
                'Financials': r'bank|lender|brokerage|crypto|bitcoin', 'Consumer Discretionary': r'retail|consumer spend|auto|EV\b',
                'Communication Services': r'advertis|streaming|social media', 'Consumer Staples': r'grocer|staples|consumer prices', 'Industrials': r'aerospace|defen[cs]e|freight|airline'}
def helious(path, params, ttl=900):
    """GET api.helious.io with the key from .env (HELIOUS_API_KEY), cached for ttl seconds to save credits"""
    key = os.getenv('HELIOUS_API_KEY', ''); cf = f'{LIVE}/helious_cache.json'; ck = path + '?' + '&'.join(f'{k}={v}' for k, v in sorted(params.items()))
    try: C = json.load(open(cf))
    except Exception: C = {}
    if ck in C and time.time() - C[ck]['t'] < ttl: return C[ck]['j']
    r = requests.get('https://api.helious.io' + path, params=params, headers={'X-Api-Key': key} if key.startswith('hel_') else {}, timeout=20)
    if r.status_code != 200: raise RuntimeError(f'helious {r.status_code}: {r.text[:120]}')
    j = r.json(); C[ck] = dict(t=time.time(), j=j)
    try: json.dump(C, open(cf, 'w'))
    except Exception: pass
    return j
def stock_news(tk, since=None, limit=20, ttl=1800, until=None):
    """recent articles on one ticker from Massive (Polygon) news, with the per-ticker sentiment + reasoning; cached"""
    key = os.getenv('MASSIVE_API_KEY', '')
    if not key: return []
    since = pd.Timestamp(since or (now_et() - pd.Timedelta(days=7))).strftime('%Y-%m-%dT%H:%M:%SZ')
    until = pd.Timestamp(until or now_et()).tz_localize('America/New_York').tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ') if pd.Timestamp(until or now_et()).tzinfo is None else pd.Timestamp(until).strftime('%Y-%m-%dT%H:%M:%SZ')
    cf = f'{LIVE}/stock_news_cache.json'; ck = f'{tk}|{since[:13]}|{until[:13]}'
    try: C = json.load(open(cf))
    except Exception: C = {}
    if ck in C and time.time() - C[ck]['t'] < ttl: return C[ck]['j']
    res, ok = [], False
    for attempt in range(3):                                              # free plan: 5 calls a minute -> wait and retry on 429
        try:
            r = requests.get('https://api.massive.com/v2/reference/news', params={'ticker': YF_SYM.get(tk, tk).replace('-', '.'), 'published_utc.gte': since, 'published_utc.lte': until,
                             'order': 'desc', 'sort': 'published_utc', 'limit': limit, 'apiKey': key}, timeout=20)
            if r.status_code == 429: time.sleep(13); continue
            res = r.json().get('results', []) if r.status_code == 200 else []; ok = r.status_code == 200
        except Exception: pass
        break
    out = []
    for it in res:
        ins = next((i for i in it.get('insights') or [] if i.get('ticker') == tk), None)
        out.append(dict(t=it.get('published_utc'), title=it.get('title', ''), pub=(it.get('publisher') or {}).get('name', ''), n=len(it.get('tickers') or []),
                        sent=(ins or {}).get('sentiment'), why=(ins or {}).get('sentiment_reasoning', ''), url=it.get('article_url', '')))
    if not ok: return out
    C[ck] = dict(t=time.time(), j=out)
    try: json.dump(C, open(cf, 'w'))
    except Exception: pass
    return out

def news_text(held, universe):
    import re
    out = []
    try:
        items = helious('/v1/feed', dict(limit=100)).get('data', {}).get('items', [])
    except Exception as ex: return [f'_News unavailable: {str(ex)[:80]}_']
    secs = {sector_of(t) for t in held}
    def hit(t, it):
        txt = f"{it.get('title', '')} {it.get('text', '')}"
        return re.search(rf'\b{t}\b' + (f"|{HEL_NAMES[t]}" if t in HEL_NAMES else ''), txt, re.I) is not None
    mine = [(t, it) for it in items for t in held if hit(t, it)]
    watch = [(t, it) for it in items for t in universe if t not in held and hit(t, it) and abs(it.get('score', 0) or 0) >= 2]
    sect = [(sc, it) for it in items for sc in secs if sc in SECTOR_WORDS and abs(it.get('score', 0) or 0) >= 1 and re.search(SECTOR_WORDS[sc], f"{it.get('title', '')} {it.get('text', '')}", re.I)
            and not any(hit(t, it) for t in held)]
    def line(tag, it):
        when = pd.Timestamp(it['created_at']).tz_convert('America/New_York').strftime('%a %H:%M')
        d = it.get('direction', ''); sc = it.get('score')
        return f"  {tag}: {it['title'][:140]} ({when}" + (f", {d.lower()}" if d and d != 'NEUTRAL' else '') + (f", score {sc:+g}" if sc else '') + ')' + (f" - {it['helious_read'][:120]}" if it.get('helious_read') else '')
    seen = set()
    if mine:
        out.append('**News on your positions**')
        for t, it in mine:
            if it['id'] in seen: continue
            seen.add(it['id']); out.append(line(t, it))
            if len(seen) >= 8: break
    if sect:
        out.append('**Sector news for what you hold**')
        for sc, it in sorted(sect, key=lambda z: -abs(z[1].get('score', 0) or 0))[:4]:
            if it['id'] in seen: continue
            seen.add(it['id']); out.append(line({'Information Technology': 'Tech', 'Consumer Discretionary': 'Discretionary', 'Communication Services': 'Comms', 'Consumer Staples': 'Staples', 'Health Care': 'Health'}.get(sc, sc), it))
    if watch:
        out.append('**Market-moving news on the universe**')
        for t, it in sorted(watch, key=lambda z: -abs(z[1].get('score', 0) or 0))[:4]:
            if it['id'] in seen: continue
            seen.add(it['id']); out.append(line(t, it))
    top = [it for it in sorted(items, key=lambda it: -abs(it.get('score', 0) or 0)) if it['id'] not in seen][:3]
    if top: out.append('**Top market news**'); out += [line(it.get('category', 'news'), it) for it in top]
    try:
        cal = helious('/v1/calendar', dict(days_back=0, days_ahead=1, country='US', limit=10), ttl=3600).get('data', {})
        ev = [e for e in (cal.get('events') or []) if e.get('impact') in ('High', 'Medium') and e.get('type') != 'auction' or e.get('impact') == 'High']
        if ev: out.append('**Calendar (US, today/tomorrow, medium/high impact)**: ' + '; '.join(
            f"{pd.Timestamp(e['datetime_utc']).tz_convert('America/New_York'):%a %H:%M} {e.get('title', '')}" + (f" (prev {e['previous']})" if e.get('previous') else '') +
            (f" = {e['actual']}" if e.get('actual') else '') for e in ev[:6]))
    except Exception: pass
    if not items: out.append('_No news on the tape right now (weekend / feed delayed 60 min on the free plan)._')
    return out
def brief_text(rows, meta, fetch=True):
    U = [r for r in rows if r['track'] == '15m' and r['base'] != 'QQQ']; held = [r for r in rows if r['action'] in ('HOLD', 'SELL') and r['px']]
    out = [f"**Screener brief** · {now_et():%a %b %d %H:%M} ET"]
    if held:
        out.append(f"**Book** ({len({r['base'] for r in held})} of {meta['slots']} slots)")
        for r in sorted(held, key=lambda r: -r['pnl']):
            x = extreme_info(r['H']) or {}
            room = r['last'] / r['stop'] - 1
            out.append(f"  {r['ticker']}: {r['pnl']:+.1%} · stop {r['stop']:.2f} " + (f"(BELOW the stop: sells at the 09:30 open)" if room < 0 or r['action'] == 'SELL' else f"({room:.1%} room)") + f" · vol {vol_word(x.get('pct', .5))}"
                       + (' · low in' if x.get('low') else '') + (' · high in' if x.get('high') else ''))
    bull = [r for r in U if r['state15'] == 1]; bull1 = [r for r in U if r['state1h'] == 1]
    out.append(f"**Universe** ({len(U)} names): {len(bull)} bull on 15m, {len(bull1)} on 1h")
    near = sorted([r for r in U if r['state15'] == -1 and r['trig'] == r['trig'] and r['last'] / r['trig'] - 1 > -.006], key=lambda r: r['last'] / r['trig'], reverse=True)
    if near: out.append('  On deck (within 0.6% of a bull flip): ' + ', '.join(f"{r['base']} {r['last'] / r['trig'] - 1:+.1%}" for r in near[:8]))
    flips = [r for r in rows if r['action'] == 'BUY']
    if flips: out.append('  Flipping now: ' + ', '.join(r['ticker'] for r in flips))
    sec = {}
    for r in U:
        H = r['H']; m = H.dt.dt.hour * 60 + H.dt.dt.minute; c = H[(m >= RTH0) & (m < RTH1)].groupby(H.dt.dt.normalize()).close.last()
        r5 = float(c.iloc[-1] / c.iloc[-6] - 1) if len(c) > 6 else np.nan
        sec.setdefault(sector_of(r['base']), []).append((r['state15'] == 1, r5))
    lines = []
    for k, v in sec.items():
        lines.append((np.nanmean([q for _, q in v]), f"{k}: {sum(b for b, _ in v)}/{len(v)} bull, 5d {np.nanmean([q for _, q in v]):+.1%}"))
    out.append('**Sectors in the universe** (5-day avg, best first)'); out += ['  ' + t for _, t in sorted(lines, key=lambda z: -(z[0] if z[0] == z[0] else -9))]
    if fetch:
        try:
            import yfinance as yf
            Y = yf.download(list(SECTOR_ETF.values()) + ['SPY', 'QQQ'], period='3mo', interval='1d', progress=False, auto_adjust=True)['Close']
            Y = Y[Y.index.tz_localize(None) < now_et().normalize()] if now_et().hour * 60 + now_et().minute < RTH1 else Y   # before the close: completed sessions only
            r1, r5 = Y.iloc[-1] / Y.iloc[-2] - 1, Y.iloc[-1] / Y.iloc[-6] - 1
            lead = sorted(SECTOR_ETF.items(), key=lambda kv: -r5.get(kv[1], -9))
            out.append('**Sector ETFs** 5d (1d): ' + ', '.join(f"{v} {r5[v]:+.1%} ({r1[v]:+.1%})" for k, v in lead[:4]) + ' … weakest: ' + ', '.join(f"{v} {r5[v]:+.1%}" for k, v in lead[-2:]))
            out.append(f"SPY 5d {r5['SPY']:+.1%} · QQQ 5d {r5['QQQ']:+.1%}")
        except Exception as ex: out.append(f'(sector ETFs unavailable: {str(ex)[:60]})')
    if fetch or os.getenv('BRIEF_NEWS', '1') == '1':
        out += news_text(sorted({r['base'] for r in held}), sorted({r['base'] for r in U}))
    return '\n'.join(out)

# ================================================================ console
def show(rows, meta):
    print(f"\n=== TREND FLIP SCREENER {now_et():%Y-%m-%d %H:%M} ET | {meta['slots']} slots, {meta['w']:.1%} each | ratio thru {meta['rat_thru']:%Y-%m-%d}, QQQ gamma thru {meta['g_thru']:%Y-%m-%d}")
    if INCLUDE_QQQ and (now_et().normalize() - meta['g_thru']).days > 4: print('  ! QQQ gamma is stale - open Theta Terminal and use --update-gamma')
    for a in ('BUY', 'SELL', 'HOLD'):
        for r in [r for r in rows if r['action'] == a]:
            extra = f"entry {r['px']:.2f}  P&L {r['pnl']:+.1%}" if r['px'] else ''
            lk = f"  lock floor on (peak {r['peak']:+.1%})" if r.get('peak', 0) >= LOCK_K else ''
            print(f"  {a:<4} {r['ticker']:<8} {r['last']:>9.2f}  stop {r['stop']:.2f} ({'1h' if r['h1'] else '15m'} close)  {extra}{lk}  {r['note']}")
    print('  FLAT: ' + ', '.join(r['ticker'] for r in rows if r['action'] == 'FLAT' and r.get('track', '15m') == '15m'))

# ================================================================ intrabar cross watch (heads-up only; the BUY still needs the 15m close)
def live_px(tk):
    from fetch_lse_ohlcv import _fetch_range
    s0 = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=6)
    for tf in ('1m', '15m'):
        try:
            rows = _fetch_range(tk, tf, s0, s0 + pd.Timedelta(hours=8))
            if rows: return float(max(rows, key=lambda x: x['ts'])['close'])
        except Exception: pass
    try:
        import yfinance as yf
        y = yf.Ticker(YF_SYM.get(tk, tk)).history(period='1d', interval='1m', prepost=True)
        if len(y): return float(y.Close.iloc[-1])
    except Exception: pass
    return None

def check_milestone(r, px, st):
    """post once each time an open position makes a new whole-percent high (+1%, +2%, ...), with the profit-lock level"""
    if not px or not r['px']: return
    ms = st.setdefault('ms', {}); key = f"{r['ticker']}|{r['px']:.4f}"; pnl = px / r['px'] - 1; lvl = int(np.floor(pnl * 100))
    if lvl < 1 or lvl <= ms.get(key, 0): return
    peak = max(r.get('peak', 0) or 0, pnl); e0 = r['px']
    if peak >= LOCK_K: lk = f" · lock ON: exit if a 15m close <= {e0 * (1 + (1 - LOCK_G) * peak):.2f} (keeps half of the +{peak:.1%} peak)"
    else: lk = f" · lock arms at +5% ({e0 * (1 + LOCK_K):.2f})"
    if post(f"\U0001F4C8 {r['ticker']} +{lvl}%  {px:.2f}  (entry {e0:.2f}, {'1h' if r.get('track') == '1h' else '15m'} trade){lk}", []):
        ms[key] = lvl
        if len(ms) > 300: [ms.pop(k) for k in list(ms)[:100]]
        _save(st)

def watch_crosses(rows, st, until):
    """every ~60 s until the next 15m close: (1) cross heads-up on flat names, (2) +1% milestones on open positions (RTH only), live prices"""
    xp = st.setdefault('xp', {})
    W = [r for r in rows if r.get('track', '15m') == '15m' and r['action'] == 'FLAT' and r['state15'] == -1 and r['trig'] == r['trig'] and 'no new entries' not in r['note']]
    P = [r for r in rows if r['action'] in ('HOLD', 'SELL') and r['px']]
    bar = now_et().floor('15min'); end = bar + pd.Timedelta('15min')
    while (W or P) and now_et() < until:
        cache = {}
        for r in W:
            tk = r['ticker']; key = f"{tk}|CROSS|{bar:%Y-%m-%d %H:%M}"
            if key in st['signals'] or pm_blocked(end): continue
            px = cache.setdefault(r['base'], live_px(r['base']))
            if px and px > r['trig']:
                full = len({q['base'] for q in rows if q['action'] == 'HOLD' and q['px']}) >= slots_at(now_et())
                tail = f" Slots are full ({slots_at(now_et())}/{slots_at(now_et())}): no buy even if it holds." if full else f" BUY only if the {end:%H:%M} 15m close holds above."
                if post(f"\u26a1 CROSS {tk} {px:.2f} is above the vector ({r['trig']:.2f}).{tail}", []):
                    st['signals'].append(key); xp[tk] = [f'{bar:%Y-%m-%d %H:%M}', r['trig']]; _save(st)
        if RTH0 <= now_et().hour * 60 + now_et().minute < RTH1:
            for r in P:
                try: check_milestone(r, cache.setdefault(r['base'], live_px(r['base'])), st)
                except Exception as ex: print(f'  ! milestone {r["ticker"]}: {ex}')
        time.sleep(max(0, min(60, (until - now_et()).total_seconds())))

def check_failed(rows, st):
    xp = st.setdefault('xp', {}); R = {r['ticker']: r for r in rows}
    for tk, (b, trig) in list(xp.items()):
        if pd.Timestamp(b) + pd.Timedelta('15min') > now_et(): continue          # cross bar still open
        r = R.get(tk); xp.pop(tk); _save(st)
        if r is None or r['action'] in ('BUY', 'HOLD'): continue                  # confirmed: the BUY card covers it
        if r['state15'] == 1 or r['last'] > trig:                                # it held, but the book didn't take it: say why
            full = len({q['base'] for q in rows if q['action'] == 'HOLD' and q['px']}) >= slots_at(now_et())
            why = 'slots are full' if full else ('no buys after 2pm' if pm_blocked(now_et()) else 'not eligible right now (breaker or options ratio)')
            post(f"\u2139\ufe0f {tk} closed {r['last']:.2f}, above the vector ({trig:.2f}), but no buy: {why}.", []); continue
        post(f"\u2717 {tk} cross failed: closed {r['last']:.2f}, below the vector ({trig:.2f}). No trade.", [])

def check_stale(rows, st):
    """Post a Discord warning when the newest completed 15m bar is too old (data feed down) - once per 30 min."""
    now = now_et(); exp = now.floor('15min') - pd.Timedelta('15min')      # start of the bar that should have completed
    old = [(r['ticker'], r['bar']) for r in rows if pd.Timestamp(r['bar']) < exp - pd.Timedelta('15min')]
    if not old: st.pop('stale', None); return
    last = st.get('stale')
    if last and now - pd.Timestamp(last) < pd.Timedelta('30min'): return
    names = ', '.join(f"{t} ({b:%H:%M})" for t, b in old[:8])
    if post(f"\u26a0\ufe0f DATA STALE - newest bars: {names}. Signals and exits can't be trusted until this clears; check positions manually.", []):
        st['stale'] = str(now); _save(st)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--capital', type=float, default=None)
    ap.add_argument('--no-fetch', action='store_true')
    ap.add_argument('--no-ledger', action='store_true', help='do not write the trade ledger (tests)')
    ap.add_argument('--update-gamma', action='store_true')
    ap.add_argument('--post-positions', action='store_true')
    ap.add_argument('--loop', action='store_true')
    ap.add_argument('--pro', default=None, help='text: vector, volatility regime and high/low-in status for one ticker (Discord /pro)')
    ap.add_argument('--brief', action='store_true', help='text: book, universe, sectors')
    ap.add_argument('--report-dir', default=None, help='render the morning report (cover + one page per position) as PNGs into this folder (Discord /brief)')
    ap.add_argument('--analysis', action='store_true', help='write the morning write-up to live/analysis.json now (website Daily Analysis tab) and refresh the site snapshot')
    ap.add_argument('--hm-card', default=None, help='render the honorable mentions (flips skipped because slots were full) to this path (Discord /hm)')
    ap.add_argument('--check-card', default=None, help='render the open-positions board to this path (Discord /check)')
    ap.add_argument('--check-ticker', default=None)
    ap.add_argument('--site-snapshot', action='store_true', help='scan once (live prices) and write live/site_snapshot.json for the website')
    ap.add_argument('--history-card', default=None, help='render every trade since the start + book vs SPY (Discord /history)')
    args = ap.parse_args()
    if args.pro:                                                    # no scan needed: one ticker
        print('<<<OUT'); print(pro_text(args.pro, not args.no_fetch)); print('OUT>>>'); sys.exit()
    if not args.loop:
        rows, meta = scan(args); show(rows, meta)
        if args.report_dir:
            import daily_report as DR
            for p in DR.build(sys.modules[__name__], mark_live(rows, not args.no_fetch), meta, args.report_dir, True): print('FILE:', p)
            sys.exit()
        if args.analysis:
            import daily_report as DR
            rows = mark_live(rows, not args.no_fetch); A = DR.analysis(sys.modules[__name__], rows, meta, not args.no_fetch)[0]
            DR.save(A, f'{LIVE}/analysis.json'); write_site_snapshot(rows, meta); print(f"analysis written: {len(A['positions'])} positions"); sys.exit()
        if args.brief:
            print('<<<OUT'); print(brief_text(mark_live(rows, not args.no_fetch), meta, not args.no_fetch)); print('OUT>>>'); sys.exit()
        if args.post_positions: post_positions(rows, meta)
        if args.site_snapshot:
            print('site snapshot written'); sys.exit()
        if args.check_card:                                       # Discord /check: open positions with their charts + book P&L
            rows = mark_live(rows, not args.no_fetch)
            R = rows
            if args.check_ticker:
                R = [r for r in rows if r['base'] == args.check_ticker.upper() and r['action'] in ('HOLD', 'SELL')]
                if not R: print(f'FLAT: no open position in {args.check_ticker.upper()}'); sys.exit()
            positions_card(R, meta, args.check_card); print(f'card -> {args.check_card}')
        if args.hm_card:                                          # Discord /hm: honorable mentions
            R = mark_live(meta['hm'], not args.no_fetch)
            if not [r for r in R if r['action'] in ('HOLD', 'SELL')]: print('FLAT: no honorable mentions right now (no flips were skipped for full slots in the last 30 days that are still open)'); sys.exit()
            positions_card(R, meta, args.hm_card, title='Honorable mentions', hm=True); print(f'card -> {args.hm_card}')
        if args.history_card:
            rows = mark_live(rows, not args.no_fetch); history_card(rows, meta, args.history_card, not args.no_fetch); print(f'card -> {args.history_card}')
        sys.exit()
    st = _state()
    while True:
        t = now_et(); mins = t.hour * 60 + t.minute; today = f'{t:%Y-%m-%d}'
        if t.dayofweek < 5 and mins > RTH1 + 10 and st.get('close') != today and st.get('morning') == today:   # ~16:10: closing positions board
            rows, meta = scan(args); show(rows, meta)
            if post_positions(rows, meta): st['close'] = today; _save(st)
        if t.dayofweek >= 5 or mins > RTH1 + 10:
            if MID_BOOK and t.dayofweek == 4 and st.get('midbuild') != today:              # Friday after close: build next week's mid/small book
                time.sleep(max(0, (t.normalize() + pd.Timedelta('16h45min') - now_et()).total_seconds()))
                with open(f'{LIVE}/mid_book.log', 'a') as lg:
                    subprocess.run([sys.executable, 'mid_book.py', '--build', '--post'], stdout=lg, stderr=subprocess.STDOUT)
                st['midbuild'] = today; _save(st)
            print('market closed - exiting'); break
        if MID_BOOK and t.dayofweek == 0 and mins >= 9 * 60 and st.get('midpost') != today:  # Monday 09:00: mid/small book card
            subprocess.Popen([sys.executable, 'mid_book.py', '--post'], stdout=open(f'{LIVE}/mid_book.log', 'a'), stderr=subprocess.STDOUT)
            st['midpost'] = today; _save(st)
        if mins >= 9 * 60 and st.get('morning') != today:                    # 09:00 ET: open positions
            args.update_gamma = INCLUDE_QQQ; rows, meta = scan(args); args.update_gamma = False; show(rows, meta)
            if post_positions(rows, meta): st['morning'] = today; _save(st)
            check_stale(rows, st); post_new_signals(rows, meta, st)             # premarket flips on held names -> exit-at-open alert
            if os.getenv('DAILY_BRIEF', '1') == '1' and st.get('brief') != today:  # 09:00 ET: the /brief, posted every weekday
                try:
                    import daily_report as DR
                    fs = DR.build(sys.modules[__name__], mark_live(rows, True), meta, f'{LIVE}/report_{today}', True)
                    try:                                                        # the website's Daily Analysis tab
                        shutil.copy(f'{LIVE}/report_{today}/analysis.json', f'{LIVE}/analysis.json'); write_site_snapshot(rows, meta)
                    except Exception as ex: print(f'  ! site analysis: {str(ex)[:80]}')
                    if post(f"\U0001f4f0 Morning report · {now_et():%a %b %d}", fs[:10], channel=os.getenv('BRIEF_CHANNEL_ID')): st['brief'] = today; _save(st)
                except Exception as ex: print(f'  ! daily brief: {str(ex)[:80]}')
        elif 9 * 60 + 15 <= mins < RTH0:                                        # ~09:16 pre-open recheck
            rows, meta = scan(args); show(rows, meta); check_stale(rows, st); post_new_signals(rows, meta, st)
        elif RTH0 + 1 <= mins <= RTH1 + 5:                                    # 9:31 (premarket flips) then after each 15m close
            rows, meta = scan(args); show(rows, meta); check_stale(rows, st); post_new_signals(rows, meta, st)
            for r in rows:
                if r['action'] == 'HOLD' and r['px'] and RTH0 < mins <= RTH1: check_milestone(r, r['last'], st)
            if mins >= RTH1 and st.get('close') != today and post_positions(rows, meta): st['close'] = today; _save(st)   # 4:01 PM closing board
        nxt = t.floor('15min') + pd.Timedelta('15min') + pd.Timedelta('75s')
        if mins < 9 * 60: nxt = min(nxt, t.normalize() + pd.Timedelta('9h'))
        if t.dayofweek < 5 and 9 * 60 <= mins < RTH1 and 'rows' in globals():
            try:
                check_failed(rows, st)
                watch_crosses(rows, st, min(nxt, now_et().floor('15min') + pd.Timedelta('14min57s')))
            except Exception as ex: print(f'  ! cross watch: {ex}')
        time.sleep(max(20, (nxt - now_et()).total_seconds()))
