"""Shared context for Discord signals (both screeners): recent news, upcoming events, market context, FINRA short share, and the mid/small setup flags.
All lookups are cached and fail soft (a missing piece is skipped, never blocks an alert).
  market_line()                        SPY / IWM vs 50d, VIX level + 1y percentile
  news_lines(tk)                       up to 3 recent Yahoo headlines that name the company/ticker (Yahoo only keeps recent items; says so when none)
  earnings_info(tk)                    next report (date, sessions away, before/after the bell), last report (date, EPS surprise)
  short_share(tk)                      FINRA Reg SHO: ShortVolume / TotalVolume averaged over the prior 20 sessions (needs the daily file cache, auto-updated)
  midsmall_flags(...)                  the 5 setup flags + 'STANDARD' / 'LOW' label (see xs/report_confidence_tiers.py for what it does and does NOT mean)
  block(tk, kind, ...)                 ready-to-post Discord text (<= ~900 chars)
Evidence notes (all in xs/): news/earnings-day entries are NOT better on average (xs/report_sub5_catalyst.py, report_post_earnings.py); the earnings skip, VWAP skip and short-share sizing are
the pieces that held up (xs/report_sub5_stack.py, report_sub5_svr_rank.py)."""
import io, json, os, time, warnings
import numpy as np, pandas as pd, requests
warnings.filterwarnings('ignore')
LIVE = 'live'; os.makedirs(LIVE, exist_ok=True); _C = {}
SVR_CUT = float(os.getenv('SVR_CUT', '0.532'))          # top third of 20-day short-sale share in the TRAIN half of the backtest
def _cached(key, ttl, fn):
    t = time.time(); v = _C.get(key)
    if v and t - v[0] < ttl: return v[1]
    try: val = fn()
    except Exception: val = None
    _C[key] = (t, val); return val

# ---------------------------------------------------------------- market context
def market_line():
    def f():
        import yfinance as yf
        D = yf.download(['SPY', 'IWM', '^VIX'], period='14mo', progress=False, auto_adjust=True)['Close'].dropna(how='all'); s, i, v = D['SPY'].dropna(), D['IWM'].dropna(), D['^VIX'].dropna()
        def seg(n, x): return f"{n} {x.iloc[-1] / x.iloc[-6] - 1:+.1%} 5d, {'above' if x.iloc[-1] > x.rolling(50).mean().iloc[-1] else 'below'} 50d"
        pct = (v.iloc[-252:] < v.iloc[-1]).mean() * 100
        return f"Market: {seg('SPY', s)} | {seg('IWM', i)} | VIX {v.iloc[-1]:.1f} ({pct:.0f}th pct of 1y)", bool(i.iloc[-1] > i.rolling(50).mean().iloc[-1])
    return _cached('mkt', 1800, f) or ('Market: n/a', None)

# ---------------------------------------------------------------- news
def news_lines(tk, days=5, n=3):
    def f():
        import yfinance as yf
        T = yf.Ticker(tk); items = T.news or []; name = ''
        try: name = (T.get_info().get('shortName') or T.fast_info.get('shortName') or '').split(' ')[0].lower()
        except Exception: pass
        out = []; now = pd.Timestamp.now(tz='UTC')
        for it in items:
            c = it.get('content', it); ts = pd.to_datetime(c.get('pubDate') or c.get('displayTime'), utc=True, errors='coerce'); title = (c.get('title') or '').strip()
            if ts is pd.NaT or not title or (now - ts) > pd.Timedelta(days=days): continue
            rel = [str(x).upper() for x in (c.get('finance', {}) or {}).get('stockTickers', [])] if isinstance(c.get('finance'), dict) else []
            if tk.upper() in rel or tk.lower() in title.lower().split() or (name and len(name) > 2 and name in title.lower()):
                out.append(f"{ts.tz_convert('America/New_York'):%b %d} - {title[:95]}")
        return out[:n]
    r = _cached(f'news|{tk}', 1200, f)
    return r if r else []

# ---------------------------------------------------------------- earnings / upcoming
def earnings_info(tk):
    def f():
        import yfinance as yf
        E = yf.Ticker(tk).get_earnings_dates(limit=10)
        if E is None or not len(E): return None
        E = E.copy(); E.index = pd.DatetimeIndex(E.index).tz_localize(None); E = E.sort_index(); now = pd.Timestamp.now(tz='America/New_York').tz_localize(None); out = {}
        fut = E[E.index > now - pd.Timedelta(hours=1)]
        if len(fut):
            d = fut.index[0]; after = d.hour >= 16; react = (d.normalize() + pd.Timedelta(days=1)) if after else d.normalize(); out['next'] = d; out['sessions'] = int(np.busday_count(now.date(), react.date())); out['when'] = 'after the close' if after else ('before the open' if d.hour < 12 else 'intraday')
        past = E[(E.index <= now) & E['Reported EPS'].notna()]
        if len(past): out['last'] = past.index[-1]; out['surp'] = float(past['Surprise(%)'].iloc[-1]) if pd.notna(past['Surprise(%)'].iloc[-1]) else None
        return out
    return _cached(f'earn|{tk}', 6 * 3600, f) or {}

# ---------------------------------------------------------------- FINRA short-sale volume (daily Reg SHO files)
FINRA_F = f'{LIVE}/finra_sv_cache.parquet'
def update_finra(days=40):
    """keep a rolling cache of the last `days` business days of FINRA CNMS short-sale volume (all symbols, compact)"""
    from concurrent.futures import ThreadPoolExecutor
    have = pd.read_parquet(FINRA_F) if os.path.exists(FINRA_F) else pd.DataFrame(columns=['Date', 'Symbol', 'ShortVolume', 'TotalVolume'])
    have['Date'] = pd.to_datetime(have['Date']); got = set(have.Date.dt.strftime('%Y%m%d')); want = [d for d in pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=days) if d.strftime('%Y%m%d') not in got]
    def g(d):
        try:
            r = requests.get(f'https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d:%Y%m%d}.txt', timeout=30)
            if r.status_code != 200: return None
            x = pd.read_csv(io.StringIO(r.text), sep='|', dtype={'Date': str})[['Date', 'Symbol', 'ShortVolume', 'TotalVolume']]; x = x[x.Date.str.len() == 8]; x['Date'] = pd.to_datetime(x.Date, format='%Y%m%d'); return x
        except Exception: return None
    with ThreadPoolExecutor(8) as ex: new = [x for x in ex.map(g, want) if x is not None]
    if new: have = pd.concat([have] + new); have = have[have.Date >= have.Date.max() - pd.Timedelta(days=days * 2)]; have.to_parquet(FINRA_F)
    return have
def short_share(tk):
    """mean ShortVolume/TotalVolume over the 20 sessions before the latest available file; None if unavailable"""
    S = _cached('finra', 3 * 3600, update_finra)
    if S is None or not len(S): return None
    x = S[S.Symbol == tk].sort_values('Date').tail(21).iloc[:-1] if (S.Symbol == tk).sum() > 21 else S[S.Symbol == tk].sort_values('Date').tail(20)
    return float((x.ShortVolume.sum() / x.TotalVolume.sum())) if len(x) >= 10 and x.TotalVolume.sum() > 0 else None

# ---------------------------------------------------------------- mid/small setup flags (honest: orders trades only coarsely)
def midsmall_flags(vr, svr20, sessions_to_earn, vwx, iwm_up):
    F = {'volume >= 2.5x': vr >= 2.5, 'short share <= 53%': (svr20 is not None and svr20 <= SVR_CUT), 'earnings > 10 sessions away': (sessions_to_earn is None or sessions_to_earn > 10), 'within 0.5 ATR of VWAP': vwx <= .5, 'IWM above 50d': bool(iwm_up)}
    n = sum(F.values()); return n, F, ('STANDARD' if n >= 3 else 'LOW')

def _fmt_earn(e):
    parts = []
    if 'next' in e: parts.append(f"next report {e['next']:%b %d} {e['when']} ({e['sessions']} sessions away)")
    if 'last' in e: parts.append(f"last report {e['last']:%b %d}" + (f", EPS surprise {e['surp']:+.0f}%" if e.get('surp') is not None else ''))
    return '; '.join(parts)
def block(tk, kind='BUY', extra=None, flags=None):
    """Discord text block. kind 'BUY'/'SELL'. extra = list of lines placed first (e.g. size note); flags = (n, dict, label, backtest_note) for the mid/small book"""
    L = []
    if extra: L += extra
    if flags:
        n, F, lab, note = flags; L.append(f"Confidence: {lab} ({n}/5 flags: " + ', '.join(('✅ ' if v else '❌ ') + k for k, v in F.items()) + f"). {note}")
    mk = market_line()[0]; L.append(mk)
    if kind == 'BUY':
        e = earnings_info(tk)
        if e: L.append('Upcoming/last: ' + _fmt_earn(e))
    if True:
        nw = news_lines(tk)
        L.append('Recent news (Yahoo headlines, unverified): ' + (' | '.join(nw) if nw else 'none found in Yahoo\'s recent feed for this name (feed is short-lived; absence is not proof of no news).'))
    out = '\n'.join(L); return out[:1700]
