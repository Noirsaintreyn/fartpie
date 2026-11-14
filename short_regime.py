"""SHORT-SQUEEZE REGIME FILTER for the mid/small screener (Nov 14 2026; research: xs/report_67b, report_69, report_69b).
Idea: stocks with high days-to-cover (FINRA short interest, DTC >= 10) only pay off when SPECULATIVE / HEAVILY SHORTED names are leading the market. 
Regime ON  = equal-weight basket of the DTC>=10 names (membership known the prior day, FINRA file published +8 days after settlement) has beaten the whole universe over the last 40 trading days (lead > 0).
Name-hot   = regime ON and the stock's own latest published DTC >= 10  -> posted as a SQUEEZE-REGIME BUY (extra alert). Everything else is unchanged unless MIDSMALL_SQUEEZE_GATE=1 (then only name-hot BUYs pass).
Evidence (daily flips on a wide $1+ universe, NOT these 1h flips; survivorship-biased): DTC>=10 flips with the basket leading averaged +13.0% per trade 2019-21 and +11.3% 2025-26 (lift in huge-move rate 1.9x / 3.5x) vs -1.6% in 2022-24 and about -1% when the basket lagged. 
Regime dependent: it failed 2022-24 (shorts were right in the tightening), the 2025-26 result is lumpy (17 huge winners in 414 flips, top 5 trades = 51% of return, mostly biotech), and the 40d>0 trigger is loose (fires in noise). Context + a prompt to look, not proof.
Data: FINRA bi-monthly short interest (cdn.finra.org/equity/otcmarket/biweekly), daily closes from Yahoo for the screener universe; both cached under live/."""
import io, os, time, datetime as dt
import numpy as np, pandas as pd, requests
LIVE = 'live'; os.makedirs(LIVE, exist_ok=True); SI_F = f'{LIVE}/finra_si_cache.pkl'; CL_F = f'{LIVE}/sq_closes.pkl'; DTC_CUT = float(os.getenv('SQ_DTC_CUT', '10')); LEAD_DAYS = int(os.getenv('SQ_LEAD_DAYS', '40')); LAG = 8; MIN_LEAD = float(os.getenv('SQ_MIN_LEAD', '0')); UA = {'User-Agent': 'Mozilla/5.0 research'}
EVIDENCE = 'Research (daily flips, wide universe, in-sample): high-short flips with the basket leading averaged +13.0% (2019-21) and +11.3% (2025-26) per trade vs -1.6% in 2022-24. Regime dependent, lumpy, mostly biotech lately; these are daily flips, not 1h.'
def settlement_dates(months=5, today=None):
    today = today or dt.date.today(); out = []; d = (today.replace(day=1) - dt.timedelta(days=31 * months)).replace(day=1)
    while d <= today:
        for day in (15, 28, 29, 30, 31):
            try: s = dt.date(d.year, d.month, day)
            except ValueError: continue
            if s + dt.timedelta(days=LAG) <= today + dt.timedelta(days=3): out.append(s)
        d = (d.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
    return out
def update_si(months=5):
    """incrementally download FINRA short-interest files (symbolCode, settlementDate, daysToCoverQuantity); each target settlement date tries up to 4 calendar days back (holiday/weekend shifts)"""
    try: SI = pd.read_pickle(SI_F)
    except Exception: SI = pd.DataFrame(columns=['symbolCode', 'settlementDate', 'daysToCoverQuantity'])
    have = set(pd.to_datetime(SI.settlementDate).dt.date) if len(SI) else set(); new = []; tried = set()
    for s in settlement_dates(months):
        if any(abs((s - h).days) <= 3 for h in have): continue
        for off in range(0, 4):
            sd = s - dt.timedelta(days=off)
            if sd.weekday() >= 5 or sd in have: continue
            try: r = requests.get(f'https://cdn.finra.org/equity/otcmarket/biweekly/shrt{sd:%Y%m%d}.csv', timeout=60, headers=UA)
            except Exception: break
            if r.status_code == 200:
                x = pd.read_csv(io.StringIO(r.text), sep='|'); x = x[['symbolCode', 'settlementDate', 'daysToCoverQuantity']]; new.append(x); have.add(sd); break
    if new: SI = pd.concat([SI] + new).drop_duplicates(['symbolCode', 'settlementDate']); SI['settlementDate'] = pd.to_datetime(SI.settlementDate); SI.to_pickle(SI_F)
    return SI
def dtc_frame(closes, si):
    """daily DTC per ticker: latest published value (settlement + LAG days), held up to 20 days"""
    si = si.copy(); si['avail'] = pd.to_datetime(si.settlementDate) + pd.Timedelta(days=LAG); si = si[si.symbolCode.isin(closes.columns)]
    out = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    for tk, g in si.groupby('symbolCode'):
        s = g.sort_values('avail').drop_duplicates('avail').set_index('avail').daysToCoverQuantity; s = s[~s.index.duplicated()]
        out[tk] = s.reindex(s.index.union(closes.index)).ffill(limit=20).reindex(closes.index)
    return out
def lead_series(closes, si):
    """basket(DTC>=cut, membership of the prior day) minus universe, compounded over LEAD_DAYS; also returns the DTC frame and basket size"""
    dtc = dtc_frame(closes, si); ret = closes.pct_change(fill_method=None).clip(-0.5, 1.0); hi = (dtc >= DTC_CUT).shift(1).fillna(False)
    bk = ret.where(hi).mean(axis=1); un = ret.mean(axis=1); lead = (1 + bk.fillna(0)).rolling(LEAD_DAYS).apply(np.prod, raw=True) - (1 + un.fillna(0)).rolling(LEAD_DAYS).apply(np.prod, raw=True)
    return lead, dtc, hi.sum(axis=1)
def fetch_closes(tks, period='1y'):
    import yfinance as yf
    cache = None
    try:
        cache = pd.read_pickle(CL_F)
        if cache['day'] == dt.date.today() and set(tks) <= set(cache['closes'].columns): return cache['closes']
    except Exception: pass
    out = {}
    for i in range(0, len(tks), 100):
        ch = tks[i:i + 100]
        try:
            D = yf.download(ch, period=period, interval='1d', group_by='ticker', threads=False, progress=False, auto_adjust=False)
            for tk in ch:
                try: s = D[tk].close.dropna() if len(ch) > 1 else D.close.dropna(); out[tk] = s if len(s) > 100 else None
                except Exception: pass
        except Exception: pass
    C = pd.DataFrame({k: v for k, v in out.items() if v is not None})
    if not len(C): return C
    ix = pd.DatetimeIndex(C.index); C.index = (ix.tz_localize(None) if ix.tz is not None else ix).normalize()
    if len(C): pd.to_pickle({'day': dt.date.today(), 'closes': C}, CL_F)
    return C
_ST = {}
def status(tks=None):
    if _ST.get('v') is not None and time.time() - _ST['t'] < 3600: return _ST['v']
    v = _status(tks); _ST.update(t=time.time(), v=v); return v
def _status(tks=None):
    """regime state for today: {'on': bool, 'lead': float, 'n_basket': int, 'asof': date, 'dtc': {tk: latest DTC}} or None if data unavailable"""
    try:
        if tks is None:
            tks = [l.strip().upper() for l in open('midsmall_universe.txt') if l.strip() and not l.startswith('#')]
        C = fetch_closes(tks); SI = update_si()
        if C is None or not len(C) or not len(SI): return None
        lead, dtc, nb = lead_series(C, SI); l = lead.dropna()
        if not len(l): return None
        d = dtc.iloc[-1].dropna(); return dict(on=bool(l.iloc[-1] > MIN_LEAD), lead=float(l.iloc[-1]), n_basket=int(nb.iloc[-1]), asof=l.index[-1].date(), dtc={k: float(v) for k, v in d.items()}, series=l)
    except Exception as e: print(f'  ! short_regime unavailable: {str(e)[:100]}'); return None
def rates_state():
    """policy-rate regime (xs/report_73/73b): 60-trading-day change in the 2y Treasury yield (FRED DGS2) ranked against all prior values since 2019. TIGHTENING = upper third (rising yields), EASING = lower third. Context only: in 2022-26 flips entered in TIGHTENING months lost (-1.6% all flips, -4.8% high-short flips with the basket leading) vs +1.8% / +2.3% in EASING months; 2019-21 did not show it (2020-21 yields rose off the lows during a speculative boom)."""
    try:
        r = requests.get('https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2', timeout=60); d = pd.read_csv(io.StringIO(r.text)); d.columns = ['date', 'y']; d['date'] = pd.to_datetime(d.date); y = d.set_index('date').y.dropna(); y = y[y.index >= '2018-06-01']
        ch = (y - y.shift(60)).dropna(); cur = float(ch.iloc[-1]); hist = ch.iloc[:-1][ch.index[:-1] >= '2019-01-01']; lo, hi = hist.quantile([1 / 3, 2 / 3])
        return dict(chg=cur, yld=float(y.iloc[-1]), cls='EASING' if cur <= lo else ('TIGHTENING' if cur >= hi else 'MID'), asof=y.index[-1].date())
    except Exception as e: print(f'  ! rates_state unavailable: {str(e)[:80]}'); return None
def rates_line():
    r = _RS.get('v') if _RS.get('v') is not None and time.time() - _RS['t'] < 6 * 3600 else None
    if r is None: r = rates_state(); _RS.update(t=time.time(), v=r)
    return '' if not r else f" Rates: 2y yield {r['yld']:.2f}% ({r['chg']:+.2f} pts over 60d = {r['cls']}; in 2022-26 high-short and sector-lead flips lost in TIGHTENING months)."
_RS = {}
def tag(tk, s):
    """per-ticker tag for a BUY: line = one-sentence status appended to the alert, hot = regime ON and the name's own DTC>=cut, alert = extra post text"""
    d = s['dtc'].get(tk); name_sq = d is not None and d >= DTC_CUT; hot = bool(s['on'] and name_sq)
    line = f"Squeeze regime {'ON' if s['on'] else 'OFF'} (high-short basket {s['lead']:+.1%} vs universe over {LEAD_DAYS}d, {s['n_basket']} names)" + (f"; {tk} days-to-cover {d:.0f}" if d is not None else '; no short-interest data for this name') + '.' + rates_line()
    alert = f"\U0001f525 SQUEEZE-REGIME BUY {tk}: days-to-cover {d:.0f} (>= {DTC_CUT:g}) while the high-short basket is leading the universe ({s['lead']:+.1%} over {LEAD_DAYS}d). {EVIDENCE}" if hot else ''
    return dict(hot=hot, name_sq=name_sq, dtc=d, line=line, alert=alert)
def change_msg(st, s):
    """one post when the regime flips (state kept in the screener's state dict)"""
    prev = st.get('sq_on'); st['sq_on'] = bool(s['on'])
    if prev is None or prev == s['on']: return None
    return (f"\U0001f525 SHORT-SQUEEZE REGIME TURNED ON: the high-short basket ({s['n_basket']} names with days-to-cover >= {DTC_CUT:g}) now leads the universe by {s['lead']:+.1%} over {LEAD_DAYS}d. High-short BUYs get the SQUEEZE-REGIME tag. " + EVIDENCE) if s['on'] else f"\U0001f4a4 Short-squeeze regime turned OFF: the high-short basket now lags the universe ({s['lead']:+.1%} over {LEAD_DAYS}d). High-short flips did not pay in this state in the research (2022-24)."
