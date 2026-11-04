"""Nov 4 2026: replay of the CURRENT live rules from Mon Oct 13 2026 to today on fresh Yahoo 15m bars (premarket+RTH), two books:
  OLD  S&P (current members) with options ratio >= 1.5, 6 slots          NEW  S&P + Nasdaq-100 + Dow with ratio >= 1.5, 10 slots
Rules = live_screener functions as they are now: vector (pine port), 15m + 1h tracks, sticky 1h stop after +2%, +5% lock (keep half), no entries 14:00-16:00,
ratio tie-break, 8% breaker. Earnings guard not applied. Both books start flat at the Oct 13 open; equity is marked to the daily close; fixed budget, slot = 1/N, 2 bp cost.
This is a replay, not the site's recorded ledger (the site ran earlier rule versions Oct 13-21)."""
import sys, os, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import live_screener as LS, yfinance as yf
T0 = pd.Timestamp('2026-10-13 09:30')
for k in ('SLOT_SWITCH', 'MODEL2', 'STICKY_START', 'CUTOVER', 'NO_PM'): setattr(LS, k, pd.Timestamp('2020-01-01') if k != 'STICKY_START' else np.datetime64('2020-01-01'))
NDX = set('ADBE ADP AMD ABNB ALNY GOOGL GOOG AMZN AEP AMGN ADI AAPL AMAT APP ARM ASML TEAM ADSK AXON BKR BKNG AVGO CDNS CHTR CTAS CSCO CCEP CTSH CMCSA CEG CPRT CSGP COST CRWD CSX DDOG DXCM FANG DASH EA EXC FAST FER FTNT GEHC GILD HON IDXX INSM INTC INTU ISRG KDP KLAC KHC LRCX LIN MAR MRVL MELI META MCHP MU MSFT MSTR MDLZ MPWR MNST NFLX NVDA NXPI ODFL ORLY PCAR PANW PAYX PYPL PDD PEP PLTR QCOM REGN ROP ROST SBUX SHOP SNPS TMUS TSLA TTWO TXN TRI VRSK VRTX WBD WDAY WMT XEL ZS'.split())
DOW = set('MMM AMZN AXP AMGN AAPL BA CAT CVX CSCO KO GS HD HON IBM JNJ JPM MCD MRK MSFT NKE NVDA PG CRM SHW TRV UNH V WMT DIS'.split())
O = pd.read_csv('live_data/optvol_ratio_live.csv', parse_dates=['date']); O = O[O.sh > 0]; O['r'] = O.opt_contracts * 100 / O.sh
R = O.pivot_table(index='date', columns='tk', values='r').sort_index().rolling(252, min_periods=60).median().shift(1)
R = R.reindex(pd.bdate_range(R.index.min(), '2026-11-05')).ffill()
now = LS.now_et(); spn = {t for t in R.columns if LS.in_sp(t, now)}
cand = sorted(t for t in R.columns if t not in LS.NO_ETF and (R.loc['2026-10-13':, t] >= 1.5).any() and (t in spn or t in NDX or t in DOW))
print('candidates', len(cand), ' S&P', sum(t in spn for t in cand), ' NDX/Dow only', [t for t in cand if t not in spn], flush=True)
YS = {t: LS.YF_SYM.get(t, t) if hasattr(LS, 'YF_SYM') else t for t in cand}
raw = yf.download(sorted(set(YS.values())), period='60d', interval='15m', prepost=True, group_by='ticker', progress=False, threads=True, auto_adjust=False)
trades = []
for t in cand:
    try: Y = raw[YS[t]].dropna(how='all')
    except Exception: continue
    if len(Y) < 500: continue
    Y = Y.reset_index(); ts = pd.to_datetime(Y.iloc[:, 0], utc=True).dt.tz_convert(LS.ET).dt.tz_localize(None)
    H = pd.DataFrame({'dt': ts, 'open': Y['Open'].values, 'high': Y['High'].values, 'low': Y['Low'].values, 'close': Y['Close'].values, 'volume': Y['Volume'].fillna(0).values})
    H.loc[H.volume <= 0, 'volume'] = 1; m = H.dt.dt.hour * 60 + H.dt.dt.minute
    H = H[(m >= 240) & (m < 960) & (H.dt.dt.dayofweek < 5) & (H.dt + pd.Timedelta('15min') <= now)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
    H = LS.add_1h(LS.vec(H)); d = H.dt.dt.normalize(); rr = R[t].reindex(d).values; elig = (rr >= 1.5) & (H.dt >= T0).values
    for tr in ('15m', '1h'):
        for x in LS.replay_track(H, elig, tr):
            if x['status'] == 'BUY' or x['t0'] is None or pd.Timestamp(x['t0']) < T0: continue
            x['tk'] = t; x['ratio'] = float(np.nan_to_num(R[t].get(pd.Timestamp(x['t0']).normalize(), 0.0))); x['sp'] = t in spn
            if x['status'] == 'CLOSED': x['r'] = H.open.values[x['x']] / x['px'] - 1 - 2e-4; x['xpx'] = H.open.values[x['x']]
            x['H'] = H[['dt', 'close']]; trades.append(x)
print('trades since Oct 13 (all candidates, before slots):', len(trades), flush=True)
def book(N, univ):
    LS.SLOTS = N; tt = [dict(x) for x in trades if univ(x)]
    for x in tt: x['t1'] = x['t1'] if x['status'] == 'CLOSED' else LS.INF
    return LS.allocate(tt)
days = pd.bdate_range('2026-10-10', now.normalize())
def curve(B, N):
    eq = []
    for dday in days:
        close_t = dday + pd.Timedelta(hours=16); v = 0.0
        for x in B:
            t0 = pd.Timestamp(x['t0'])
            if t0 > close_t: continue
            if x['status'] == 'CLOSED' and pd.Timestamp(x['t1']) <= close_t: v += x['r'] / N; continue
            h = x['H']; c = h[h.dt <= close_t - pd.Timedelta('15min')].close
            if len(c): v += (c.iloc[-1] / x['px'] - 1 - 1e-4) / N
        eq.append(v)
    return pd.Series(eq, index=days)
OLD = book(6, lambda x: x['sp']); NEW = book(10, lambda x: x['sp'] or x['tk'] in NDX or x['tk'] in DOW)
eo, en = curve(OLD, 6), curve(NEW, 10)
bx = yf.download(['SPY', 'QQQ'], start='2026-10-08', interval='1d', progress=False, auto_adjust=True)['Close']; bx.index = bx.index.normalize()
bx = bx.reindex(days).ffill(); spy, qqq = bx.SPY / bx.SPY.iloc[0] - 1, bx.QQQ / bx.QQQ.iloc[0] - 1
out = pd.DataFrame({'SPY': spy, 'QQQ': qqq, 'NEW 10-slot S&P+NDX+Dow': en, 'OLD 6-slot S&P': eo}) * 100
print(out.round(2).to_string(), flush=True); out.to_csv('/tmp/claude-0/ml/since_oct13_curve.csv')
for nm, B in (('OLD', OLD), ('NEW', NEW)):
    print(f'\n{nm}: {len(B)} trades taken, {sum(x["status"] == "CLOSED" for x in B)} closed, open now:', sorted(f"{x['tk']} {x['track']}" for x in B if x['status'] != 'CLOSED'))
    for x in sorted(B, key=lambda z: pd.Timestamp(z['t0'])):
        last = x['H'].close.iloc[-1]; r = x['r'] if x['status'] == 'CLOSED' else last / x['px'] - 1
        print(f"   {x['tk']:<6}{x['track']:>4} {pd.Timestamp(x['t0']):%m-%d %H:%M} @ {x['px']:.2f} -> {('closed ' + pd.Timestamp(x['t1']).strftime('%m-%d %H:%M')) if x['status'] == 'CLOSED' else 'open  ':<18} {r * 100:+.1f}%  ratio {x['ratio']:.1f}")
