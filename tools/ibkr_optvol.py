"""Daily option volume from IBKR (replaces the Theta pull). TWS / IB Gateway running + logged in, API on (port 7497), read-only.
For every name in the screener's universe pool (S&P members now + names in the live ratio file), streams generic tick 100
(option call / put volume for the day) in batches and appends today's row to data/theta_vol/<TK>.csv in the same layout the ratio
builder already reads (date, opt_contracts, opt_premium, call_c, tk, month). Run after the close (~16:15 ET), weekdays.
  python3 tools/ibkr_optvol.py            all names        python3 tools/ibkr_optvol.py --test AAPL MSFT   print only, write nothing"""
import os, sys, json, time, math, pandas as pd
from ib_insync import IB, Stock, util
PORT = int(os.getenv('IB_PORT', '7497')); util.logToConsole(level=40); ib = IB()
TEST = '--test' in sys.argv; args = [a for a in sys.argv[1:] if not a.startswith('--')]
ETF = {'GLD', 'SLV', 'TLT', 'QQQ', 'SPY', 'IWM', 'IEF', 'DIA'}
SYM = {'BRKB': 'BRK B', 'BFB': 'BF B'}                                          # IBKR class-share symbols
def universe():
    M = json.load(open('sp500_pit_membership.json')); now = {('META' if t == 'FB' else t.replace('.', '')) for t in M[max(M)]}
    live = set(pd.read_csv('live_data/optvol_ratio_live.csv', usecols=['tk']).tk)
    return sorted((now | live) - ETF)
tks = args or universe()
for k in range(20):
    try: ib.connect('127.0.0.1', PORT, clientId=81, readonly=True); break
    except Exception as ex: print(f'waiting for TWS ({str(ex)[:60]})', flush=True); time.sleep(15)
else: sys.exit('TWS not reachable - open TWS / IB Gateway, API on port 7497')
ib.reqMarketDataType(1)
today = pd.Timestamp.now(tz='America/New_York').normalize().tz_localize(None)
got, miss = {}, []
B = 80                                                                           # stay under the 100 market-data-line limit
for i in range(0, len(tks), B):
    batch = tks[i:i + B]; cs = []
    for tk in batch:
        c = Stock(SYM.get(tk, tk), 'SMART', 'USD')
        try: ib.qualifyContracts(c); cs.append((tk, c, ib.reqMktData(c, genericTickList='100', snapshot=False)))
        except Exception as ex: miss.append(tk)
    t0 = time.time()
    while time.time() - t0 < 12:                                                 # wait for the option-volume ticks to arrive
        ib.sleep(1)
        if all(not math.isnan(t.callVolume) and not math.isnan(t.putVolume) for _, _, t in cs): break
    for tk, c, t in cs:
        cv, pv = t.callVolume, t.putVolume
        if math.isnan(cv) or math.isnan(pv) or cv + pv <= 0: miss.append(tk)
        else: got[tk] = (float(cv), float(pv))
        ib.cancelMktData(c)
    print(f'{min(i + B, len(tks))}/{len(tks)} names, {len(got)} with option volume', flush=True)
ib.disconnect()
if TEST:
    for tk, (cv, pv) in got.items(): print(f'{tk}: calls {cv:,.0f}  puts {pv:,.0f}  total {cv + pv:,.0f}')
    print('missing:', miss); sys.exit()
os.makedirs('data/theta_vol', exist_ok=True); n = 0
for tk, (cv, pv) in got.items():
    f = f'data/theta_vol/{tk}.csv'
    row = pd.DataFrame([dict(date=today.strftime('%Y-%m-%d'), opt_contracts=cv + pv, opt_premium=float('nan'), call_c=cv, tk=tk, month=today.strftime('%Y-%m'))])
    if os.path.exists(f):
        old = pd.read_csv(f); old = old[old.date.astype(str) != row.date.iloc[0]]; row = pd.concat([old, row])
    row.to_csv(f, index=False); n += 1
print(f'{today:%Y-%m-%d}: wrote option volume for {n} names; missing {len(miss)}: {" ".join(miss[:30])}')
if len(got) < 0.5 * len(tks): sys.exit(f'only {len(got)}/{len(tks)} names returned option volume - check the IBKR options market-data subscription')
