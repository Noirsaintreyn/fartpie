"""Nov 4 2026: options-ratio ranking on other universes. DECLARED BEFORE RESULTS. Trades = wide_T 'base+sticky' (live exits) for 168 names, ratio >= 1.5 (asof, as live), entries before 14:00, same 6-slot book / ratio tie-break / 8% breaker,
40 matched runs, both periods. Membership: Nasdaq-100 (Feb 2026 list), Dow 30 (list fetched; GOOGL removed as suspect), S&P = the sp flag in wide_T (live definition), sectors = my hand GICS-style tags (current, static -> hindsight).
Universes: SP (live) | DOW | NDX | NON-SP | SP+NDX | ALL ex-ETF | ALL ex-ETF liq | ETFs only | sectors (book if >= 1500 trades)."""
import numpy as np, pandas as pd, zlib, warnings; warnings.filterwarnings('ignore')
from collections import deque
src = open('xs/stop_1h_ev.py').read(); exec(src[src.index('def stat2'):src.index('def line(')].replace('IDX =', 'IDX0 =')) if False else None
exec(src[:src.index("B0 = ev(XL)")])
NDX = set('ADBE ADP AMD ABNB ALNY GOOGL GOOG AMZN AEP AMGN ADI AAPL AMAT APP ARM ASML TEAM ADSK AXON BKR BKNG AVGO CDNS CHTR CTAS CSCO CCEP CTSH CMCSA CEG CPRT CSGP COST CRWD CSX DDOG DXCM FANG DASH EA EXC FAST FER FTNT GEHC GILD HON IDXX INSM INTC INTU ISRG KDP KLAC KHC LRCX LIN MAR MRVL MELI META MCHP MU MSFT MSTR MDLZ MPWR MNST NFLX NVDA NXPI ODFL ORLY PCAR PLTR PANW PAYX PYPL PDD PEP QCOM REGN ROP ROST STX SHOP SBUX SNPS TTWO TSLA TXN TRI TMUS VRSK VRTX WMT WBD WDC WDAY XEL ZS'.split())
DOW = set('MMM AMZN AXP AMGN AAPL BA CAT CVX CSCO KO GS HD HON IBM JNJ JPM MCD MRK MSFT NKE NVDA PG CRM SHW TRV UNH V WMT DIS'.split())
SEC = {}
for s, ts in {'Tech/Software': 'AAPL MSFT ADBE CRM CRWD PANW PLTR SNOW ZM TTD TWLO SHOP APP DDOG IBM ORCL SMCI SQ AI FSLY BB', 'Semis': 'NVDA AMD AVGO MU LRCX QCOM TSM MRVL ARM',
              'Comm/Media': 'GOOGL META NFLX DIS SNAP ROKU PINS FUBO', 'Cons Disc': 'AMZN TSLA BKNG ABNB HD CMG LULU WYNN CVNA UPST? DKNG PENN CZR LCID SPCE LYFT UBER MCD SBUX NKE TGT DLTR BYND CELH CPRI JD BABA PTON',
              'Staples': 'COST WMT', 'Financials': 'GS JPM C PYPL COIN HOOD SOFI AFRM UPST MSTR BRKB RILY', 'Crypto miners': 'MARA RIOT CIFR IREN CLSK', 'Health': 'LLY UNH TLRY ACB CGC SENS NVAX MRNA',
              'Industrials': 'BA CAT FDX UPS LMT GEV ENPH SPWR FCEL BLNK CHPT QS', 'Energy/Mat': 'OXY CCJ', 'ETF': 'QQQ GLD SLV TLT', 'Meme/other': 'AMC GME BBBY BKKT'}.items():
    for t in ts.replace('?', '').split(): SEC.setdefault(t, s)
W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); W = W[(W['var'] == 'base+sticky') & ((W.entry.dt.hour * 60 + W.entry.dt.minute) < 840) & (W.entry >= '2019-01-01') & (W.ratio >= 1.5)].copy()
W['sec'] = W.tk.map(SEC).fillna('Unclassified'); W['ndx'] = W.tk.isin(NDX); W['dow'] = W.tk.isin(DOW); W['etf'] = W.sec.eq('ETF'); W = W.sort_values('entry').reset_index(drop=True)
print('unclassified tickers:', sorted(W[W.sec == 'Unclassified'].tk.unique()))
print('parity S&P subset trades', int(W.sp.sum()), 'vs XL', len(XL))
def stats(nm, Y):
    a, b = Y[Y.entry < '2022-01-01'], Y[Y.entry >= '2022-01-01']
    print(f'  {nm}: {Y.tk.nunique()} tickers, {len(Y)} trades ({len(a)}/{len(b)}), avg {a.ret.mean()*100:+.2f}% / {b.ret.mean()*100:+.2f}%, win {a.ret.gt(0).mean():.0%} / {b.ret.gt(0).mean():.0%}, runners>=15% {(Y.ret >= .15).sum()}', flush=True)
B0 = ev(W[W.sp].reset_index(drop=True)); print('\n40 matched runs, 2019-21 / 2022-26'); line('L live (S&P)', B0)
UN = [('Dow 30', W.dow), ('Nasdaq-100', W.ndx), ('Non-S&P', ~W.sp & ~W.etf), ('S&P + Nasdaq-100', W.sp | W.ndx), ('ALL ex-ETF', ~W.etf), ('ALL ex-ETF, liq flag', ~W.etf & W.liq), ('ETFs only', W.etf)]
for nm, m in UN: Y = W[m].reset_index(drop=True); stats(nm, Y); line(nm, ev(Y), B0)
print('\nSECTORS (all names in the sector, ratio >= 1.5): per-trade, and book where >= 1500 trades')
for s, g in W[~W.etf].groupby('sec'):
    stats(s, g.reset_index(drop=True))
    if len(g) >= 1500: line(f'   book: {s}', ev(g.reset_index(drop=True)), B0)
