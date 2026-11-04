"""Nov 4 2026: equity curves for charts. Books (backtest, realized P&L, compounding, 2 bp cost, same engine as the yardstick):
  LIVE   S&P, ratio >= 1.5, 6 slots                     SPNDX  50% S&P book + 50% Nasdaq-100 book (two separate 6-slot books, no rebalancing)
  NDXDOW 50% Nasdaq-100 book + 50% Dow book              plus SPY and QQQ (Yahoo, dividends included)
Each book's curve = per-day MEDIAN equity over the 40 matched runs (same ~5% of signals dropped in every book). Windows restart every book (and the 50/50 split) at the window start.
Backtest data ends 2026-10-21."""
_s = open('xs/universe_test.py').read(); exec(_s[:_s.index("B0 = ev(W[W.sp]")].replace("print('unclassified", "pass #").replace("print('parity", "pass #"))
import yfinance as yf
LIVE = W[W.sp].reset_index(drop=True); DOWF = W[W.dow].reset_index(drop=True); NDXF = W[W.ndx].reset_index(drop=True)
def runs(Y):
    HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(Y.tk, Y.entry.astype('int64'))], dtype=np.uint64); out = []
    for i in range(1, 41):
        m = ~(((HSH * 2654435761 + i * 40503) % 1000) < 50)
        s = sim(Y.entry.values[m], Y.exit.values[m], Y.ret.values[m], Y.ratio.values[m].astype(float), Y.tk.values[m])
        out.append(np.exp(s.groupby(s.index.normalize()).last().reindex(IDX, method='ffill').fillna(0)).values)
    return np.array(out)                                   # 40 x days, equity from 2019-01-01
RS = {k: runs(Y) for k, Y in (('LIVE', LIVE), ('NDX', NDXF), ('DOW', DOWF))}
bx = yf.download(['SPY', 'QQQ'], start='2018-12-20', end='2026-10-23', interval='1d', progress=False, auto_adjust=True)['Close']; bx.index = bx.index.normalize(); bx = bx.reindex(IDX.union(bx.index)).ffill().reindex(IDX)
WIN = {'since 2019': ('2019-01-01', '2026-10-21'), '2019-2023': ('2019-01-01', '2023-12-29'), 'since 2023': ('2023-01-01', '2026-10-21')}
rows = []; curves = {}
for wn, (a, b) in WIN.items():
    k = (IDX >= a) & (IDX <= b); ix = IDX[k]; i0 = np.argmax(k)
    def rb(E): return E[:, k] / E[:, [i0]]                  # restart each run at the window start
    C = {'Current book (S&P 6-slot)': np.median(rb(RS['LIVE']), axis=0),
         'S&P + NDX split 50/50': np.median(.5 * rb(RS['LIVE']) + .5 * rb(RS['NDX']), axis=0),
         'NDX + Dow split 50/50': np.median(.5 * rb(RS['NDX']) + .5 * rb(RS['DOW']), axis=0),
         'SPY': (bx.SPY[k] / bx.SPY[k].iloc[0]).values, 'QQQ': (bx.QQQ[k] / bx.QQQ[k].iloc[0]).values}
    D = pd.DataFrame(C, index=ix); curves[wn] = D
    yr = (ix[-1] - ix[0]).days / 365.25
    for c in D: e = D[c]; rows.append((wn, c, e.iloc[-1] - 1, e.iloc[-1] ** (1 / yr) - 1, (e / e.cummax() - 1).min()))
S = pd.DataFrame(rows, columns=['window', 'series', 'total', 'cagr', 'maxdd'])
print(S.assign(total=(S.total * 100).round(1), cagr=(S.cagr * 100).round(1), maxdd=(S.maxdd * 100).round(1)).to_string(index=False))
for wn, D in curves.items():
    Wk = D.resample('W-FRI').last() if wn != '2019-2023' else D.resample('W-FRI').last()
    Wk = ((Wk - 1) * 100).round(1); Wk.to_csv(f'/tmp/claude-0/ml/curve_{wn.replace(" ", "_")}.csv')
print('weekly points', {wn: len(D.resample('W-FRI').last()) for wn, D in curves.items()})
