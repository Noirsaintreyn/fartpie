"""Trade lists behind the vector-vs-random tests, for checking against charts. Bar times are US/Eastern, bar START (1h bars: 09:30, 10:30 ...; 04:00-16:00 session). Prices are Yahoo unadjusted bar OPENS (fills at the open of the bar after the flip).
Files: xs/trades_random10_1h.csv (10 random stocks, vector flip trades, long on bull flip / short on bear flip, exit at the open after the opposite flip)
       xs/trades_leveraged_1h.csv (signal = vector on the underlying; trades in the bull ETF while bull / the inverse ETF while bear; RTH bars only; entry = open of the first RTH bar the position is held, exit = open of the next bar after it ends)."""
import os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_vector_leveraged.py').read(); exec(src[:src.index('SS = {}')])
SUF = '_rth' if os.environ.get('RTH_ONLY') == '1' else ''
rows = []
for tk in PICK:
    B = yf_bars(tk, '1h', '730d')
    if B is None or len(B) < 300: continue
    P = prep(B); idx = B.index
    for (a, b, d, rr) in trades(P):
        rows.append(dict(ticker=tk, side='long' if d > 0 else 'short', entry_time=idx[a], entry_price=round(float(P['o'][a]), 4), exit_time=idx[b], exit_price=round(float(P['o'][b]), 4), ret_pct=round(rr * 100, 3)))
R = pd.DataFrame(rows).sort_values(['ticker', 'entry_time']); R.to_csv(f'xs/trades_random10_1h{SUF}.csv', index=False); print('random10 trades', len(R), R.groupby('ticker').size().to_dict())
rows = []
for U, L, I in PAIRS:
    BU = get(U)
    if BU is None or len(BU) < 500: continue
    P = prep(BU); reg = pd.Series(P['reg'], index=BU.index); X = {'U': rth_df(BU)}
    for nm, tk in (('L', L), ('I', I)):
        if tk: X[nm] = rth_df(get(tk))
    idx = X['U'].index
    for k in X: idx = idx.intersection(X[k].index)
    idx = idx.sort_values(); st = np.r_[0.0, reg.reindex(idx).values.astype(float)[:-1]]; n = len(idx)
    for nm, want, tkr in (('U', 1, U), ('L', 1, L), ('I', -1, I)):
        if nm not in X: continue
        o = X[nm].reindex(idx).open.values; i = 0
        while i < n:
            if st[i] == want:
                j = i
                while j + 1 < n and st[j + 1] == want: j += 1
                if j + 1 < n: rows.append(dict(signal=U, instrument=tkr, entry_time=idx[i], entry_price=round(float(o[i]), 4), exit_time=idx[j + 1], exit_price=round(float(o[j + 1]), 4), ret_pct=round((o[j + 1] / o[i] - 1) * 100, 3)))
                i = j + 1
            else: i += 1
R2 = pd.DataFrame(rows).sort_values(['signal', 'instrument', 'entry_time']); R2.to_csv(f'xs/trades_leveraged_1h{SUF}.csv', index=False); print('leveraged trades', len(R2), R2.groupby('instrument').size().to_dict())
