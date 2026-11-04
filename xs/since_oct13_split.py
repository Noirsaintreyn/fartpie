"""Nov 4 2026: since-deployment (Oct 13 2026) replay of the split books on fresh Yahoo bars, same rules as xs/since_oct13_replay.py.
Books: LIVE S&P 6 slots | S&P book 50% + Nasdaq-100 book 50% (each 6 slots) | Nasdaq-100 book 50% + Dow book 50% (each 6 slots). Nasdaq/Dow books use the names in the live options-ratio file."""
_s = open('xs/since_oct13_replay.py').read(); exec(_s[:_s.index("OLD = book(")])
SP6, NDX6, DOW6 = book(6, lambda x: x['sp']), book(6, lambda x: x['tk'] in NDX), book(6, lambda x: x['tk'] in DOW)
cs, cn, cd = curve(SP6, 6), curve(NDX6, 6), curve(DOW6, 6)
bx = yf.download(['SPY', 'QQQ'], start='2026-10-08', interval='1d', progress=False, auto_adjust=True)['Close']; bx.index = bx.index.normalize(); bx = bx.reindex(days).ffill()
out = pd.DataFrame({'Current book (S&P 6-slot)': cs, 'S&P + NDX split 50/50': .5 * cs + .5 * cn, 'NDX + Dow split 50/50': .5 * cn + .5 * cd,
                    'SPY': bx.SPY / bx.SPY.iloc[0] - 1, 'QQQ': bx.QQQ / bx.QQQ.iloc[0] - 1}) * 100
print(out.round(2).to_string()); out.round(2).to_csv('/tmp/claude-0/ml/curve_since_oct13.csv')
for nm, B in (('NDX book', NDX6), ('Dow book', DOW6)):
    print(nm, len(B), 'trades; tickers', sorted({x['tk'] for x in B}), '; open:', sorted(f"{x['tk']} {x['track']}" for x in B if x['status'] != 'CLOSED'))
