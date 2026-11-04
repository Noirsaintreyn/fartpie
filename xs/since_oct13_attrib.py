"""Nov 4 2026: since-deployment attribution of the replay books (xs/since_oct13_split.py): per-ticker contribution to each book's return, Oct 13 - Nov 4."""
_s = open('xs/since_oct13_replay.py').read(); exec(_s[:_s.index("OLD = book(")])
SP6, NDX6, DOW6 = book(6, lambda x: x['sp']), book(6, lambda x: x['tk'] in NDX), book(6, lambda x: x['tk'] in DOW)
def contrib(B, w):
    out = {}
    for x in B:
        r = x['r'] if x['status'] == 'CLOSED' else x['H'].close.iloc[-1] / x['px'] - 1
        out[x['tk']] = out.get(x['tk'], 0) + r * w
    return pd.Series(out)
BOOKS = {'Current book': contrib(SP6, 1 / 6), 'S&P+NDX split': contrib(SP6, 1 / 12).add(contrib(NDX6, 1 / 12), fill_value=0), 'NDX+Dow split': contrib(NDX6, 1 / 12).add(contrib(DOW6, 1 / 12), fill_value=0)}
T = pd.DataFrame(BOOKS).fillna(0) * 100; T.loc['TOTAL'] = T.sum()
print(T.round(2).sort_values('Current book', ascending=False).to_string())
for nm, B in (('S&P', SP6), ('NDX', NDX6), ('DOW', DOW6)):
    n = len(B); w = sum((x['r'] if x['status'] == 'CLOSED' else x['H'].close.iloc[-1] / x['px'] - 1) > 0 for x in B)
    print(f'{nm} book: {n} trades, {w} winners, avg {np.mean([(x["r"] if x["status"] == "CLOSED" else x["H"].close.iloc[-1] / x["px"] - 1) for x in B])*100:+.2f}%')
