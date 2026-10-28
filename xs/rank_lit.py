"""Oct 28 2026: options signals from the stock-return literature, as tie-breaks (universe stays ratio >= 1.5). Known at the prior close.
 cps   call - put ATM IV spread (Cremers & Weinbaum 2010: high spread -> stock outperforms)          [opt_features, all names, ~half covered]
 vrp   ATM IV - 20d realized vol (Bali & Hovakimian 2009: high IV-RV -> stock outperforms)
 wall  distance from price up to the call wall (max call OI strike), in % (room to run)            [opt_summary, 23 names]
 pwall distance down to the put wall (max put OI strike)"""
exec(open('/home/claude/fartpie/xs/rank_opts.py').read().split("FA = []")[0])
exec('def attach' + open('/home/claude/fartpie/xs/rank_opts.py').read().split('def attach')[1].split("print('A)")[0])
Fo = pd.read_csv(f'{EXT}/opt_features.csv', parse_dates=['date']).rename(columns={'sym': 'tk', 'date': 'd'}).sort_values(['tk', 'd'])
RV = []
for tk in L.tk.unique():
    b = bars(tk)
    if b is None: continue
    r = np.log(b[1]).diff(); RV.append(pd.DataFrame({'d': r.index, 'rv': (r.rolling(20).std() * np.sqrt(252)).values, 'tk': tk}))
Fo = Fo.merge(pd.concat(RV), on=['tk', 'd'], how='left')
Fo['cps'] = Fo.iv_c50 - Fo.iv_p50; Fo['vrp'] = (Fo.iv_c50 + Fo.iv_p50) / 2 - Fo.rv
for c in ('cps', 'vrp'): Fo[c] = Fo.groupby('tk')[c].shift(1)
ZB = attach(L, Fo[['tk', 'd', 'cps', 'vrp']], []).sort_values('entry').reset_index(drop=True)
print(f'B) all names, coverage {ZB[ZB.entry >= "2019-03-01"].cps.notna().mean():.0%}'); block(ZB, ['cps', 'vrp'])
FA = []
for f in glob.glob(f'{EXT}/opt_summary/*.csv'):
    tk = os.path.basename(f)[:-4]
    if tk not in set(L.tk): continue
    o = pd.read_csv(f, parse_dates=['date']).set_index('date').sort_index()
    X = pd.DataFrame({'wall': o.call_wall / o.spot - 1, 'pwall': 1 - o.put_wall / o.spot}).shift(1); X['tk'] = tk; FA.append(X.reset_index().rename(columns={'date': 'd'}))
ZA = attach(L[L.tk.isin([os.path.basename(f)[:-4] for f in glob.glob(f'{EXT}/opt_summary/*.csv')])], pd.concat(FA), []).sort_values('entry').reset_index(drop=True)
print(f'\nA) 23 names, OI walls'); block(ZA, ['wall', 'pwall'])
