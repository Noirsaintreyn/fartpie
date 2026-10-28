"""Oct 28 2026: open interest instead of volume as the tie-break (23 names with daily OI, opt_summary 2018+; book on those names only).
 oir    total OI / 20d avg share volume, vs its own 252d median (positioning built up relative to normal)
 doi5   5d % change in total OI (new money opening positions)       dcoi5  5d % change in CALL OI
 coishare  call OI / total OI                                         dcs5   5d change in call OI share"""
exec(open('/home/claude/fartpie/xs/rank_opts.py').read().split("FA = []")[0])
FA = []
for f in glob.glob(f'{EXT}/opt_summary/*.csv'):
    tk = os.path.basename(f)[:-4]
    if tk not in set(L.tk): continue
    o = pd.read_csv(f, parse_dates=['date']).set_index('date').sort_index(); sh = shares(tk).reindex(o.index)
    oi = o.call_oi + o.put_oi; x = oi / sh.rolling(20).mean(); cs = o.call_oi / oi
    X = pd.DataFrame({'oir': x / x.rolling(252).median(), 'doi5': oi / oi.shift(5) - 1, 'dcoi5': o.call_oi / o.call_oi.shift(5) - 1,
                      'coishare': cs, 'dcs5': cs - cs.shift(5)}).shift(1)
    X['tk'] = tk; FA.append(X.reset_index().rename(columns={'date': 'd'}))
FA = pd.concat(FA)
exec('def attach' + open('/home/claude/fartpie/xs/rank_opts.py').read().split('def attach')[1].split("print('A)")[0])
ZA = attach(L[L.tk.isin(FA.tk.unique())], FA, []).sort_values('entry').reset_index(drop=True)
print(f'23 names, coverage oir {ZA.oir.notna().mean():.0%}   2019-03..2021 / 2022-26'); block(ZA, ['oir', 'doi5', 'dcoi5', 'coishare', 'dcs5'])
