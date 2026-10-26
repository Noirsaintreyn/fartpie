import pandas as pd, numpy as np
T = pd.read_parquet('xs/ext/range_intraday.parquet'); T['y'] = T.d.dt.year
T = T[T.to_tp > 0]   # expected extreme not already passed at entry
def book(Z, a, b):
    Z = Z[(Z.d >= a) & (Z.d < b)].sort_values('entry'); r = []
    for d, g in Z.groupby('d'):
        g = g.head(6); r.append((d, np.log1p(g.ret.sum() / 6)))
    s = pd.Series(dict(r)); days = pd.bdate_range(a, min(pd.Timestamp(b), Z.d.max())); s = s.reindex(days, fill_value=0)
    eq = np.exp(s.cumsum()); y = len(s) / 252; return eq.iloc[-1] ** (1 / y) - 1, (eq / eq.cummax() - 1).min()
P = [('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01')]
print(f'{"variant":<52}{"trades":>8}{"avg":>8}{"win":>6}' + ''.join(f'{a[:4]+"-"+b[2:4]:>16}' for a, b in P))
for side, sn in ((1, 'long'), (-1, 'short'), (0, 'both sides')):
    for ctx, cn in ((None, 'any 15m flip'), (True, '15m + 1h agree')):
        for sl, tp in (('sl_exp', 'tp'), ('sl_day', 'tp'), ('no_sl', 'tp'), ('sl_exp', 'no_tp'), ('no_sl', 'no_tp')):
            z = T[(T.sl == sl) & (T.tp == tp)]
            if side: z = z[z.side == side]
            if ctx: z = z[z.both]
            s = ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (book(z, a, b) for a, b in P))
            lab = f'{sn}, {cn}, {"SL exp low" if sl=="sl_exp" else "SL day low" if sl=="sl_day" else "no SL"}, {"TP exp high" if tp=="tp" else "close at 16:00"}'
            print(f'{lab:<52}{len(z):>8}{z.ret.mean()*100:>+7.2f}%{(z.ret>0).mean():>6.0%}' + s)
        print()
z = T[(T.sl == 'sl_exp') & (T.tp == 'tp') & (T.side == 1)]
print('long, SL exp low, TP exp high: median distance to TP %.2f%%, to SL %.2f%%, avg trade by year:' % (z.to_tp.median()*100, z.to_sl.median()*100))
print((z.groupby('y').ret.mean() * 100).round(3).to_string())
