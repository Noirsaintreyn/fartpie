src = open('/home/claude/fartpie/xs/stop_1h_ev.py').read().split("def build(var):")[0].replace('stop1h_V.parquet', 'twobots_V.parquet')
src = src.replace("XL = build('L')", "")
exec(src.split("XL = ")[0])
def build(nm):
    S = pd.concat([V[V['var'] == nm], V[V['var'] == nm + '_H1']]); S = S[(S.entry >= '2019-01-01') & ((S.entry.dt.hour * 60 + S.entry.dt.minute) < 840)].copy(); S['d'] = S.entry.dt.normalize()
    S = pd.merge_asof(S.sort_values('d'), RT, on='d', by='tk', direction='nearest', tolerance=pd.Timedelta(days=5)); return S[(S.sp == True) & (S.ratio >= 1.5)].sort_values('entry').reset_index(drop=True)
rest = open('/home/claude/fartpie/xs/stop_1h_ev.py').read().split("XL = build('L')")[1].split("B0 = ev(XL)")[0]; exec(rest.split('\n', 1)[1])
YTD = (pd.Timestamp('2025-12-31'), pd.Timestamp('2026-09-15'))
XP, XR = build('PRE'), build('RTH'); print('trades: PRE', len(XP), 'RTH', len(XR), '| avg trade PRE', f'{XP.ret.mean():+.3%}', 'RTH', f'{XR.ret.mean():+.3%}')
BP = ev(XP); BR = ev(XR); print('40 matched runs, 2019-21 / 2022-26'); line('PRE  (backtest / Mac: premarket + regular bars)', BP); line('RTH  (Render screener: regular-session bars only)', BR, BP)
k1 = set(zip(XP.tk, XP.entry, XP.h1)); k2 = set(zip(XR.tk, XR.entry, XR.h1)); print(f'trades identical in both: {len(k1 & k2) / len(k1 | k2):.0%} of the union; only-PRE {len(k1 - k2)}, only-RTH {len(k2 - k1)}')
for nm, X_ in (('PRE', XP), ('RTH', XR)):
    for yr in (2022, 2023, 2024, 2025, 2026):
        d = X_[X_.entry.dt.year == yr]; print(f'   {nm} {yr}: trades {len(d)}, avg {d.ret.mean():+.3%}', end='')
    print()
