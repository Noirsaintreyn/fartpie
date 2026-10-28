"""Oct 28 2026: after-close reports - sell in the after-hours session once the news is bad, vs premarket / the system's 09:45 exit.
Decision uses only what is known then: the 16:30 price (close of the 16:15 bar) is <= -5% vs the 16:00 close. Prices = 15m bar closes."""
import os, glob, numpy as np, pandas as pd
rows = []
for f in glob.glob('/tmp/claude-0/ml/earn/*.csv'):
    tk = os.path.basename(f)[:-4]; fb = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    try: e = pd.read_csv(f)
    except Exception: continue
    if not fb or not len(e): continue
    B = pd.read_parquet(fb[0]); B['dt'] = pd.to_datetime(B.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    B = B.drop_duplicates('dt').set_index('dt').sort_index(); C, O = B.close, B.open
    m = B.index.hour * 60 + B.index.minute; R = B[(m >= 570) & (m < 960)]; dc = R.groupby(R.index.normalize()).close.last(); D = dc.index
    for tt in pd.to_datetime(e.iloc[:, 0], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None):
        if tt.hour < 12: continue
        d = tt.normalize()
        if d not in D or d < pd.Timestamp('2019-01-01'): continue
        k = D.get_loc(d)
        if k + 6 >= len(D): continue
        g = D[k + 1]; c0 = dc.iloc[k]; px = lambda t: C.get(t - pd.Timedelta('15min'), np.nan)    # price at t = close of the bar ending at t
        r = dict(tk=tk, d=d, a1630=px(d + pd.Timedelta('16h30min')), a1700=px(d + pd.Timedelta('17h')), a1800=px(d + pd.Timedelta('18h')), a1945=px(d + pd.Timedelta('19h45min')),
                 pm8=px(g + pd.Timedelta('8h')), o930=O.get(g + pd.Timedelta('9h30min'), np.nan), o945=O.get(g + pd.Timedelta('9h45min'), np.nan), close=dc.iloc[k + 1], d5=dc.iloc[k + 6])
        rows.append({k_: (v / c0 - 1 if k_ not in ('tk', 'd') else v) for k_, v in r.items()})
A = pd.DataFrame(rows).dropna(subset=['a1630', 'o945']); A['h'] = np.where(A.d.dt.year <= 2021, '2019-21', '2022-26')
cols = ['a1630', 'a1700', 'a1800', 'a1945', 'pm8', 'o930', 'o945', 'close', 'd5']
print(f'after-close reports with after-hours data: {len(A)}')
for th in (-.05, -.08):
    X = A[A.a1630 <= th]
    print(f'\n16:30 price <= {th:.0%}: n={len(X)}   mean return vs the 16:00 close by exit time')
    print((X.groupby('h')[cols].mean() * 100).round(2).to_string())
    print('share better than the 09:45 exit'); print(pd.DataFrame({c: (X[c] > X.o945).groupby(X.h).mean() for c in cols if c != 'o945'}).round(2).to_string())
X = A[A.a1630 <= -.05]; print('\nmedian (<= -5%)'); print((X[cols].median() * 100).round(2).to_string())
print('\nfalse alarms: 16:30 <= -5% but 09:45 better than the 16:30 price by 3%+:', f"{((X.o945 - X.a1630) > .03).mean():.0%}")
