"""Book test of the ATR-smoothing variants vs live (xs/vol_scale.py output), 40 matched runs, both periods."""
import glob
src = open('xs/stop_1h_ev.py').read(); exec(src[:src.index("B0 = ev(XL)")])
A = pd.concat(pd.read_parquet(f) for f in glob.glob('/tmp/claude-0/ml/volsc/*.parquet')); print('tickers', A.tk.nunique(), 'trades', len(A))
def bld(nm):
    S = A[A['var'] == nm].copy(); S = S[(S.entry >= '2019-01-01') & ((S.entry.dt.hour * 60 + S.entry.dt.minute) < 840)]; S['d'] = S.entry.dt.normalize()
    S = pd.merge_asof(S.sort_values('d'), RT, on='d', by='tk', direction='nearest', tolerance=pd.Timedelta(days=5)); return S[(S.sp == True) & (S.ratio >= 1.5)].sort_values('entry').reset_index(drop=True)
B0 = ev(XL); print('40 matched runs, 2019-21 / 2022-26'); line('L live (Wilder ATR14 (live))', B0)
for nm in ('natr', 'park', 'gk', 'gjr'):
    S = bld(nm); print(f'  {nm}: trades {len(S)}, avg {S.ret.mean()*100:+.3f}% (live {XL.ret.mean()*100:+.3f}%), runners>=15% {(S.ret >= .15).sum()} (live {(XL.ret >= .15).sum()})'); line(f'vol {nm}', ev(S), B0)
