"""report_25 rerun on the London Strategic Edge 15m cache (lse_clean/, 9 names: AAPL AMZN AVGO GOOGL META MSFT NVDA QQQ TSLA, 2018-01 -> 2026-09/10, extended hours, times UTC->ET). Same A/B rules and controls as report_25. Strict split: era A 2018-2021 (train) vs era B 2022+ (test). Sessions: REGULAR (09:30-15:59 ET bars only) vs EXTENDED (04:00-19:59 ET bars; the cache has no 20:00-04:00 ET overnight bars). Also 1h bars (regular, resampled from 15m)."""
import glob
src=open('/home/claude/fartpie/xs/report_25_vector_pivot.py').read(); exec("\n".join(l for l in src.split("\n") if not l.startswith(("D1=fetch","R1=run","R15=run"))))
def load(f):
    d=pd.read_parquet(f); d['datetime']=pd.to_datetime(d.datetime); d=d.drop_duplicates('datetime').set_index('datetime').sort_index()
    d.index=d.index.tz_localize('UTC').tz_convert('America/New_York').tz_localize(None); return d[['open','high','low','close','volume']].astype(float)
F={f.split('/')[-1].split('_')[0]:f for f in sorted(glob.glob('/home/claude/fartpie/lse_clean/*.parquet'))}
print('names',list(F),flush=True)
raw={t:load(f) for t,f in F.items()}
for t,x in raw.items(): print(t,x.index.min(),x.index.max(),len(x),'hours',sorted(set(x.index.hour))[:3],'..',sorted(set(x.index.hour))[-3:])
def mk(sess,tf):
    out={}
    for t,x in raw.items():
        y=x.between_time('09:30','15:59') if sess=='regular' else x.between_time('04:00','19:59')
        if tf=='1h': y=y.between_time('09:30','15:59').resample('60min',offset='30min').agg(dict(open='first',high='max',low='min',close='last',volume='sum')).dropna(subset=['close'])
        out[t]=y
    return out
def runN(D,tf,Ns):
    R=[]
    for t,z in D.items(): R+=trades(z,t,tf,Ns)
    R=pd.DataFrame(R); R['half']=np.where(R.t<pd.Timestamp('2022-01-01'),'H1','H2'); return R
for sess,tf in (('regular','15m'),('extended','15m'),('regular','1h')):
    Ns=[4,8,16]; R=runN(mk(sess,tf),tf,Ns); print(f'\n>>>>>>>>>> {sess} {tf}  (H1 = 2018-21 train, H2 = 2022+ test)'); rep(R,f'{sess} {tf}',Ns); R.to_pickle(SC+f'vp_lse_{sess}_{tf}.pkl')
