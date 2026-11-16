"""Report 215: placebo for the vector on LARGE caps (S&P point-in-time universe, no survivorship in the trade list; 15m bars available for the 65 cached names). DECLARED before results.
For every live-universe flip (wide_T: sp & ratio>=1.5) on a cached name: forward return from the entry bar's open over 1, 5 and 20 trading days (26 / 130 / 520 RTH 15m bars), vs 5 random RTH bars (before 14:00) of the SAME ticker within +-20 trading days of the flip. Flip minus random = the entry edge of the flip beyond owning the name in that period. By track (15m, 1H) and era (train<2022 | test). Welch t on per-flip differences (flip return minus the mean of its 5 matched randoms), clustered by entry day (mean per day)."""
import glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import stats
rng=np.random.default_rng(3); W=pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); T=W[W.sp&(W.ratio>=1.5)].copy(); rows=[]
for f in glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/*_15m_*.parquet'):
    tk=os.path.basename(f).split('_15m_')[0]; g=T[T.tk==tk]
    if not len(g): continue
    B=pd.read_parquet(f,columns=['datetime','open']); B['dt']=pd.to_datetime(B.datetime,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m=B.dt.dt.hour*60+B.dt.dt.minute; B=B[(m>=570)&(m<960)].drop_duplicates('dt').sort_values('dt'); dts=B.dt.values; o=B.open.values; n=len(o); mm=(B.dt.dt.hour*60+B.dt.dt.minute).values
    for e,h1 in zip(g.entry.values,g.h1.values):
        k=np.searchsorted(dts,e)
        if k>=n or dts[k]!=e or k+520>=n or k<520: continue
        cand=np.arange(k-520,k+520); cand=cand[(mm[cand]<840)&(abs(cand-k)>26)&(cand+520<n)]
        if len(cand)<5: continue
        rk=rng.choice(cand,5,replace=False); d={}
        for H in (26,130,520): d[f'f{H}']=o[k+H]/o[k]-1; d[f'r{H}']=np.mean(o[rk+H]/o[rk]-1)
        rows.append(dict(tk=tk,e=pd.Timestamp(e),h1=bool(h1),**d))
R=pd.DataFrame(rows); R['era']=np.where(R.e<'2022-01-01','train','test'); print('flips matched',len(R),'names',R.tk.nunique(),R.groupby(['h1','era']).size().to_dict())
for nm,m in (('15m track',~R.h1),('1H track',R.h1),('both',np.ones(len(R),bool))):
    for e in ('train','test'):
        s=R[m&(R.era==e)]; out=[]
        for H,lab in ((26,'1d'),(130,'5d'),(520,'20d')):
            d=(s[f'f{H}']-s[f'r{H}']); dd=d.groupby(s.e.dt.normalize()).mean(); t=dd.mean()/(dd.std(ddof=1)/np.sqrt(len(dd))); out.append(f'{lab}: flip {s[f"f{H}"].mean()*100:5.2f}% random {s[f"r{H}"].mean()*100:5.2f}% diff {d.mean()*100:+5.2f}pp t{t:+5.1f}')
        print(f'  {nm:10s}{e:6s} n{len(s):5d} | '+' | '.join(out))
