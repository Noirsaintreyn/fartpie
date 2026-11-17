"""Report 276 (declared before results). New causal leads for the mean-reversion filler, all measured as drift (no stops) first. For checkpoints 10:30 11:30 12:30 13:30 14:30 (5m closes): dev = (close - session VWAP)/R20 (prior-20d mean RTH range), cross = number of VWAP crossings since 10:00 (two-sided-day state, NOT the vector flip count of 254), inOR = price still inside the first-hour range (checked from 11:00), gamma = prior-close QQQ net GEX > 0. Outcome fade = -sign(dev) x (close 60 min later - close now), pts per NQ contract, no cost (net of 1pt shown too). Cells: top third |dev| (cut from train) split by cross >= 3 vs <= 1, by inOR, by gamma sign, and the combination gamma+ & cross>=3. Per-day averages across checkpoints, t over days, train 2019-21 | test 2018+2022-26. A cell is a lead only if positive with the same sign in both splits and n_days >= 80 in train."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index(); rth=b.between_time('09:30','15:55'); G=rth.groupby(rth.index.normalize())
dd=pd.DataFrame({'h':G.high.max(),'l':G.low.min(),'n':G.size()}); dd=dd[dd.n>=70]; dd['R20']=(dd.h-dd.l).shift(1).rolling(20).mean()
gx=pd.read_csv('xs/ext/gex/QQQ.csv',parse_dates=['date']).set_index('date').sort_index(); kg=np.clip(gx.index.searchsorted(dd.index.values.astype('datetime64[ns]'))-1,0,len(gx)-1); dd['gpos']=gx.net_gex.values[kg]>0
CK=['10:30','11:30','12:30','13:30','14:30']; rows=[]
for d,g in G:
    if d not in dd.index or not np.isfinite(dd.R20[d]): continue
    c=g.close.values;h=g.high.values;l=g.low.values;v=g.volume.values.astype(float); tp=(h+l+c)/3; vw=np.cumsum(tp*v)/np.maximum(np.cumsum(v),1); t=np.array(g.index.strftime('%H:%M')); R=dd.R20[d]
    sg=np.sign(c-vw)
    for ck in CK:
        w=np.where(t==ck)[0]; w0=np.where(t=='10:00')[0]
        if len(w)==0 or len(w0)==0: continue
        j=int(w[0]); k0=int(w0[0])
        if j+12>=len(c): continue
        cr=int((np.diff(sg[k0:j+1])!=0).sum()); dev=(c[j]-vw[j])/R
        orh=h[:12].max(); orl=l[:12].min(); inor=bool(h[:j+1].max()<=orh and l[:j+1].min()>=orl) if j>=12 else np.nan
        f=-np.sign(dev)*(c[j+12]-c[j]); rows.append((d,ck,dev,cr,inor,dd.gpos[d],f))
T=pd.DataFrame(rows,columns=['d','ck','dev','cross','inor','gpos','f']); T['sp']=np.where((T.d.dt.year>=2019)&(T.d.dt.year<=2021),'train','test')
cut=T[T.sp=='train'].dev.abs().quantile(2/3); T['big']=T.dev.abs()>=cut
print('top-third |dev| cut (R20 units):',round(cut,3),' obs',len(T))
def cell(m):
    s=[]
    for sp in ('train','test'):
        q=T[m&(T.sp==sp)].groupby('d').f.mean(); s.append(f'days{len(q):4d} {q.mean():+6.2f} (t{(q.mean()/(q.std()/np.sqrt(len(q)))) if len(q)>2 else np.nan:+.1f})')
    return ' | '.join(s)
print('fade of VWAP deviation, 60-min, pts/contract gross (train | test)')
print(' all obs                    ',cell(T.dev.notna()))
print(' big |dev|                  ',cell(T.big))
print(' big |dev| & gamma+         ',cell(T.big&T.gpos)); print(' big |dev| & gamma-         ',cell(T.big&~T.gpos))
print(' big & cross>=3             ',cell(T.big&(T.cross>=3))); print(' big & cross<=1             ',cell(T.big&(T.cross<=1)))
print(' big & cross>=3 & gamma+    ',cell(T.big&(T.cross>=3)&T.gpos)); print(' big & cross>=3 & gamma-    ',cell(T.big&(T.cross>=3)&~T.gpos))
print(' big & inOR (>=11:00)       ',cell(T.big&(T.inor==True))); print(' big & outside OR           ',cell(T.big&(T.inor==False)))
print(' by checkpoint (big & gamma+):'); 
for ck in CK: print('   ',ck,cell(T.big&T.gpos&(T.ck==ck)))
print(' by checkpoint (big & cross>=3):')
for ck in CK: print('   ',ck,cell(T.big&(T.cross>=3)&(T.ck==ck)))
