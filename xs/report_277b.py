"""Report 277 (declared before results; follow-up to 276, post-hoc lead). 276 found: fading a large VWAP deviation over 60 min earns +4.5|+5.4 pts (t 4.0|3.6) and the effect is concentrated in ONE-SIDED days (<=1 VWAP crossing since 10:00: +9.0|+10.4, t 5.1|5.6), not two-sided ones. Checks: (1) by year, (2) by checkpoint, (3) dose-response in |dev| (train quintile cuts), (4) overlap with the main algo (are these in-trend pullbacks? share of dev sign agreeing with the day's final direction), (5) tradeable form: enter at the checkpoint close toward VWAP, stop at a multiple of R20 beyond the entry (S = .10 / .15 / .20 R20), target = VWAP, time exit +60 min, 1pt cost; pts/trade, win rate, n/yr, train | test. Pass bar: positive in BOTH splits net, positive in >= 6 of 8 years."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index(); rth=b.between_time('09:30','15:55'); G=rth.groupby(rth.index.normalize())
dd=pd.DataFrame({'h':G.high.max(),'l':G.low.min(),'c':G.close.last(),'o':G.open.first(),'n':G.size()}); dd=dd[dd.n>=70]; dd['R20']=(dd.h-dd.l).shift(1).rolling(20).mean()
CK=['10:30','11:30','12:30','13:30','14:30']; rows=[]
for d,g in G:
    if d not in dd.index or not np.isfinite(dd.R20[d]): continue
    c=g.close.values;h=g.high.values;l=g.low.values;v=g.volume.values.astype(float); tp=(h+l+c)/3; vw=np.cumsum(tp*v)/np.maximum(np.cumsum(v),1); t=np.array(g.index.strftime('%H:%M')); R=dd.R20[d]; sg=np.sign(c-vw); dday=np.sign(dd.c[d]-dd.o[d])
    w0=np.where(t=='10:00')[0]
    if len(w0)==0: continue
    k0=int(w0[0])
    for ck in CK:
        w=np.where(t==ck)[0]
        if len(w)==0: continue
        j=int(w[0])
        if j+12>=len(c): continue
        cr=int((np.diff(sg[k0:j+1])!=0).sum()); dev=(c[j]-vw[j])/R; s=-np.sign(dev)
        if s==0: continue
        e=c[j]; tgt=vw[j]; res={}
        for S in (.10,.15,.20,.35,.50,9.0):
            stop=e-s*S*R*1.0  # adverse = against fade direction: price keeps going away from VWAP
            stop=e+(-s)*S*R if False else (e+ (-s)*S*R)
            r=None
            for q in range(j+1,min(j+13,len(c))):
                hs=(h[q]>=stop) if s==-1 else (l[q]<=stop); ht=(l[q]<=tgt) if s==-1 else (h[q]>=tgt)
                if hs: r=-S*R; break
                if ht: r=abs(e-tgt); break
            if r is None: r=s*(c[min(j+12,len(c)-1)]-e)
            res[S]=r-1
        rows.append((d,ck,dev,cr,s==dday,res[.10],res[.15],res[.20],s*(c[j+12]-e),res[.35],res[.50],res[9.0],-min(0,(s*(np.where(s==1,l[j+1:j+13],-h[j+1:j+13]) - s*e*0 - (e if s==1 else -e))).min()) if False else 0))
T=pd.DataFrame(rows,columns=['d','ck','dev','cross','withday','r10','r15','r20','f60','r35','r50','rNo','x']); T['yr']=T.d.dt.year; T['sp']=np.where((T.yr>=2019)&(T.yr<=2021),'train','test')
cut=T[T.sp=='train'].dev.abs().quantile([.2,.4,.6,.8]).values; big=T.dev.abs()>=T[T.sp=='train'].dev.abs().quantile(2/3); one=T.cross<=1
def cell(m,col='f60'):
    s=[]
    for sp in ('train','test'):
        q=T[m&(T.sp==sp)].groupby('d')[col].mean(); s.append(f'days{len(q):4d} {q.mean():+6.2f} (t{(q.mean()/(q.std()/np.sqrt(len(q)))) if len(q)>2 else np.nan:+.1f})')
    return ' | '.join(s)
M=big&one
print('(1) by year: fade drift (60m pts, gross), big & cross<=1')
print('  ',{y:(round(T[M&(T.yr==y)].groupby("d").f60.mean().mean(),1),int(T[M&(T.yr==y)].d.nunique())) for y in range(2018,2027)})
print('(2) by checkpoint, big & cross<=1'); 
for ck in CK: print('  ',ck,cell(M&(T.ck==ck)))
print('(3) dose-response |dev| quintile (cross<=1)')
q=np.digitize(T.dev.abs(),cut)
for i in range(5): print('  Q',i+1,cell(one&(q==i)))
print('(4) share of fades whose direction agrees with the day final direction (fade is WITH the day): train/test',
      [round(T[M&(T.sp==sp)].withday.mean(),2) for sp in ('train','test')],' (base in trend pullbacks would be ~.5+)')
print('(5) tradeable: stop S x R20 beyond entry, target VWAP, 60-min exit, net 1pt (big & cross<=1)')
for col,S in (('r10','.10'),('r20','.20'),('r35','.35'),('r50','.50'),('rNo','none')):
    z=T[M]; s=[]
    for sp in ('train','test'):
        x=z[z.sp==sp][col]; s.append(f'n{len(x):4d} ({len(x)/z[z.sp==sp].yr.nunique():.0f}/yr) {x.mean():+5.1f} (t{x.mean()/(x.std()/np.sqrt(len(x))):+.1f}) win {(x>0).mean():.0%}')
    print(f'  S={S}: '+' | '.join(s)); print('     by year',{y:round(z[z.yr==y][col].mean(),1) for y in range(2018,2027) if (z.yr==y).any()})
