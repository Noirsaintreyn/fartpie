"""ORDER-FLOW DELTA (CORRECTED SIGN: export has buy/sell swapped -> true delta = sell_vol - buy_vol) vs the system + new trade types. MNQ 1-min, 2024-12..2025-11. Nov 10 2026."""
import pandas as pd, numpy as np, pickle
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
X=pd.read_parquet('/mnt/user-data/uploads/fartpie/mnq_orderflow_1min.parquet')
X['et']=X.datetime.dt.tz_convert('America/New_York').dt.tz_localize(None); X['day']=X.et.dt.normalize(); X['m']=X.et.dt.hour*60+X.et.dt.minute
X['delta']=X.sell_vol-X.buy_vol; X['vol']=X.buy_vol+X.sell_vol
R=X[(X.m>=570)&(X.m<960)]
B={d:g.set_index('m') for d,g in R.groupby('day') if len(g)>300}
def tstat(x): x=pd.Series(x).dropna(); return x.mean()/(x.std()/np.sqrt(len(x))) if len(x)>8 and x.std()>0 else np.nan
def halves(df,col='pts'): 
    mid=df.day.sort_values().iloc[len(df)//2]; a=df[df.day<mid][col]; b=df[df.day>=mid][col]; return a,b
# ---- day features
rows=[]
for d,g in B.items():
    px=g.last_price; o=px.iloc[0]
    def ratio(a,b): w=g[(g.index>=a)&(g.index<b)]; return w.delta.sum()/max(w.vol.sum(),1)
    def ret(a,b): 
        w=g[(g.index>=a)&(g.index<b)]; return w.last_price.iloc[-1]-w.last_price.iloc[0] if len(w)>2 else np.nan
    rows.append(dict(day=d,dr30=ratio(570,600),dr60=ratio(570,630),r30=ret(570,600),r60=ret(570,630),vol30=g[(g.index<600)].vol.sum()))
F=pd.DataFrame(rows).set_index('day')
F['agree60']=np.sign(F.dr60)==np.sign(F.r60)
F['effi']=F.dr60.abs()/F.dr60.abs().expanding().median()  # relative size
TD=pd.read_pickle(SC+'tiers_days.pkl'); TD.index=pd.to_datetime(TD.index).normalize()
D=TD.join(F,how='inner'); D['T']=D.tier.str[0].astype(int)
print(f'days with delta+system outcomes: {len(D)}')
print('\n== DAY-LEVEL: system day pts vs morning delta (corrected)')
for c in ('dr30','dr60'):
    q=D[c].abs().quantile([1/3,2/3]).values
    print(f' |{c}| corr w/ day pts {D[c].abs().corr(D.pts):+.2f}; signed corr {D[c].corr(D.pts):+.2f}')
    for nm,m in (('small',D[c].abs()<=q[0]),('mid',(D[c].abs()>q[0])&(D[c].abs()<q[1])),('large',D[c].abs()>=q[1])):
        print(f'   {nm:6} n {m.sum():3d} mean {D[m].pts.mean():+6.1f} terrible {np.mean(D[m]["T"]==1):.0%} great {np.mean(D[m]["T"]==5):.0%}')
print(' delta agrees with price over first hour:')
for k,m in (('agree',D.agree60),('DIVERGE (absorption)',~D.agree60)):
    print(f'   {k:22} n {m.sum():3d} mean {D[m].pts.mean():+6.1f} t {tstat(D[m].pts):+.2f} terrible {np.mean(D[m]["T"]==1):.0%} great {np.mean(D[m]["T"]==5):.0%}')
# ---- live legs: confirm/oppose at entry, leverage rule
O,N=pickle.load(open(SC+'s6.pkl','rb')); L=dict(zip(N,O['V2']))
rows=[]
for nm in ('RA','V15a'):
    T=L[nm][0]; T=T[T.sz>0]; 
    for r in T.itertuples():
        d=pd.Timestamp(r.day)
        if d not in B: continue
        g=B[d]; w=g[(g.index>=570)&(g.index<int(r.m))]
        if len(w)<5: continue
        c=-w.delta.sum()*0+w.delta.sum()/max(w.vol.sum(),1)*int(r.dir)
        rows.append((nm,d,r.pts,r.sz,c))
E=pd.DataFrame(rows,columns=['leg','day','pts','sz','cvd'])
print(f'\n== LIVE LEGS (RA+V15a) n {len(E)}  mean pts*sz {np.mean(E.pts*E.sz):+.2f}')
base=E.pts*E.sz
for nm,fn in (('veto opposing (x0)',lambda c:np.where(c>0,1,0)),('halve opposing',lambda c:np.where(c>0,1,.5)),('x1.5 confirm / x0.5 oppose',lambda c:np.where(c>0,1.5,.5)),('x2 confirm / x0 oppose',lambda c:np.where(c>0,2,0)),('x1.5 confirm / x0 oppose',lambda c:np.where(c>0,1.5,0))):
    p=base*fn(E.cvd); a,b=halves(E.assign(p=p),'p')
    print(f' {nm:28} total {p.sum():+8.0f} vs base {base.sum():+8.0f} | mean/trade {p.mean():+6.2f} | halves {a.sum():+7.0f}/{b.sum():+7.0f} (base {halves(E.assign(p=base),"p")[0].sum():+.0f}/{halves(E.assign(p=base),"p")[1].sum():+.0f}) | worst trade {p.min():+.0f} (base {base.min():+.0f})')
# ---- NEW TRADE TYPES on MNQ price, stop S pts, exit 15:55
def sim(g,m0,dr,S,m1=955):
    px=g.last_price; e=px.get(m0,np.nan)
    if np.isnan(e): return np.nan
    w=px[(px.index>m0)&(px.index<=m1)]
    for v in w.values:
        if (v-e)*dr<=-S: return -S
    return (w.iloc[-1]-e)*dr
def new_types(S):
    out={'A: delta momentum 10:00':[], 'B: absorption fade 10:30':[], 'C: delta+price strong 10:30 follow':[]}
    for d,g in B.items():
        f=F.loc[d]
        if abs(f.dr30)>=F.dr30.abs().median() and np.sign(f.dr30)==np.sign(f.r30) and not np.isnan(f.r30): out['A: delta momentum 10:00'].append((d,sim(g,600,int(np.sign(f.dr30)),S)))
        if not f.agree60 and abs(f.dr60)>=F.dr60.abs().median(): out['B: absorption fade 10:30'].append((d,sim(g,630,int(np.sign(f.dr60)),S)))
        if f.agree60 and abs(f.dr60)>=F.dr60.abs().quantile(2/3): out['C: delta+price strong 10:30 follow'].append((d,sim(g,630,int(np.sign(f.dr60)),S)))
    return out
for S in (40,60):
    print(f'\n== NEW TRADE TYPES, stop {S} pts, exit 15:55, 1 MNQ-pt = 1 NQ pt')
    for k,v in new_types(S).items():
        T=pd.DataFrame(v,columns=['day','pts']).dropna(); a,b=halves(T)
        print(f' {k:36} n {len(T):3d} mean {T.pts.mean():+6.1f} win {np.mean(T.pts>0):3.0%} t {tstat(T.pts):+.2f} | halves {a.mean():+6.1f}(t {tstat(a):+.1f}) / {b.mean():+6.1f}(t {tstat(b):+.1f}) | total {T.pts.sum():+.0f}')
pickle.dump((D,E),open(SC+'delta_out.pkl','wb'))
