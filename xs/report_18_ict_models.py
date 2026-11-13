"""Nov 13 2026: ICT model batch on NQ 5m (Rey: test all, see if it pairs with the algo / adds trades when the algo is off). DECLARED BEFORE RESULTS.
Event = same sweep -> MSS+displacement chain as report_17 (sweep of PDH/PDL/ONH/ONL/Asia/London H/L by an RTH 5m bar 09:35-14:00 that closes back; within 12 bars a bar closes beyond the last 3-bar swing with body >= 0.6 ATR14 and an FVG). One trade per day per model.
ENTRY variants:  FVG = limit at FVG midpoint (report_17) | OB = limit at the 50% of the last opposite-colour candle before the displacement bar (<=15 bars back) | BRK (my definition of a breaker) = limit at the 50% of the last candle of the OLD trend direction before the sweep extreme (longs: last bullish candle before the lowest low of the 10 bars up to the sweep, now retested from above).
TARGET variants: 2R | DOL = nearest opposing liquidity pool (PDH/ONH/AsiaH/LonH for longs, lows for shorts) at >= 1R away; if none within 6R fall back to 2R.
Stop = sweep extreme -/+2 pts (skip risk <6 or >40). Limit must fill within 12 bars (0.25 pt through). Flat 15:55. Cost 0.5 pt. 1 contract, NQ points.
TIME tags on every trade: macro (entry minute-of-hour in 50-59 or 0-10), 90-min quarter (position of the fill inside its 90-min cycle counted from 18:00 ET: Q1..Q4).
PASS = same sign, |t|>=2 in both eras (A 2018-02..2021-12, B 2022-01..2026-01), n>=30 each. 6 models + tag slices: expect false positives.
PAIRING: each model's daily P&L at 0.5 contract vs main V2 legs (s6.pkl): correlation, combined $/month proxy (pts), max DD, worst month; and trades on days the core RTH legs are idle (idle_state)."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute
by=dict(tuple(b.groupby('d'))); days=sorted(by); COST=.5
RT=b[(b.m>=570)&(b.m<960)].groupby('d').agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last'))
def pivots(h,l,k):
    sh=sl=np.nan
    for p in range(k-4,2,-1):
        if np.isnan(sh) and h[p]==h[p-3:p+4].max(): sh=h[p]
        if np.isnan(sl) and l[p]==l[p-3:p+4].min(): sl=l[p]
        if not np.isnan(sh) and not np.isnan(sl): break
    return sh,sl
def sim(h,l,c,f,ent,stop,dr,xend,tgt):
    worst=0.0
    for q in range(f,xend+1):
        worst=min(worst,((l[q] if dr>0 else h[q])-ent)*dr)
        if (l[q]<=stop) if dr>0 else (h[q]>=stop): return -abs(ent-stop)-COST,worst-COST
        if (h[q]>=tgt) if dr>0 else (l[q]<=tgt): return abs(tgt-ent)-COST,worst-COST
    return (c[xend]-ent)*dr-COST,worst-COST
R=[]
for i,D in enumerate(days):
    if i<21 or D<pd.Timestamp('2018-02-01') or D>pd.Timestamp('2026-01-30') or days[i-1] not in RT.index or D not in RT.index: continue
    p=by[days[i-1]]; p=p[p.m>=1080]; cD=by[D]; cD=cD[cD.m<960]
    if len(p)<30 or len(cD)<150: continue
    x=pd.concat([p,cD]).reset_index(drop=True); m=x.m.values; msx=np.where(m>=1080,m-1080,m+360)
    h,l,c,o=[x[k].values.astype(float) for k in ('high','low','close','open')]; tr5=np.maximum.reduce([h-l,np.abs(h-np.r_[c[0],c[:-1]]),np.abs(l-np.r_[c[0],c[:-1]])]); atr=pd.Series(tr5).rolling(14).mean().values
    i930=int(np.searchsorted(msx,930)); i03=int(np.searchsorted(msx,540)); i08=int(np.searchsorted(msx,840)); ie=np.where(m==840)[0]; xe=np.where(m==955)[0]
    if not len(ie) or not len(xe): continue
    ie=int(ie[-1]); xend=int(xe[-1]); pd_=RT.loc[days[i-1]]; PDH,PDL=pd_.h,pd_.l; ONH,ONL=h[:i930].max(),l[:i930].min(); AH,AL=h[:i03].max(),l[:i03].min(); LH,LL=h[i03:i08].max(),l[i03:i08].min()
    lv_lo={'PDL':PDL,'ONL':ONL,'AsiaL':AL,'LonL':LL}; lv_hi={'PDH':PDH,'ONH':ONH,'AsiaH':AH,'LonH':LH}; highs=np.array(list(lv_hi.values())); lows=np.array(list(lv_lo.values()))
    done=False
    for k in range(i930+1,ie+1):
        if done: break
        for dr,lv in ((1,lv_lo),(-1,lv_hi)):
            hit=[n for n,v in lv.items() if (l[k]<v and c[k]>v)] if dr>0 else [n for n,v in lv.items() if (h[k]>v and c[k]<v)]
            if not hit: continue
            ext=l[k] if dr>0 else h[k]; sh,sl=pivots(h,l,k); piv=sh if dr>0 else sl
            if np.isnan(piv): continue
            for j in range(k+1,min(k+13,len(x)-1)):
                body=abs(c[j]-o[j]); ok=(c[j]>piv and c[j]>o[j]) if dr>0 else (c[j]<piv and c[j]<o[j])
                if not(ok and body>=.6*atr[j] and (l[j]>h[j-2] if dr>0 else h[j]<l[j-2])): continue
                # entry candidates
                ents={}
                ents['FVG']=((l[j]+h[j-2])/2) if dr>0 else ((h[j]+l[j-2])/2)
                for q in range(j-1,max(j-16,0),-1):
                    if (c[q]<o[q]) if dr>0 else (c[q]>o[q]): ents['OB']=(o[q]+c[q])/2; break
                e0=int(np.argmin(l[max(k-10,0):k+1])+max(k-10,0)) if dr>0 else int(np.argmax(h[max(k-10,0):k+1])+max(k-10,0))
                for q in range(e0-1,max(e0-11,0),-1):
                    if (c[q]>o[q]) if dr>0 else (c[q]<o[q]): ents['BRK']=(o[q]+c[q])/2; break
                stop=ext-2*dr; placed=False
                for en,ent in ents.items():
                    if (dr>0 and ent>=c[j]) or (dr<0 and ent<=c[j]): continue
                    f=None
                    for q in range(j+1,min(j+13,len(x))):
                        if (l[q]<=ent-.25) if dr>0 else (h[q]>=ent+.25): f=q; break
                    risk=abs(ent-stop)
                    if f is None or not(6<=risk<=40): continue
                    cand=(highs[highs>=ent+risk] if dr>0 else lows[lows<=ent-risk]); dol=(cand.min() if dr>0 else cand.max()) if len(cand) else np.nan
                    if np.isfinite(dol) and abs(dol-ent)>6*risk: dol=np.nan
                    t2=ent+dr*2*risk; tdol=dol if np.isfinite(dol) else t2; mm=int(m[f]); mof=mm%60; q4=int(((mm-1080)%1440%90)//22.5)
                    placed=True
                    for tn,tg in (('2R',t2),('DOL',tdol)):
                        pts,mae=sim(h,l,c,f,ent,stop,dr,xend,tg); R.append(dict(d=D,model=en+'-'+tn,dr=dr,m=mm,pts=pts,mae=mae,risk=risk,usedDOL=float(np.isfinite(dol)),macro=float(mof>=50 or mof<=10),q=q4))
                if placed: done=True
                break
            if done: break
T=pd.DataFrame(R); T['d']=pd.to_datetime(T.d); T['era']=np.where(T.d<pd.Timestamp('2022-01-01'),'A','B'); T['yr']=T.d.dt.year; T.to_pickle(SC+'ict_models_T.pkl')
def t_(a): return a.mean()/(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>5 and a.std()>0 else np.nan
def line(nm,g):
    a,bb=g[g.era=='A'].pts,g[g.era=='B'].pts
    if len(a)<8 or len(bb)<8: return '%-26s n%d/%d'%(nm,len(a),len(bb))
    ok=' <== PASS' if (a.mean()*bb.mean()>0 and abs(t_(a))>=2 and abs(t_(bb))>=2 and min(len(a),len(bb))>=30) else ''
    return '%-26s A n %3d mean %+6.2f win %2.0f%% t %+5.1f | B n %3d mean %+6.2f win %2.0f%% t %+5.1f | yrs>0 %2.0f%% | tot A %+5.0f B %+5.0f%s'%(nm,len(a),a.mean(),(a>0).mean()*100,t_(a),len(bb),bb.mean(),(bb>0).mean()*100,t_(bb),(g.groupby('yr').pts.mean()>0).mean()*100,a.sum(),bb.sum(),ok)
print('events/trades',len(T),'days',T.d.nunique())
for mname,g in T.groupby('model',sort=False): print(line(mname,g))
print('\n-- how often DOL target existed (share of trades): %.0f%%'%(100*T[T.model.str.endswith('DOL')].usedDOL.mean()))
print('\n-- TIME tags (all models pooled per model family FVG-2R shown separately)')
for mname in ('FVG-2R','OB-2R','FVG-DOL'):
    g=T[T.model==mname]; print(mname); print(line('  macro (:50-:10)',g[g.macro==1])); print(line('  not macro',g[g.macro==0]))
    for q in range(4): print(line('  90m quarter Q%d'%(q+1),g[g.q==q]))
# pairing
O,N=pickle.load(open(SC+'s6.pkl','rb')); rows=[]
for n,(Tt,u,_) in zip(N,O['V2']):
    t=Tt.copy(); t['day']=pd.to_datetime(t.day); t['pnl']=t.pts*t.sz*u; rows.append(t[['day','pnl']])
M=pd.concat(rows); bd=pd.bdate_range('2018-02-01','2026-01-30'); md=M.groupby('day').pnl.sum().reindex(bd).fillna(0)
idl=pd.read_pickle(SC+'idle_state.pkl')['trade'].astype(bool); idl.index=pd.to_datetime(idl.index)
def st(s):
    mo=s.groupby([s.index.year,s.index.month]).sum(); eq=s.cumsum(); return s.sum(),mo.mean(),mo.min(),(eq-eq.cummax()).min()
print('\n-- PAIRING with main V2 legs (pts at 0.5 contract): main total %+.0f /mo %+.1f worst mo %+.0f maxDD %+.0f'%((lambda r:(r[0],r[1],r[2],r[3]))(st(md))))
for mname,g in T.groupby('model',sort=False):
    cd=(g.groupby('d').pts.sum()*0.5).reindex(bd).fillna(0); c_=np.corrcoef(md,cd)[0,1]; s=st(md+cd); gi=g[g.d.map(idl)==False]
    print('%-9s chain-only %+6.0f pts | corr w/ main %+.2f | main+it: total %+6.0f (/mo %+5.1f) worst mo %+5.0f maxDD %+5.0f | on core-IDLE days: n %3d mean %+5.2f'%(mname,cd.sum(),c_,s[0],s[1],s[2],s[3],len(gi),gi.pts.mean() if len(gi) else np.nan))
