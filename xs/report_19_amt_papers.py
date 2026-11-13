"""Nov 13 2026: AMT rules from Rey's papers as STANDALONE NQ models, then pairing with the main algo. DECLARED BEFORE RESULTS (rules below are lifted from his text; thresholds are mine, listed here).
NQ 5m ETH, RTH 09:30-15:55 ET, settle = prior RTH close (15:55 bar close). Entry = limit/next-bar open as stated, flat 15:55, cost 0.5 pt, 1 contract, NQ points. ADR = 20-day mean RTH range. Eras A 2018-02..2021-12, B 2022-01..2026-01. PASS = same sign |t|>=2 both eras n>=30 each.
OVERNIGHT INVENTORY ('about 70% of the time a counter-auction'): inv = share of overnight bars (18:00-09:25) closing above settle; strong LONG >=0.9, strong SHORT <=0.1.
  INV-F   fade the inventory at 09:35 open (long inv -> short), exit 10:30 close, stop 25  | INV-F2 same, exit 15:55
  INV-C   continuation: strong inv and NO counter-auction in 09:30-10:30 (max adverse move from the open < 0.15 ADR) -> follow at 10:35 open, exit 15:55, stop 25
  INV-S   sweep-reclaim: long inv, RTH trades through the ON low then a 5m bar closes back above within 6 bars -> long (mirror for short inv/ON high); stop = sweep extreme -2 (risk 6-40), target 2R else 15:55 ('sold to the ON low, reclaimed, rallied 200')
VALUE AREA (prior RTH volume profile, 1-pt bins, 70% VA around POC): 
  VA-F    day opens inside VA; first touch of VAH (VAL) after 10:00 -> fade, limit at the edge, stop edge+0.15 ADR (cap 40), target POC, else 15:55 ('responsive trades at bracket extremes')
  VA-80   day opens OUTSIDE VA; two consecutive 30-min closes back inside -> go to the opposite edge: entry next 5m open, stop = entered edge -/+0.1 ADR beyond (cap 40), target opposite edge, else 15:55
INITIAL BALANCE: IB-B  IB = 09:30-10:30 range; first 5m close beyond IB between 10:35 and 14:00 -> follow at next open, stop = IB midpoint (cap 40), target = 1 x IB range beyond the edge, else 15:55. IB-Bn same only when IB < 0.35 ADR (narrow, expansion).
SHAPES (prior day): POC location p=(POC-L)/(H-L): P-shape p>=0.67 (short covering -> bearish next day), b-shape p<=0.33 (long liquidation -> bullish): SHAPE trade at 09:35 open against/with that bias, stop 40, flat 15:55.
VA-B (added after seeing VA-F lose in both eras): exact mirror of VA-F - same touch, go WITH the break, stop at the POC, target = VA-F's stop, +1.0 pt extra slippage (stop-order fill). It is a post-hoc mirror: treat as hypothesis, not a finding, until confirmed on new data.
POOR EXTREMES: poor high = prior day high within 2 pts reached by >=2 of the 13 30-min periods (unfinished business -> magnet): POOR-M long at 09:35 toward a poor high above open (target = the high), mirror for a poor low; if both poor skip; stop 25, else 15:55.
PAIRING: daily P&L at 0.5 contract vs main V2 legs (s6.pkl): corr, main+it per month, max DD, and trades on core-idle days."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute
by=dict(tuple(b.groupby('d'))); days=sorted(by); COST=.5
RTH={}
for d in days:
    g=by[d]; g=g[(g.m>=570)&(g.m<960)]
    if len(g)==78: RTH[d]=g.reset_index(drop=True)
def profile(g):
    h,l,v=g.high.values,g.low.values,g.volume.values.astype(float); lo=int(np.floor(l.min())); hi=int(np.ceil(h.max())); vol=np.zeros(hi-lo+2)
    for hh,ll,vv in zip(h,l,v):
        a,bb=int(np.floor(ll))-lo,int(np.floor(hh))-lo; vol[a:bb+1]+=vv/(bb-a+1)
    poc=int(np.argmax(vol)); tot=vol.sum(); acc=vol[poc]; i0=i1=poc
    while acc<.7*tot and (i0>0 or i1<len(vol)-1):
        up=vol[i1+1] if i1+1<len(vol) else -1; dn=vol[i0-1] if i0>0 else -1
        if up>=dn: i1+=1; acc+=up
        else: i0-=1; acc+=dn
    return lo+poc,lo+i1,lo+i0   # POC, VAH, VAL
PRO={d:profile(g) for d,g in RTH.items()}
def sim(h,l,c,f,ent,stop,dr,xend,tgt=None):
    worst=0.0
    for q in range(f,xend+1):
        worst=min(worst,((l[q] if dr>0 else h[q])-ent)*dr)
        if (l[q]<=stop) if dr>0 else (h[q]>=stop): return -abs(ent-stop)-COST,worst-COST
        if tgt is not None and ((h[q]>=tgt) if dr>0 else (l[q]<=tgt)): return abs(tgt-ent)-COST,worst-COST
    return (c[xend]-ent)*dr-COST,worst-COST
R=[]; STAT=[]
dl=[d for d in days if d in RTH]
for i,D in enumerate(dl):
    if i<21 or D<pd.Timestamp('2018-02-01') or D>pd.Timestamp('2026-01-30'): continue
    P=dl[i-1]
    if (D-P).days>5: continue
    g=RTH[D]; pg=RTH[P]; o,h,l,c=[g[k].values.astype(float) for k in ('open','high','low','close')]; adr=np.mean([RTH[d].high.max()-RTH[d].low.min() for d in dl[i-20:i]])
    settle=pg.close.values[-1]; pO,pvah,pval=PRO[P]; PH,PL=pg.high.max(),pg.low.min()
    on=by[P]; on=on[on.m>=1080]; on=pd.concat([on,by[D][by[D].m<570]])
    if len(on)<60: continue
    onc=on.close.values; inv=float((onc>settle).mean()); onh,onl=on.high.max(),on.low.min()
    XE=77; day=D
    def add(name,pts,mae,dr,m,**kw): R.append(dict(d=D,model=name,dr=dr,pts=pts,mae=mae,m=m,**kw))
    sl=1 if inv>=.9 else (-1 if inv<=.1 else 0)
    if sl!=0:
        # INV-F / F2 : fade at 09:35 open (bar 1)
        e=1; ent=o[e]; dr=-sl; stp=ent-dr*25
        p1,m1=sim(h,l,c,e,ent,stp,dr,12); add('INV-F',p1,m1,dr,575)   # exit bar 12 = 10:30 close
        p2,m2=sim(h,l,c,e,ent,stp,dr,XE); add('INV-F2',p2,m2,dr,575)
        # INV-C
        adv=(o[0]-l[:12].min()) if sl>0 else (h[:12].max()-o[0])
        if adv<.15*adr:
            e=13; ent=o[e]; p3,m3=sim(h,l,c,e,ent,ent-sl*25,sl,XE); add('INV-C',p3,m3,sl,630)
        # INV-S
        lvl=onl if sl>0 else onh
        for k in range(1,50):
            thr=(l[k]<lvl and c[k]>lvl) if sl>0 else (h[k]>lvl and c[k]<lvl)
            if thr:
                ext=l[k] if sl>0 else h[k]; e=k+1; ent=o[e]; stop=ext-2*sl; risk=abs(ent-stop)
                if 6<=risk<=40: p4,m4=sim(h,l,c,e,ent,stop,sl,XE,ent+sl*2*risk); add('INV-S',p4,m4,sl,570+5*e)
                break
    # VA-F
    if pval<=o[0]<=pvah:
        done=False
        for k in range(6,76):
            if done: break
            for dr,lvl in ((-1,pvah),(1,pval)):
                if (h[k]>=lvl-.25 and dr<0) or (l[k]<=lvl+.25 and dr>0):
                    stop=lvl-dr*min(.15*adr,40); ent=lvl; tg=pO
                    if abs(ent-stop)<6 or (dr<0 and tg>=ent) or (dr>0 and tg<=ent): done=True; break
                    p5,m5=sim(h,l,c,k,ent,stop,dr,XE,tg); add('VA-F',p5,m5,dr,570+5*k)
                    if abs(ent-pO)>=6:
                        pB,mB=sim(h,l,c,k,ent,pO,-dr,XE,stop); add('VA-B',pB-1.0,mB-1.0,-dr,570+5*k)   # MIRROR (continuation through the edge), +1 pt extra slippage
                    done=True; break
    elif o[0]>pvah or o[0]<pval:
        side=1 if o[0]<pval else -1   # opened below VA -> expect up
        cl30=[c[5+6*j] for j in range(13)]
        for j in range(1,12):
            a_,b_=cl30[j-1],cl30[j]; ins=lambda x: pval<=x<=pvah
            if ins(a_) and ins(b_):
                k=5+6*j; e=k+1; ent=o[e]; tgt=pvah if side>0 else pval; stop=(pval-min(.1*adr,40)) if side>0 else (pvah+min(.1*adr,40))
                if abs(ent-stop)>=6 and ((tgt>ent) if side>0 else (tgt<ent)):
                    p6,m6=sim(h,l,c,e,ent,stop,side,XE,tgt); add('VA-80',p6,m6,side,570+5*e)
                break
    # IB-B
    ibh,ibl=h[:12].max(),l[:12].min(); ibw=ibh-ibl
    for k in range(13,66):
        up=c[k]>ibh; dn=c[k]<ibl
        if up or dn:
            dr=1 if up else -1; e=k+1; ent=o[e]; mid=(ibh+ibl)/2; stop=mid if abs(ent-mid)<=40 else ent-dr*40
            if abs(ent-stop)>=6:
                tg=(ibh+ibw) if dr>0 else (ibl-ibw)
                p7,m7=sim(h,l,c,e,ent,stop,dr,XE,tg); add('IB-B',p7,m7,dr,570+5*e)
                if ibw<.35*adr: add('IB-Bn',p7,m7,dr,570+5*e)
            break
    # SHAPE
    pp=(pO-PL)/max(PH-PL,1e-9)
    if pp>=.67: dr=-1
    elif pp<=.33: dr=1
    else: dr=0
    if dr!=0:
        e=1; ent=o[e]; p8,m8=sim(h,l,c,e,ent,ent-dr*40,dr,XE); add('SHAPE',p8,m8,dr,575,shape='P' if dr<0 else 'b')
    # POOR-M
    per_h=[pg.high.values[6*j:6*j+6].max() for j in range(13)]; per_l=[pg.low.values[6*j:6*j+6].min() for j in range(13)]
    ph_=sum(1 for x in per_h if x>=PH-2)>=2; pl_=sum(1 for x in per_l if x<=PL+2)>=2
    if ph_!=pl_:
        if ph_ and PH>o[1]+10: dr=1; tg=PH
        elif pl_ and PL<o[1]-10: dr=-1; tg=PL
        else: dr=0
        if dr!=0:
            e=1; ent=o[e]; p9,m9=sim(h,l,c,e,ent,ent-dr*25,dr,XE,tg); add('POOR-M',p9,m9,dr,575)
        STAT.append((D, ph_, pl_, float(h.max()>=PH-1), float(l.min()<=PL+1)))
T=pd.DataFrame(R); T['era']=np.where(T.d<pd.Timestamp('2022-01-01'),'A','B'); T['yr']=T.d.dt.year; T.to_pickle(SC+'amt_models_T.pkl')
def t_(a): return a.mean()/(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>5 and a.std()>0 else np.nan
def line(nm,g):
    a,bb=g[g.era=='A'].pts,g[g.era=='B'].pts
    if len(a)<8 or len(bb)<8: return '%-9s n%d/%d'%(nm,len(a),len(bb))
    ok=' <== PASS' if (a.mean()*bb.mean()>0 and abs(t_(a))>=2 and abs(t_(bb))>=2 and min(len(a),len(bb))>=30) else ''
    return '%-9s A n %4d mean %+6.2f win %2.0f%% t %+5.1f | B n %4d mean %+6.2f win %2.0f%% t %+5.1f | yrs>0 %2.0f%% | tot A %+6.0f B %+6.0f%s'%(nm,len(a),a.mean(),(a>0).mean()*100,t_(a),len(bb),bb.mean(),(bb>0).mean()*100,t_(bb),(g.groupby('yr').pts.mean()>0).mean()*100,a.sum(),bb.sum(),ok)
print('trades',len(T))
for mname,g in T.groupby('model',sort=False): print(line(mname,g))
S=T[T.model=='SHAPE']
for sh in ('P','b'): print(line('SHAPE-'+sh,S[S['shape']==sh]))
for dr,nm in ((1,'long'),(-1,'short')):
    for mname in ('INV-F2','INV-C','VA-F','IB-B'):
        g=T[(T.model==mname)&(T.dr==dr)]; print(line(mname+' '+nm,g))
# poor extreme stats
S=pd.DataFrame(STAT,columns=['d','poorH','poorL','hitH','hitL']); S['era']=np.where(S.d<pd.Timestamp('2022-01-01'),'A','B')
print('\nPOOR extremes revisited next session (share):')
for e,g in S.groupby('era'):
    ph,pl=g[g.poorH],g[g.poorL]; print('  era',e,'poor high revisited %.0f%% (n %d)'%(100*ph.hitH.mean(),len(ph)),'| poor low revisited %.0f%% (n %d)'%(100*pl.hitL.mean(),len(pl)),'| all days high revisited %.0f%% low revisited %.0f%%'%(100*g.hitH.mean(),100*g.hitL.mean()))
O,N=pickle.load(open(SC+'s6.pkl','rb')); rows=[]
for n,(Tt,u,_) in zip(N,O['V2']):
    t=Tt.copy(); t['day']=pd.to_datetime(t.day); t['pnl']=t.pts*t.sz*u; rows.append(t[['day','pnl']])
M=pd.concat(rows); bd=pd.bdate_range('2018-02-01','2026-01-30'); md=M.groupby('day').pnl.sum().reindex(bd).fillna(0)
idl=pd.read_pickle(SC+'idle_state.pkl')['trade'].astype(bool); idl.index=pd.to_datetime(idl.index)
def st(s):
    mo=s.groupby([s.index.year,s.index.month]).sum(); eq=s.cumsum(); return s.sum(),mo.mean(),mo.min(),(eq-eq.cummax()).min()
mm=st(md); print('\n-- PAIRING (0.5 ct): main total %+.0f /mo %+.1f worst mo %+.0f maxDD %+.0f'%mm)
for mname,g in T.groupby('model',sort=False):
    cd=(g.groupby('d').pts.sum()*0.5).reindex(bd).fillna(0); c_=np.corrcoef(md,cd)[0,1]; s=st(md+cd); gi=g[g.d.map(idl)==False]
    print('%-8s alone %+6.0f pts | corr %+.2f | main+it: /mo %+5.1f worst mo %+5.0f maxDD %+5.0f | core-IDLE days: n %3d mean %+5.2f'%(mname,cd.sum(),c_,s[1],s[2],s[3],len(gi),gi.pts.mean() if len(gi) else np.nan))
