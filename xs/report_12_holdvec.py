"""Nov 12 2026: DOES THE FLIP HOLD THE VECTOR? classify flips as NEW-LEG vs EXIT-OF-PRIOR-LEG (chop), and test a RETEST entry near the vector. DECLARED BEFORE RESULTS.
Rey's mechanism: a bull flip after a long bear leg is often the END of the move (shorts covering/exit), followed by consolidation - not the start of a new leg. A flip inside an uptrend (shallow pullback, MAs aligned) is continuation. The best entries are near the vector (small risk) but only if the flip is likely to hold.
Per 1h bull flip (bar fi; bear regime started at b0): dn_atr = size of the bear leg just completed (peak-to-trough in ATR), dn_bars, bounce = retracement of that leg at the flip close, up_prev_atr = bull leg before it, nflip60 = flips in the last 60 bars (chop), dist_trough = (close - trough)/ATR,
 trend: ema5020 (1h EMA50>EMA200), a200 (close>1h EMA200), a4 (close>4h EMA200), bD/b4 (daily/4h vector bull, last completed), kal (Kalman slope>0), rv (flip volume ratio), ext ((close-level)/ATR), atrp, lpx.
Hypotheses (signs): dn_atr high -> exit-of-move, WORSE hold/return (-); trend flags (ema5020, a200, a4, bD, b4, kal) -> hold (+); nflip60 high (chop) -> worse (-); bounce high -> worse (-).
Labels: HOLD12 = no bear flip within the next 12 bars (the flip 'holds the vector'); WIN = V0 trade return > 0 (next open entry, exit next open after bear flip, 40bp); MOVE = max high in next 24 bars >= entry + 3 ATR.
Models (fixed): logistic (C=.1) and LightGBM(4 leaves,150 trees). Train < 2025-07-01, test after. Reports AUC, quintile calibration on test, and filter effects.
RETEST ENTRY (floor-trader 'buy the pullback to value'): limit at vector level + 0.3 ATR(fi), live bars fi+1..fi+12 but only while still bull (fill needs bar index < bear-flip bar); fill = min(open, limit) if low <= limit and open > level; exit as usual; compared with V0 on the SAME filled flips and on all flips; R = (exit - fill)/(fill - level).
Books: $1-5 and BIG proxy (median daily $vol >= $50M, px>=10, cost 40bp for $1-5, 15bp for BIG)."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'xs'); sys.path.insert(0,'.')
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from vector_pine import run_state_machine
from vector_core import compute_atr_series
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data=pickle.load(open(SC+'uni_cache.pkl','rb')); SPLIT=pd.Timestamp('2025-07-01')
def htf(Y,tf):
    day=Y.index.normalize(); g=[day,(Y.index.hour>=13).astype(int)] if tf=='4h' else [day]
    G=Y.groupby(g); o=G.agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum'))
    first=G.apply(lambda x:x.index[0]); last=G.apply(lambda x:x.index[-1]); o.index=pd.DatetimeIndex(first.values)
    end=pd.DatetimeIndex(last.values)+pd.to_timedelta(np.where(pd.DatetimeIndex(last.values).hour==15,30,60),unit='m')
    reg=run_state_machine(o.assign(datetime=o.index))[0]; return pd.Series(reg,index=end).sort_index(), pd.Series(o.close.ewm(span=200,adjust=False).mean().values,index=end).sort_index()
def kslope(c):
    l=np.log(c); r=np.diff(l); R=np.var(r) if len(r)>3 else 1e-4; q1,q2=.05*R,.001*R; x0=l[0]; x1=0.; P00,P01,P11=R,0.,R; out=np.zeros(len(l))
    for i in range(len(l)):
        x0p=x0+x1; x1p=x1; P00p=P00+2*P01+P11+q1; P01p=P01+P11; P11p=P11+q2; S=P00p+R; K0,K1=P00p/S,P01p/S; y=l[i]-x0p; x0=x0p+K0*y; x1=x1p+K1*y; P00=(1-K0)*P00p; P01=(1-K0)*P01p; P11=P11p-K1*P01p; out[i]=x1
    return out
TK={tk:Y.dropna() for tk,Y in data.items() if len(Y.dropna())>=1500}
dv={tk:(Y.close*Y.volume).groupby(Y.index.normalize()).sum().median() for tk,Y in TK.items()}; big={t for t,v in dv.items() if v>=5e7}
rows=[]
for tk,Y in TK.items():
    idx=Y.index; o,hi,lo,c,vol=[Y[k].values for k in ('open','high','low','close','volume')]; atr=compute_atr_series(hi,lo,c,14); reg,lev,_,fl=run_state_machine(Y.assign(datetime=idx))
    e50=pd.Series(c).ewm(span=50,adjust=False).mean().values; e200=pd.Series(c).ewm(span=200,adjust=False).mean().values; ks=kslope(c); r4,e4=htf(Y,'4h'); rd,_=htf(Y,'1d'); vm=pd.Series(vol).rolling(20).mean().shift().values
    fl_i=[i for i,_ in fl]; fl_s=dict(fl); pos=None; last_bear=None; prev_bull_start=None
    for k,(i,s) in enumerate(fl):
        if s==-1: last_bear=i; 
        if s==1 and pos is None: pos=i
        elif s==-1 and pos is not None:
            fi=pos; xb=i+1; pos=None
            if xb>=len(Y) or fi<250 or not np.isfinite(atr[fi]) or not np.isfinite(vm[fi]): continue
            e=fi+1; px=o[e]; sm=1<=px<=5; bg=(tk in big and px>=10)
            if not(sm or bg): continue
            prev=[(j,t_) for j,t_ in fl if j<fi]; bears=[j for j,t_ in prev if t_==-1]; 
            if not bears: continue
            b0=bears[-1]; bulls_before=[j for j,t_ in prev if t_==1 and j<b0]; bs=bulls_before[-1] if bulls_before else max(b0-60,0)
            A=atr[fi]; tr_lo=lo[b0:fi+1].min(); pk=hi[max(b0-6,0):b0+1].max(); dn=pk-tr_lo
            end_t=idx[fi]+pd.Timedelta(minutes=30 if idx[fi].hour==15 else 60); a4=r4[r4.index<=end_t]; ad=rd[rd.index<=end_t]; v4=e4[e4.index<=end_t]
            if not len(a4) or not len(ad): continue
            hold=int(not any(j>fi and j<=fi+12 and t_==-1 for j,t_ in fl)) if fi+12<len(Y) else np.nan
            move=float(hi[e:min(e+24,len(Y))].max()>=px+3*A)
            cost=.004 if sm else .0015
            f=dict(tk=tk,t=idx[e],tx=idx[xb],book='1-5' if sm else 'BIG',ret=o[xb]/px-1-cost,hold=hold,move=move,
               dn_atr=dn/A,dn_pct=dn/pk,dn_bars=fi-b0,bounce=(c[fi]-tr_lo)/max(dn,1e-9),up_prev=(hi[bs:b0+1].max()-lev[bs] if np.isfinite(lev[bs]) else np.nan)/A,nflip60=sum(1 for j in fl_i if fi-60<j<=fi),dist_tr=(c[fi]-tr_lo)/A,
               ema5020=float(e50[fi]>e200[fi]),a200=float(c[fi]>e200[fi]),a4=float(c[fi]>v4.iat[-1]),bD=float(ad.iat[-1]==1),b4=float(a4.iat[-1]==1),kal=float(ks[fi]>0),rv=vol[fi]/vm[fi],ext=(c[fi]-lev[fi])/A,atrp=A/c[fi],lpx=np.log(px))
            # retest entry
            lim=lev[fi]+0.3*A; ent=None
            for kk in range(1,13):
                b=fi+kk
                if b>=i: break
                if lo[b]<=lim and o[b]>lev[fi]: ent=(b,min(o[b],lim)); break
            if ent:
                fp=ent[1]; f.update(rt_ret=o[xb]/fp-1-cost,rt_R=(o[xb]-fp)/max(fp-lev[fi],0.1*A),rt_wait=ent[0]-fi)
            rows.append(f)
T=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan); T['tr']=T.t<SPLIT; T.to_pickle(SC+'holdvec_T.pkl'); print('flips',len(T),T.groupby('book').size().to_dict())
FE=['dn_atr','dn_pct','dn_bars','bounce','up_prev','nflip60','dist_tr','ema5020','a200','a4','bD','b4','kal','rv','ext','atrp','lpx']
def st(g,col='ret'):
    if len(g)<5: return 'n%d'%len(g)
    r=g[col].sort_values(); return 'n%-4d win %2.0f%% mean %+5.1f%% med %+5.1f%% ex-top1 %+5.1f%%'%(len(g),(g[col]>0).mean()*100,g[col].mean()*100,g[col].median()*100,r.iloc[:int(len(r)*.99)].mean()*100)
for bk in ('1-5','BIG'):
    D=T[T.book==bk].dropna(subset=FE); tr,te=D[D.tr],D[~D.tr]; print('\n================ %s train %d test %d | HOLD12 rate train %.0f%% test %.0f%% | MOVE(+3ATR/24b) train %.0f%% test %.0f%%'%(bk,len(tr),len(te),tr.hold.mean()*100,te.hold.mean()*100,tr.move.mean()*100,te.move.mean()*100))
    print('-- single-feature test AUC (>0.5 => higher value = more of the label); train/test, sign must agree')
    for lab in ('hold','move'):
        a_,b_=tr[tr[lab].notna()],te[te[lab].notna()]
        print('  '+lab+': '+', '.join('%s %.2f/%.2f'%(f_,roc_auc_score(a_[lab].astype(int),a_[f_]),roc_auc_score(b_[lab].astype(int),b_[f_])) for f_ in FE))
    Zs=StandardScaler().fit(tr[FE]); 
    for lab in ('hold','move'):
        ok_t,ok_s=tr[lab].notna(),te[lab].notna(); ytr,yte=tr[lab][ok_t].astype(int),te[lab][ok_s].astype(int)
        lg=LogisticRegression(C=.1,max_iter=500).fit(np.clip(Zs.transform(tr[FE][ok_t]),-5,5),ytr); pl=lg.predict_proba(np.clip(Zs.transform(te[FE][ok_s]),-5,5))[:,1]
        gb=lgb.LGBMClassifier(num_leaves=4,n_estimators=150,learning_rate=.03,min_child_samples=50,subsample=.7,subsample_freq=1,colsample_bytree=.7,reg_lambda=5,verbose=-1,random_state=0).fit(tr[FE][ok_t],ytr); pg=gb.predict_proba(te[FE][ok_s])[:,1]
        print('-- label %s: test AUC logit %.3f  lgbm %.3f  (train lgbm %.3f)'%(lab,roc_auc_score(yte,pl),roc_auc_score(yte,pg),roc_auc_score(ytr,gb.predict_proba(tr[FE][ok_t])[:,1])))
        S=te[ok_s].assign(p=pg); S['q']=pd.qcut(S.p,5,labels=False,duplicates='drop')
        print('   calibration (test, lgbm quintiles low->high): '+' | '.join('Q%d %s rate %.0f%% ret mean %+.1f%% win %.0f%% n%d'%(q,lab,g[lab].mean()*100,g.ret.mean()*100,(g.ret>0).mean()*100,len(g)) for q,g in S.groupby('q')))
        imp=pd.Series(gb.booster_.feature_importance('gain'),index=FE); imp/=imp.sum(); print('   top features: '+', '.join('%s %.0f%%'%(k_,v*100) for k_,v in imp.sort_values(ascending=False).head(6).items()))
    # exit-of-move vs new-leg: big prior decline terciles
    q=tr.dn_atr.quantile([1/3,2/3]).values
    for nm,m in (('train',D.tr),('test ',~D.tr)):
        d=D[m]; print('  prior-leg size terciles (dn_atr) %s: '%nm+' | '.join('%s %s hold %.0f%%'%(lbl,st(g),g.hold.mean()*100) for lbl,g in (('small',d[d.dn_atr<=q[0]]),('mid',d[(d.dn_atr>q[0])&(d.dn_atr<=q[1])]),('LARGE',d[d.dn_atr>q[1]]))))
    # trend aligned
    al=(D.ema5020==1)&(D.a200==1)&(D.bD==1)
    for nm,m in (('train',D.tr),('test ',~D.tr)): d=D[m]; print('  trend-aligned (EMA50>200, >EMA200, daily bull) %s: yes %s hold %.0f%% | no %s hold %.0f%%'%(nm,st(d[al[m]]),d[al[m]].hold.mean()*100,st(d[~al[m]]),d[~al[m]].hold.mean()*100))
    print('  RETEST entry (limit vector+0.3ATR, 12 bars):')
    for nm,m in (('train',D.tr),('test ',~D.tr)):
        d=D[m]; f=d[d.rt_ret.notna()]; print('   %s fill rate %.0f%% (%d/%d) | retest %s | V0 same flips %s | V0 all %s'%(nm,len(f)/len(d)*100,len(f),len(d),st(f,'rt_ret'),st(f),st(d)))
        print('       R multiple: median %.2f mean %.2f, share R>=2: %.0f%%, share stopped (R<=-0.9) %.0f%% | trend-aligned retest %s'%(f.rt_R.median(),f.rt_R.mean(),(f.rt_R>=2).mean()*100,(f.rt_R<=-.9).mean()*100,st(f[al[m][f.index]],'rt_ret')))
