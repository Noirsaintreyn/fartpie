"""report_127 (Nov 15 2026): CONTEXT LOOK-BACK MODEL (Rey: look at the last K trades of this stock / sector / cap-bucket / regime, and analogs, to forecast the next flip). All strictly point-in-time: a past flip enters a history only after its EXIT date (entry d + DUR bars) is before today's flip date.
Look-back features (winsorized net return at +100%; BIG hit rate): LB_tk (last 5 matured flips same ticker), LB_sec (last 10 same SIC2), LB_cell (last 20 same market-cap tercile x price bucket), LB_secpx (last 10 same SIC2 & px<5), LB_brd (last 20 flips whose market breadth tercile matches today's). KNN analog: monthly refit, 100 nearest matured flips in train-ECDF rank space (ATRP, EXT, LTURN, LMC, LPX, F11, F7, F4); mean winsorized return and P(BIG) of the neighbours.
Declared: each look-back [+] (recent peers' outcomes predict the next). Pass bar: test IC with BIG/net >0 and top-vs-bottom tercile month-clustered t>=2 on px<5, and incremental AUC over the frozen model."""
import numpy as np, pandas as pd, heapq, warnings; warnings.filterwarnings('ignore')
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'
O=pd.read_pickle(X+'combined_scores.pkl'); P=pd.read_pickle(X+'physics_flips.pkl')[['tk','d','DUR']]; V=pd.read_pickle(X+'vh_features.pkl')[['tk','d','BRD']]
O=O.merge(P,on=['tk','d'],how='left').merge(V,on=['tk','d'],how='left'); sic=pd.read_csv(X+'sic.csv'); sic['s2']=(sic.sic.fillna(0).astype(int)//100); O=O.merge(sic[['tk','s2']],on='tk',how='left')
O=O.sort_values('d').reset_index(drop=True); O['DUR']=O.DUR.fillna(O.DUR.median()).clip(lower=1)
O['exit']=[d+pd.offsets.BDay(int(k)) for d,k in zip(O.d,O.DUR)]
tr=O[~O.test]; O['mcb']=pd.cut(O.LMC,[-1,tr.LMC.quantile(.33),tr.LMC.quantile(.67),99],labels=False); O['pxb']=np.where(O.px<5,0,np.where(O.px<20,1,2)); O['cell']=O.mcb.fillna(-1).astype(int)*3+O.pxb
O['brdt']=pd.cut(O.BRD,[-1,tr.BRD.quantile(.33),tr.BRD.quantile(.67),2],labels=False)
O['w']=O.n.clip(upper=1.0); O['bg']=O.BIG.astype(float)
from collections import defaultdict, deque
H={k:defaultdict(lambda:deque(maxlen=K)) for k,K in (('tk',5),('s2',10),('cell',20),('s2px',10),('brd',20))}
O['s2px']=np.where(O.px<5,O.s2.fillna(-1)*10+1,np.nan); heap=[]; out={k:np.full(len(O),np.nan) for k in ('LB_tk','LB_sec','LB_cell','LB_secpx','LB_brd','LBb_tk','LBb_sec','LBb_cell')}
keys={'tk':'LB_tk','s2':'LB_sec','cell':'LB_cell','s2px':'LB_secpx','brd':'LB_brd'}
for i,r in enumerate(O.itertuples()):
    while heap and heap[0][0]<r.d:
        _,j=heapq.heappop(heap); q=O.iloc[j]
        for g,col in (('tk','tk'),('s2','s2'),('cell','cell'),('s2px','s2px'),('brd','brdt')):
            kv=q[col]
            if kv==kv: H[g][kv].append((q.w,q.bg))
    for g,col in (('tk','tk'),('s2','s2'),('cell','cell'),('s2px','s2px'),('brd','brdt')):
        kv=getattr(r,col)
        if kv==kv and kv in H[g] and len(H[g][kv])>=(2 if g=='tk' else 5):
            a=np.array(H[g][kv]); out[keys[g]][i]=a[:,0].mean()
            if g in('tk','s2','cell'): out['LBb_'+{'tk':'tk','s2':'sec','cell':'cell'}[g]][i]=a[:,1].mean()
    heapq.heappush(heap,(r.exit,i))
for k,v in out.items(): O[k]=v
# KNN monthly
rf=['ATRP','EXT','LTURN','LMC','LPX','F11','F7','F4']
def ecdf(c): s=np.sort(tr[c].dropna().values); return lambda x: np.searchsorted(s,x,side='right')/len(s)
R=np.column_stack([ecdf(f)(O[f].fillna(tr[f].median()).values) for f in rf]); O['KN_ret']=np.nan; O['KN_big']=np.nan
O['m']=O.d.dt.to_period('M')
for m,idx in O.groupby('m').groups.items():
    ms=m.to_timestamp(); mat=np.where(O.exit.values<np.datetime64(ms))[0]
    if len(mat)<400: continue
    nn=NearestNeighbors(n_neighbors=100).fit(R[mat]); _,nb=nn.kneighbors(R[idx]); O.loc[idx,'KN_ret']=O.w.values[mat][nb].mean(1); O.loc[idx,'KN_big']=O.bg.values[mat][nb].mean(1)
O.to_pickle(X+'lookback.pkl')
feats=['LB_tk','LB_sec','LB_cell','LB_secpx','LB_brd','LBb_tk','LBb_sec','LBb_cell','KN_ret','KN_big','m_BIG_0.5','lr','OS']
def t_(d): return d.mean()/(d.std()/np.sqrt(len(d))) if len(d)>5 and d.std()>0 else np.nan
for un,mk in (('ALL flips',O.px>0),('px<5',O.px<5)):
    print('\n####',un)
    for era,e in (('2020-21',(O.d>='2020-01-01')&(O.d<'2022-01-01')),('TEST 2022+',O.test)):
        U=O[mk&e]
        for f in feats:
            z=U.dropna(subset=[f])
            if len(z)<200: print(f'  {era} {f:11s} n {len(z)} (thin)'); continue
            q1,q2=z[f].quantile([1/3,2/3]); hi=z[z[f]>=q2]; lo=z[z[f]<=q1]; dd=(hi.groupby('mo').n.mean()-lo.groupby('mo').n.mean()).dropna()
            print(f'  {era} {f:11s} n {len(z):5d} AUC big {roc_auc_score(z.BIG,z[f]):.3f} huge {roc_auc_score(z.HUGE,z[f]):.3f} IC {spearmanr(z[f],z.n)[0]:+.3f} | hiT {hi.n.mean()*100:+6.1f}% med {hi.n.median()*100:+5.1f} loT {lo.n.mean()*100:+6.1f}% med {lo.n.median()*100:+5.1f} t {t_(dd):+.2f}')
# incremental: rank-blend frozen + knn on test
for un,mk in (('ALL',O.px>0),('px<5',O.px<5)):
    U=O[mk&O.test].dropna(subset=['KN_big','m_BIG_0.5']).copy(); rk=lambda c:U[c].rank(pct=True)
    for nm,sc in (('frozen m_BIG',rk('m_BIG_0.5')),('KNN P(BIG)',rk('KN_big')),('blend 50/50',rk('m_BIG_0.5')+rk('KN_big'))):
        c=sc.quantile(.8); top=U[sc>=c]; print(f'BLEND {un:5s} {nm:14s} AUC big {roc_auc_score(U.BIG,sc):.3f} huge {roc_auc_score(U.HUGE,sc):.3f} | top20 mean {top.n.mean()*100:+6.1f}% med {top.n.median()*100:+5.1f}% catch big {top.BIG.sum()/U.BIG.sum()*100:4.1f}% huge {top.HUGE.sum()/U.HUGE.sum()*100:4.1f}%')
