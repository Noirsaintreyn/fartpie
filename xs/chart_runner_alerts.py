import sys, pickle, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; sys.path.insert(0,'xs'); sys.path.insert(0,'.')
from vector_pine import run_state_machine
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; D5=pickle.load(open(SC+'lowpx_5m.pkl','rb')); DD=pickle.load(open(SC+'lowpx_daily.pkl','rb')); E=pd.read_pickle(SC+'alert5m_E.pkl')
B=E[(E.VM==20)&(E.GM==.15)].drop_duplicates(['tk','day'])
def prep(X):
    X=X.copy(); X.index=X.index.tz_convert('America/New_York').tz_localize(None); X=X.rename(columns=str.lower)[['open','high','low','close','volume']].astype(float); return X[~X.index.duplicated()]
def alert_time(tk,day):
    X=prep(D5[tk]); d=DD[tk]; d=d[~d.index.duplicated()]; j=d.index.get_loc(pd.Timestamp(day)); med=d.Volume.shift(1).rolling(20).median().iloc[j]; pcl=d.Close.iloc[j-1]
    g=X[X.index.normalize()==pd.Timestamp(day)]; ok=np.where((g.volume.cumsum().values>=20*med)&(g.close.values>=pcl*1.15))[0]; return g.index[ok[0]] if len(ok) else None, pcl
rows=[]
for _,r in B.iterrows():
    t,pcl=alert_time(r.tk,r.day)
    if t is not None: rows.append((t,r.tk,pcl,r))
rows.sort(key=lambda x:x[0]); rows=rows[-10:]
def panel(ax,X,tk,t,pcl,title,show=70):
    reg,lev,_,_=run_state_machine(X.assign(datetime=X.index)); lev=pd.Series(lev,index=X.index); reg=pd.Series(reg,index=X.index)
    ia=X.index.searchsorted(t); s=max(0,ia-show//2); e=min(len(X),ia+show); W=X.iloc[s:e]; n=len(W); x=np.arange(n)
    L=lev.iloc[s:e].values; R=reg.iloc[s:e].values; RR=R.copy()
    for k,(i,r) in enumerate(W.iterrows()):
        col={1:'#3d8bff',-1:'#ff4d6d'}.get(int(R[k]),'#7a7f8c'); ax.plot([k,k],[r.low,r.high],color=col,lw=1); ax.add_patch(plt.Rectangle((k-.38,min(r.open,r.close)),.76,max(abs(r.close-r.open),(r.high-r.low)*.02,1e-9),color=col))
    for k in range(n-1):
        if np.isfinite(L[k]) and R[k]!=0: ax.plot([k,k+1],[L[k],L[k]],color='#b6ff3c' if R[k]==1 else '#ffb3c1',lw=2.2,drawstyle='steps-post')
    for k in range(1,n):
        if R[k]==1 and R[k-1]!=1: ax.annotate('\u25B2 BULL',(k,W.low.iloc[k]),xytext=(0,-12),textcoords='offset points',color='#b6ff3c',fontsize=8,ha='center',weight='bold')
        if R[k]==-1 and R[k-1]!=-1: ax.annotate('\u25BC BEAR',(k,W.high.iloc[k]),xytext=(0,6),textcoords='offset points',color='#ff8fa3',fontsize=8,ha='center',weight='bold')
    ka=ia-s; ax.axvline(ka,color='#ffd23c',lw=1,ls=':'); ax.axhline(pcl,color='#888',lw=.7,ls='--')
    ax.set_title(title,color='#ddd',fontsize=9,loc='left'); ax.set_facecolor('#0b0d12'); ax.tick_params(colors='#999',labelsize=7)
    d=W.index; tk_=[k for k in range(0,n,max(n//6,1))]; ax.set_xticks(tk_); ax.set_xticklabels([f'{d[k]:%d %H:%M}' for k in tk_],rotation=0)
    for sp in ax.spines.values(): sp.set_color('#333')
def fig_for(chunk,path):
    fig,axs=plt.subplots(len(chunk),2,figsize=(16,4.4*len(chunk)),facecolor='#0b0d12')
    for row,(t,tk,pcl,r) in zip(axs,chunk):
        X5=prep(D5[tk]); X15=X5.resample('15min').agg(dict(open='first',high='max',low='min',close='last',volume='sum')).dropna(subset=['close'])
        info=f'{tk}  alert {t:%a %b %d %H:%M}  (+{r.gain_at*100:.0f}% at alert, day high +{r.day_hi*100:.0f}%, to-close {r.rc*100:+.0f}%)'
        panel(row[0],X5,tk,t,pcl,'5m  '+info,60); panel(row[1],X15,tk,t,pcl,'15m  BLUE candle=bull vector, RED=bear, grey=none | line=vector level | yellow dotted=alert',40)
    fig.tight_layout(); fig.savefig(path,dpi=100,facecolor='#0b0d12'); plt.close(fig)
fig_for(rows[:5],SC+'runner_alerts_A.png'); fig_for(rows[5:],SC+'runner_alerts_B.png'); print([(f'{t:%m-%d %H:%M}',tk) for t,tk,_,_ in rows])
