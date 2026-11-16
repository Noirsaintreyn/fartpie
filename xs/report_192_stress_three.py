"""Report 192: stress test of the three model groups (core models, FIG-V on 1m fill-resolved bars, S1 premium fade regime-on) + chart.
Tests: (1) block bootstrap of daily $ (10-day blocks, 3000 paths) -> equity fan, P(total>0), 5th pct total, 95th pct max drawdown; (2) extra slippage per trade 0..3 pts (trade-level, scaled by size) -> breakeven slippage; (3) sign-flip placebo: random +/- sign on every trade (5000 draws) -> p-value of the real total; (4) random missed trades 20% (2000 draws); (5) leave-one-year-out total. Sized as in the deploy config (core legs u*sz, FIG-V 0.5 NQ, premium fade 0.2 NQ), no prop-account rules."""
import sys; sys.path.insert(0,'/home/claude/fartpie/xs')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, warnings; warnings.filterwarnings('ignore')
import report_179_fractal as f
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
TR=f.prep(); U={'RA':1.0,'M1':.3,'R1':.5,'ON+':.4,'ON-':.1,'RULES':1.0,'V15a':.3}
core=pd.concat([t.assign(day=pd.to_datetime(t.day),k=t.sz*U[n]*20) for n,t in TR.items()])[['day','pts','k']]; core=core[core.k>0]
K=pd.read_pickle(SP+'ifvg_1m189.pkl').dropna(subset=['pts1m']); figv=pd.DataFrame({'day':pd.to_datetime(K.d),'pts':K.pts1m.values,'k':.5*20})
S=pd.read_pickle(SP+'s1_trades162.pkl'); S=S[S.reg>0]; s1=pd.DataFrame({'day':pd.to_datetime(S.day).dt.normalize(),'pts':S.pts.values,'k':.2*20})
G={'Core models':core,'FIG-V (1m fill-resolved)':figv,'Premium fade (regime on)':s1}; idx=pd.bdate_range('2018-02-01','2026-01-30'); rng=np.random.default_rng(7)
def clip(t): return t[(t.day>='2018-02-01')&(t.day<='2026-01-30')].copy()
res={}
for nm,t in G.items():
    t=clip(t); t['usd']=t.pts*t.k; D=t.groupby('day').usd.sum().reindex(idx).fillna(0).values; N=len(D); tot=D.sum(); cs=np.cumsum(D); mdd=(cs-np.maximum.accumulate(cs)).min()
    # block bootstrap
    B=10; nb=int(np.ceil(N/B)); paths=np.empty((3000,N))
    for p in range(3000):
        st=rng.integers(0,N-B,nb); paths[p]=np.concatenate([D[s:s+B] for s in st])[:N]
    pc=np.cumsum(paths,1); ttot=pc[:,-1]; pmd=(pc-np.maximum.accumulate(pc,1)).min(1)
    # slippage
    sl=np.linspace(0,3,31); stot=[(t.usd-(x*t.k)).sum() for x in sl]; be=next((x for x,v in zip(sl,stot) if v<=0),np.nan)
    # sign flip placebo
    flips=np.array([(t.usd.values*rng.choice([-1,1],len(t))).sum() for _ in range(5000)]); pv=(flips>=tot).mean()
    # missed trades
    miss=np.array([t.usd.values[rng.random(len(t))>.2].sum() for _ in range(2000)])
    yrs=t.groupby(t.day.dt.year).usd.sum(); loyo=[tot-v for v in yrs.values]
    res[nm]=dict(D=D,cs=cs,tot=tot,mdd=mdd,pc=pc,ttot=ttot,pmd=pmd,sl=sl,stot=stot,be=be,flips=flips,pv=pv,miss=miss,yrs=yrs,n=len(t),mean_pts=t.pts.mean(),loyo=loyo)
    print(f'{nm}: trades {len(t)}, avg {t.pts.mean():+.2f} pts, total ${tot:,.0f}, max DD ${mdd:,.0f}')
    print(f'   bootstrap: P(total>0) {np.mean(ttot>0):.1%}, 5th pct total ${np.percentile(ttot,5):,.0f}, median ${np.median(ttot):,.0f}, 95th pct max DD ${np.percentile(pmd,5):,.0f}')
    print(f'   breakeven extra slippage {be:.1f} pts/trade (total at +1 pt: ${stot[10]:,.0f}, +2 pts: ${stot[20]:,.0f})')
    print(f'   sign-flip placebo p = {pv:.4f} (null mean ${flips.mean():,.0f}, sd ${flips.std():,.0f}); 20% missed trades: P(total>0) {np.mean(miss>0):.1%}, 5th pct ${np.percentile(miss,5):,.0f}')
    print('   leave-one-year-out totals: min $%s, worst year contributes $%s of $%s'%(f'{min(loyo):,.0f}',f'{yrs.max():,.0f}',f'{tot:,.0f}'))
    print('   by year:',{int(a):int(round(b)) for a,b in yrs.items()})
C='#1f2937'; plt.rcParams.update({'font.family':'DejaVu Sans','axes.edgecolor':'#9ca3af','axes.labelcolor':C,'xtick.color':'#4b5563','ytick.color':'#4b5563'})
fig,ax=plt.subplots(3,3,figsize=(16,11),facecolor='white'); cols={'Core models':'#2f6fb0','FIG-V (1m fill-resolved)':'#d98324','Premium fade (regime on)':'#2a9d6f'}
for j,(nm,r) in enumerate(res.items()):
    c=cols[nm]; a=ax[0,j]; lo,mid,hi=np.percentile(r['pc'],[5,50,95],axis=0)
    a.fill_between(idx,lo,hi,color=c,alpha=.2,lw=0,label='5-95% of 3,000 reshuffled histories'); a.plot(idx,mid,color=c,lw=1,ls='--',label='median reshuffle'); a.plot(idx,r['cs'],color='#111827',lw=2,label='actual')
    a.set_title(nm,loc='left',fontsize=13,color=C,fontweight='bold'); a.grid(axis='y',color='#e5e7eb',lw=.8); [a.spines[s].set_visible(False) for s in ('top','right')]
    a.text(.02,.97,f"P(profit) {np.mean(r['ttot']>0):.0%} | 5th pct ${np.percentile(r['ttot'],5):,.0f}",transform=a.transAxes,va='top',fontsize=9,color='#374151'); 
    if j==0: a.set_ylabel('Cumulative $ (bootstrap)'); a.legend(loc='center left',frameon=False,fontsize=8)
    a=ax[1,j]; a.plot(r['sl'],r['stot'],color=c,lw=2.4); a.axhline(0,color='#6b7280',lw=1); a.grid(axis='y',color='#e5e7eb',lw=.8); [a.spines[s].set_visible(False) for s in ('top','right')]
    a.set_xlabel('Extra slippage, points per trade'); 
    if j==0: a.set_ylabel('Total $ after slippage')
    txt='stays positive past 3 pts' if np.isnan(r['be']) else f"breaks even at {r['be']:.1f} pts"; a.text(.98,.95,txt,transform=a.transAxes,ha='right',va='top',fontsize=10,color=C)
    a=ax[2,j]; a.hist(r['flips'],bins=50,color='#9ca3af',alpha=.8); a.axvline(r['tot'],color=c,lw=3); a.set_xlabel('Total $ if every trade had a random sign'); [a.spines[s].set_visible(False) for s in ('top','right')]
    if j==0: a.set_ylabel('Placebo draws')
    a.text(.98,.95,f"real ${r['tot']:,.0f}\nplacebo p = {r['pv']:.3f}",transform=a.transAxes,ha='right',va='top',fontsize=10,color=C)
fig.suptitle('Stress tests: bootstrap, slippage and placebo for the three model groups (2018-2026, before prop-account rules)',x=.01,ha='left',fontsize=15,color=C)
plt.tight_layout(rect=[0,0,1,.96]); plt.savefig(SP+'stress_three.png',dpi=140)
