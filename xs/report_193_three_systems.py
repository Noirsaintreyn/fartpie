"""Report 193: one chart, three systems: (1) NQ prop algo (core models + premium fade, FIG-V excluded after the fill-bar audit), (2) large-cap options-ratio screener (S&P 500 point-in-time, ratio>=1.5, 6 slots, daily MTM 2019-01..2026-09-25, xs/ext/site_wide_daily.parquet), (3) mid/small-cap book (website series live_data/site_series.json 'mid', monthly). Stress: 3-month block bootstrap of monthly returns (5,000 paths).
Prop algo shown as % of a $50K account at base deploy sizes, NOT compounded, before prop-account rules (a one-account view). Benchmarks: prop = Nasdaq (NQ) buy&hold scaled to the same daily volatility; large-cap = SPY; mid/small = equal-weight of the same names ('ew')."""
import sys, json; sys.path.insert(0,'/home/claude/fartpie/xs')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, warnings; warnings.filterwarnings('ignore')
import report_179_fractal as f
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; rng=np.random.default_rng(11)
# --- prop algo daily $ (core + S1 regime-on 0.2 NQ), as % of 50K
TR=f.prep(); U={'RA':1.0,'M1':.3,'R1':.5,'ON+':.4,'ON-':.1,'RULES':1.0,'V15a':.3}
core=pd.concat([t.assign(day=pd.to_datetime(t.day)).assign(usd=lambda x:x.pts*x.sz*U[n]*20)[['day','usd']] for n,t in TR.items()])
S=pd.read_pickle(SP+'s1_trades162.pkl'); S=S[S.reg>0]; s1=pd.DataFrame({'day':pd.to_datetime(S.day).dt.normalize(),'usd':S.pts.values*.2*20})
idx=pd.bdate_range('2018-02-01','2026-01-30'); dly=lambda x:x.groupby('day').usd.sum().reindex(idx).fillna(0); P=dly(core)+dly(s1)
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['d']=b.dt.dt.normalize(); cl=b.groupby('d').close.last().reindex(idx).ffill(); nq=cl.diff().fillna(0)*20; nq=nq*(P.std()/nq.std())
prop_m=(P.groupby(P.index.to_period('M')).sum()/50000); bm_prop=(nq.groupby(nq.index.to_period('M')).sum()/50000)
# --- large-cap
L=pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/site_wide_daily.parquet'); L.index=pd.to_datetime(L.index)
lc_m=(1+L.m).groupby(L.index.to_period('M')).prod()-1; bm_lc=(1+L.SPY).groupby(L.index.to_period('M')).prod()-1
# --- mid/small (monthly growth index)
j=json.load(open('/home/claude/fartpie/live_data/site_series.json'))['mid']; mi=pd.PeriodIndex(j['d'],freq='M'); gb=pd.Series(j['b'],index=mi); ge=pd.Series(j['ew'],index=mi)
mid_m=gb.pct_change().fillna(gb.iloc[0]-1); bm_mid=ge.pct_change().fillna(ge.iloc[0]-1)
SYS={'NQ prop algo':(prop_m,bm_prop,'Nasdaq futures, same volatility','#2f6fb0',False),'Large-cap options-ratio screener':(lc_m,bm_lc,'SPY','#d98324',True),'Mid/small-cap book':(mid_m,bm_mid,'Equal-weight, same names','#2a9d6f',True)}
def curve(r,comp): return (1+r).cumprod() if comp else 1+r.cumsum()
def mdd(c): return (c/c.cummax()-1).min() if True else 0
out={}
for nm,(r,bm,bn,col,comp) in SYS.items():
    c=curve(r,comp); cb=curve(bm,comp); yrs=len(r)/12; cagr=(c.iloc[-1])**(1/yrs)-1 if comp else r.sum()/yrs; sh=r.mean()/r.std()*np.sqrt(12); dd=(c/c.cummax()-1) ; ddb=(cb/cb.cummax()-1)
    B=3; n=len(r); nb=int(np.ceil(n/B)); paths=np.empty((5000,n)); v=r.values
    for p in range(5000):
        st=rng.integers(0,n-B+1,nb); paths[p]=np.concatenate([v[s:s+B] for s in st])[:n]
    pc=(1+np.cumsum(paths,1)) if not comp else np.cumprod(1+paths,1); pmd=((pc/np.maximum.accumulate(pc,1))-1).min(1); end=pc[:,-1]
    out[nm]=dict(r=r,c=c,cb=cb,dd=dd,ddb=ddb,pc=pc,pmd=pmd,end=end,cagr=cagr,sh=sh,bn=bn,col=col,comp=comp,yrs=yrs)
    ann=(lambda e:(e**(1/yrs)-1)) if comp else (lambda e:(e-1)/yrs)
    print(f'{nm}: {r.index[0]}..{r.index[-1]} ({n} months) {"CAGR" if comp else "avg yearly % of 50K"} {cagr:.1%}, Sharpe {sh:.2f}, max DD {dd.min():.1%} (benchmark {ddb.min():.1%}), +months {np.mean(r>0):.0%}, worst month {r.min():.1%}')
    print(f'   bootstrap: P(end below start) {np.mean(end<1):.1%}, 5th pct yearly {np.percentile(ann(end),5):.1%}, median {np.median(ann(end)):.1%}, 95th pct max DD {np.percentile(pmd,5):.1%}; benchmark yearly {((cb.iloc[-1])**(1/yrs)-1 if comp else (cb.iloc[-1]-1)/yrs):.1%}')
C='#1f2937'; plt.rcParams.update({'font.family':'DejaVu Sans','axes.edgecolor':'#9ca3af','axes.labelcolor':C,'xtick.color':'#4b5563','ytick.color':'#4b5563'})
fig,ax=plt.subplots(3,3,figsize=(16,11),facecolor='white')
for j_,(nm,o) in enumerate(out.items()):
    col=o['col']; ix=o['c'].index.to_timestamp()
    a=ax[0,j_]; a.plot(ix,o['c'].values,color=col,lw=2.4,label=nm); a.plot(ix,o['cb'].values,color='#6b7280',lw=1.6,label=o['bn']); a.set_title(nm,loc='left',fontsize=13,color=C,fontweight='bold'); a.grid(axis='y',color='#e5e7eb',lw=.8); [a.spines[s].set_visible(False) for s in ('top','right')]
    if j_==0: a.set_ylabel('Growth of 1 (prop algo: 1 + P&L / $50K, not compounded)')
    a.legend(loc='upper left',frameon=False,fontsize=9); ctext=('CAGR' if o['comp'] else 'avg/yr'); a.text(.98,.05,f"{ctext} {o['cagr']:.0%} | Sharpe {o['sh']:.1f} | max DD {o['dd'].min():.0%}",transform=a.transAxes,ha='right',fontsize=9,color='#374151')
    a=ax[1,j_]; lo,mid,hi=np.percentile(o['pc'],[5,50,95],axis=0); a.fill_between(ix,lo,hi,color=col,alpha=.2,lw=0,label='5-95% of 5,000 reshuffled histories'); a.plot(ix,mid,color=col,ls='--',lw=1,label='median reshuffle'); a.plot(ix,o['c'].values,color='#111827',lw=2,label='actual'); a.grid(axis='y',color='#e5e7eb',lw=.8); [a.spines[s].set_visible(False) for s in ('top','right')]
    a.text(.02,.97,f"P(finish below start) {np.mean(o['end']<1):.1%} | worst-5% max DD {np.percentile(o['pmd'],5):.0%}",transform=a.transAxes,va='top',fontsize=9,color='#374151')
    if j_==0: a.set_ylabel('Bootstrap of monthly returns'); a.legend(loc='center left',frameon=False,fontsize=8)
    a=ax[2,j_]; a.fill_between(ix,o['ddb'].values*100,0,color='#9ca3af',alpha=.6,lw=0,label=o['bn']); a.fill_between(ix,o['dd'].values*100,0,color=col,alpha=.7,lw=0,label=nm); a.grid(axis='y',color='#e5e7eb',lw=.8); [a.spines[s].set_visible(False) for s in ('top','right')]
    if j_==0: a.set_ylabel('Drawdown %')
    a.legend(loc='lower left',frameon=False,fontsize=8)
fig.suptitle('Three systems: growth, stress test and drawdown vs benchmark',x=.01,ha='left',fontsize=15,color=C)
fig.text(.01,.005,'Prop algo 2018-02..2026-01 (before prop-account rules, FIG-V excluded). Large-cap 2019-01..2026-09 (point-in-time S&P 500). Mid/small-cap: website series on a hand-picked universe (survivorship), short history: read the level with care.',fontsize=8,color='#4b5563')
plt.tight_layout(rect=[0,.02,1,.96]); plt.savefig(SP+'three_systems.png',dpi=140)
