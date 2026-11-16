"""Report 222: charts of the measured alpha. (A) large-cap options-ratio screener vs SPY: growth and beta-adjusted cumulative alpha (daily OLS on SPY, 2019-01..2026-09). (B) rolling 252-day alpha. (C) NQ prop algo (core legs + premium fade, FIG-V excluded, % of a $50K account at base sizes, before account rules) and its beta-adjusted alpha vs Nasdaq futures. (D) entry edge = flip minus random entries on the same names, 20-day forward return (reports 215, 217)."""
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/fartpie/xs'); import report_179_fractal as f
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
def ols(y,x):
    X=np.column_stack([np.ones(len(x)),x]); b=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b; se=np.sqrt(e.var(ddof=2)*np.linalg.inv(X.T@X)[0,0]); return b[0],b[1],b[0]/se
L=pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/site_wide_daily.parquet'); L.index=pd.to_datetime(L.index); a,bt,t=ols(L.m.values,L.SPY.values); al=L.m-bt*L.SPY
roll=pd.Series([ols(L.m.values[i-252:i],L.SPY.values[i-252:i])[0]*252 for i in range(252,len(L)+1)],index=L.index[251:])
print(f'large-cap: alpha {a*252:.1%}/yr beta {bt:.2f} t {t:.1f}; rolling 1y alpha min {roll.min():.1%} max {roll.max():.1%} share>0 {np.mean(roll>0):.0%}')
TR=f.prep(); U={'RA':1.0,'M1':.3,'R1':.5,'ON+':.4,'ON-':.1,'RULES':1.0,'V15a':.3}
core=pd.concat([t_.assign(day=pd.to_datetime(t_.day)).assign(usd=lambda x:x.pts*x.sz*U[n]*20)[['day','usd']] for n,t_ in TR.items()])
S=pd.read_pickle(SP+'s1_trades162.pkl'); S=S[S.reg>0]; s1=pd.DataFrame({'day':pd.to_datetime(S.day).dt.normalize(),'usd':S.pts.values*.2*20})
idx=pd.bdate_range('2018-02-01','2026-01-30'); dly=lambda x:x.groupby('day').usd.sum().reindex(idx).fillna(0); P=(dly(core)+dly(s1))/50000
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['d']=b.dt.dt.normalize(); cl=b.groupby('d').close.last().reindex(idx).ffill(); nq=cl.pct_change().fillna(0)
ap,bp,tp=ols(P.values,nq.values); alp=P-bp*nq; print(f'prop: alpha {ap*252:.1%}/yr of $50K, beta {bp:.3f}, t {tp:.1f}, corr with NQ {np.corrcoef(P,nq)[0,1]:.2f}')
C='#1f2937'; BL='#2f6fb0'; OR='#d98324'; GR='#6b7280'; plt.rcParams.update({'font.family':'DejaVu Sans','axes.edgecolor':'#9ca3af','axes.labelcolor':C,'xtick.color':'#4b5563','ytick.color':'#4b5563','font.size':10})
fig,ax=plt.subplots(2,2,figsize=(15,10),facecolor='#fcfcfb')
def clean(a):
    a.set_facecolor('#fcfcfb'); a.grid(axis='y',color='#e5e7eb',lw=.8); a.set_axisbelow(True); [a.spines[s].set_visible(False) for s in ('top','right')]
A=ax[0,0]; clean(A); g=(1+L.m).cumprod(); gs=(1+L.SPY).cumprod(); A.plot(g.index,g.values,color=BL,lw=2,label='Large-cap screener'); A.plot(gs.index,gs.values,color=GR,lw=1.6,label='SPY'); A.set_yscale('log'); A.set_yticks([1,2,5,10]); A.set_yticklabels(['1x','2x','5x','10x']); A.set_title('Large-cap screener vs SPY (growth of 1, log scale)',loc='left',fontweight='bold',color=C); A.legend(frameon=False,loc='upper left')
A.text(.98,.05,f'ends {g.iloc[-1]:.1f}x vs SPY {gs.iloc[-1]:.1f}x',transform=A.transAxes,ha='right',color='#374151')
B=ax[0,1]; clean(B); ca=al.cumsum()*100; B.plot(ca.index,ca.values,color=BL,lw=2); B.axhline(0,color='#9ca3af',lw=1); B.set_title('Screener alpha: return not explained by SPY (cumulative, % points)',loc='left',fontweight='bold',color=C)
B.text(.02,.93,f'alpha {a*252:.0%} a year, beta {bt:.2f}, t-stat {t:.1f}',transform=B.transAxes,color='#374151'); B2=roll*100
B.text(.02,.86,f'rolling 1-year alpha positive {np.mean(roll>0):.0%} of the time (worst {roll.min():.0%}, best {roll.max():.0%})',transform=B.transAxes,color='#374151')
Cx=ax[1,0]; clean(Cx); cp=P.cumsum()*100; ca2=alp.cumsum()*100; Cx.plot(cp.index,cp.values,color=BL,lw=2,label='Prop algo P&L (% of $50K, not compounded)'); Cx.plot(ca2.index,ca2.values,color=OR,lw=1.8,ls='--',label='after removing Nasdaq exposure'); Cx.axhline(0,color='#9ca3af',lw=1); Cx.set_title('NQ prop algo: almost all of it is alpha',loc='left',fontweight='bold',color=C); Cx.legend(frameon=False,loc='upper left')
Cx.text(.025,.78,f'alpha {ap*252:.0%} of $50K a year, beta to Nasdaq {bp:.2f}, t-stat {tp:.1f}',transform=Cx.transAxes,ha='left',color='#374151')
Dx=ax[1,1]; clean(Dx); labs=['Large-cap\n15m flips','Large-cap\n1H flips','Sub-$5\n1H flips','Sub-$5\ndaily flips']; e1=[1.04,0.46,-1.48,-1.08]; e2=[0.93,-0.29,-1.33,-2.53]; t1=[2.4,0.1,-2.9,-0.6]; t2=[3.4,-2.6,-2.8,-0.8]; x=np.arange(4); w=.36
r1=Dx.bar(x-w/2-.01,e1,w,color=BL,label='earlier period'); r2=Dx.bar(x+w/2+.01,e2,w,color=OR,label='later period'); Dx.axhline(0,color='#6b7280',lw=1); Dx.set_xticks(x); Dx.set_xticklabels(labs); Dx.set_ylabel('flip minus random entry, 20-day return (% points)')
for r,tv in list(zip(r1,t1))+list(zip(r2,t2)):
    h=r.get_height(); Dx.text(r.get_x()+r.get_width()/2,h+(.08 if h>=0 else -.08),f'{h:+.2f}\nt {tv:+.1f}',ha='center',va='bottom' if h>=0 else 'top',fontsize=8.5,color='#374151')
Dx.set_ylim(-3.6,2.2); Dx.set_title('Entry edge: does the flip beat a random entry on the same stock?',loc='left',fontweight='bold',color=C); Dx.legend(frameon=False,loc='upper right')
fig.suptitle('Where the measured alpha is',x=.01,ha='left',fontsize=16,color=C,fontweight='bold')
fig.text(.01,.006,'Hypothetical backtest results. Screener: S&P 500 point-in-time, 2019-01 to 2026-09, 2bp per trade. Prop algo: 2018-02 to 2026-01, FIG-V leg excluded, before prop-account rules.\nEntry edge: large caps 2019-21 vs 2022-26 (54 names); sub-$5 2024-02 to 2025-06 vs 2025-07 to 2026-11 (299 surviving names).',fontsize=8,color='#4b5563')
plt.tight_layout(rect=[0,.04,1,.95]); plt.savefig(SP+'alpha_charts.png',dpi=150,facecolor='#fcfcfb')
