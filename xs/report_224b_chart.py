"""Report 224b: chart of the random-trading Monte Carlo on the 50K eval rules (report 224)."""
import pickle, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; EVt,FUt=pickle.load(open(SP+'random_eval224.pkl','rb')); SZ=[1,2,3,5,8,10,15,20,30,40]
G=[((50,50),'Coin flip, stop 50 / target 50 points','#2f6fb0','o'),((50,150),'Coin flip, stop 50 / target 150 points','#d98324','s'),((None,None),'Coin flip, hold to the close','#2e8b6a','^')]
C='#1f2937'; plt.rcParams.update({'font.family':'DejaVu Sans','axes.edgecolor':'#9ca3af','axes.labelcolor':C,'xtick.color':'#4b5563','ytick.color':'#4b5563','font.size':10.5})
fig,ax=plt.subplots(1,3,figsize=(17,6.2),facecolor='#fcfcfb'); x=np.arange(len(SZ))
def clean(a,t,yl):
    a.set_facecolor('#fcfcfb'); a.grid(axis='y',color='#e5e7eb',lw=.8); a.set_axisbelow(True); [a.spines[s].set_visible(False) for s in ('top','right')]; a.set_xticks(x); a.set_xticklabels(SZ); a.set_xlabel('micro contracts (MNQ) per trade'); a.set_title(t,loc='left',fontweight='bold',color=C,fontsize=12); a.set_ylabel(yl)
clean(ax[0],'1. Chance of passing the eval','% of evals passed'); clean(ax[1],'2. Payouts collected per funded account','expected total payout (\\$)'); clean(ax[2],'3. Net result per \\$88 eval','expected profit per eval bought (\\$)')
for g,nm,col,mk in G:
    p=[EVt[(g,n)][0]*100 for n in SZ]; e=[FUt[(g,n)][0] for n in SZ]; net=[EVt[(g,n)][0]*FUt[(g,n)][0]-88 for n in SZ]
    ax[0].plot(x,p,color=col,lw=2,marker=mk,ms=6,label=nm); ax[1].plot(x,e,color=col,lw=2,marker=mk,ms=6,label=nm); ax[2].plot(x,net,color=col,lw=2,marker=mk,ms=6,label=nm)
    k=int(np.argmax(net)); ax[2].annotate(f'peak +\\${net[k]:.0f}\nat {SZ[k]} MNQ',(x[k],net[k]),textcoords='offset points',xytext=(8,8),fontsize=9,color='#374151')
ax[2].axhline(0,color='#6b7280',lw=1.2); ax[2].text(x[-1],4,'break-even',ha='right',fontsize=9,color='#4b5563'); ax[0].legend(frameon=False,loc='lower center',fontsize=9.5); ax[0].set_ylim(0,36)
fig.suptitle('No strategy at all: what coin-flip trading does on a \\$50K prop eval',x=.01,ha='left',fontsize=16,color=C,fontweight='bold')
fig.text(.01,.905,'One trade a day on real Nasdaq futures prices 2018-2026, direction chosen by coin flip. The only thing that changes along each line is position size.',fontsize=10.5,color='#374151')
fig.text(.01,.012,'Monte Carlo, 3,000 random paths per point, 1 point cost per trade. Rules modeled: \\$2,000 trailing drawdown (end of day, locks at +\\$100), \\$3,000 target, best day at most half of the profit, 5 winning days of \\$150+ for a payout,\npayout = half the balance up to \\$2,000 (90% kept), up to 5 payouts, 40 MNQ cap in the eval and 20-40 funded. No monthly or activation fees. Panel 3 uses the same size in the eval and the funded account. Hypothetical simulation.',fontsize=8,color='#4b5563')
ax[2].set_ylim(-95,140); plt.tight_layout(rect=[0,.07,1,.88]); plt.savefig(SP+'coinflip_eval_convexity.png',dpi=150,facecolor='#fcfcfb')
for g,nm,col,mk in G: print(nm,[round(EVt[(g,n)][0]*FUt[(g,n)][0]-88) for n in SZ])
