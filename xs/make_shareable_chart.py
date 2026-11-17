import pickle, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; from matplotlib.patches import FancyBboxPatch
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
Z=pickle.load(open(SP+'shareable250.pkl','rb')); M,ALG,PROP=Z['M'],Z['ALG'],Z['PROP']; ev=PROP['evals']
EVAL_FEE=90; MONTH_FEE=100
years=list(range(2018,2027)); months_active={y:12 for y in years}; months_active[2018]=11; months_active[2026]=1
Y=pd.DataFrame(index=years)
Y['evals']=[sum(e['start'].year==y for e in ev) for y in years]; Y['passed']=[sum(e['start'].year==y and e['passed'] for e in ev) for y in years]
Y['npay']=[sum(1 for e in ev for d,x in e['pays'] if d.year==y) for y in years]; Y['pay']=[sum(x for e in ev for d,x in e['pays'] if d.year==y) for y in years]
Y['fee_eval']=Y.evals*EVAL_FEE; Y['fee_mo']=[months_active[y]*MONTH_FEE for y in years]; Y['fees']=Y.fee_eval+Y.fee_mo; Y['net']=Y.pay-Y.fees
bd={y:len(pd.bdate_range(max(pd.Timestamp(f'{y}-01-01'),pd.Timestamp('2018-02-01')),min(pd.Timestamp(f'{y}-12-31'),pd.Timestamp('2026-01-30')))) for y in years}
Y['ptsday']=[sum(r['by_year_pts'].get(y,0) for _,r in M.iterrows())/bd[y] for y in years]; Y['twin']=[ALG['by_year_tradewin'].get(y,np.nan) for y in years]; Y['dwin']=[ALG['by_year_dayw'].get(y,np.nan) for y in years]
tot_pay=Y.pay.sum(); tot_fee=Y.fees.sum(); net=tot_pay-tot_fee; yrs=ALG['yrs']; npay=int(Y.npay.sum())
print(Y.round(2).to_string()); print('payouts',tot_pay,'fees',tot_fee,'net',net,'roi',net/tot_fee,'payouts/mo',npay/(yrs*12))
# cumulative net by date
evts=[(e0['start'],-EVAL_FEE) for e0 in ev]+[(d,x) for e0 in ev for d,x in e0['pays']]+[(pd.Timestamp(f'{y}-{m:02d}-01'),-MONTH_FEE) for y in years for m in range(1,13) if pd.Timestamp(f'{y}-{m:02d}-01')>=pd.Timestamp('2018-02-01') and pd.Timestamp(f'{y}-{m:02d}-01')<=pd.Timestamp('2026-01-30')]
E=pd.DataFrame(evts,columns=['d','x']).sort_values('d'); E['c']=E.x.cumsum()
BLUE,ORG,INK,MUT,GRY='#2f6fdb','#e07b00','#1d2433','#6b7385','#e6e9f0'
plt.rcParams.update({'text.parse_math':False,'font.family':'DejaVu Sans','text.color':INK,'axes.edgecolor':GRY})
fig=plt.figure(figsize=(12,17),dpi=200,facecolor='white')
fig.text(.05,.968,'Prop algo backtest: NQ futures, 8 models',fontsize=25,fontweight='bold',ha='left')
fig.text(.05,.949,f'Lucid 50K simulation, one account at a time, Feb 2018 - Jan 2026 ({yrs:.1f} years). Backtest, not live results.',fontsize=11.5,color=MUT,ha='left')
tiles=[('Payouts received',f'${tot_pay:,.0f}',f'{npay} payouts, {npay/(yrs*12):.2f} a month'),('Fees paid',f'${tot_fee:,.0f}',f'{len(ev)} evals x ${EVAL_FEE} + ${MONTH_FEE}/month'),('Net after fees',f'${net:,.0f}',f'about ${net/yrs:,.0f} a year'),('Return on fees',f'{net/tot_fee:.1f}x',f'${tot_pay/tot_fee:.1f} paid out per $1 of fees'),
       ('Trade win rate',f'{ALG["win"]*100:.0f}%',f'{ALG["trades"]:,} trades ({ALG["per_yr"]:.0f} a year)'),('Winning days',f'{ALG["day_win"]*100:.0f}%',f'{ALG["days_traded"]:,} trading days'),('Points per day',f'{ALG["pts_day_all"]:.1f}',f'all models, 1 contract each ({ALG["pts_day_all"]*ALG["ndays"]:,.0f} pts)'),('Evals passed',f'{sum(e["passed"] for e in ev)} of {len(ev)}',f'{sum(e["passed"] for e in ev)/len(ev)*100:.0f}% pass rate')]
for i,(a,b,c) in enumerate(tiles):
    r,cc=divmod(i,4); x=.05+cc*.2325; y=.865-r*.077
    ax=fig.add_axes([x,y,.2175,.067]); ax.axis('off'); ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle='round,pad=0,rounding_size=.06',fc='#f5f7fb',ec=GRY,transform=ax.transAxes))
    ax.text(.06,.74,a,fontsize=10.5,color=MUT,transform=ax.transAxes); ax.text(.06,.32,b,fontsize=21,fontweight='bold',transform=ax.transAxes); ax.text(.06,.08,c,fontsize=8,color=MUT,transform=ax.transAxes)
# bars
ax=fig.add_axes([.075,.6,.4,.155]); xx=np.arange(len(years)); w=.38
ax.bar(xx-w/2,Y.pay,w,color=BLUE,label='Payouts'); ax.bar(xx+w/2,Y.fees,w,color=ORG,label='Fees (evals + monthly)')
for i,(p,f,nn) in enumerate(zip(Y.pay,Y.fees,Y.net)): ax.text(i-w/2,p+600,f'${p/1000:.1f}k',ha='center',fontsize=7.5)
ax.set_xticks(xx); ax.set_xticklabels([str(y) if y not in (2018,2026) else f"{y}*" for y in years],fontsize=9); ax.set_title('Per year: money paid out vs fees',loc='left',fontsize=12.5,fontweight='bold'); ax.legend(frameon=False,fontsize=9,loc='upper left'); ax.yaxis.set_major_formatter(lambda v,p:f'${v/1000:.0f}k'); ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',color=GRY,lw=.6); ax.set_axisbelow(True); ax.tick_params(labelsize=9)
ax2=fig.add_axes([.56,.6,.39,.155]); ax2.step(E.d,E.c,where='post',color=BLUE,lw=2); ax2.fill_between(E.d,E.c,step='post',color=BLUE,alpha=.12)
ax2.set_title('Cumulative net (payouts minus fees)',loc='left',fontsize=12.5,fontweight='bold'); ax2.yaxis.set_major_formatter(lambda v,p:f'${v/1000:.0f}k'); ax2.spines[['top','right']].set_visible(False); ax2.grid(axis='y',color=GRY,lw=.6); ax2.tick_params(labelsize=9); ax2.text(E.d.iloc[-1],E.c.iloc[-1],f'  ${E.c.iloc[-1]/1000:.1f}k',fontsize=9,va='center',color=BLUE,fontweight='bold'); ax2.set_xlim(E.d.iloc[0],E.d.iloc[-1]+pd.Timedelta(days=300))
# year table
def table(x,y,w,h,title,cols,rows,widths,aligns,fs=9.5,total=None):
    ax=fig.add_axes([x,y,w,h]); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.text(0,1.02,title,fontsize=12.5,fontweight='bold',transform=ax.transAxes,va='bottom')
    n=len(rows)+1+(1 if total else 0); rh=1/n; xs=np.r_[0,np.cumsum(widths)]/sum(widths)
    for j,c in enumerate(cols): ax.text(xs[j+1]-.01 if aligns[j]=='r' else xs[j]+.01,1-rh*.5,c,fontsize=fs-.5,fontweight='bold',color=MUT,ha='right' if aligns[j]=='r' else 'left',va='center')
    ax.plot([0,1],[1-rh,1-rh],color=INK,lw=1)
    allr=rows+([total] if total else [])
    for i,r in enumerate(allr):
        yy=1-rh*(i+1.5); 
        if total and i==len(allr)-1: ax.plot([0,1],[1-rh*(i+1)]*2,color=INK,lw=1)
        elif i%2==1: ax.add_patch(plt.Rectangle((0,1-rh*(i+2)),1,rh,fc='#f5f7fb',ec='none'))
        for j,v in enumerate(r): ax.text(xs[j+1]-.01 if aligns[j]=='r' else xs[j]+.01,yy,v,fontsize=fs,ha='right' if aligns[j]=='r' else 'left',va='center',fontweight='bold' if (total and i==len(allr)-1) else 'normal')
rows=[[str(y)+('*' if y in (2018,2026) else ''),f'{int(r.evals)}',f'{int(r.passed)}',f'{int(r.npay)}',f'${r.pay:,.0f}',f'${r.fees:,.0f}',(f'${r.net:,.0f}' if r.net>=0 else f'-${-r.net:,.0f}'),f'{r.twin*100:.0f}%',f'{r.dwin*100:.0f}%',f'{r.ptsday:.1f}'] for y,r in Y.iterrows()]
tot=['Total',f'{int(Y.evals.sum())}',f'{int(Y.passed.sum())}',f'{npay}',f'${tot_pay:,.0f}',f'${tot_fee:,.0f}',f'${net:,.0f}',f'{ALG["win"]*100:.0f}%',f'{ALG["day_win"]*100:.0f}%',f'{ALG["pts_day_all"]:.1f}']
table(.05,.335,.9,.19,'Year by year',['Year','Evals','Passed','Payouts','Paid out','Fees','Net','Trade win','Day win','Pts/day'],rows,[1.1,1,1,1,1.5,1.3,1.5,1.2,1.2,1.1],['l','r','r','r','r','r','r','r','r','r'],total=tot)
mr=[[r.model,f'{r.per_yr:.0f}',f'{r.win*100:.0f}%',f'{r.avg_pts:+.1f}',f'{r.tot_pts:,.0f}',f'{r.pts_day:.2f}',f'{r.pf:.2f}',f'{r.usd/M.usd.sum()*100:.0f}%'] for _,r in M.sort_values('model').iterrows()]
table(.05,.1,.9,.2,'Per model (1 contract, after costs)',['Model','Trades a year','Win rate','Avg pts per trade','Total pts','Pts per day','Profit factor','Share of profit'],mr,[1.2,1.2,1,1.5,1.2,1.1,1.1,1.2],['l','r','r','r','r','r','r','r'])
fig.text(.05,.066,'* 2018 starts Feb 1 (11 months of fees); 2026 is January only. NQ points per 1 contract after costs; Pts/day sums all models.',fontsize=8.6,color=MUT,wrap=True)
fig.text(.05,.045,f'Fees assume ${EVAL_FEE} per evaluation attempt and ${MONTH_FEE}/month. Payouts follow Lucid 50K rules (5 qualifying days, 50% cap, $2,000 max).',fontsize=8.6,color=MUT)
fig.text(.05,.03,'Simulation of a rule-based algorithm on historical data; past results do not guarantee future results. Not financial advice.',fontsize=8.6,color=MUT)
fig.savefig('/home/claude/fartpie/xs/algo_stats_shareable.png',dpi=200,facecolor='white'); print('saved')
