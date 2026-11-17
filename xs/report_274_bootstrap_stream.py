"""Report 274 (declared before results). Stream view: daily $ P&L of the 8-model algo at base sizes (shareable250 daily, 2018-02..2026-01). Simplified Lucid 50K Flex chain on DAILY pnl (approximation: no intraday MAE, no E2/F0 sizing policies, daily loss clipped at -$1,000): eval floor = min(peak-2000, +100), pass +3000; funded same floor, qualifying day >= $150, payout after 5 qualifying days = min(50% of profit, $2,000) if >= paymin, 90% kept, graduate after 5 payouts, then a new eval; fee $90 a eval + $100 a month. (1) VALIDATION on the actual path vs the full sim. (2) Stationary block bootstrap (mean block 20 days, 1,000 paths, length of the history = 2,087 days) from (a) all days, (b) 2022+ days. Metrics across paths: payouts a month (mean, p10, p90), longest gap in months (median, p90), months with >= 1 payout, months with >= 2, net $ after fees. (3) Variants applied day-wise BEFORE bootstrap: global size x.75/x1.25/x1.5; RANGE-TARGET size = clip(median R20 / R20_t, .6, 1.8) (prior-20d mean RTH range, causal; pnl assumed linear in contracts); EVENT size x1.5 on NFP / big-tech reaction days (post-2022 era only, flagged). Rules tested: Lucid Flex only (other firms' parameters not verified)."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
s=pickle.load(open(SP+'shareable250.pkl','rb')); d=s['daily'].copy(); d.index=d.index.to_timestamp() if hasattr(d.index,'to_timestamp') else pd.to_datetime(d.index); d=d[d.index<='2026-01-30']
e=pickle.load(open(SP+'ev256.pkl','rb')); e.index=pd.to_datetime(e.index); e['rng']=e.h-e.l; e['R20']=e.rng.shift(1).rolling(20).mean()
df=pd.DataFrame({'p':d}).join(e[['R20','nfp','react','yr']],how='left'); ref=df.R20[:'2021'].median(); df['R20']=df.R20.fillna(ref)
def run(p,paymin):
    n=len(p); i=0; pays=[]; ev=0; p=np.maximum(p,-1000.0)
    while i<n:
        ev+=1; eq=pk=0.0; ok=False
        while i<n:
            eq+=p[i]; pk=max(pk,eq); i+=1
            if eq<=min(pk-2000,100): break
            if eq>=3000: ok=True; break
        if not ok: continue
        eq=pk=0.0; npay=0; q=0
        while i<n and npay<5:
            x=p[i]; eq+=x; pk=max(pk,eq); i+=1
            if eq<=min(pk-2000,100): break
            if x>=150: q+=1
            if q>=5:
                amt=min(.5*eq,2000.0)
                if amt>=paymin: eq-=amt; pays.append((i,amt*.9)); npay+=1; q=0
    return pays,ev
def metr(p,paymin,nmo):
    pays,ev=run(p,paymin); n=len(p); idx=[a for a,_ in pays]; tot=sum(b for _,b in pays)
    g=np.diff([0]+idx+[n])*1.45/30.4 if idx else np.array([n*1.45/30.4])
    mo=np.zeros(nmo,int)
    for a in idx: mo[min(a*nmo//n,nmo-1)]+=1
    return len(pays)/nmo, g.max(), (mo>=1).mean(), (mo>=2).mean(), tot-ev*90-nmo*100
NM=96
def boot(x,paymin,B=1000,L=20,seed=0):
    rng=np.random.default_rng(seed); n=2087; out=[]
    for _ in range(B):
        idx=[]; 
        while len(idx)<n:
            st=rng.integers(0,len(x)); ln=rng.geometric(1/L); idx+=list((np.arange(st,st+ln))%len(x))
        out.append(metr(x[np.array(idx[:n])],paymin,NM))
    return np.array(out)
def show(tag,x,paymin,hist=None):
    r=boot(x,paymin); h=''
    if hist is not None: m=metr(hist,paymin,NM); h=f' | history {m[0]:.2f}/mo gap {m[1]:.1f}mo net ${m[4]:,.0f}'
    print(f'{tag:34s} ${paymin:4.0f} | payouts/mo mean {r[:,0].mean():.2f} p10 {np.percentile(r[:,0],10):.2f} p90 {np.percentile(r[:,0],90):.2f} | longest gap med {np.median(r[:,1]):.1f} p90 {np.percentile(r[:,1],90):.1f} mo | mo>=1 {r[:,2].mean():.0%} mo>=2 {r[:,3].mean():.0%} | net med ${np.median(r[:,4]):,.0f}{h}',flush=True)
base=df.p.values; late=df[df.yr>=2022].p.values
print('(1) validation on the actual path (full sim earlier: $500 rule ~.5/mo, longest gap ~300d ~10mo; $2,000 rule max gap 197d ~6.5mo, $100.8k)')
for pm in (500,2000): m=metr(base,pm,NM); print(f'  history paymin {pm}: {m[0]:.2f}/mo, longest gap {m[1]:.1f} mo, mo>=1 {m[2]:.0%}, net ${m[4]:,.0f}')
print('\n(2) block bootstrap, all days vs 2022+ pool')
for pm in (500,2000): show('all days',base,pm); show('2022+ days',late,pm)
print('\n(3) variants (all-days pool; history in last column)')
V={'size x0.75':base*.75,'size x1.25':base*1.25,'size x1.5':base*1.5,
   'range-target':base*np.clip(ref/df.R20.values,.6,1.8),
   'event x1.5 (nfp/react, 2022+)':base*np.where(((df.nfp==True)|(df.react==True))&(df.yr>=2022),1.5,1.0)}
for pm in (500,2000):
    show('base',base,pm,base)
    for k,x in V.items(): show(k,x,pm,x)
