"""Report 217: placebo for the 1H vector on sub-$5 names (the timeframe Rey trades; reports 199-216 used DAILY flips). DECLARED before results.
Data = report 210 (Yahoo 1H, 2023-12..2026-11, 299 survivor names, flips with entry >= 2024-02 and entry price < $5). Forward return from the entry bar's open over 1 / 5 / 20 trading days (7 / 35 / 140 RTH 1H bars), vs 5 random 1H bars of the SAME ticker within +-140 bars (>= 7 bars away, open < $5). Per-flip difference (flip minus mean of its randoms), clustered by entry day; halves H1 = 2024-02..2025-06, H2 = 2025-07..2026-11. Same for daily flips on the same names (horizons 1 / 5 / 20 / 40 days vs random days +-60 trading days)."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
E='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; SP=E[:-4]; rng=np.random.default_rng(5)
P=pickle.load(open(E+'lowpx_1h_3y.pkl','rb')); D=pd.read_pickle(E+'wide_daily_ohlcv.pkl'); T=pd.read_pickle(SP+'flips_1h_vs_d210.pkl'); MID=pd.Timestamp('2025-07-01'); rows=[]
for tk,g in T.groupby('tk'):
    x=P[tk].copy(); x.columns=['open','high','low','close','volume']; x.index=x.index.tz_convert('America/New_York').tz_localize(None); x=x[~x.index.duplicated()].dropna(); m=x.index.hour*60+x.index.minute; x=x[(m>=570)&(m<960)]; o=x.open.values; ix=x.index.values; n=len(o)
    d=D[tk]; d=d[~d.index.duplicated()]; od=d.open.values; idd=d.index.values; nd=len(od)
    for tf,e in zip(g.tf,g.entry):
        if tf=='1H':
            k=np.searchsorted(ix,np.datetime64(e))
            if k>=n or k+140>=n or k<140: continue
            c=np.arange(k-140,min(k+140,n-141)); c=c[(abs(c-k)>=7)&(o[c]<5)]
            if len(c)<5: continue
            rk=rng.choice(c,5,replace=False); rows.append(dict(tf=tf,e=e,**{f'f{H}':o[k+H]/o[k]-1 for H in (7,35,140)},**{f'r{H}':np.mean(o[rk+H]/o[rk]-1) for H in (7,35,140)}))
        else:
            k=np.searchsorted(idd,np.datetime64(e))
            if k>=nd or k+40>=nd or k<60: continue
            c=np.arange(k-60,min(k+60,nd-41)); c=c[(abs(c-k)>=3)&(od[c]<5)]
            if len(c)<5: continue
            rk=rng.choice(c,5,replace=False); rows.append(dict(tf=tf,e=e,**{f'f{H}':od[k+H]/od[k]-1 for H in (1,5,20,40)},**{f'r{H}':np.mean(od[rk+H]/od[rk]-1) for H in (1,5,20,40)}))
R=pd.DataFrame(rows); R['half']=np.where(R.e<MID,'H1','H2')
for tf,Hs in (('1H',((7,'1d'),(35,'5d'),(140,'20d'))),('D',((1,'1d'),(5,'5d'),(20,'20d'),(40,'40d')))):
    for hf in ('H1','H2'):
        s=R[(R.tf==tf)&(R.half==hf)]; out=[]
        for H,lab in Hs:
            dd=(s[f'f{H}']-s[f'r{H}']); cl=dd.groupby(s.e.dt.normalize()).mean(); out.append(f'{lab}: flip {s[f"f{H}"].mean()*100:5.2f}% random {s[f"r{H}"].mean()*100:5.2f}% diff {dd.mean()*100:+5.2f}pp t{cl.mean()/(cl.std(ddof=1)/np.sqrt(len(cl))):+5.1f} (median diff {dd.median()*100:+5.2f}pp)')
        print(f'  {tf:2s} flips {hf} n{len(s):5d} | '+' | '.join(out))
