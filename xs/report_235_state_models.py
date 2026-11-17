"""Report 235: STATE-CONDITIONED models (Rey Nov 17: 'fair value mean reversion... model the market in a state or regime and analyze that individual'). DECLARED before results; state is known BEFORE the trade (prior close / prior days only), each state is judged on its own.
STATE variables: GAMMA (prior-close QQQ net GEX < 0 'neg' / >= 0 'pos'); CHOP (Kaufman efficiency ratio of the prior 10 daily closes, |net| / sum of |daily changes|; 'chop' = bottom third, 'trend' = top third, cut points from 2019-21 only).
MODEL A fair-value reversion: fair value FV = (FV1) prior RTH day's VWAP or (FV5) volume-weighted VWAP of the prior 5 RTH days; deviation z = (5m close - FV) / mean RTH range of the prior 14 days; after 10:00, first 5m close with |z| >= K (K = .5, .75) -> fade toward FV: enter next bar open, target = distance to FV, stop = the same distance on the other side (1:1), flat 15:55, cost 1 pt, one trade at a time.
MODEL B intraday momentum into the close (the documented late-day momentum / dealer-hedging effect): at 15:30 take the direction of the day so far (15:30 open - 09:30 open), hold to 15:55 close, protective stop 40 pts.
Cells = model x state. PASS: same sign both eras, n >= 30 each, |t| >= 2 test; with this many cells read t ~3 as the bar. Eras: train 2019-21, test 2018 + 2022-26. FILL value: per-day pts on the main algo's drought days and correlation with the main algo."""
import sys; src=open('/home/claude/fartpie/xs/report_230_fill_models_gex.py').read(); exec(src[:src.index("for rule,tag in")])
rth=(m>=570)&(m<=955); tp_=(h+l+c)/3
D=pd.DataFrame(dict(day=day[rth],pv=(tp_*v)[rth],vv=v[rth],hi=h[rth],lo=l[rth],cl=c[rth]))
dd=D.groupby('day').agg(pv=('pv','sum'),vv=('vv','sum'),hi=('hi','max'),lo=('lo','min'),cl=('cl','last')); dd['vwap']=dd.pv/dd.vv; dd['rng']=dd.hi-dd.lo
dd['atr14']=dd.rng.rolling(14).mean().shift(1); dd['fv1']=dd.vwap.shift(1)
dd['fv5']=(dd.pv.rolling(5).sum()/dd.vv.rolling(5).sum()).shift(1)
net=(dd.cl-dd.cl.shift(10)).abs(); path=dd.cl.diff().abs().rolling(10).sum(); dd['er']=(net/path).shift(1)
tr_=dd[(dd.index>='2019-01-01')&(dd.index<'2022-01-01')].er; q1,q2=tr_.quantile(1/3),tr_.quantile(2/3); print(f'ER cuts (train): chop < {q1:.2f}, trend > {q2:.2f}')
ST={d:('chop' if r.er<q1 else 'trend' if r.er>q2 else 'mid') for d,r in dd.iterrows() if np.isfinite(r.er)}
def stfilt(sigs,gm,ch):
    out=[]
    for x in sigs:
        d=day[x[0]]; a,_=GXD[d]
        if not np.isfinite(a) or pd.Timestamp(d) not in ST: continue
        if gm!='all' and ((gm=='neg')!=(a<0)): continue
        if ch!='all' and ST[pd.Timestamp(d)]!=ch: continue
        out.append(x)
    return out
dayidx={}
for i in range(n): dayidx.setdefault(day[i],[]).append(i)
print('\n=== MODEL A: fair-value reversion (1:1 to FV) ===',flush=True)
for fvn in ('fv1','fv5'):
    for K in (.5,.75):
        sigs=[]
        for d,ix in dayidx.items():
            td=pd.Timestamp(d)
            if td not in dd.index or not np.isfinite(dd.loc[td,fvn]) or not np.isfinite(dd.loc[td,'atr14']): continue
            fv=dd.loc[td,fvn]; at=dd.loc[td,'atr14']
            for i in ix:
                if m[i]<600 or m[i]>925 or i+1>=n: continue
                z=(c[i]-fv)/at
                if abs(z)>=K:
                    e=i+1; dist=abs(o[e]-fv)
                    if dist>=3: sigs.append((e,-1 if z>0 else 1,dist,dist))
                    break
        for gm in ('all','pos','neg'):
            for ch in ('all','chop','trend'):
                if gm=='all' and ch=='all' or (gm!='all' and ch!='all') or True:
                    report(f'A {fvn} K{K} gam={gm} {ch}',sim(stfilt(sigs,gm,ch)))
print('\n=== MODEL B: late-day momentum (15:30 -> 15:55, stop 40) ===',flush=True)
sigs=[]
for d,ix in dayidx.items():
    a=[i for i in ix if m[i]==570]; z=[i for i in ix if m[i]==930]
    if not a or not z: continue
    r=o[z[0]]-o[a[0]]
    if r!=0: sigs.append((z[0],1 if r>0 else -1,999.0,40.0))
for gm in ('all','pos','neg'):
    for ch in ('all','chop','trend'): report(f'B late mom gam={gm} {ch}',sim(stfilt(sigs,gm,ch)))
