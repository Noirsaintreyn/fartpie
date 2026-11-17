"""Report 236: robustness of the one lead from 235 (reversion to multi-day VWAP when dealers are net short gamma). Rey Nov 17: run it. Spec for the lead: neg gamma (prior-close QQQ net GEX < 0), fade first 5m close after 10:00 with |close - FV| >= K x 14d range, target FV, stop same distance, flat 15:55, cost 1.
No fresh data is left (2018-2026 all seen), so the test is the PARAMETER SURFACE and the STATE BREAKDOWN: (1) FV window 3 / 5 / 10 prior days x K .5 / .75 / 1.0 on neg-gamma days (a real effect should be a smooth surface, not one lucky cell); (2) the ER state split incl. 'mid' (235 showed chop and trend positive, all-ER negative in test, so mid must be strongly negative: is that a story or noise?); (3) gamma defined as bottom-third rank instead of sign; (4) year by year for the chosen cell. Reported: n, net pts, t, per era, drought-day pts, corr with main."""
import sys; src=open('/home/claude/fartpie/xs/report_235_state_models.py').read(); exec(src[:src.index("print('\\n=== MODEL A")])
dayidx={}
for i in range(n): dayidx.setdefault(day[i],[]).append(i)
def fvsig(W,K):
    fvW=(dd.pv.rolling(W).sum()/dd.vv.rolling(W).sum()).shift(1); sigs=[]
    for d,ix in dayidx.items():
        td=pd.Timestamp(d)
        if td not in dd.index or not np.isfinite(fvW.get(td,np.nan)) or not np.isfinite(dd.loc[td,'atr14']): continue
        fv=fvW[td]; at=dd.loc[td,'atr14']
        for i in ix:
            if m[i]<600 or m[i]>925 or i+1>=n: continue
            z=(c[i]-fv)/at
            if abs(z)>=K:
                e=i+1; dist=abs(o[e]-fv)
                if dist>=3: sigs.append((e,-1 if z>0 else 1,dist,dist))
                break
    return sigs
print('=== (1) surface: neg gamma, all ER ===')
S={}
for W in (3,5,10):
    for K in (.5,.75,1.0):
        S[(W,K)]=fvsig(W,K); report(f'FV{W}d K{K} neg gam all-ER',sim(stfilt(S[(W,K)],'neg','all')))
print('\n=== (2) ER state split, FV5 K.75 and K.5, neg gamma ===')
for K in (.5,.75):
    for ch in ('chop','mid','trend'): report(f'FV5 K{K} neg {ch}',sim(stfilt(S[(5,K)],'neg',ch)))
print('\n=== (3) gamma as bottom-third rank (low3) instead of sign, FV5 ===')
def lowf(sigs):
    out=[]
    for x in sigs:
        a,r=GXD[day[x[0]]]
        if np.isfinite(r) and r<=1/3: out.append(x)
    return out
for K in (.5,.75,1.0): report(f'FV5 K{K} gamma low3',sim(lowf(S[(5,K)])))
print('\n=== (4) year by year, FV5 K.75 neg gamma, and pos for contrast ===')
for gm in ('neg','pos'):
    R=sim(stfilt(S[(5,.75)],gm,'all')); R['y']=pd.to_datetime(R.day).dt.year
    print(gm,' '.join(f'{y}:{g.pts.mean():+.0f}({len(g)})' for y,g in R.groupby('y')))
R=sim(stfilt(S[(5,.75)],'neg','all')); R.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/fv5_neg_trades.pkl')
