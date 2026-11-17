"""Report 241: FAILED BREAKOUT OF THE PRIOR-DAY RANGE on 1H bars, with flow and gamma state (Rey Nov 17: 'try that'). DECLARED before results.
Prior-day high/low = previous RTH session (09:30-16:00). Signal: a 1H bar whose high is ABOVE the prior-day high but CLOSES back below it (a sweep that is rejected), or low below the prior-day low and close back above. Fade it: entry = next 5m open after that 1H bar closes, stop = the sweep bar's extreme (R = distance, >= 15 pts here since this is the large-stop regime), target 1R or 2R, flat 15:55, cost 1 pt, entries 10:00-15:30, one trade at a time, at most one per day per side.
TRAP = cumulative estimated delta (CLV-signed volume) of the 6 hourly-equivalent bars before the sweep is in the sweep direction (crowd positioned with the push, report 237/238 logic). Gamma: prior-close QQQ net GEX. IVHI: prior-close ATM IV >= trailing-252 median.
Cells (x 2 exits): all, TRAP, pos gamma, neg gamma, TRAP&pos, TRAP&neg, IVHI. 14 cells -> ~3 is the bar. Train 2019-21 | test 2018+2022-26; drought-day pts and corr with main as in 229-230."""
import sys; src=open('/home/claude/fartpie/xs/report_230_fill_models_gex.py').read(); exec(src[:src.index("for rule,tag in")])
dsrc=open('/home/claude/fartpie/xs/dte_greeks.py').read(); seg=dsrc[dsrc.index("U = '/mnt"):dsrc.index("DAYF =")].replace("f'{U}/xs/ext/gex/QQQ.csv'","'/home/claude/fartpie/xs/ext/gex/QQQ.csv'"); exec(seg)
dl=v*(2*c-h-l)/np.maximum(h-l,1e-9); cd=np.cumsum(dl)
rth=(m>=570)&(m<=955); DR=pd.DataFrame(dict(day=day[rth],hi=h[rth],lo=l[rth])).groupby('day').agg(hi=('hi','max'),lo=('lo','min')); PDH=DR.hi.shift(1); PDL=DR.lo.shift(1)
g=b.set_index('dt').resample('60min',label='right',closed='left').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna(); ends=g.index.values; GH,GL,GC=g.high.values,g.low.values,g.close.values
SW=[]; seen=set()
for i in range(1,len(g)):
    e=np.searchsorted(t5,ends[i])
    if e>=n or t5[e]!=ends[i] or e<90 or not (600<=m[e]<=930): continue
    d=pd.Timestamp(day[e]); ph,pl=PDH.get(d,np.nan),PDL.get(d,np.nan)
    if not (np.isfinite(ph) and np.isfinite(pl)): continue
    pre=cd[e-13]-cd[e-85]
    if GH[i]>ph and GC[i]<ph and (d,'U') not in seen:
        R=GH[i]-o[e]
        if R>=15: SW.append((e,-1,R,pre>0,ivhi:=bool(prior(d) is not None and np.isfinite(prior(d).ivp) and prior(d).ivp>=.5))); seen.add((d,'U'))
    elif GL[i]<pl and GC[i]>pl and (d,'D') not in seen:
        R=o[e]-GL[i]
        if R>=15: SW.append((e,1,R,pre<0,bool(prior(d) is not None and np.isfinite(prior(d).ivp) and prior(d).ivp>=.5))); seen.add((d,'D'))
print('sweep signals',len(SW),' median R',np.median([x[2] for x in SW]))
def sel(k,gm=None,trap=None,iv=None):
    out=[]
    for e,s,R,tr,ih in SW:
        a,_=GXD[day[e]]
        if not np.isfinite(a): continue
        if gm=='pos' and not a>=0: continue
        if gm=='neg' and not a<0: continue
        if trap and not tr: continue
        if iv and not ih: continue
        out.append((e,s,k*R,R))
    return out
for k in (1.0,2.0):
    print(f'\n=== exit target {k}R, stop = sweep extreme ===')
    for nm,kw in (('all',{}),('TRAP',dict(trap=1)),('pos gamma',dict(gm='pos')),('neg gamma',dict(gm='neg')),('TRAP & pos',dict(trap=1,gm='pos')),('TRAP & neg',dict(trap=1,gm='neg')),('IVHI',dict(iv=1))): report(f'{k}R {nm}',sim(sel(k,**kw)))
