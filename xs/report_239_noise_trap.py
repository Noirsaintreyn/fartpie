"""Report 239: NOISE-BAND TRAPPED BREAKOUT / CONTINUATION with flow and gamma state (Rey Nov 17: leverage absorption + gamma + IV on a different model for other days, not 'just fade the vector', use the noise-range idea). DECLARED before results.
Noise band (as in orb_noise_vector / M1): sigma(t) = mean over the prior 14 sessions of |close(t)/open - 1| at that 5m slot; UB = max(open, prior close) x (1 + sigma), LB = min(open, prior close) x (1 - sigma). Signals after 10:00, entries next 5m open, flat 15:55, 1 pt cost, one trade at a time, FIX 1R.
MODEL REJ (trapped breakout): a 5m bar whose high pokes through UB (or low through LB) but CLOSES back inside = a failed breakout. Fade it: entry next open, stop = the poke bar's extreme, target 1R (R >= 3 pts). Mechanism: the band is where momentum traders put stops/entries; if the push is absorbed the crowd that bought it is trapped and price returns, and dealers short gamma would amplify a real break while long gamma dampens it. TRAP = the cumulative estimated delta of the 6 bars before the poke is in the poke direction (crowd positioned with the move, report 237 ABSORB logic).
MODEL CONT (absorbed-then-break): first 5m CLOSE beyond a band (not a poke) -> go with it; entry next open, stop = the band, target 1R. ABS = the 6-bar cumulative delta before the break is AGAINST it (crowd on the wrong side, report 237 ABSORB logic).
Cells (declared): REJ all, pos gamma, neg gamma, TRAP, pos&TRAP, IVHI, pos&TRAP&IVHI; CONT all, neg gamma, ABS, neg&ABS, neg&ABS&IVHI. NEG = prior-close QQQ net GEX < 0, IVHI = prior-close ATM IV >= trailing-252 median. 12 cells -> ~3 is the bar. Train 2019-21 | test 2018+2022-26; drought-day pts and correlation with the main algo as in 229-230."""
import sys; src=open('/home/claude/fartpie/xs/report_230_fill_models_gex.py').read(); exec(src[:src.index("for rule,tag in")])
dsrc=open('/home/claude/fartpie/xs/dte_greeks.py').read(); seg=dsrc[dsrc.index("U = '/mnt"):dsrc.index("DAYF =")].replace("f'{U}/xs/ext/gex/QQQ.csv'","'/home/claude/fartpie/xs/ext/gex/QQQ.csv'"); exec(seg)
rng_=np.maximum(h-l,1e-9); delta=v*(2*c-h-l)/rng_; cd=np.cumsum(delta)
# daily open / prior close, sigma(t)
slots=np.arange(570,960,5); S=pd.DataFrame({'day':day,'m':m,'o':o,'c':c}); S=S[(S.m>=570)&(S.m<=955)]
op=S[S.m==570].set_index('day').o; pcl=S[S.m==955].set_index('day').c.shift(1)
rel=S.assign(r=lambda x:(x.c/x.day.map(op)-1).abs()).pivot_table(index='day',columns='m',values='r')
sig=rel.rolling(14).mean().shift(1)
dayidx={}
for i in range(n): dayidx.setdefault(day[i],[]).append(i)
def ivhi(d):
    f=prior(d); return bool(f is not None and np.isfinite(f.ivp) and f.ivp>=.5)
REJ=[];CONT=[]
for d,ix in dayidx.items():
    td=pd.Timestamp(d)
    if td not in sig.index or td not in op.index: continue
    o0=op[td]; pc0=pcl.get(td,np.nan); pc0=o0 if not np.isfinite(pc0) else pc0
    got_r=False; got_c=False
    for i in ix:
        if m[i]<600 or m[i]>925 or i+1>=n or i<8: continue
        sg=sig.loc[td].get(m[i],np.nan)
        if not np.isfinite(sg): continue
        UB=max(o0,pc0)*(1+sg); LB=min(o0,pc0)*(1-sg); pre=cd[i-1]-cd[i-7]
        e=i+1
        if not got_r:
            if h[i]>=UB and c[i]<UB:
                R=h[i]-o[e]
                if R>=3: REJ.append((e,-1,R,R,pre>0,ivhi(td))); got_r=True
            elif l[i]<=LB and c[i]>LB:
                R=o[e]-l[i]
                if R>=3: REJ.append((e,1,R,R,pre<0,ivhi(td))); got_r=True
        if not got_c:
            if c[i]>UB:
                R=o[e]-UB
                if R>=3: CONT.append((e,1,R,R,pre<0,ivhi(td))); got_c=True
            elif c[i]<LB:
                R=LB-o[e]
                if R>=3: CONT.append((e,-1,R,R,pre>0,ivhi(td))); got_c=True
def sel(L,gm=None,flag=None,iv=None):
    out=[]
    for x in L:
        a,_=GXD[day[x[0]]]
        if not np.isfinite(a): continue
        if gm=='neg' and not a<0: continue
        if gm=='pos' and not a>=0: continue
        if flag is not None and not x[4]: continue
        if iv is not None and not x[5]: continue
        out.append(x[:4])
    return out
print(f'REJ signals {len(REJ)}, CONT signals {len(CONT)}')
print('=== REJ: trapped breakout fade ===')
for nm,kw in (('REJ all',{}),('REJ pos gamma',dict(gm='pos')),('REJ neg gamma',dict(gm='neg')),('REJ TRAP',dict(flag=1)),('REJ pos & TRAP',dict(gm='pos',flag=1)),('REJ IVHI',dict(iv=1)),('REJ pos&TRAP&IVHI',dict(gm='pos',flag=1,iv=1))): report(nm,sim(sel(REJ,**kw)))
print('\n=== CONT: break of the noise band ===')
for nm,kw in (('CONT all',{}),('CONT neg gamma',dict(gm='neg')),('CONT ABS',dict(flag=1)),('CONT neg & ABS',dict(gm='neg',flag=1)),('CONT neg&ABS&IVHI',dict(gm='neg',flag=1,iv=1))): report(nm,sim(sel(CONT,**kw)))
