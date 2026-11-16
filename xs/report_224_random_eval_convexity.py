"""Report 224: pure randomness on the Lucid 50K eval math (no strategy). Where is the payout convexity peak? DECLARED before results.
Rules exactly as in the fleet harness (fleet_levers.ACCX / report_194 EV, FU): eval starts at 0 with a floor at -$2,000 that trails the END-OF-DAY peak until it locks at +$100; breach = balance + intraday adverse move <= floor; pass = balance >= $3,000 with the best single day <= 50% of the balance; fee $88. Funded: same floor rule until the first payout; a payout needs 5 days of >= $150 profit, balance > 0; amount = min(50% of balance, $2,000), minimum $500, trader keeps 90%; after a payout the floor is fixed at +$100; 5 payouts = graduated; contract cap 40 MNQ in the eval, 20 / 30 / 40 MNQ funded by profit (< $1,000 / < $2,000 / above).
Randomness: one trade a day on real NQ 5-minute bars 2018-2026, direction by coin flip, entry 09:35 open, exit at a stop S or target T (stop wins a bar that touches both) or the 15:55 close; cost 1.0 point; $2 per point per MNQ; fixed size n MNQ. Zero edge by construction (minus cost). Start days drawn at random. Grid: geometry (S, T) x size n. Eval and funded are simulated separately because the expected value separates: EV per eval = P(pass | eval config) x E[payout $ | funded config] - $88. 3,000 random paths per cell."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
rng=np.random.default_rng(11); b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['d']=b.dt.dt.normalize(); m=b.dt.dt.hour*60+b.dt.dt.minute; b=b[(m>=575)&(m<=955)&(b.d>='2018-01-01')]
GEO=[(25,25),(50,50),(50,100),(50,150),(100,50),(150,50),(100,25),(None,None)]; gname=lambda g:'hold to close' if g[0] is None else f'stop {g[0]} / target {g[1]}'
days=[]; O={g:[] for g in GEO}
for d,x in b.groupby('d'):
    if len(x)<60: continue
    o=x.open.values[0]; h=x.high.values; l=x.low.values; c=x.close.values[-1]; days.append(d)
    for g in GEO:
        S,T=g; res=[]
        for sgn in (1,-1):
            adv=(o-l) if sgn==1 else (h-o); fav=(h-o) if sgn==1 else (o-l)      # adverse / favourable excursion per bar
            ks=np.argmax(adv>=S) if S is not None and (adv>=S).any() else 10**6; kt=np.argmax(fav>=T) if T is not None and (fav>=T).any() else 10**6
            if ks<=kt and ks<10**6: pts=-S; mae=S
            elif kt<10**6: pts=T; mae=adv[:kt+1].max()
            else: pts=sgn*(c-o); mae=adv.max()
            res.append((pts-1.0,-max(mae,0)-1.0))
        O[g].append(res)
N=len(days); O={g:np.array(v) for g,v in O.items()}; print('NQ days',N,days[0].date(),days[-1].date())
for g in GEO: p=O[g][:,:,0].mean(); w=(O[g][:,:,0]>0).mean(); print(f'  {gname(g):22s} mean per trade {p:+.2f} pts (cost included), win {w:.0%}, sd {O[g][:,:,0].std():.0f} pts')
def ev(g,n,paths=3000):
    A=O[g]; res=np.zeros(3); dd=[]
    for _ in range(paths):
        k=rng.integers(0,N-400); bal=0.; floor=-2000.; peak=0.; best=0.; out=0
        for t in range(400):
            s=rng.integers(0,2); pts,mae=A[k+t,s]; pnl=pts*2*n
            if bal+mae*2*n<=floor: out=-1; break
            bal+=pnl; best=max(best,pnl)
            if bal>=3000 and best<=.5*bal: out=1; break
            if floor<100: peak=max(peak,bal); floor=min(max(floor,peak-2000),100.)
        res[out+1]+=1; dd.append(t+1)
    return res[2]/paths, res[0]/paths, np.mean(dd)
def fu(g,n,paths=3000):
    A=O[g]; tot=[]; npay=[]; first=[]
    for _ in range(paths):
        k=rng.integers(0,N-700); bal=0.; floor=-2000.; peak=0.; fixed=False; wins=0; since=0.; pays=[]
        for t in range(700):
            cap=20 if bal<1000 else 30 if bal<2000 else 40; nn=min(n,cap); s=rng.integers(0,2); pts,mae=A[k+t,s]; pnl=pts*2*nn
            if bal+mae*2*nn<=floor: break
            bal+=pnl; since+=pnl; wins+=pnl>=150
            if not fixed: peak=max(peak,bal); floor=min(max(floor,peak-2000),100.)
            if wins>=5 and bal>0 and (not pays or since>0):
                amt=min(.5*bal,2000.)
                if amt>=500:
                    pays.append(.9*amt); bal-=amt; floor=100.; fixed=True; wins=0; since=0.
                    if len(pays)==1: first.append(t+1)
                    if len(pays)>=5: break
        tot.append(sum(pays)); npay.append(len(pays))
    return np.mean(tot), np.mean(np.array(npay)>0), np.mean(npay), (np.mean(first) if first else np.nan)
SZ=[1,2,3,5,8,10,15,20,30,40]; EVt={}; FUt={}
print('\nEVAL: pass rate by geometry x size (MNQ) [mean days to resolve]')
print(' '*24+''.join(f'{n:>10d}' for n in SZ))
for g in GEO:
    row=[]
    for n in SZ: p,f,dy=ev(g,n); EVt[(g,n)]=(p,f,dy); row.append(f'{p*100:4.0f}%[{dy:3.0f}]')
    print(f'{gname(g):24s}'+''.join(f'{r:>10s}' for r in row),flush=True)
print('\nFUNDED: expected total payout $ per funded account (share paid at least once) by geometry x size')
print(' '*24+''.join(f'{n:>12d}' for n in SZ))
for g in GEO:
    row=[]
    for n in SZ: e,p1,npy,fd=fu(g,n); FUt[(g,n)]=(e,p1,npy,fd); row.append(f'${e:5.0f}({p1*100:2.0f}%)')
    print(f'{gname(g):24s}'+''.join(f'{r:>12s}' for r in row),flush=True)
print('\nNET per eval = pass x expected payout - $88, best funded size for each eval cell; top 12 cells')
best=[]
for (g,n),(p,f,dy) in EVt.items():
    for (g2,n2),(e,p1,npy,fd) in FUt.items(): best.append((p*e-88,gname(g),n,gname(g2),n2,p,e,p1,dy,fd))
best.sort(reverse=True)
for r in best[:12]: print(f'  net ${r[0]:6.0f} | eval: {r[1]} x {r[2]} MNQ (pass {r[5]*100:.0f}%, {r[8]:.0f} days) | funded: {r[3]} x {r[4]} MNQ (E[payout] ${r[6]:.0f}, paid once {r[7]*100:.0f}%, first payout day {r[9]:.0f})')
import pickle; pickle.dump((EVt,FUt),open('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/random_eval224.pkl','wb'))
