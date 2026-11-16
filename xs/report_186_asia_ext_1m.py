"""Report 186 (1-MINUTE REDO of 185 and of the 184 Asia EXT fade): report 184/185 used 5m bars with a conservative stop-first rule on the FILL bar, which is an artifact (the bar low can precede the touch; diagnosis: 149 of 277 fills were counted as stopped on the fill bar). Everything below is simulated on 1m bars. Original header: ASIA range extension CONTINUATION. DECLARED BEFORE RESULTS (post-hoc from report 184: fading +/-1.5..2.5 x Asia width extensions lost in both eras, ASIA EXT 2.0: -4.5 (t-3.2) | -7.8 (t-5.7), which is a strict-split PASS with the NEGATIVE sign).
Mechanism: the 20:00-24:00 Asia box is narrow; a move of 2 box-widths beyond it by the NY session = an expanding/trending overnight (vol clustering), which keeps trending (momentum in range-expansion days) rather than reverting.
Trade: first touch during RTH (09:30-12:00 ET) of Asia high + k*W / low - k*W, k in (1.5, 2, 2.5), entry in the direction of the extension at max(open, level) (gap fills at the open, realistic for a stop-entry), stop = level -/+ sw*W (sw 0.5, 1.0), target = level +/- tw*W (tw 0.5, 1, 2); risk 6..60 pts else skip; stop first in a bar; EOD exit; cost 0.5; one trade/day/config. Split by whether the open was already beyond the level (GAPPED) or not (CROSSED).
Primary pre-declared config: k=2, sw=1, tw=1. 18 configs total; null by sign-shuffle (200 reps) as in report 184. Daily correlation with system daily points."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
b=pd.read_csv('/home/claude/nqdrive/NQ_1m_ETH.csv',parse_dates=['timestamp']); b['dt']=b.timestamp-pd.Timedelta(hours=6); b=b.sort_values('dt').drop_duplicates('dt').reset_index(drop=True); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute
asia=b[b.m>=1200].assign(d=lambda x:(x.dt+pd.Timedelta(days=1)).dt.normalize()).groupby('d').agg(ah=('high','max'),al=('low','min'),n=('high','size'))
rt=b[(b.m>=570)&(b.m<=955)]; DAYS={d:g for d,g in rt.groupby('d') if len(g)>=350}
COST=.5; WEND=720; rows=[]
for d,g in DAYS.items():
    if d<pd.Timestamp('2018-01-01') or d>pd.Timestamp('2026-01-30') or d not in asia.index or asia.n[d]<120: continue
    H,L=asia.ah[d],asia.al[d]; W=H-L
    if W<3: continue
    o,h,l,c=g.open.values,g.high.values,g.low.values,g.close.values; n=len(h)
    for k in (1.5,2.,2.5):
        for sg,lvl in ((1,H+k*W),(-1,L-k*W)):
            t=None
            for u in range(n):
                if u+570>WEND: break
                if (h[u]>=lvl if sg>0 else l[u]<=lvl): t=u; break
            if t is None: continue
            gap=(o[t]>=lvl) if sg>0 else (o[t]<=lvl); ent=max(o[t],lvl) if sg>0 else min(o[t],lvl)
            for sw in (.5,1.):
                for tw in (.5,1.,2.):
                    stop=lvl-sg*sw*W; tgt=lvl+sg*tw*W; risk=sg*(ent-stop)
                    if risk<6 or risk>60 or sg*(tgt-ent)<=0: continue
                    res=None; worst=0.
                    # fill bar: stop only (conservative)
                    if (l[t]<=stop if sg>0 else h[t]>=stop): res=-risk-COST
                    else:
                        for v in range(t+1,n):
                            if (l[v]<=stop if sg>0 else h[v]>=stop): res=-risk-COST; break
                            if (h[v]>=tgt if sg>0 else l[v]<=tgt): res=sg*(tgt-ent)-COST; break
                        if res is None: res=sg*(c[-1]-ent)-COST
                    rows.append((d,k,sw,tw,sg,gap,res))
            # FADE of the same extension (report 184 config: stop 0.5W beyond, target back 1.0W)
            ent=max(o[t],lvl) if sg>0 else min(o[t],lvl); stop=lvl+sg*.5*W; tgt=lvl-sg*1.0*W; risk=sg*(stop-ent)
            if 6<=risk<=60:
                res=None
                if (h[t]>=stop if sg>0 else l[t]<=stop): res=-risk-COST
                else:
                    for v in range(t+1,n):
                        if (h[v]>=stop if sg>0 else l[v]<=stop): res=-risk-COST; break
                        if (l[v]<=tgt if sg>0 else h[v]>=tgt): res=sg*(ent-tgt)-COST; break
                    if res is None: res=sg*(ent-c[-1])-COST
                rows.append((d,k,-1.,-1.,sg,gap,res))
X=pd.DataFrame(rows,columns=['d','k','sw','tw','sg','gap','pts']).sort_values('d'); 
X['t']=0; X=X.drop_duplicates(['d','k','sw','tw'])      # earliest-listed (up before down on same day: ordering by loop), documented
def st(s): n=len(s); return n,s.mean(),(s.mean()/(s.std()/np.sqrt(n)) if n>2 and s.std()>0 else 0)
def pas(a,b_): return np.sign(a[1])==np.sign(b_[1]) and abs(a[2])>=2 and abs(b_[2])>=2 and a[0]>=30 and b_[0]>=30
print('k sw tw | 2018-21 n mean t | 2022+ n mean t'); npass=0; keys=[]
for (k,sw,tw),Y in X.groupby(['k','sw','tw']):
    a=st(Y.pts[Y.d<'2022-01-01']); b_=st(Y.pts[Y.d>='2022-01-01']); ok=pas(a,b_); npass+=ok; keys.append((k,sw,tw))
    print(f'{k} {sw} {tw} | {a[0]:4d} {a[1]:+6.2f} t{a[2]:+5.1f} | {b_[0]:4d} {b_[1]:+6.2f} t{b_[2]:+5.1f}{"  <== PASS" if ok else ""}')
rng=np.random.default_rng(2); null=[]
for _ in range(200):
    c_=0
    for (k,sw,tw),Y in X.groupby(['k','sw','tw']):
        p=Y.pts.values*rng.choice([-1,1],len(Y)); a=st(pd.Series(p[(Y.d<'2022-01-01').values])); b_=st(pd.Series(p[(Y.d>='2022-01-01').values])); c_+=pas(a,b_)
    null.append(c_)
print(f'\nreal PASS {npass} of {len(keys)}; null mean {np.mean(null):.1f}, 95th {np.percentile(null,95):.0f}')
P=X[(X.k==2.)&(X.sw==1.)&(X.tw==1.)]; PF=X[(X.k==2.)&(X.sw==-1.)]
print('\nPRIMARY k2 sw1 tw1 by GAPPED/CROSSED:')
for gp,nm in ((True,'open already beyond level'),(False,'crossed in RTH')):
    Y=P[P.gap==gp]; a=st(Y.pts[Y.d<'2022-01-01']); b_=st(Y.pts[Y.d>='2022-01-01']); print(f'  {nm:26s} A {a[1]:+6.2f} t{a[2]:+.1f} n{a[0]} | B {b_[1]:+6.2f} t{b_[2]:+.1f} n{b_[0]}')
P.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/asia_ext_cont185.pkl')
import report_179_fractal as f179
TR=f179.prep(); SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
I=pd.read_pickle(SP+'ext/uni_ifvg.pkl'); I=I[I.model=='IFVG']; S1=pd.read_pickle(SP+'s1_trades162.pkl'); S1=S1[S1.reg>0]
parts=[t[['day','pts']] for t in TR.values()]+[pd.DataFrame({'day':pd.to_datetime(I.d).dt.normalize(),'pts':I.pts.values}),pd.DataFrame({'day':pd.to_datetime(S1.day).dt.normalize(),'pts':S1.pts.values*0.4})]
sysd=pd.concat(parts).groupby('day').pts.sum(); mine=P.groupby('d').pts.sum()
for lab,sl in (('2018-21',slice('2018','2021')),('2022+',slice('2022',None))):
    ix=sysd.loc[sl].index.union(mine.loc[sl].index); a=sysd.reindex(ix).fillna(0); b_=mine.reindex(ix).fillna(0); print(f'  daily corr with system {lab}: {np.corrcoef(a,b_)[0,1]:+.3f} (system days {int((a!=0).sum())}, strat days {int((b_!=0).sum())}, overlap {int(((a!=0)&(b_!=0)).sum())})')
