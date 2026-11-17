"""Report 248: (1) VECTOR AS THE DIRECTION FILTER ON THE LEVEL REACTIONS, by time frame, and (2) other options-flow techniques as direction/quality filters (Rey Nov 17). DECLARED before results. Candidates = report-246 stage-2 rejections (10,250; outcome 2R after 1 pt cost; cand246.pkl), and the absorbed subset (stage 3, 1,103).
(1) Vector regime of the trade's own direction on 5m / 15m / 30m / 1H / 4H / daily bars, read at the last CLOSED bar before the signal: WITH = regime sign equals the trade direction (a pullback-style bounce), AGAINST = fade of the vector. Also the number of the 6 time frames that agree with the trade (0-6): bucket 0-1 / 2-3 / 4-6. Cells: WITH / AGAINST per time frame for all and for absorbed; alignment buckets.
(2) Options flow at the prior close (QQQ), read in the trade direction: dealer delta (DEX) agrees, vanna flow agrees, charm agrees, put/call volume high (contrarian: more puts = support holds for longs), skew high, IV high, net gamma negative; for all and for absorbed.
Per era net pts per rejection (not sequenced, so n is large) and t; train 2019-21 | test 2018+2022-26; ~50 cells, bar ~3."""
import sys, warnings; warnings.filterwarnings('ignore'); src=open('/home/claude/fartpie/xs/report_232_vx_exits.py').read(); exec(src[:src.index("MODES=")])
dsrc=open('/home/claude/fartpie/xs/dte_greeks.py').read(); seg=dsrc[dsrc.index("U = '/mnt"):dsrc.index("DAYF =")].replace("f'{U}/xs/ext/gex/QQQ.csv'","'/home/claude/fartpie/xs/ext/gex/QQQ.csv'"); exec(seg)
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
C=pd.read_pickle(SP+'cand246.pkl')
def tstat(x):
    d=x.groupby('day').pts.sum(); return d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>5 else np.nan
out=[]
for i_,r in C.iterrows():
    e=int(r.e)
    if e>=n-2 or not (575<=m[e]<=930): out.append(np.nan); continue
    gp,mfe,j=trade(e,int(r.s),r.R,'fix',2.0); out.append(gp-1.0)
C['pts']=out; C=C.dropna(subset=['pts']).reset_index(drop=True); C['y']=C.day.dt.year
tr=(C.y>=2019)&(C.y<=2021); te=(C.y==2018)|(C.y>=2022)
# vector regimes on several time frames; signal bar j = e-1 (5m); regime read from the last bar CLOSED at or before the signal bar's open+5m => bars whose end <= t5[e]
TF={'5m':'5min','15m':'15min','30m':'30min','1H':'60min','4H':'240min','1D':'1D'}; RG={}
for nm,rule in TF.items():
    gg=b.set_index('dt').resample(rule,label='right',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    r_=run_state_machine(gg.assign(datetime=gg.index)); rg=np.asarray(r_[0]); en=gg.index.values
    k=np.searchsorted(en,t5[C.e.values.astype(int)],'right')-1; RG[nm]=np.where(k>=0,rg[np.clip(k,0,None)],0)*C.s.values
    print(nm,'regime share with trade',round(float((RG[nm]==1).mean()),2),'against',round(float((RG[nm]==-1).mean()),2),flush=True)
def cell(nm,mk):
    res=[]
    for era,m_ in (('train',tr),('test',te)):
        x=C[mk&m_]; res.append(f'{era}: n{len(x):5d} win{np.mean(x.pts>0)*100:3.0f}% net {x.pts.mean():+6.1f} t{tstat(x):+5.1f}' if len(x)>20 else f'{era}: n{len(x)}')
    print(f'{nm:34s}',' || '.join(res),flush=True)
print('\n=== (1) vector direction by time frame, 2R, all rejections / absorbed ===')
cell('ALL rejections',C.pts==C.pts); cell('ABSORBED',C.ab)
for nm in TF:
    w=pd.Series(RG[nm]==1,index=C.index); a=pd.Series(RG[nm]==-1,index=C.index)
    cell(f'{nm} vector WITH',w); cell(f'{nm} vector AGAINST',a); cell(f'{nm} vector WITH & absorbed',w&C.ab); cell(f'{nm} vector AGAINST & absorbed',a&C.ab)
cnt=sum((RG[nm]==1).astype(int) for nm in TF); C['agree']=cnt
for nm,mk in (('agree 0-1 TFs',C.agree<=1),('agree 2-3 TFs',(C.agree>=2)&(C.agree<=3)),('agree 4-6 TFs',C.agree>=4)): cell(nm,mk); cell(nm+' & absorbed',mk&C.ab)
print('\n=== (2) options flow at the prior close, in the trade direction ===')
rows=[]
for i_,r in C.iterrows():
    f=prior(pd.Timestamp(r.day)); s=int(r.s)
    if f is None: rows.append((False,)*8); continue
    ok=lambda x:bool(np.isfinite(x))
    rows.append((ok(f.dex) and np.sign(f.dex)==s, ok(f.vflow) and f.vflow==s, ok(f.net_charm) and np.sign(f.net_charm)==s, ok(f.pcvp) and f.pcvp>=.5, ok(f.skewp) and f.skewp>=.5, ok(f.ivp) and f.ivp>=.5, bool(r.neg), ok(f.pcvp) and f.pcvp>=.5 and s==1))
O=pd.DataFrame(rows,columns=['DEX','VANNA','CHARM','PCVHI','SKEWHI','IVHI','NEG','PCVHI_long'],index=C.index)
for nm in O.columns:
    cell(f'{nm}',O[nm]); cell(f'{nm} & absorbed',O[nm]&C.ab); cell(f'not {nm} & absorbed',(~O[nm])&C.ab)
C.to_pickle(SP+'cand248.pkl')
