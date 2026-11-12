"""BUY-THE-FEAR (lead from report_mr_lit): long NQ at the 09:30 open, exit 15:55, when prior-close VIX spiked >8% in a day OR VIX/VIX3M > 1 (backwardation). Checks: both eras, per year, idle vs core days, day-stop, gap-down confluence, overlap."""
_s=open('xs/report_mr_lit.py').read(); exec(_s[:_s.index("print('=== 1. FADE")])
import yfinance as yf
v=yf.download(['^VIX','^VIX3M'],start='2017-06-01',progress=False,auto_adjust=False)['Close']; v.index=pd.to_datetime(v.index).normalize()
V=pd.DataFrame({'vix':v['^VIX'],'v3m':v['^VIX3M']}); V['term']=V.vix/V.v3m; V['chg']=V.vix.pct_change(); V=V[['vix','term','chg']].shift(1).reindex(F.day,method='ffill'); V.index=F.index; F=F.join(V)
idle=~pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/idle_state.pkl')['trade'].astype(bool); F['idle']=F.day.map(idle)
# day-stop variant: path from 5m bars
def long_path(d,stop):
    g=G[d]; e=g.open.iloc[0]
    for lo,c in zip(g.low.values,g.close.values):
        if lo<=e-stop: return -stop-.5
    return g.close.iloc[-1]-e-.5
def ln(x,col='pts'): return f'n {len(x):4d}' if len(x)<12 else f'n {len(x):4d} avg {x[col].mean():+6.1f} win {(x[col]>0).mean():.0%} t {x[col].mean()/(x[col].std()/len(x)**.5):+.1f} sum {x[col].sum():+.0f}'
F['pts']=F.oc-.5
S={'spike>8%':F.chg>.08,'backwd>1':F.term>1.0,'spike OR backwd':(F.chg>.08)|(F.term>1.0),'spike AND backwd':(F.chg>.08)&(F.term>1.0),'spike, gap-down':(F.chg>.08)&(F.gap<0),'baseline all days':F.chg==F.chg}
for k,m in S.items():
    z=F[m]; print(f'{k:18s}: '+' || '.join(f'{e} {ln(z[z.era==e])}' for e in ('18-21','22-26'))+' | idle days: '+ln(z[z.idle==True])+' | core days: '+ln(z[z.idle==False]))
z=F[S['spike OR backwd']].copy(); print('--- by year (spike OR backwardation):')
for y,x in z.groupby('yr'): print(f'   {y}: {ln(x)}')
for stop in (60,100,150):
    z['ps']=[long_path(d,stop) for d in z.day]; print(f'--- with {stop}-pt stop: '+' || '.join(f'{e} {ln(z[z.era==e],"ps")}' for e in ('18-21','22-26')))
print('trades per year (spike OR backwd):',round(S['spike OR backwd'].sum()/ (F.yr.nunique()),1))
