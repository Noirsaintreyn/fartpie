"""Nov 13 2026: MACRO-CONDITIONED VALUE Z-SCORE ('value the company, set the range from the state of the economy and volatility; trade the extension'). DECLARED BEFORE RESULTS.
Anchor: annual sales per share (point-in-time: latest 10-K with filed <= date, revenue / diluted weighted shares of the same period). lm = log(price / sales_per_share).
Macro state m_t: log VIX, 10y yield (^TNX), curve (^TNX - ^IRX), credit risk appetite (60d return HYG - LQD).
Fair multiple: lm_it = a_i + b'.m_t + e_it, pooled with ticker fixed effects. FIT on 2018-01..2021-12 monthly obs only (fixed betas and a_i), z_it = e_it / sd_i(train). Evaluated in-sample (2018-21, flagged) and OUT OF SAMPLE 2022+. Robustness: expanding-window refit each month (>=24 obs).
Controls: z0 = lm - a_i (own-history z, NO macro), PS = raw cross-sectional log P/S. If macro conditioning adds nothing over z0, the idea is not macro-adding.
Outcome: next 21 / 63 trading-day return minus the universe mean that month. Monthly cross-section (first trading day). Report: Spearman IC (cheap = low z should win -> IC of -z), D1(cheapest decile)-D10 spread, by era. Survivorship: Yahoo prices only for tickers still trading (flagged)."""
import numpy as np, pandas as pd, json, warnings, pickle, yfinance as yf; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
F=pd.read_parquet('/home/claude/fartpie/xs/data/sec_facts.parquet'); M=json.load(open('/home/claude/fartpie/sp500_ticker_cik_map.json'))
for c in ('start','end','filed'): F[c]=pd.to_datetime(F[c])
F['dur']=(F.end-F.start).dt.days; A=F[(F.form=='10-K')&(F.dur.between(350,380))].sort_values(['tk','kind','end','filed'])
A=A.drop_duplicates(['tk','kind','end','tag'],keep='first')
def pick(k):
    x=A[A.kind==k]; 
    best=x.groupby(['tk','tag']).size().reset_index(name='n').sort_values('n',ascending=False).drop_duplicates('tk')[['tk','tag']]
    return x.merge(best,on=['tk','tag'])
rev=pick('rev')[['tk','end','filed','val']].rename(columns={'val':'rev'})
sh=pick('sh')[['tk','end','val']].rename(columns={'val':'sh'})
fund=rev.merge(sh,on=['tk','end'],how='inner'); fund['sps']=fund.rev/fund.sh; fund=fund[fund.sps>0].sort_values(['tk','filed']); print('tickers with sales/share',fund.tk.nunique(),flush=True)
try: PX=pickle.load(open(SC+'macroz_px.pkl','rb'))
except Exception:
    tk=sorted(fund.tk.unique()); PX={}
    for i in range(0,len(tk),100):
        d=yf.download([t.replace('.','-') for t in tk[i:i+100]],start='2017-06-01',interval='1d',auto_adjust=True,group_by='ticker',progress=False,threads=True)
        for t in tk[i:i+100]:
            try: PX[t]=d[t.replace('.','-')]['Close'].dropna()
            except Exception: pass
        print('px',i,len(PX),flush=True)
    MAC=yf.download(['^VIX','^TNX','^IRX','HYG','LQD'],start='2017-06-01',auto_adjust=True,progress=False)['Close']; PX['_MAC']=MAC; pickle.dump(PX,open(SC+'macroz_px.pkl','wb'))
MAC=PX.pop('_MAC'); MAC=MAC.ffill()
mac=pd.DataFrame({'lvix':np.log(MAC['^VIX']),'y10':MAC['^TNX'],'curve':MAC['^TNX']-MAC['^IRX'],'cred':(MAC['HYG']/MAC['HYG'].shift(60)-MAC['LQD']/MAC['LQD'].shift(60))})
P=pd.DataFrame(PX).sort_index(); P=P[P.index>='2017-06-01']
dts=pd.Series(P.index,index=P.index); mstart=dts.groupby([dts.dt.year,dts.dt.month]).first().values  # first trading day each month
rows=[]
for d in mstart:
    d=pd.Timestamp(d)
    if d<pd.Timestamp('2018-01-01'): continue
    m=mac.loc[:d].iloc[-1]
    if m.isna().any(): continue
    i0=P.index.get_loc(d)
    for t in P.columns:
        f=fund[(fund.tk==t)&(fund.filed<=d-pd.Timedelta(days=1))]
        if f.empty: continue
        sps=f.sps.iloc[-1]; px=P[t].iloc[i0]
        if not np.isfinite(px) or px<=0: continue
        r={}
        for h in (21,63):
            j=i0+h
            r[f'r{h}']=P[t].iloc[j]/px-1 if j<len(P) and np.isfinite(P[t].iloc[j]) else np.nan
        rows.append(dict(d=d,tk=t,lm=np.log(px/sps),**m.to_dict(),**r))
Z=pd.DataFrame(rows); print('obs',len(Z),'months',Z.d.nunique(),flush=True); Z.to_pickle(SC+'macroz_obs.pkl')
X=['lvix','y10','curve','cred']
def fit(tr):
    tr=tr.copy(); mu=tr.groupby('tk').lm.transform('mean'); tr['lmd']=tr.lm-mu
    Xd=tr[X]-tr.groupby('tk')[X].transform('mean'); b=np.linalg.lstsq(Xd.values,tr.lmd.values,rcond=None)[0]
    a=(tr.lm-tr[X].values@b).groupby(tr.tk).mean(); res=tr.lm-tr[X].values@b-tr.tk.map(a); sd=res.groupby(tr.tk).std()
    return b,a,sd
tr=Z[(Z.d<'2022-01-01')]; cnt=tr.groupby('tk').size(); keep=cnt[cnt>=24].index; Z=Z[Z.tk.isin(keep)].copy(); tr=Z[Z.d<'2022-01-01']
b,a,sd=fit(tr); print('train betas',dict(zip(X,np.round(b,3))))
Z['z']=(Z.lm-Z[X].values@b-Z.tk.map(a))/Z.tk.map(sd)
a0=tr.groupby('tk').lm.mean(); sd0=tr.groupby('tk').lm.std(); Z['z0']=(Z.lm-Z.tk.map(a0))/Z.tk.map(sd0)
# expanding refit (robustness)
ze=[]
for d,g in Z.groupby('d'):
    h=Z[(Z.d<d)]; 
    if h.d.nunique()<24: ze.append(pd.Series(np.nan,index=g.index)); continue
    cn=h.groupby('tk').size(); h=h[h.tk.isin(cn[cn>=18].index)]; be,ae,se=fit(h)
    ze.append((g.lm-g[X].values@be-g.tk.map(ae))/g.tk.map(se))
Z['ze']=pd.concat(ze)
def ev(col,h,era,sign=-1):
    ics=[];sp=[]
    for d,g in Z[(Z.d>=era[0])&(Z.d<era[1])].groupby('d'):
        g=g.dropna(subset=[col,f'r{h}'])
        if len(g)<60: continue
        y=g[f'r{h}']-g[f'r{h}'].mean(); ics.append(spearmanr(sign*g[col],y)[0])
        q=g[col].rank(pct=True); sp.append(y[q<=.1].mean()-y[q>=.9].mean())
    ics=np.array(ics); sp=np.array(sp); 
    return f'IC {ics.mean():+.3f} (t {ics.mean()/(ics.std()/len(ics)**.5):+.1f}) | D1-D10 {sp.mean()*100:+.2f}% (t {sp.mean()/(sp.std()/len(sp)**.5):+.1f}) n={len(ics)}mo'
eras={'2018-21 IN-SAMPLE':('2018-01-01','2022-01-01'),'2022+ OUT-OF-SAMPLE':('2022-01-01','2027-01-01')}
for h in (21,63):
    print(f'\n==== forward {h}d excess return; cheap (low z) should win ====')
    for col,nm in (('z','macro-conditioned z (fixed train fit)'),('ze','macro z, expanding refit'),('z0','own-history z, NO macro'),('lm','raw log P/S cross-section')):
        for en,er in eras.items(): print(f'{nm:40s} {en:22s} {ev(col,h,er)}')
Z.to_pickle(SC+'macroz_Z.pkl')
