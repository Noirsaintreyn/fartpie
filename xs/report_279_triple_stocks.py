"""Report 279 (declared before results). Rey confirmed the idea: volume building + absorption + positive gamma at a rejected extreme. Frequency on NQ is ~3-4 trades a year, so test the SAME pre-declared triple on 19 liquid names that have their own daily net GEX (xs/ext/gex, to 2025-12): AAPL ADBE AMD CAT COST CRM DIS HD IBM MCD NVDA ORCL QQQ SBUX TGT TSLA UNH UPS WMT, 15m RTH bars. This is a different-asset out-of-sample check and a frequency fix.
Event: bar (start 10:00-14:00 ET) that makes a new session high (short) or low (long), >= .5 R20 from the open (R20 = that name's prior-20d mean daily range), rejects (closes back with >= 50% of its range as the wick on the push side). V1 = event-bar volume >= mean volume of EARLIER new-extreme bars the same day (>=1 earlier) and >= 1.3x the prior-8-bar mean; V2 = est. delta (vol x (2c-h-l)/(h-l)) summed over the event bar and 2 before is in the push direction (against the fade) and event volume >= 1.3x prior-8-bar mean; G = the name's OWN prior-close net_gex > 0. Entry next bar open, stop = extreme +- .02 R20, target = extreme -+ .618 R20 (toward the day), skip if risk > .4 R20 or reward < 1 x risk, flat 15:45, cost 2 bps of price round trip, max 2 per day per side per name. Output per cell: n, hold rate (event extreme stays the session extreme), mean net R, day-clustered t (cluster = date across names), train 2019-21 | test 2018 + 2022-25. Cells: all events; G; V1&V2; V1&V2&G; V1&G; V2&G. Pass: V1&V2&G beats 'all events' in both splits on hold rate and net R with n >= 150 per split."""
import glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
T_=['AAPL','ADBE','AMD','CAT','COST','CRM','DIS','HD','IBM','MCD','NVDA','ORCL','QQQ','SBUX','TGT','TSLA','UNH','UPS','WMT']; rows=[]
for t in T_:
    f=glob.glob(f'/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/{t}_15m_*.parquet')+glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{t}_15m_*.parquet')
    if not f: print('missing',t); continue
    d=pd.read_parquet(f[0]); d['datetime']=pd.to_datetime(d.datetime).dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None); d=d.set_index('datetime').sort_index().between_time('09:30','15:45'); d['d']=d.index.normalize()
    gx=pd.read_csv(f'xs/ext/gex/{t}.csv',parse_dates=['date']).set_index('date').sort_index(); gx.index=gx.index.normalize()
    G=d.groupby('d'); dd=pd.DataFrame({'h':G.high.max(),'l':G.low.min(),'n':G.size()}); dd=dd[dd.n>=24]; dd['R20']=(dd.h-dd.l).shift(1).rolling(20).mean()
    kg=np.clip(gx.index.searchsorted(dd.index.values.astype('datetime64[ns]'))-1,0,len(gx)-1); dd['g']=(gx.net_gex.values[kg]>0).astype(float); dd.loc[dd.index>gx.index.max(),'g']=np.nan
    for day,g in G:
        if day not in dd.index or not np.isfinite(dd.R20[day]) or pd.isna(dd.g[day]): continue
        R=dd.R20[day]; o=g.open.values;h=g.high.values;l=g.low.values;c=g.close.values;v=g.volume.values.astype(float); mm=g.index.hour*60+g.index.minute; dl=v*(2*c-h-l)/np.maximum(h-l,1e-9)
        o0=o[0]; rh=-np.inf; rl=np.inf; ev={1:[],-1:[]}; cnt={1:0,-1:0}; dh=h.max(); dlow=l.min()
        for i in range(len(g)-2):
            for dr,new in ((-1,h[i]>rh),(1,l[i]<rl)):
                if not new: continue
                pv=list(ev[dr]); ev[dr].append(v[i])
                if not(600<=mm[i]<840) or cnt[dr]>=2: continue
                rng=h[i]-l[i]; 
                if rng<=0: continue
                if dr==-1:
                    if h[i]-o0<.5*R or (h[i]-c[i])<.5*rng: continue
                    ext=h[i]; stop=ext+.02*R; tgt=ext-.618*R
                else:
                    if o0-l[i]<.5*R or (c[i]-l[i])<.5*rng: continue
                    ext=l[i]; stop=ext-.02*R; tgt=ext+.618*R
                e=o[i+1]; risk=abs(stop-e); rew=abs(e-tgt)
                if risk>.4*R or rew<risk or risk<=0: continue
                cnt[dr]+=1
                pm=v[max(0,i-8):i].mean(); rel=v[i]/pm if pm>0 else np.nan
                v1=bool(len(pv)>=1 and v[i]>=np.mean(pv) and rel>=1.3); ds=dl[max(0,i-2):i+1].sum(); v2=bool(-dr*ds>0 and rel>=1.3)
                res=None
                for j in range(i+1,len(g)):
                    hs=(h[j]>=stop) if dr==-1 else (l[j]<=stop); ht=(l[j]<=tgt) if dr==-1 else (h[j]>=tgt)
                    if hs: res=-risk;break
                    if ht: res=rew;break
                if res is None: res=(e-c[-1]) if dr==-1 else (c[-1]-e)
                cost=2e-4*e; net=(res-cost)/risk; held=(ext>=dh) if dr==-1 else (ext<=dlow)
                rows.append((t,day,dr,v1,v2,bool(dd.g[day]>0),bool(held),net))
E=pd.DataFrame(rows,columns=['t','d','dr','v1','v2','g','held','R']); E['split']=np.where((E.d.dt.year>=2019)&(E.d.dt.year<=2021),'train',np.where((E.d.dt.year==2018)|(E.d.dt.year>=2022),'test','x'))
print('events',len(E),' tickers',E.t.nunique(),' split',E.split.value_counts().to_dict())
def cell(m):
    s=[]
    for sp in ('train','test'):
        z=E[m&(E.split==sp)]
        if len(z)<10: s.append(f'n{len(z):5d} -'); continue
        ds=z.groupby('d').R.agg(['sum','size']); mean=z.R.mean(); se=np.sqrt(((ds['sum']-mean*ds['size'])**2).sum())/len(z)
        s.append(f'n{len(z):5d} hold {z.held.mean():4.0%} netR {mean:+.3f} (t{mean/se:+.1f})')
    return ' | '.join(s)
for nm,m in (('all events',E.R.notna()),('G',E.g),('~G',~E.g),('V1',E.v1),('V2',E.v2),('V1&V2',E.v1&E.v2),('V1&V2&G',E.v1&E.v2&E.g),('V1&V2&~G',E.v1&E.v2&~E.g),('V1&G',E.v1&E.g),('V2&G',E.v2&E.g)): print(f' {nm:10s}',cell(m))
z=E[E.v1&E.v2&E.g]; print('\nV1&V2&G by year (n, mean R):',{y:(int((z.d.dt.year==y).sum()),round(z[z.d.dt.year==y].R.mean(),2)) for y in range(2018,2026)})
print('V1&V2&G by ticker n>=8:',{t:(len(g),round(g.R.mean(),2)) for t,g in z.groupby('t') if len(g)>=8})
