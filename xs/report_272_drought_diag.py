"""Report 272 (declared before results): why do payout droughts happen? Windows (from earlier sim): W1 2019-03..2020-01, W2 2021-06..2021-12, W3 2024-03..2024-09 vs everything else. For each window: algo $/day (base sizes, shareable250 daily), share of days >= $150 (a Lucid qualifying day), share of days <= -$150, month-level count of qualifying days, share of MR / MID / TREND days (ex-post ER tertile label), mean RTH range in pts, and algo $/day by day type. Hypothesis: droughts = few qualifying days because MR share is high and/or range is low (a size problem, not a loss problem). If droughts are low-range, sizing by forecast range (HAR+IV) is the lever; if MR-heavy, only a mean-reversion leg helps."""
import pickle, numpy as np, pandas as pd
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
s=pickle.load(open(SP+'shareable250.pkl','rb')); daily=s['daily'].copy(); daily.index=pd.to_datetime(daily.index.to_timestamp() if hasattr(daily.index,'to_timestamp') else daily.index)
e=pickle.load(open(SP+'ev256.pkl','rb')); e.index=pd.to_datetime(e.index); e['rng']=e.h-e.l
df=pd.DataFrame({'pnl':daily}).join(e[['type','er','rng']],how='left'); df['pnl']=df.pnl.fillna(0)
W={'W1 2019-03..2020-01':('2019-03-01','2020-01-31'),'W2 2021-06..2021-12':('2021-06-01','2021-12-31'),'W3 2024-03..2024-09':('2024-03-01','2024-09-30')}
df['w']='rest'
for k,(a,b) in W.items(): df.loc[a:b,'w']=k
def row(g): return pd.Series(dict(days=len(g),usd_day=g.pnl.mean(),q150=(g.pnl>=150).mean(),l150=(g.pnl<=-150).mean(),MR=(g.type=='MR').mean(),MID=(g.type=='MID').mean(),TREND=(g.type=='TREND').mean(),rng=g.rng.mean()))
print(df.groupby('w').apply(row).round(3).to_string())
print('\nalgo $/day by day type, window vs rest'); print(df.pivot_table(index='w',columns='type',values='pnl',aggfunc='mean').round(1).to_string())
print('\nqualifying days (>=$150) per calendar month: windows vs rest'); m=df.pnl.ge(150).groupby(df.index.to_period('M')).sum(); mw=df.w.groupby(df.index.to_period('M')).agg(lambda x:x.mode()[0])
print(pd.DataFrame({'q':m,'w':mw}).groupby('w').q.describe().round(1).to_string()); print('share of months with <5 qualifying days:',pd.DataFrame({'q':m,'w':mw}).groupby('w').q.apply(lambda x:(x<5).mean()).round(2).to_dict())
print('\nby year: $/day | q150 | MR share | mean range'); print(df.groupby(df.index.year).apply(row)[['usd_day','q150','MR','TREND','rng']].round(3).to_string())
# range-forecast relation: is algo $/day proportional to range?
df['yr']=df.index.year; c=df.groupby('yr').apply(lambda g:np.corrcoef(g.pnl,g.rng.fillna(g.rng.mean()))[0,1]); print('\ncorr(daily algo $, that day range) by year',c.round(2).to_dict())
