"""TRADABLE version of the idle-day scan: the idle label is ex-post (look-ahead), so gate with the walk-forward RF 'accepts-new-value' prob at +60m (10:30 ET, OOS from 2019).
Gate = prob in the bottom 40% of its expanding past distribution (balance day call). Scan all rules with entry >=10:30; compare with ungated."""
import numpy as np, pandas as pd, runpy, io, contextlib
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
buf=io.StringIO()
with contextlib.redirect_stdout(buf): g=runpy.run_path('xs/report_idle_scan.py')
keep,P,res,n,era2=g['keep'],g['P'],g['res'],g['n'],g['era2']
Z=pd.read_pickle(SC+'regime_scores.pkl')[60].dropna(subset=['m2_rf']).copy(); Z['day']=pd.to_datetime(Z.day); Z=Z.sort_values('day').set_index('day')
p=Z.m2_rf; rk=p.expanding(min_periods=100).apply(lambda s:(s.iloc[:-1]<s.iloc[-1]).mean() if len(s)>1 else np.nan,raw=False)
gate=(rk<=.4).reindex(keep).fillna(False).values.astype(bool); have=p.reindex(keep).notna().values
print('days with OOS score',have.sum(),' gated(balance call)',gate.sum(),' era1',(gate&~era2).sum(),' era2',(gate&era2).sum())
def st(mask):
    X=P[:,mask]; k=(~np.isnan(X)).sum(1); mu=np.nanmean(X,1); sd=np.nanstd(X,1,ddof=1); return k,mu,mu/np.where(sd>0,sd/np.sqrt(np.maximum(k,1)),np.nan)
for nm,m in (('g',gate),('g1',gate&~era2),('g2',gate&era2),('u',have),('u1',have&~era2),('u2',have&era2)):
    k,mu,t=st(m); res['n_'+nm]=k; res['mu_'+nm]=mu; res['t_'+nm]=t
ok=res[(res.ent>=630)&(res.n_g>=60)]
rng=np.random.default_rng(1); Pg=P[:,gate]; mx=[]
for _ in range(300):
    X=Pg*rng.choice([-1,1],size=Pg.shape[1]); k=(~np.isnan(X)).sum(1); mu=np.nanmean(X,1); sd=np.nanstd(X,1,ddof=1); t=mu/np.where(sd>0,sd/np.sqrt(np.maximum(k,1)),np.nan); mx.append(np.nanmax(t[(res.ent>=630).values&(k>=60)]))
print(f'null max-t (gated, ent>=10:30): median {np.median(mx):.2f} 95% {np.quantile(mx,.95):.2f}; configs {len(ok)}')
cols=['feat','ent','thr','mode','ex','stop','n_g','mu_g','t_g','mu_g1','mu_g2','mu_u','mu_u1','mu_u2']
b=ok[(ok.mu_g1>0)&(ok.mu_g2>0)].sort_values('t_g',ascending=False)
print('positive both eras on gated days:',len(b),'of',len(ok)); print(b[cols].head(25).round(2).to_string(index=False))
print('\nof those, ones that also BEAT ungated in both eras (gate adds value):')
bb=b[(b.mu_g1>b.mu_u1)&(b.mu_g2>b.mu_u2)]; print(len(bb)); print(bb[cols].head(15).round(2).to_string(index=False))
print('\nfeature-family on gated days: share cfgs positive both eras')
print(ok.assign(pb=(ok.mu_g1>0)&(ok.mu_g2>0)).groupby(['feat','mode']).agg(cfgs=('pb','size'),pos_both=('pb','mean'),mu=('mu_g','mean')).round(3).to_string())
