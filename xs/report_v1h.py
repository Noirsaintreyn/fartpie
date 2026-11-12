"""Add a 1h-vector leg to the NQ algo. V1h = same engine as V15a (run_sig2: flip entries, exit on opposite flip, 0.15 ADR stop, <=4 per day), vector on 1h bars (RTH, 09:30 anchored)."""
_t = open('/home/claude/fartpie/xs/trail_tiers.py').read(); exec(_t[:_t.rindex("if __name__ == '__main__':")])
import pickle
V15 = run_sig2('s_vector'); print('V15 cols', V15.columns.tolist(), len(V15))
B15_ = B.copy()
H['b60'] = H.dt.dt.floor('60min') + pd.Timedelta(minutes=30); H.loc[H.dt.dt.hour*60+H.dt.dt.minute < 570, 'b60'] = pd.NaT
H['b60'] = (H.dt - pd.Timedelta(minutes=30)).dt.floor('60min') + pd.Timedelta(minutes=30)
B60 = H.groupby('b60').agg(open=('open','first'), high=('high','max'), low=('low','min'), close=('close','last'), day=('day','first'))
B60['reg'] = np.asarray(vec(B60[['open','high','low','close']])); B60['s_vector'] = B60.reg; B60['end'] = B60.index + pd.Timedelta(minutes=60)
B = B60
V60 = run_sig2('s_vector'); B = B15_
print('V60 trades', len(V60), 'V15 trades', len(V15))
pickle.dump(dict(V15=V15, V60=V60), open('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/v1h.pkl', 'wb'))
def st(nm, x):
    for e, m in (('2014-21', x.day < '2022-01-01'), ('2022+', x.day >= '2022-01-01')):
        z = x[m]; yrs = max((z.day.max() - z.day.min()).days / 365.25, .1) if len(z) else 1
        print(f'  {nm} {e}: n {len(z)} avg {z.pts.mean():+.2f} pts win {(z.pts>0).mean():.0%} total {z.pts.sum():+.0f} ({z.pts.sum()/yrs:+.0f}/yr) trade-days {z.day.nunique()} ({z.day.nunique()/yrs:.0f}/yr)')
V15g = V15[~hig(V15)]; V60g = V60[~hig(V60)]
st('V15a', V15g); st('V60 ', V60g)
d15 = set(V15g.day); new = V60g[~V60g.day.isin(d15)]; print('V60 on days V15a is idle:'); st('V60-new-days', new)
print('days with V15a', len(d15), 'days with V60', V60g.day.nunique(), 'union', len(d15 | set(V60g.day)))
