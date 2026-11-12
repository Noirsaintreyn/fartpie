"""High-volume days with a vector flip that day, for the 5 vol-forecast names (KGS LCID RIOT Z IREN). High volume = RTH volume >= 1.5x its trailing 20-day mean (daily Yahoo). Flips: 15m vector on pre+RTH bars, 1h vector on RTH bars (09:30-anchored), last 60 sessions (Yahoo 15m limit). Times ET = bar close at which the state changed (fills next bar open)."""
import sys, numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
from vector_pine import run_state_machine
rows = []
for tk in ('KGS', 'LCID', 'RIOT', 'Z', 'IREN'):
    dd = yf.download(tk, period='1y', interval='1d', auto_adjust=False, progress=False); dd.columns = [c[0] for c in dd.columns]; dd.index = dd.index.normalize()
    rel = dd.Volume / dd.Volume.rolling(20).mean().shift(); dret = dd.Close / dd.Open - 1
    d15 = yf.download(tk, period='60d', interval='15m', prepost=True, progress=False, auto_adjust=False); d15.columns = [c[0].lower() for c in d15.columns]
    d15.index = d15.index.tz_convert('America/New_York').tz_localize(None); d15 = d15[['open', 'high', 'low', 'close']].dropna(); d15 = d15[~d15.index.duplicated()]
    m = d15.index.hour * 60 + d15.index.minute; d15 = d15[(m >= 240) & (m < 960)]
    reg15 = run_state_machine(d15.assign(datetime=d15.index))[0]
    rth = d15[(d15.index.hour * 60 + d15.index.minute) >= 570]; h = rth.resample('60min', origin='start_day', offset='30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    reg1 = run_state_machine(h.assign(datetime=h.index))[0]
    def fl(B, reg, tag):
        out = []
        for i in range(1, len(B)):
            if reg[i] != reg[i - 1] and reg[i] != 0 and B.index[i] >= pd.Timestamp('2026-08-20'):
                out.append((B.index[i], tag, 'BULL' if reg[i] > 0 else 'BEAR', float(B.close.iloc[i])))
        return out
    F = fl(d15, reg15, '15m') + fl(h, reg1, '1h'); 
    for t, v in rel.dropna().items():
        if v >= 1.5 and t >= pd.Timestamp('2026-08-20'):
            f = [x for x in F if x[0].normalize() == t]
            rows.append(dict(tk=tk, day=t.date(), relvol=round(float(v), 2), day_ret_pct=round(float(dret[t]) * 100, 1), flips='; '.join(f'{x[1]} {x[2]} {x[0]:%H:%M} @ {x[3]:.2f}' for x in f) or '-', n_flips=len(f)))
R = pd.DataFrame(rows).sort_values(['tk', 'day']); R.to_csv('xs/highvol_flip_days.csv', index=False)
pd.set_option('display.width', 250, 'display.max_colwidth', 120); print(R.to_string(index=False))
print('\nshare of high-volume days with >=1 flip:', R.groupby('tk').apply(lambda g: f'{(g.n_flips>0).mean():.0%} ({(g.n_flips>0).sum()}/{len(g)})').to_dict())
