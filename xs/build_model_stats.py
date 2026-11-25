"""Backtest numbers shown on signal-model alerts (live_data/model_stats.json). Per model x timeframe: all names, and per ticker. n, mean bp, win, month-clustered t. Rebuild after new research."""
import pandas as pd, numpy as np, json
exec(open('xs/report_457_optfeat_flips.py').read().split("NAMES = dict")[0].split('"""')[2])
_Z = {}
for _tk in TK:
    _x = pd.read_csv(UP + f'data/theta_feat2/{_tk}.csv', parse_dates=['date']).set_index('date').sort_index().pc_vol; _r = _x.rolling(252, min_periods=60)
    _Z[_tk] = ((_x - _r.mean().shift(1)) / _r.std().shift(1)).shift(1).rename('pcz')
W = W.merge(pd.concat(_Z, names=['tk', 'day']).reset_index(), on=['tk', 'day'], how='left')
def st(x):
    if len(x) < 5: return None
    m = x.groupby('mon').ret.mean(); t = m.mean() / (m.std() / np.sqrt(len(m))) if len(m) > 2 and m.std() > 0 else None
    return dict(n=int(len(x)), bp=round(float(x.ret.mean() * 1e4), 1), win=round(float((x.ret > 0).mean() * 100)), t=None if t is None else round(float(t), 2))
out = {}
for h1 in (False, True):
    tf = '1h' if h1 else '15m'; D = W[(W.h1 == h1) & W.gex.notna()].copy()
    D['NEG'] = D.gex < 0; D['PCV'] = D.r_pc_vol >= 2 / 3; D['PCZ'] = D.pcz > 1; D['SKW'] = D.r_skew >= 2 / 3; D['CHM'] = D.r_charm >= 2 / 3
    for k in ('NEG', 'PCV', 'PCZ', 'SKW', 'CHM'):
        if k in ('PCV', 'PCZ') and h1: continue
        x = D[D[k].fillna(False)]; out[f'{k}|{tf}|ALL'] = st(x)
        for tk, g in x.groupby('tk'): out[f'{k}|{tf}|{tk}'] = st(g)
    out[f'BASE|{tf}|ALL'] = st(D)
    for tk, g in D.groupby('tk'): out[f'BASE|{tf}|{tk}'] = st(g)
json.dump({k: v for k, v in out.items() if v}, open('live_data/model_stats.json', 'w'), indent=0)
print(len(out))
