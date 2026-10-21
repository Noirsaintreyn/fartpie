"""Oct 21 2026: exit ideas on the full-S&P screener (ratio >= 1.5, live rules, 6 slots, 8% breaker).
 TS  time stop: if the trade is below THR at the close N trading days after entry, sell at the next open (frees the slot)
 PP  partial profit: sell half when price first touches entry x (1+P) (fill at that price); the rest follows the rule
 VW  weekend flat: if VIX (prior close) > V on Friday, sell at Friday's last-bar close and buy back Monday 09:30 (slot kept)
Each variant changes a trade's return by (alt path - rule path) measured on the same clean 15m bars, so costs/fills stay comparable."""
from collections import deque
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('xs/slot_tests.py').read().split('P = [(np.datetime64')[0])
exec('def sim3' + open('xs/equity_models.py').read().split('def sim3')[1].split('STRAT_MED =')[0])
STRAT_MED = 1
T = T.sort_values('entry').reset_index(drop=True)
vix = pd.read_parquet('etf_daily/VIX.parquet').close; vix.index = pd.to_datetime(vix.index); vix.index = (vix.index.tz_localize(None) if vix.index.tz is not None else vix.index).normalize()
BARS = {}
def bars(tk):
    if tk not in BARS:
        f = glob.glob(f'lse_clean/{tk}_15m_2018-01-01_*.parquet')
        if not f: BARS[tk] = None; return None
        B = pd.read_parquet(f[0], columns=['datetime', 'open', 'high', 'low', 'close']); B['dt'] = pd.to_datetime(B.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        m = B.dt.dt.hour * 60 + B.dt.dt.minute; B = B[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True); BARS[tk] = B
    return BARS[tk]
VARS = {f'TS{n}d<{thr:+.0%}': ('ts', n, thr) for n in (1, 2, 3, 5) for thr in (0.0, .01)}
VARS.update({f'PP half at +{p:.0%}': ('pp', p) for p in (.03, .05, .08)})
VARS.update({f'VW flat weekends VIX>{v}': ('vw', v) for v in (25, 30)})
ALT = {k: (T.ret.values.copy(), T.exit.values.copy()) for k in VARS}; nb = 0
for tk, g in T.groupby('tk'):
    B = bars(tk)
    if B is None: continue
    dt = B.dt.values; o, h, c = B.open.values, B.high.values, B.close.values; day = B.dt.dt.normalize().values
    for i, e, x in zip(g.index, g.entry.values, g.exit.values):
        a = np.searchsorted(dt, e); b = np.searchsorted(dt, x)
        if a >= len(dt) or dt[a] != e or b >= len(dt) or b <= a: nb += 1; continue
        p0 = o[a]; base = o[b] / p0 - 1
        days = np.unique(day[a:b + 1])
        for k, v in VARS.items():
            r_, x_ = ALT[k]
            if v[0] == 'ts':
                n, thr = v[1], v[2]
                if len(days) > n + 1:
                    dN = days[n]; last = np.where(day[a:b] == dN)[0]
                    if len(last):
                        j = a + last[-1]
                        if c[j] / p0 - 1 < thr and j + 1 < b: r_[i] += o[j + 1] / p0 - 1 - base; x_[i] = dt[j + 1]
            elif v[0] == 'pp':
                hit = np.where(h[a:b] >= p0 * (1 + v[1]))[0]
                if len(hit): r_[i] += 0.5 * (v[1] - base)
            else:
                lvl = v[1]; adj = 0.0
                for dd in days[:-1]:
                    if pd.Timestamp(dd).dayofweek != 4: continue
                    vv = vix[vix.index < dd]
                    if not len(vv) or vv.iloc[-1] <= lvl: continue
                    fr = np.where(day[a:b] == dd)[0]; nx = np.where(day[a:b + 1] > dd)[0]
                    if not len(fr) or not len(nx): continue
                    jf = a + fr[-1]; jm = a + nx[0]
                    adj += np.log(c[jf] / o[jm])          # sell Friday close, rebuy Monday open: removes the weekend gap
                r_[i] = (1 + r_[i]) * np.exp(adj) - 1 if adj else r_[i]
print('trades without matching bars:', nb, 'of', len(T))
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z):
    print(f'  {nm:<46}' + ''.join(f'{cc:>+9.1%} /{dd:>5.0%}' for cc, dd in (sim3(Z, a, b) for a, b in P)), flush=True)
print(f'\n{"exits (S&P, ratio >= 1.5, live rules)":<48}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
show('live', T)
for k in VARS:
    Z = T.copy(); Z['ret'], Z['exit'] = ALT[k]; show(k, Z)
