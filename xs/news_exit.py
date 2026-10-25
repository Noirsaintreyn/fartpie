"""Oct 25 2026: sell right after a bad earnings reaction, in extended hours, instead of waiting for the 09:30 open.
Trades that hold through a report (full-S&P screener, ratio >= 1.5, live rules). Reaction = price at a checkpoint after the report vs the
last regular-session close before it. If reaction <= -X%, sell at that checkpoint price (extended-hours fill, 15 bp extra cost); otherwise
the trade runs as usual. Checkpoints:  fast = 30 min after (after-close reports: 16:30; pre-open reports: 08:30)
evening = 20:00 (pre-open reports: 08:30)   preopen = 09:15 the next morning (pre-open reports: 09:15)."""
from collections import deque
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('xs/slot_tests.py').read().split('\nP = [(np.datetime64')[0])
exec('def sim3' + open('xs/equity_models.py').read().split('def sim3')[1].split('STRAT_MED =')[0]); STRAT_MED = 1
T = T.sort_values('entry').reset_index(drop=True)
TD = pd.DatetimeIndex(sorted(pd.to_datetime(pd.read_parquet('etf_daily/SPY.parquet').datetime).dt.tz_localize(None).dt.normalize().unique()))
REP = {}
for f in glob.glob('xs/ext/earn/*.csv'):
    tk = f.split('/')[-1][:-4]
    if tk.startswith('_'): continue
    try: e = pd.read_csv(f); t = pd.to_datetime(e.iloc[:, 0], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); s = pd.to_numeric(e['Surprise(%)'], errors='coerce')
    except Exception: continue
    REP[tk] = [(tt, tt.hour >= 12) for tt, ss in zip(t, s) if ss == ss]                  # (report time, after close?)
FULL = {}
def full(tk):
    if tk not in FULL:
        f = glob.glob(f'lse_clean/{tk}_15m_2018-01-01_*.parquet')
        if not f: FULL[tk] = None; return None
        B = pd.read_parquet(f[0], columns=['datetime', 'open', 'close']); B['dt'] = pd.to_datetime(B.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        FULL[tk] = B.drop_duplicates('dt').sort_values('dt').set_index('dt')
    return FULL[tk]
def px_at(B, t):                                                               # last close of a bar ending at or before t
    k = B.index.searchsorted(t - pd.Timedelta('15min'), 'right') - 1
    return (B.close.iloc[k], B.index[k] + pd.Timedelta('15min')) if k >= 0 and t - B.index[k] < pd.Timedelta('60min') else (None, None)
CK = {'fast': (pd.Timedelta('16h30min'), pd.Timedelta('8h30min'), 0), 'evening': (pd.Timedelta('20h'), pd.Timedelta('8h30min'), 0), 'preopen': (pd.Timedelta('9h15min'), pd.Timedelta('9h15min'), 1)}
VARS = [(ck, X, prof) for ck in CK for X in (.01, .02, .03, .05) for prof in (False,)] + [('fast', .02, True), ('preopen', .02, True)]
ALT = {v: (T.ret.values.copy(), T.exit.values.copy()) for v in VARS}; fired = {v: 0 for v in VARS}; ev = []
for tk, g in T.groupby('tk'):
    B = full(tk); R = REP.get(tk)
    if B is None or not R: continue
    rth = B[(B.index.hour * 60 + B.index.minute >= 570) & (B.index.hour * 60 + B.index.minute < 960)]
    for i, e, x in zip(g.index, g.entry, g.exit):
        if e not in rth.index or x not in rth.index: continue
        p0 = rth.open[e]; pexit = rth.open[x]
        for rt, ac in R:
            d = rt.normalize(); pre_day = d if ac else TD[TD.searchsorted(d) - 1]
            pre_close_t = pre_day + pd.Timedelta('16h')
            gap_open = (TD[min(TD.searchsorted(d, 'right'), len(TD) - 1)] if ac else d) + pd.Timedelta('9h30min')
            if not (e < pre_close_t and x >= gap_open): continue
            pc, _ = px_at(B, pre_close_t)
            if pc is None: break
            ev.append(dict(tk=tk, rt=rt, ac=ac, up=pc / p0 - 1))
            for v in VARS:
                ck, X, prof = v; a_, b_, nxt = CK[ck]
                tchk = (d + a_ if ac else d + b_) if not (ck == 'preopen' and ac) else gap_open.normalize() + a_
                if ck == 'evening' and not ac: tchk = d + b_
                p, tt = px_at(B, tchk)
                if p is None or (prof and pc / p0 - 1 < .03): continue
                if p / pc - 1 <= -X:
                    r_, x_ = ALT[v]; r_[i] += (p * (1 - .0015)) / p0 - pexit / p0; x_[i] = np.datetime64(tt); fired[v] += 1
            break
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z): print(f'  {nm:<56}' + ''.join(f'{cc:>+9.1%} /{dd:>5.0%}' for cc, dd in (sim3(Z, a, b) for a, b in P)), flush=True)
E = pd.DataFrame(ev); print(f'{len(E)} trades held into a report ({E.ac.mean():.0%} after-close reports)\n')
print(f'{"sell after a bad reaction":<58}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}'); show('live (rule exit, usually the 09:30 open)', T)
for v in VARS:
    Z = T.copy(); Z['ret'], Z['exit'] = ALT[v]; show(f'{v[0]:<8} reaction <= -{v[1]:.0%}' + (' (only if up >= 3%)' if v[2] else '') + f'  [{fired[v]}]', Z)
