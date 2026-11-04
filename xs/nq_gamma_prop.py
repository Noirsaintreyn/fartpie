"""Nov 4 2026: vector + QQQ negative gamma, NY AM session, as an NQ futures prop-firm strategy. DECLARED BEFORE RESULTS.
DATA   QQQ 5m bars 2018 - Sep 2026 (premarket + RTH, ET) as the NQ proxy: points = QQQ return x NQ level (Yahoo NQ=F prior close). Check on real MNQ 1-minute prints (Dec 2024 - Nov 2025).
       Gamma = QQQ net GEX at the PRIOR close (xs/ext/gex/QQQ.csv, the file the live bot used); negative = net_gex < 0.
RULES  vector (live settings) on 5m or 15m bars, state from premarket + RTH bars. Entry at the open of a bar from 09:30 to 11:30 when the state at the prior close just flipped
       (long on a bull flip; 'both' also shorts bear flips). Exit at the open after the opposite flip, or 12:00 flat (NY AM only). One position at a time. Cost 1.0 NQ point round trip.
VARIANTS  bars 5m / 15m  x  long-only / both  x  days ALL / NEG gamma / POS gamma.  Placebo: same entry times and exits, direction by coin flip (200 draws).
PROP   Topstep-style 50K: +$3,000 target, $2,000 trailing max loss on end-of-day balance (stops trailing at the start balance), $1,000 daily loss -> stop for the day.
       Sizes: 1 NQ ($20/pt) and 2 MNQ ($4/pt). Every trading day is a start date; pass / fail / unfinished; median days to pass. Fees for resets not counted."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/stop_1h.py').read(); exec(src[:src.index("if __name__ == '__main__':")])
import yfinance as yf
ROOT = '/mnt/user-data/uploads/fartpie'
G = pd.read_csv(f'{ROOT}/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').net_gex.sort_index()
NQd = yf.download('NQ=F', start='2017-12-01', interval='1d', progress=False, auto_adjust=False)['Close'].squeeze(); NQd.index = NQd.index.normalize()
def bars(kind):
    if kind == 'QQQ':
        H = load(glob.glob(f'{ROOT}/lse_ohlcv_cache/QQQ_5m_*.parquet')[0])
    else:
        M = pd.read_parquet(f'{ROOT}/mnq_orderflow_1min.parquet'); M['dt'] = pd.to_datetime(M.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        M = M[['dt', 'last_price']].dropna().sort_values('dt'); x = M.set_index('dt').last_price
        H = pd.DataFrame({'open': x.resample('5min').first(), 'high': x.resample('5min').max(), 'low': x.resample('5min').min(), 'close': x.resample('5min').last()}).dropna().reset_index()
        H['m'] = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(H.m >= 240) & (H.m < 960) & (H.dt.dt.dayofweek < 5)].reset_index(drop=True)
    return H
def agg15(H):
    x = H.set_index('dt')[['open', 'high', 'low', 'close']]
    r = x.groupby([x.index.normalize(), pd.Grouper(freq='15min', origin='start_day', offset='4h')]).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    r.index = r.index.get_level_values(1); r = r.reset_index(); r.columns = ['dt', 'open', 'high', 'low', 'close']; r['m'] = r.dt.dt.hour * 60 + r.dt.dt.minute; return r
def trades(H, both, futures):
    reg = np.asarray(vec(H[['open', 'high', 'low', 'close']])); o = H.open.values; m = H.m.values; d = H.dt.dt.normalize().values; out = []; pos = 0
    for j in range(2, len(H)):
        flip = reg[j - 1] != reg[j - 2] and reg[j - 2] != 0
        if pos != 0:
            noon = m[j] >= 720 or d[j] != d[e]
            if noon or (flip and reg[j - 1] == -pos):
                px = o[j] if not (d[j] != d[e]) else o[j - 1]
                out.append((pd.Timestamp(d[e]), pd.Timestamp(H.dt.values[e]), pos, o[e], px)); pos = 0
        if pos == 0 and flip and 570 <= m[j] < 690 and (both or reg[j - 1] == 1):
            pos, e = int(reg[j - 1]), j
    T = pd.DataFrame(out, columns=['day', 'entry', 'dir', 'px0', 'px1'])
    if futures: T['pts'] = (T.px1 - T.px0) * T.dir - 1.0
    else:
        lvl = NQd.sort_index().reindex(pd.DatetimeIndex(T.day) - pd.Timedelta(days=1), method='ffill').values
        T['pts'] = (T.px1 / T.px0 - 1) * T.dir * lvl - 1.0
    gx = G.reindex(pd.DatetimeIndex(T.day) - pd.Timedelta(days=1), method='ffill').values; T['gex'] = gx
    return T
def daily(T, days): return T.groupby('day').pts.sum().reindex(days).fillna(0.0)
def prop(pnl_days, usd, target=3000, mll=2000, dll=1000):
    v = pnl_days.values * usd; res = []
    for s in range(len(v)):
        bal, peak, floor = 0.0, 0.0, -mll; out = None
        for k in range(s, len(v)):
            day = max(v[k], -dll); bal += day                      # daily loss limit: the day stops at -$1,000 (approximation: trades are not split)
            if bal <= floor: out = ('fail', k - s + 1); break
            if bal >= target: out = ('pass', k - s + 1); break
            peak = max(peak, bal); floor = min(max(floor, peak - mll), 0.0)
        res.append(out or ('open', len(v) - s))
    R = pd.DataFrame(res, columns=['r', 'days']); done = R[R.r != 'open']
    return (done.r == 'pass').mean(), done[done.r == 'pass'].days.median(), len(done)
def stats(nm, T, days):
    if not len(T): print(f'  {nm:<40} no trades'); return
    p = daily(T, days); yrs = (days[-1] - days[0]).days / 365.25; eq = p.cumsum(); dd = (eq - eq.cummax()).min()
    h1, h2 = T[T.day < '2022-01-01'], T[T.day >= '2022-01-01']
    s = f'  {nm:<40} n{len(T):>5} win {T.pts.gt(0).mean():.0%} avg {T.pts.mean():+6.1f} pts | 2018-21 {h1.pts.sum():+8.0f} / 2022-26 {h2.pts.sum():+8.0f} pts | per yr {p.sum() / yrs:+7.0f} pts  maxDD {dd:+7.0f} pts  Sharpe {p.mean() / p.std() * np.sqrt(252) if p.std() else 0:.2f}'
    print(s, flush=True); return p
if __name__ == '__main__':
    H5 = bars('QQQ'); H15 = agg15(H5); days = pd.DatetimeIndex(sorted(H5[(H5.m >= 570)].dt.dt.normalize().unique()))
    print(f'QQQ 5m bars {len(H5):,}, days {len(days)}, gex coverage {G.index.min():%Y-%m-%d} - {G.index.max():%Y-%m-%d}, neg-gamma days {(G < 0).mean():.0%}', flush=True)
    res = {}
    for bn, H in (('5m', H5), ('15m', H15)):
        for both in (False, True):
            T = trades(H, both, False); tag = f"{bn} {'both' if both else 'long'}"
            print(f'\n[{tag}]', flush=True)
            for gn, msk in (('ALL', np.ones(len(T), bool)), ('NEG gamma', T.gex.values < 0), ('POS gamma', T.gex.values >= 0)):
                p = stats(f'{tag} {gn}', T[msk], days)
                if p is not None: res[(tag, gn)] = (T[msk], p)
            Tn = T[T.gex.values < 0]; rng = np.random.default_rng(0); sims = []
            for _ in range(200):
                Z = Tn.copy(); fl = rng.choice([-1, 1], len(Z)); Z['pts'] = (Z.pts + 1.0) * fl - 1.0; sims.append(Z.pts.sum())
            print(f'  {"coin-flip direction, same NEG trades":<40} total {np.median(sims):+.0f} pts (90% {np.percentile(sims, 5):+.0f} .. {np.percentile(sims, 95):+.0f}); vector {Tn.pts.sum():+.0f}', flush=True)
    print('\nPROP 50K (target $3,000, $2,000 EOD trailing max loss, $1,000 daily loss): pass rate among finished attempts | median days to pass | attempts', flush=True)
    for (tag, gn), (T, p) in res.items():
        for usd, lab in ((20, '1 NQ'), (4, '2 MNQ')):
            pr, md, n = prop(p, usd); print(f'  {tag:<10} {gn:<10} {lab:<6} pass {pr:.0%}  median {md:.0f} days  n {n}', flush=True)
    print('\nCHECK on real MNQ 1-min prints (Dec 2024 - Nov 2025), 5m bars', flush=True)
    HM = bars('MNQ'); dm = pd.DatetimeIndex(sorted(HM[HM.m >= 570].dt.dt.normalize().unique()))
    for both in (False, True):
        T = trades(HM, both, True); Tq = trades(H5[(H5.dt >= HM.dt.min()) & (H5.dt <= HM.dt.max())].reset_index(drop=True), both, False)
        for gn, f in (('ALL', lambda X: np.ones(len(X), bool)), ('NEG gamma', lambda X: X.gex.values < 0)):
            a, b = T[f(T)], Tq[f(Tq)]
            print(f"  {'both' if both else 'long'} {gn:<10} MNQ: n{len(a)} {a.pts.sum():+.0f} pts win {a.pts.gt(0).mean():.0%} | QQQ proxy same months: n{len(b)} {b.pts.sum():+.0f} pts", flush=True)
