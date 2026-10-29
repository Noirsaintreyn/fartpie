"""Oct 29 2026: in consolidation, raise the bar - skip 15m flips, take only 1h-track flips (everything else live). Consolidation known before entry:
 S-tight   stock's 20d range (high-low / close) in the bottom 30% of its own past year      S-flips  stock's daily vector flipped >= 3 times in 20 days
 M-tight   QQQ 20d range in the bottom 30% of its past year                                  M-flips  QQQ 15m vector flipped >= N times in the last 3 sessions
 B-cold    the book's last K closed trades were all losers (sequential, inside the sim)"""
import os, glob, numpy as np, pandas as pd
from collections import deque
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
src = open('/home/claude/fartpie/xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
import sys; sys.path.insert(0, '/home/claude/fartpie'); cwd = os.getcwd(); os.chdir('/home/claude/fartpie'); import live_screener as LS; os.chdir(cwd)
L = pd.read_parquet('/tmp/claude-0/ml/mtf_states.parquet').sort_values('entry').reset_index(drop=True); L['d'] = L.entry.dt.normalize()
def et(df): df = df.copy(); df['dt'] = pd.to_datetime(df.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); return df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
def daily(X):
    m = X.dt.dt.hour * 60 + X.dt.dt.minute; R = X[(m >= 570) & (m < 960)]
    return R, R.set_index('dt').resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
def tight(D):
    r = (D.high.rolling(20).max() - D.low.rolling(20).min()) / D.close
    return r.rolling(252, min_periods=120).rank(pct=True).shift(1)          # known at the prior close
F = []
for tk, g in L.groupby('tk'):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    R, D = daily(et(pd.read_parquet(f[0])))
    V = LS.vec(D.reset_index().rename(columns={'index': 'dt'})[['dt', 'open', 'high', 'low', 'close', 'volume']]); fl = pd.Series((V.reg.diff().abs() > 0).astype(int).values, index=D.index)
    X = pd.DataFrame({'s_tight': tight(D), 's_flips': fl.rolling(20).sum().shift(1)}); X['tk'] = tk; F.append(X.reset_index().rename(columns={'dt': 'd', 'index': 'd'}))
F = pd.concat(F); F['d'] = F.d.astype('datetime64[ns]'); L['d'] = L.d.astype('datetime64[ns]')
L = L.merge(F, on=['tk', 'd'], how='left')
Rq, Dq = daily(et(pd.read_parquet(glob.glob('/mnt/user-data/uploads/fartpie/lse_clean/QQQ_15m_*.parquet')[0])))
mq = pd.DataFrame({'m_tight': tight(Dq)}); mq.index = mq.index.astype('datetime64[ns]'); L = L.join(mq, on='d')
Vq = LS.vec(Rq[['dt', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)); fq = pd.Series((Vq.reg.diff().abs() > 0).astype(int).values, index=Vq.dt + pd.Timedelta('15min'))
cf = fq.rolling(78).sum()                                                    # flips in the last 3 sessions (26 bars/day)
k_ = cf.index.searchsorted(L.entry.values, side='right') - 1; L['m_flips'] = np.where(k_ >= 0, cf.values[np.clip(k_, 0, None)], np.nan)
print('m_flips quantiles', L.m_flips.quantile([.25, .5, .75, .9]).to_dict())
P = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def sh(nm, Z, on=None):
    frac = '' if on is None else f'  on {on.mean():.0%} of signals, drops {((on) & (~L.h1)).mean():.0%}'
    print(f'  {nm:<36}' + ''.join(f'{c:>+8.1%} /{d:>4.0%}   ' for c, d in (sim(Z, a, b) for a, b in P)) + frac, flush=True)
print('yearly return / max DD   2019-21   2022-26')
sh('live', L)
sh('only 1h flips, always', L[L.h1])
C = {'S-tight (<30%)': L.s_tight < .3, 'S-tight (<20%)': L.s_tight < .2, 'S-flips >= 3': L.s_flips >= 3, 'S-flips >= 4': L.s_flips >= 4,
     'M-tight (<30%)': L.m_tight < .3, 'M-tight (<20%)': L.m_tight < .2}
for q in (.5, .75, .9):
    v = L.m_flips.quantile(q); C[f'M-flips >= {v:.0f} (top {1 - q:.0%})'] = L.m_flips >= v
for nm, on in C.items():
    on = on.fillna(False); sh(nm, L[~(on & ~L.h1)], on)
def sim_cold(Z, a, b, K, n=6, brk=.08):                     # B-cold: skip 15m flips while the last K closed trades all lost
    Z = Z[(Z.entry >= a) & (Z.entry < b)].sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; last = deque(maxlen=K)
    for e, x, r_, h1 in zip(Z.entry.values, Z.exit.values, Z.ret.values, Z.h1.values):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / n); hist.append((xt, real)); win.append((xt, real)); last.append(rr)
        op = [o for o in op if o > e]
        if len(op) >= n: continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
        if not h1 and len(last) == K and all(r <= 0 for r in last): continue
        op.append(x); pend.append((x, r_))
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr / n); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
    eq = np.exp(s.groupby(s.index.normalize()).last()); y = (eq.index[-1] - eq.index[0]).days / 365.25
    return eq.iloc[-1] ** (1 / y) - 1, (eq / eq.cummax() - 1).min()
for K in (2, 3, 4, 5):
    print(f'  {f"B-cold: last {K} closed all lost":<36}' + ''.join(f'{c:>+8.1%} /{d:>4.0%}   ' for c, d in (sim_cold(L, a, b, K) for a, b in P)), flush=True)
rng = np.random.default_rng(3); R = []
for k in range(30): R.append([sim(L.drop(index=rng.choice(L.index[~L.h1], int((~L.h1).sum() * .15), replace=False)), a, b)[0] for a, b in P])
R = np.array(R); print(f'  luck band: drop 15% of 15m flips at random x30   2019-21 {R[:, 0].min():+.1%}..{R[:, 0].max():+.1%}   2022-26 {R[:, 1].min():+.1%}..{R[:, 1].max():+.1%}')
