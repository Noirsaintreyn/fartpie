"""Nov 9 2026: RANK THE SCREENER'S SIMULTANEOUS FLIPS BY EXPECTED MOVE - Yang-Zhang, HAR, or the DAILY options ratio (Rey). DECLARED BEFORE RESULTS.
Live rule: when several bull flips arrive in the same bar, the highest 252-day-median options ratio gets the slot first (xs/scr_slots.py: worth ~7-10 pts of CAGR
over random). Question: does ranking by how much the stock is EXPECTED TO MOVE do better?
Daily OHLC from Yahoo (split/dividend adjusted) for the 54 names that ever pass the live filter; every feature as of the PRIOR close (known before the open):
  YZ20     Yang-Zhang volatility, last 20 days
  HAR      HAR forecast of tomorrow's high-low range (% of price): log range on yesterday / last week / last month, pooled over the 54 names, fit 2018-21
  HARREL   HAR forecast / that stock's own trailing-year median forecast (about to move more than usual for it)
  RATIO    the live 252-day median options ratio
Pick orders (6 slots, same stand-in daily path as xs/scr_slots.py): ratio (live) | YZ20 | HAR | HARREL | rank-sum ratio + HAR | rank-sum ratio + HARREL.
CHOSEN on 2019-22, JUDGED on 2023-26 (Sep 25). Also: per-trade return and |return| by YZ third, and the in-bar rank table.
DAILY options ratio (yesterday's contracts x 100 / shares) only exists from Jul 2025 (live_data/optvol_ratio_live.csv): ranked vs the 252-day ratio on Jul 15 2025 - Sep 2026 only."""
import os, numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
CACHE = '/home/claude/fartpie/data/yahoo/daily'; os.makedirs(CACHE, exist_ok=True); TKS = sorted(T0.tk.unique()); YSYM = {'BRKB': 'BRK-B'}
def daily_ohlc(tk):
    f = f'{CACHE}/{tk}.parquet'
    if os.path.exists(f): return pd.read_parquet(f)
    Y = yf.download(YSYM.get(tk, tk), start='2017-06-01', end='2026-10-25', interval='1d', auto_adjust=True, progress=False)
    if isinstance(Y.columns, pd.MultiIndex): Y.columns = Y.columns.get_level_values(0)
    Y = Y.rename(columns=str.lower)[['open', 'high', 'low', 'close']].dropna(); Y.index = pd.DatetimeIndex(Y.index).tz_localize(None).normalize(); Y.to_parquet(f); return Y
def yz(Y, n=20):
    o, h, l, c = np.log(Y.open), np.log(Y.high), np.log(Y.low), np.log(Y.close); co = o - c.shift(1); oc = c - o
    rs = (h - c) * (h - o) + (l - c) * (l - o); k = .34 / (1.34 + (n + 1) / (n - 1))
    return np.sqrt(co.rolling(n).var() + k * oc.rolling(n).var() + (1 - k) * rs.rolling(n).mean())
F = []
for tk in TKS:
    Y = daily_ohlc(tk)
    if len(Y) < 300: print('  no data', tk); continue
    r = np.log(Y.high / Y.low).clip(lower=1e-4); F.append(pd.DataFrame({'tk': tk, 'yz': yz(Y), 'lr': np.log(r), 'd': np.log(r).shift(0), 'w': np.log(r.rolling(5).mean()), 'm': np.log(r.rolling(22).mean())}, index=Y.index))
F = pd.concat(F); F['y'] = F.groupby('tk').lr.shift(-1)                                     # tomorrow's log range, predicted from today's d / w / m
fit = F[(F.index >= '2018-01-01') & (F.index < '2022-01-01')].dropna(subset=['y', 'd', 'w', 'm'])
X_ = np.c_[np.ones(len(fit)), fit[['d', 'w', 'm']].values]; beta = np.linalg.lstsq(X_, fit.y.values, rcond=None)[0]
F['har'] = np.exp(beta[0] + F[['d', 'w', 'm']].values @ beta[1:])                            # forecast for the NEXT session, known at today's close
F['harrel'] = F.har / F.groupby('tk').har.transform(lambda s_: s_.rolling(252, min_periods=120).median())
F = F.reset_index().rename(columns={'index': 'date', 'Date': 'date'})
F['date'] = pd.to_datetime(F['date']); F['nxt'] = F.groupby('tk').date.shift(-1)                # the session these features are used for
FE = F.dropna(subset=['nxt']).set_index(['tk', 'nxt'])[['yz', 'har', 'harrel']]
T0['day'] = T0.entry.dt.normalize(); T0 = T0.join(FE, on=['tk', 'day']); base_T0 = T0.copy()
def order(kind):
    X = base_T0.copy(); g = X.groupby('entry')
    if kind == 'ratio': X['_k'] = -X.ratio
    elif kind in ('yz', 'har', 'harrel'): X['_k'] = -X[kind].fillna(-1)
    elif kind == 'ratio+har': X['_k'] = g.ratio.rank(ascending=False) + g.har.rank(ascending=False, na_option='bottom')
    elif kind == 'ratio+harrel': X['_k'] = g.ratio.rank(ascending=False) + g.harrel.rank(ascending=False, na_option='bottom')
    return X.sort_values(['entry', '_k', 'tk'], kind='stable')
def st2(s):
    out = []
    for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')):
        z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25
        out.append(f'{e.iloc[-1] ** (1 / y) - 1:6.1%} / DD {(e / e.cummax() - 1).min():6.1%}')
    e = (1 + s).cumprod(); return ' | '.join(out) + f' | all {e.iloc[-1] ** (1 / ((s.index[-1] - s.index[0]).days / 365.25)) - 1:6.1%}'
if __name__ == '__main__':
    print(f'features found for {T0.har.notna().mean():.0%} of {len(T0)} candidate signals; HAR betas {np.round(beta, 3)}', flush=True)
    print('\n=== pick order (6 slots, stand-in path) | CAGR / max DD 2019-22 (CHOOSE) | 2023-26 (JUDGE) | all ===', flush=True); R = {}
    for kind in ('ratio', 'yz', 'har', 'harrel', 'ratio+har', 'ratio+harrel'):
        T0 = order(kind); tk_, _ = taken(); s_, _ = daily(tk_, 6); R[kind] = s_; print(f'  {kind:<14} {st2(s_)}', flush=True)
    T0 = base_T0; q = T0[T0.day < '2023-01-01'].yz.quantile([1 / 3, 2 / 3]).values; T0['yzt'] = np.where(T0.yz < q[0], 'low', np.where(T0.yz < q[1], 'mid', 'high'))
    print('\n  all candidates by YZ20 third: ' + ' | '.join(f"{k}: avg {g.ret.mean():+.2%}, avg |move| {g.ret.abs().mean():.2%}, win {np.mean(g.ret > 0):.0%}" for k, g in T0.groupby('yzt')), flush=True)
    g = T0.groupby('entry'); T0['n'] = g.tk.transform('size')
    for f in ('ratio', 'har', 'harrel'):
        T0['rk'] = g[f].rank(ascending=False, method='first', na_option='bottom'); X = T0[T0.n >= 7]
        print(f'  crowded bars (>= 7 flips), avg return by {f} rank: ' + ' '.join(f'{int(k)}:{v:+.2%}' for k, v in X.groupby(X.rk.clip(upper=8)).ret.mean().items()), flush=True)
    O = pd.read_csv('/home/claude/fartpie/live_data/optvol_ratio_live.csv', parse_dates=['date']); O['dr'] = O.opt_contracts * 100 / O.sh
    O['nxt'] = O.sort_values('date').groupby('tk').date.shift(-1); DR = O.dropna(subset=['nxt']).set_index(['tk', 'nxt']).dr
    W = base_T0[(base_T0.day >= '2025-07-15')].copy(); W['dr'] = [DR.get((t, d), np.nan) for t, d in zip(W.tk, W.day)]
    print(f'\n=== Jul 15 2025 - Sep 2026 only: daily options ratio (prior day) found for {W.dr.notna().mean():.0%} of {len(W)} signals ===', flush=True)
    for lab, key in (('252-day ratio (live)', -W.ratio), ('daily ratio, yesterday', -W.dr.fillna(-1)), ('HAR', -W.har.fillna(-1))):
        T0 = W.assign(_k=key.values).sort_values(['entry', '_k', 'tk'], kind='stable'); tk_, _ = taken(); s_, _ = daily(tk_, 6); s_ = s_[s_.index >= '2025-07-15']
        e = (1 + s_).cumprod(); print(f'  {lab:<24} return {e.iloc[-1] - 1:+6.1%}, max DD {(e / e.cummax() - 1).min():6.1%}', flush=True)
