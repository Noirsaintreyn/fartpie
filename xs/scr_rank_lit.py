"""Nov 9 2026: RANKING THE SCREENER'S SIMULTANEOUS FLIPS WITH WHAT THE LITERATURE / PRACTITIONERS SAY PREDICTS SHORT-TERM RETURNS (Rey). DECLARED BEFORE RESULTS.
Same slot sim as xs/scr_slots.py / xs/scr_rank_vol.py (6 slots, stand-in daily path); every feature known before the entry. Direction declared up front:
  RVOL    prior-day volume / its 50-day average, HIGHER first      (high-volume return premium, Gervais-Kaniel-Mingelgrin; 'stocks in play', Zarattini-Barbon-Aziz)
  GAP     today's open vs prior close, HIGHER first                 (opening strength)
  HI52    prior close / 252-day high, HIGHER first                  (52-week-high momentum, George-Hwang)
  RS20    20-day return minus SPY's, HIGHER first                   (relative strength)
  REV1    prior-day return, LOWER first                             (short-term reversal)
  IVOL    60-day residual vol vs SPY, LOWER first                   (idiosyncratic-volatility puzzle, Ang-Hodrick-Xing-Zhang)
  options, only for the 23 names with ThetaData history (data/theta_compact), universe restricted to them for these rows:
  PCR     prior-day put / call volume, LOWER first                  (Pan-Poteshman)
  VRP     realized 20-day vol minus ATM implied vol (straddle approx, ~30 DTE), HIGHER first   (Bali-Hovakimian)
Each as the pick order alone and as rank-sum with the live options ratio. CHOSEN on 2019-22, JUDGED on 2023-26 against the live ratio order."""
import os, numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/scr_rank_vol.py').read(); exec(_s[:_s.index("\nif __name__ == '__main__':")])
CV = '/home/claude/fartpie/data/yahoo/daily_v'; os.makedirs(CV, exist_ok=True)
def dv(tk):
    f = f'{CV}/{tk}.parquet'
    if os.path.exists(f): return pd.read_parquet(f)
    Y = yf.download(YSYM.get(tk, tk), start='2017-01-01', end='2026-10-25', interval='1d', auto_adjust=True, progress=False)
    if isinstance(Y.columns, pd.MultiIndex): Y.columns = Y.columns.get_level_values(0)
    Y = Y.rename(columns=str.lower)[['open', 'high', 'low', 'close', 'volume']].dropna(); Y.index = pd.DatetimeIndex(Y.index).tz_localize(None).normalize(); Y.to_parquet(f); return Y
SPYd = dv('SPY'); spr = SPYd.close.pct_change()
G = []
for tk in TKS:
    Y = dv(tk); r = Y.close.pct_change(); z = pd.concat([r, spr], axis=1, keys=['r', 's']).dropna()
    beta = z.r.rolling(60).cov(z.s) / z.s.rolling(60).var(); res = z.r - beta * z.s
    X = pd.DataFrame({'tk': tk, 'rvol': Y.volume / Y.volume.rolling(50).mean(), 'hi52': Y.close / Y.close.rolling(252, min_periods=200).max(),
                      'rs20': Y.close.pct_change(20) - SPYd.close.reindex(Y.index).pct_change(20), 'rev1': r, 'ivol': res.rolling(60).std().reindex(Y.index),
                      'rv20': np.log(Y.close).diff().rolling(20).std() * np.sqrt(252), 'open_nx': Y.open.shift(-1), 'close': Y.close}, index=Y.index)
    X['gap'] = X.open_nx / X.close - 1; G.append(X)
G = pd.concat(G).reset_index().rename(columns={'index': 'date', 'Date': 'date'}); G['date'] = pd.to_datetime(G['date']); G['nxt'] = G.groupby('tk').date.shift(-1)
GE = G.dropna(subset=['nxt']).set_index(['tk', 'nxt'])[['rvol', 'hi52', 'rs20', 'rev1', 'ivol', 'gap', 'rv20']]
OPT = []
for tk in TKS:
    f = f'/mnt/user-data/uploads/fartpie/data/theta_compact/{tk}.parquet'
    if not os.path.exists(f): continue
    Q = pd.read_parquet(f); Q = Q[(Q.bid > 0) & (Q.ask > 0)]; pv = Q.groupby(['d', 'r']).vol.sum().unstack()
    pcr = (pv.get(-1) / pv.get(1)).rename('pcr')
    A = Q[(Q.dte >= 20) & (Q.dte <= 45)].copy(); A['mny'] = (A.K / A.S - 1).abs(); A['mid'] = (A.bid + A.ask) / 2
    best = A.loc[A.groupby(['d', 'r']).mny.idxmin()]; st_ = best.pivot_table(index='d', columns='r', values=['mid', 'S', 'dte'])
    iv = ((st_['mid'][1] + st_['mid'][-1]) / st_['S'][1] * np.sqrt(2 * np.pi / (st_['dte'][1] / 365)) / 2).rename('iv')
    O_ = pd.concat([pcr, iv], axis=1); O_['tk'] = tk; OPT.append(O_.reset_index().rename(columns={'d': 'date'}))
OPT = pd.concat(OPT); OPT['date'] = pd.to_datetime(OPT['date']); OPT = OPT.sort_values(['tk', 'date']); OPT['nxt'] = OPT.groupby('tk').date.shift(-1)
OE = OPT.dropna(subset=['nxt']).set_index(['tk', 'nxt'])[['pcr', 'iv']]; OTK = set(OPT.tk)
T0 = base_T0.join(GE, on=['tk', 'day']).join(OE, on=['tk', 'day']); T0['vrp'] = T0.rv20 - T0.iv; base2 = T0.copy()
DIR = {'rvol': -1, 'gap': -1, 'hi52': -1, 'rs20': -1, 'rev1': 1, 'ivol': 1, 'pcr': 1, 'vrp': -1}
def order2(X, f, mix):
    X = X.copy(); g = X.groupby('entry'); fr = g[f].rank(ascending=DIR[f] == 1, na_option='bottom')
    X['_k'] = (g.ratio.rank(ascending=False) + fr) if mix else fr
    return X.sort_values(['entry', '_k', 'tk'], kind='stable')
if __name__ == '__main__':
    print(f'features found: ' + ', '.join(f'{f} {T0[f].notna().mean():.0%}' for f in DIR) + f' | options names: {len(OTK)}', flush=True); RES = []
    for uni, X0, feats_ in (('all 54 names', base2, ('rvol', 'gap', 'hi52', 'rs20', 'rev1', 'ivol')), (f'{len(OTK)} names with options data', base2[base2.tk.isin(OTK)], ('pcr', 'vrp', 'rvol', 'hi52'))):
        print(f'\n=== {uni}: CAGR / max DD 2019-22 (CHOOSE) | 2023-26 (JUDGE) | all ===', flush=True)
        T0 = X0.assign(_k=-X0.ratio).sort_values(['entry', '_k', 'tk'], kind='stable'); tk_, _ = taken(); s_, _ = daily(tk_, 6); base_s = st2(s_); print(f'  {"live ratio order":<26} {base_s}', flush=True)
        b_a = (1 + s_[s_.index < '2023-01-01']).prod()
        for f in feats_:
            for mix in (False, True):
                T0 = order2(X0, f, mix); tk_, _ = taken(); s_, _ = daily(tk_, 6); lab = f"{f}{' + ratio' if mix else ' alone'}"
                RES.append((uni, lab, (1 + s_[s_.index < '2023-01-01']).prod(), s_)); print(f'  {lab:<26} {st2(s_)}', flush=True)
        best = max([r for r in RES if r[0] == uni], key=lambda r: r[2]); print(f'  -> chosen on 2019-22: {best[1]} ({"beats" if best[2] > b_a else "does not beat"} the live order there) | 2019-22 / 2023-26: {st2(best[3])}', flush=True)
