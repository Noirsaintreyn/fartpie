"""Nov 5 2026: the anchored-volatility filter re-run on the HONEST live harness (xs/site_backtest_wide.py: point-in-time S&P 500, options ratio >= 1.5,
sticky 1h, LOCK, 6 slots, highest-ratio tie-break, 8% breaker, no buys after 14:00, daily mark-to-market, 2 bp). xs/screener_anchor.py used 40 fixed names with
no ratio gate and no breaker, which is why its base numbers (+27% / +13%) did not match the live model's (+52% / +29%). DECLARED BEFORE RESULTS.
Filter (same as live_screener.anchor_blocked): skip an entry more than k x sigma_d above the day's RTH low so far, or k x sigma_d x sqrt5 above the week's earlier RTH lows
(sigma_d = 20-day stdev of daily closes, prior close). k = 1 is the rule; k = 0.75 / 1.25 shown only as sensitivity."""
import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
ROOT = '/mnt/user-data/uploads/fartpie'; N = 6; BRK = .08; COST = 2e-4
T0 = pd.read_parquet(f'{ROOT}/xs/ext/wide_T.parquet'); T0 = T0[T0.sp & (T0.ratio >= 1.5)].copy(); T0['entry'] = pd.to_datetime(T0.entry); T0['exit'] = pd.to_datetime(T0.exit)
OP, CL, FE = {}, {}, []
for tk in sorted(T0.tk.unique()):
    H = pd.read_parquet(glob.glob(f'{ROOT}/lse_clean/{tk}_15m_2018-01-01_*.parquet')[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
    OP[tk] = H.set_index('dt').open; CL[tk] = H.set_index('dt').close.groupby(lambda x: x.normalize()).last()
    day = H.dt.dt.normalize(); cm = H.low.groupby(day).cummin(); lowD = cm.groupby(day).shift(1).fillna(np.inf)
    D = H.groupby(day).agg(l=('low', 'min'), c=('close', 'last')); sig = np.log(D.c).diff().rolling(20).std().shift(1)
    wl = D.l.groupby(D.index.to_period('W-FRI')).transform(lambda s: s.cummin().shift(1)).fillna(np.inf)
    F = pd.DataFrame({'dt': H.dt, 'px': H.open, 'lowD': lowD.values, 'lowW': wl.reindex(day).values, 'sig': sig.reindex(day).values})
    Z = T0[T0.tk == tk][['tk', 'h1', 'entry']].merge(F, left_on='entry', right_on='dt', how='left'); FE.append(Z)
FE = pd.concat(FE); ld = np.minimum(FE.lowD, FE.px); lw = np.minimum(FE.lowW, ld)
FE['usedD'] = (FE.px - ld) / (FE.px * FE.sig); FE['usedW'] = (FE.px - lw) / (FE.px * FE.sig * np.sqrt(5))
T0 = T0.merge(FE[['tk', 'h1', 'entry', 'usedD', 'usedW']], on=['tk', 'h1', 'entry'], how='left')
print(f'{len(T0)} candidate trades, {T0.tk.nunique()} names; features found {T0.usedD.notna().mean():.1%}; usedD > 1 {(T0.usedD > 1).mean():.0%}, usedW > 1 {(T0.usedW > 1).mean():.0%}', flush=True)
def taken(T):
    T = T.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable'); op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; out = []
    for e, x, r, tk in zip(T.entry.values, T.exit.values, T.ret.values, T.tk.values):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / N); hist.append((xt, real)); win.append((xt, real))
        op = [o for o in op if o > e]
        if len(op) >= N: continue
        if hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-BRK): continue
        op.append(x); pend.append((x, r)); out.append((tk, pd.Timestamp(e), pd.Timestamp(x), r))
    return out
SPY = pd.read_parquet(f'{ROOT}/etf_daily/SPY.parquet'); SPY['d'] = pd.to_datetime(SPY.datetime).dt.tz_localize(None).dt.normalize(); DAYS = SPY.drop_duplicates('d').set_index('d').index
def mtm(TK):
    D = DAYS[(DAYS >= '2019-01-01') & (DAYS <= max(t[2] for t in TK).normalize())]; Rr = pd.Series(0.0, index=D)
    for tk, e, x, r in TK:
        p0, p1 = OP[tk].get(e), OP[tk].get(x)
        if p0 is None or p1 is None or not p0 > 0: p0, p1 = 1.0, 1.0 + r
        d0, d1 = e.normalize(), x.normalize(); c = CL[tk]; mid = c[(c.index >= d0) & (c.index < d1)]
        path = pd.concat([pd.Series([p0], index=[d0 - pd.Timedelta(seconds=1)]), mid, pd.Series([p1 * (1 - COST)], index=[d1])])
        rr = path.pct_change().dropna(); rr.index = rr.index.normalize(); rr = rr.groupby(level=0).apply(lambda z: (1 + z).prod() - 1)
        Rr = Rr.add(rr.reindex(D).fillna(0) / N, fill_value=0)
    return Rr
def st(s):
    e = (1 + s).cumprod(); y = (s.index[-1] - s.index[0]).days / 365.25
    return (e.iloc[-1] ** (1 / y) - 1 if y >= 1 else e.iloc[-1] - 1), (e / e.cummax() - 1).min(), s.mean() / s.std() * 252 ** .5
V = {'LIVE (should match +51.7 / +29.1)': T0}
for k in (1.0, .75, 1.25):
    tag = '' if k == 1.0 else f' (k {k}, sensitivity)'
    V[f'SKIP-D{tag}'] = T0[~(T0.usedD > k)]; V[f'SKIP-W{tag}'] = T0[~(T0.usedW > k)]; V[f'SKIP-DW{tag}'] = T0[~((T0.usedD > k) | (T0.usedW > k))]
print(f'\n{"6 slots, honest harness":<34}{"2019-21":>22}{"2022-26":>22}{"2026 YTD":>22}{"2019-26":>22}  taken', flush=True)
for nm, T in V.items():
    TK = taken(T); s = mtm(TK)
    cells = [st(s.loc[a:b]) for a, b in (('2019', '2021'), ('2022', '2026'), ('2026', '2026'), ('2019', '2026'))]
    print(f'  {nm:<32}' + ''.join(f'{c:>+8.1%} /{d:>5.0%} S{sh:4.2f}' for c, d, sh in cells) + f'{len(TK):>7,}', flush=True)
