"""Nov 8 2026: SCREENER GATES on the honest live harness (point-in-time S&P 500, options ratio >= 1.5, sticky 1h, LOCK, 6 slots, ratio tie-break, 8% breaker,
no buys after 14:00, daily mark-to-market). The LIVE row must reproduce +51.7% / +29.1% first. DECLARED BEFORE RESULTS.
  BREADTH  share of the eligible universe (the 54 names) whose 15m vector is bull at the bar before the entry; skip entries when breadth < 50% / < 65% / < 80%
           (80% is the 'cohesion' gate from the pasted note)
  RVOL     the stock's 15m bar before the entry, volume vs the same 15-minute slot's 20-day average; skip when RVOL < 1.5 / < 2.5
  QQQVOL   QQQ Yang-Zhang 20d vol percentile (1 year, prior close): skip when below 2/3, and the 'top 5% days only' gate (>= 95th percentile)
  COMBO    breadth >= 65% and RVOL >= 1.5"""
_aw = open('xs/screener_anchor_wide.py').read(); exec(_aw[:_aw.index("V = {'LIVE")])
exec(open('/home/claude/fartpie/xs/ah_sticky.py').read().split('def run(')[0])
import sys as _s; _s.path.insert(0, '/home/claude/fartpie/tools'); import account_signals as AS
REG, VOLB = {}, {}
for tk in sorted(T0.tk.unique()):
    Hb = bars(glob.glob(f'{ROOT}/lse_clean/{tk}_15m_2018-01-01_*.parquet')[0], 960)
    REG[tk] = pd.Series(Hb.reg.values, index=pd.DatetimeIndex(Hb.dt.values)); VOLB[tk] = Hb.set_index('dt').volume if 'volume' in Hb else None
R = pd.DataFrame(REG).sort_index().ffill(); BREADTH = (R == 1).sum(axis=1) / R.notna().sum(axis=1)
def prev_bar(tk_or_none, e):
    idx = R.index.searchsorted(e) - 1; return R.index[idx] if idx >= 0 else None
br, rv = [], []
for tk, e in zip(T0.tk.values, T0.entry.values):
    pb = prev_bar(None, e); br.append(BREADTH.loc[pb] if pb is not None else np.nan)
    v = VOLB.get(tk)
    if v is None or pb is None or pb not in v.index: rv.append(np.nan); continue
    slot = v.index.hour * 60 + v.index.minute; same = v[(slot == pb.hour * 60 + pb.minute) & (v.index < pb.normalize())].iloc[-20:]
    rv.append(v.loc[pb] / same.mean() if len(same) >= 5 and same.mean() > 0 else np.nan)
T0['breadth'], T0['rvol'] = br, rv
q_, y_ = AS.data('2016-06-01'); yz_, yzp_, *_ = AS.series(q_, y_)
T0['qqqvol'] = yzp_.reindex(pd.DatetimeIndex(T0.entry.dt.normalize() - pd.Timedelta(days=1)), method='ffill').values
print(f'breadth at entries: median {T0.breadth.median():.0%}, share >= 80%: {(T0.breadth >= .8).mean():.0%} | RVOL median {T0.rvol.median():.2f}, share >= 2.5: {(T0.rvol >= 2.5).mean():.0%} | QQQ vol pct >= 95%: {(T0.qqqvol >= .95).mean():.0%}', flush=True)
V = {'LIVE (should match +51.7 / +29.1)': T0}
for th in (.5, .65, .8): V[f'BREADTH >= {th:.0%}'] = T0[~(T0.breadth < th)]
for th in (1.5, 2.5): V[f'RVOL >= {th}'] = T0[~(T0.rvol < th)]
V['QQQVOL >= 2/3'] = T0[~(T0.qqqvol < 2 / 3)]; V['QQQVOL >= 95th pct (top 5% days)'] = T0[~(T0.qqqvol < .95)]
V['COMBO breadth >= 65% + RVOL >= 1.5'] = T0[~((T0.breadth < .65) | (T0.rvol < 1.5))]
print(f'\n{"6 slots, honest harness":<38}{"2019-21":>22}{"2022-26":>22}{"2026 YTD":>22}{"2019-26":>22}  taken', flush=True)
for nm, T in V.items():
    TK = taken(T)
    if len(TK) < 20: print(f'  {nm:<36} only {len(TK)} trades', flush=True); continue
    s = mtm(TK); cells = [st(s.loc[a:b]) for a, b in (('2019', '2021'), ('2022', '2026'), ('2026', '2026'), ('2019', '2026'))]
    print(f'  {nm:<36}' + ''.join(f'{c:>+8.1%} /{d:>5.0%} S{sh:4.2f}' for c, d, sh in cells) + f'{len(TK):>7,}', flush=True)
