"""Nov 5 2026: do Rule A trades do better when the OVERNIGHT (Globex) vector agrees? Only data with overnight bars: MNQ 1-minute Dec 2024 - Nov 2025 (15m Globex vector)
and yfinance NQ=F 1h, Nov 2024 - Nov 2026 (1h Globex vector). Per trade (1 lot, 75 / 50) agree vs disagree, and what x1.5 on agree would do to the per-trade numbers.
No prop sim: two years of overlap is too short for eval / funded rates."""
_rt = open('xs/nq_reach_tp.py').read(); exec(_rt[:_rt.rindex("if __name__ == '__main__':")])
full = seq('T75'); nets = pd.Series({i: v[0] - 1.0 for i, v in full.items()})
R = XA.loc[nets.index, ['day', 'j', 'dir']].copy(); R['net'] = nets; R['t'] = pd.DatetimeIndex(t5[R.j.values])          # entry bar time (next 5m open after the flip)
M = pd.read_parquet('/mnt/user-data/uploads/fartpie/mnq_orderflow_1min.parquet'); M['dt'] = pd.to_datetime(M.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
M = M.dropna(subset=['last_price']).sort_values('dt').drop_duplicates('dt'); c = M.last_price.values; o = np.r_[c[0], c[:-1]]; vw = M.vwap.fillna(M.last_price).values
M['open'], M['high'], M['low'], M['close'] = o, np.maximum.reduce([o, c, vw]), np.minimum.reduce([o, c, vw]), c
mm = M.dt.dt.hour * 60 + M.dt.dt.minute; G = M[~((mm >= 1020) & (mm < 1080)) & (M.dt.dt.dayofweek < 5)].set_index('dt')[['open', 'high', 'low', 'close']]
G = G.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna().reset_index(); G['reg'] = np.asarray(vec(G[['open', 'high', 'low', 'close']])); G['end'] = G.dt + pd.Timedelta('15min')
N1 = pd.read_parquet('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/yf1h/NQ_F.parquet'); N1 = N1[N1.index.dayofweek < 5]
N1 = N1.reset_index().rename(columns={N1.index.name or 'index': 'dt'}); N1.columns = ['dt'] + list(N1.columns[1:]); N1['reg'] = np.asarray(vec(N1[['open', 'high', 'low', 'close']])); N1['end'] = N1.dt + pd.Timedelta('1h')
def state(B, t): i = np.searchsorted(B.end.values, np.asarray(t, dtype='datetime64[ns]'), side='right') - 1; return np.where(i >= 0, B.reg.values[np.clip(i, 0, None)], 0)
for nm, B, a, b in (('15m Globex (MNQ 1m)', G, G.dt.min(), G.dt.max()), ('1h Globex (NQ=F)', N1, N1.dt.min(), N1.dt.max())):
    S = R[(R.t > a + pd.Timedelta(days=5)) & (R.t < b)].copy(); S['ag'] = state(B, S.t) == S.dir
    for lab, g in (('agrees', S[S.ag]), ('disagrees', S[~S.ag])):
        print(f'  {nm:<22} {lab:<9} n{len(g):>3} win {g.net.gt(0).mean():4.0%} avg {g.net.mean():+6.1f} pts | total {g.net.sum():+6.0f}', flush=True)
    up = np.where(S.ag, 1.5, 1.0); print(f'  {nm:<22} all n{len(S)} avg {S.net.mean():+.1f} | x1.5 on agree: total {np.sum(S.net * up):+.0f} vs {S.net.sum():+.0f}, worst 5-trade run {pd.Series(S.net.values * up).rolling(5).sum().min():+.0f} vs {S.net.rolling(5).sum().min():+.0f}', flush=True)
