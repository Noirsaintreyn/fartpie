"""Nov 8 2026: MOMENTUM and the YANG-ZHANG LADDER on the prop trades and every flip. DECLARED BEFORE RESULTS.
Momentum at the prior close, in the trade's direction (with = the trade goes the way the market has been going): QQQ 20-day return, 63-day return, close vs 50-day and 200-day
  average; prior-day return; today's gap; intraday drift (entry vs today's open, in 20-day ADR, thirds).
The ladder now in the prop system: 10 micros, 5 if ONE flag, 3 if BOTH: YZH = prior-day Yang-Zhang volatility in the top third of its trailing year, VWAPX = entry stretched
  past session VWAP in the trade's direction (top third of 2018-21). Per trade: average and the share of trades that take the full 50 stop.
Outcomes, NQ points per trade: prop trades (Rule A, 75 / 50 noon, 1 contract) and every RTH flip (TH = held to vector exit, F75 = 75 / 50). Periods 18-21 / 2022 ... 2026."""
_l = open('xs/nq_location_gates.py').read(); exec(_l[:_l.rindex("if __name__ == '__main__':")])
dqc = Hq.groupby('day').close.last(); DM = pd.DataFrame({'m20': dqc.pct_change(20), 'm63': dqc.pct_change(63), 'a50': dqc / dqc.rolling(50).mean() - 1, 'a200': dqc / dqc.rolling(200).mean() - 1}).shift(1)
def mom(days_, dirs):
    M = DM.reindex(pd.DatetimeIndex(days_).normalize()); s = lambda c: pd.Series(np.where(M[c].isna(), None, np.where(np.sign(M[c].values) * dirs > 0, 'with', 'against')))
    return pd.DataFrame({'M20': s('m20'), 'M63': s('m63'), 'MA50': s('a50'), 'MA200': s('a200')})
XF = pd.read_parquet('/tmp/claude-0/ml/nq_flip_features_opt.parquet'); XF['t'] = Hq.dt.values[XF.j.values]; XF = XF.reset_index(drop=True); XF['per'] = perof(XF.day)
XF = pd.concat([XF, mom(XF.day, XF.dir.values)], axis=1)
KEY = {(np.datetime64(t, 'ns'), int(d)): i for i, (t, d) in enumerate(zip(XF.t.values, XF.dir.values))}
T = XA.loc[nets.index, ['day', 'j', 'dir']].copy(); T['pts'] = nets.values; T = T.reset_index().rename(columns={'index': 'xa'}); T['per'] = perof(T.day)
ix = [KEY.get((np.datetime64(t5[int(j)], 'ns'), int(d))) for j, d in zip(T.j.values, T.dir.values)]
for c in ('prev', 'gap', 'drift'): T[c] = [XF.at[i, c] if i is not None else np.nan for i in ix]
T = pd.concat([T, mom(T.day, T.dir.values)], axis=1)
T['YZH'] = np.array([bool(YZF.iloc[dpos[pd.Timestamp(d)]]) if pd.Timestamp(d) in dpos else False for d in T.day]); T['VWAPX'] = X.loc[T.xa, 'VWAPX'].values.astype(bool)
T['ladder'] = np.select([~T.YZH & ~T.VWAPX, T.YZH & T.VWAPX], ['10 micros (no flag)', '3 micros (both)'], '5 micros (one flag)'); T['stop'] = T.pts <= -50.5
sgn = lambda s: pd.Series(np.where(s.isna(), None, np.where(s > 0, 'with', 'against')), index=s.index)
if __name__ == '__main__':
    print(f'prop trades {len(T)}, matched to flip features {np.mean([i is not None for i in ix]):.0%}; periods ' + ' '.join(f'{p:>6}' for p in PERS), flush=True)
    RES = []
    for Z, nm, cols in ((T, 'PROP TRADES (Rule A, 75/50 noon)', ['pts']), (XF, 'EVERY RTH FLIP', ['th', 'f75'])):
        print(f'\n{nm}', flush=True)
        for col in cols:
            RES += [table(Z, f'20-day momentum', Z.M20, col), table(Z, '63-day momentum', Z.M63, col), table(Z, 'vs 50-day average', Z.MA50, col), table(Z, 'vs 200-day average', Z.MA200, col),
                    table(Z, 'prior-day return', sgn(Z.prev), col), table(Z, "today's gap", sgn(Z.gap), col), table(Z, 'intraday drift (thirds)', T3(Z.drift), col)]
    print('\nYANG-ZHANG LADDER on the prop trades (1 contract): average | full-stop share, by period', flush=True)
    for c in ('YZH', 'VWAPX', 'ladder'):
        for v, g in T.groupby(c):
            print(f'  {c:<6} {str(v):<22} n{len(g):>4} avg {g.pts.mean():+6.1f} stops {g.stop.mean():4.0%} | ' + ' '.join(f'{g[g.per == p].pts.mean():+6.1f}/{g[g.per == p].stop.mean():3.0%}' if (g.per == p).sum() >= 5 else '      .    ' for p in PERS), flush=True)
    print('\nSUMMARY (* = best beats worst in every period with 8+ trades)', flush=True)
    for r_ in [x for x in RES if x]: print(f'  {"*" if r_[3] else " "} {r_[0]:<24} {r_[1]:<4} spread {r_[2]:+6.1f}  best {r_[4]:<12} worst {r_[5]}', flush=True)
