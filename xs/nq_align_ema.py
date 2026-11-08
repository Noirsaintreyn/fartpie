"""Nov 8 2026: VECTOR ALIGNMENT (NQ vs QQQ, and across timeframes) and EMA FLIPS. DECLARED BEFORE RESULTS.
Every RTH 15m QQQ vector flip 2018 - Oct 2026 (held to the 15m vector exit = TH, and 75 / 50 = F75; NQ points, 1 contract, after cost), plus the Rule A morning subset.
Alignment at entry (known then), 'agrees' = same direction as the flip:
  V5m   QQQ 5m vector          V1h   QQQ 1h vector (last closed hour)        VD   QQQ daily vector (yesterday)
  NQ1h  NQ futures 1h vector incl. overnight (Jun 2024 - Nov 2026 only)       ALL3  5m + 1h + daily all agree
EMA (QQQ RTH bars, carried across days): 15m EMA 9 vs 21 agrees; 15m close vs EMA 50; 1h close vs EMA 20; daily close vs EMA 20 (yesterday);
  EMA FLIP CONFLUENCE: the 15m EMA 9/21 crossed the same way within the last 2 bars (the vector flip and an EMA flip together).
EMA FLIP AS ITS OWN SIGNAL: 15m EMA 9/21 cross, enter at the next 5m open, exit at the opposite cross (or stop 50, flat 15:55, <= 4 a day), vs the vector flip held the same way.
A label counts when its best bucket beats its worst in every period with 8+ flips (18-21 / 22-23 / 24-25 / 2026)."""
_f = open('xs/nq_floor.py').read(); exec(_f[:_f.rindex("if __name__ == '__main__':")])
PER = ['18-21', '22-23', '24-25', '2026']; X['per'] = np.select([X.day < '2022-01-01', X.day < '2024-01-01', X.day < '2026-01-01'], PER[:3], '2026')
tE = pd.DatetimeIndex(t5[X.j.values])
def state_at(ends, vals, times):
    s = pd.Series(vals, index=pd.DatetimeIndex(ends)).sort_index(); s = s[~s.index.duplicated(keep='last')]; i = s.index.searchsorted(times, side='right') - 1
    return np.where(i >= 0, s.values[np.clip(i, 0, None)], np.nan)
r5 = np.asarray(vec(H[['open', 'high', 'low', 'close']])); X['V5m'] = state_at(H.dt.values + np.timedelta64(5, 'm'), r5, tE) * X.dir.values
Q1 = load(glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/QQQ_1h_*.parquet')[0]); Q1 = Q1[Q1.dt >= '2017-06-01'].reset_index(drop=True)
Q1['reg'] = np.asarray(vec(Q1[['open', 'high', 'low', 'close']])); X['V1h'] = state_at(Q1.dt.values + np.timedelta64(60, 'm'), Q1.reg.values, tE) * X.dir.values
Dd = H.groupby('day').agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')); Dd['reg'] = np.asarray(vec(Dd[['open', 'high', 'low', 'close']]))
X['VD'] = state_at(Dd.index.values + np.timedelta64(16, 'h'), Dd.reg.values, tE) * X.dir.values
N1 = pd.read_parquet('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/yf1h/NQ_F.parquet'); N1 = N1[N1.index.dayofweek < 5].copy()
N1['reg'] = np.asarray(vec(N1[['open', 'high', 'low', 'close']])); nq1 = state_at(N1.index.values + np.timedelta64(60, 'm'), N1.reg.values, tE)
X['NQ1h'] = np.where(tE >= N1.index.min() + pd.Timedelta(days=30), nq1 * X.dir.values, np.nan)
X['ALL3'] = np.where((X.V5m > 0) & (X.V1h > 0) & (X.VD > 0), 1.0, -1.0)
ema = lambda s, n: s.ewm(span=n, adjust=False).mean()
B['e9'], B['e21'], B['e50'] = ema(B.close, 9), ema(B.close, 21), ema(B.close, 50); B['x'] = np.sign(B.e9 - B.e21); B['flip'] = B.x != B.x.shift(1)
X['E15'] = state_at(B.end.values, B.x.values, tE) * X.dir.values; X['E50'] = state_at(B.end.values, np.sign(B.close - B.e50).values, tE) * X.dir.values
rec = B.x.where(B.flip).ffill(); age = B.groupby(B.flip.cumsum()).cumcount(); conf = np.where(age <= 1, B.x, 0)
X['EFLIP'] = state_at(B.end.values, conf, tE) * X.dir.values
Q1['e20'] = ema(Q1.close, 20); X['E1h'] = state_at(Q1.dt.values + np.timedelta64(60, 'm'), np.sign(Q1.close - Q1.e20).values, tE) * X.dir.values
Dd['e20'] = ema(Dd.close, 20); X['ED20'] = state_at(Dd.index.values + np.timedelta64(16, 'h'), np.sign(Dd.close - Dd.e20).values, tE) * X.dir.values
def table(Z, name, lab, col):
    out = []
    for v in sorted(lab.dropna().unique()):
        g = Z[lab == v]; out.append((v, len(g), g[col].mean(), [g[g.per == p][col].mean() if (g.per == p).sum() >= 8 else np.nan for p in PER]))
    if len(out) < 2: return None
    out.sort(key=lambda o: -o[2]); best, worst = out[0], out[-1]; ok = [(b, w) for b, w in zip(best[3], worst[3]) if np.isfinite(b) and np.isfinite(w)]
    cons = len(ok) >= 3 and all(b > w for b, w in ok)
    print(f'  -- {name} [{col}] spread {best[2] - worst[2]:+.1f}{"  * every period" if cons else ""}', flush=True)
    for v, n, m, pp in out: print(f'    {str(v):<10} n{n:>4} {m:+6.1f} | ' + ' '.join(f'{x:+6.1f}' if np.isfinite(x) else '     .' for x in pp), flush=True)
    return (name, col, best[2] - worst[2], cons, best[0])
lab = lambda c: X[c].map({1.0: 'agrees', -1.0: 'against', 0.0: 'none'})
def ema_strategy(S=50.0):
    """15m EMA 9/21 cross: enter next 5m open, exit at the opposite cross (next open), stop S, flat 15:55, <= 4 a day, no overlap"""
    xs = pd.Series(B.x.values, index=B.end.values); fl = B[B.flip & (B.me >= 585) & (B.me <= 930)]; pos5 = {t: i for i, t in enumerate(t5)}; rows = []; busy = {}
    for r in fl.itertuples():
        j = pos5.get(np.datetime64(r.end, 'ns'))
        if j is None or d5[j] != pd.Timestamp(r.end).normalize(): continue
        d = int(r.x); day = d5[j]; b = busy.get(day, (-1, 0))
        if j <= b[0] or b[1] >= 4: continue
        e = o5[j]; kk = k5[j]; out = None
        for q in range(j, len(H)):
            if d5[q] != day or m5[q] >= 955: out = ((c5[q - 1] - e) * d * kk, q - 1); break
            if ((l5[q] if d > 0 else h5[q]) - e) * d * kk <= -S: out = (-S, q); break
            if m5[q] % 15 == 10 and q + 1 < len(H) and d5[q + 1] == day:
                st = xs.get(t5[q] + np.timedelta64(5, 'm'))
                if st is not None and st == -d: out = ((o5[q + 1] - e) * d * kk, q); break
        if out is None: continue
        rows.append((day, out[0] - 1.0)); busy[day] = (out[1], b[1] + 1)
    return pd.DataFrame(rows, columns=['day', 'pts'])
if __name__ == '__main__':
    print(f'flips {len(X)}; Rule A morning {int(((X.ruleA == 1) & (X.me <= 690)).sum())}; NQ 1h coverage {X.NQ1h.notna().sum()} flips', flush=True)
    print('periods: ' + ' '.join(f'{p:>6}' for p in PER), flush=True); RES = []
    for nm, Z in (('EVERY RTH FLIP', X), ('RULE A MORNING FLIPS', X[(X.ruleA == 1) & (X.me <= 690)])):
        print(f'\n{nm}', flush=True)
        for col in ('th', 'f75'):
            for c in ('V5m', 'V1h', 'VD', 'ALL3', 'NQ1h', 'E15', 'E50', 'EFLIP', 'E1h', 'ED20'):
                RES.append(table(Z, f'{nm[:5]} {c}', Z[c].map({1.0: 'agrees', -1.0: 'against', 0.0: 'none'}), col))
    print('\nSUMMARY (* = best beats worst in every period with 8+ flips, at least 3 periods)', flush=True)
    for r in [r for r in RES if r]: print(f'  {"*" if r[3] else " "} {r[0]:<12} {r[1]:<4} spread {r[2]:+6.1f}  best {r[4]}', flush=True)
    print('\nEMA 9/21 FLIP AS ITS OWN SIGNAL vs the vector flip (held to its opposite flip, stop 50, flat 15:55, <= 4 a day), 1 contract', flush=True)
    E = ema_strategy(); EP = E.groupby(pd.DatetimeIndex(E.day)).pts.sum().reindex(AD).fillna(0.0)
    VT = []
    for day, g in X.groupby('day', sort=True):
        busy = -1; n = 0
        for r in g.itertuples():
            if n >= 4 or r.j <= busy: continue
            VT.append((day, r.th)); busy = r.q; n += 1
    VT = pd.DataFrame(VT, columns=['day', 'pts']); VP = VT.groupby(pd.DatetimeIndex(VT.day)).pts.sum().reindex(AD).fillna(0.0)
    for nm, T, P in (('EMA 9/21 flip', E, EP), ('vector flip', VT, VP)):
        cum = P.cumsum(); ty = pd.DatetimeIndex(T.day).year
        print(f'  {nm:<14} trades {len(T):5d} ({len(T) / len(AD):.2f}/day) win {np.mean(T.pts > 0):3.0%} avg {T.pts.mean():+5.2f} | {P.mean():+5.2f} pts/day, Sharpe {P.mean() / P.std() * np.sqrt(252):5.2f}, '
              f'max DD {(cum - cum.cummax()).min():+6.0f} | 2018-21 {T.pts[ty < 2022].mean():+5.2f} / 2022-26 {T.pts[ty >= 2022].mean():+5.2f} per trade | corr with vector {np.corrcoef(P, VP)[0, 1]:+.2f}', flush=True)
    X.to_parquet('/tmp/claude-0/ml/nq_align_ema_flips.parquet')
