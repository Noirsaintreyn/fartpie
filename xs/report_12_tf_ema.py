"""Nov 12 2026: stack = daily vector bull (D) + Kalman trend up (K) on $1-2 vol>=1.5x flips, then EMA alignment on 1h / 4h / daily bars. DECLARED BEFORE RESULTS.
EMA context at the flip (close vs EMA of that timeframe's last COMPLETED bar): EMA50 and EMA200 on 1h, 4h, daily (4h/daily built from the 1h cache; daily EMA200 needs 200 sessions so it is n/a for early trades - n shown).
'count' = how many of the three EMA200s (1h,4h,D) the flip closes above. 15m: Yahoo keeps ~60 days of 15m bars, so not testable here. Same split / costs / 8-slot book as before; nothing fitted."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.'); sys.path.insert(0, 'xs')
import midsmall_screener as M
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb')); COST = .004
T = pd.read_pickle(SC + 'smooth_T.pkl')
def htf(Y, tf):
    day = Y.index.normalize(); g = [day, (Y.index.hour >= 13).astype(int)] if tf == '4h' else [day]; G = Y.groupby(g)
    cl = G.close.last(); last = G.apply(lambda x: x.index[-1]); end = pd.DatetimeIndex(last.values) + pd.to_timedelta(np.where(pd.DatetimeIndex(last.values).hour == 15, 30, 60), unit='m')
    s = pd.Series(cl.values); return pd.DataFrame({'end': end, 'e50': s.ewm(span=50, adjust=False).mean().where(s.expanding().count() >= 50).values, 'e200': s.ewm(span=200, adjust=False).mean().where(s.expanding().count() >= 200).values}).sort_values('end')
rows = []
for tk, Z in T.groupby('tk'):
    Y = M.closed_only(data[tk].dropna(subset=['open', 'close'])); H = {'4h': htf(Y, '4h'), 'd': htf(Y, '1d')}; q = pd.DataFrame({'t': Z.et.values})
    r = Z.copy()
    for k, h in H.items():
        m = pd.merge_asof(q.sort_values('t'), h, left_on='t', right_on='end', direction='backward'); m.index = Z.sort_values('et').index
        r.loc[m.index, f'{k}50'] = r.loc[m.index, 'sig'] > m.e50; r.loc[m.index, f'{k}200'] = np.where(m.e200.isna(), np.nan, r.loc[m.index, 'sig'] > m.e200)
    rows.append(r)
T = pd.concat(rows); T['a200'] = T.a200.astype(float)
def book(Z, K=8):
    eq = 1.0; op = []; curve = []
    for r in sorted(Z.to_dict('records'), key=lambda r: (r['et'], -r['vr'])):
        still = []
        for p in op:
            if p['xt'] <= r['et']: eq += p['sz'] * (p['ret'] - COST); curve.append((p['xt'], eq))
            else: still.append(p)
        op = still
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r)
    for p in op: eq += p['sz'] * (p['ret'] - COST); curve.append((p['xt'], eq))
    return pd.Series(dict(curve)).sort_index()
def rep(nm, Z):
    if len(Z) < 15: print(f'  {nm:<34} {len(Z)} trades (too few)'); return
    r = Z.ret - COST; a, b = r[Z.et < '2025-07-01'], r[Z.et >= '2025-07-01']; s = book(Z); yrs = (s.index[-1] - Z.et.min()).days / 365.25; cg = s.iloc[-1] ** (1 / max(yrs, .5)) - 1
    print(f'  {nm:<34} {len(Z):>4} tr {len(Z)/yrs:>3.0f}/yr | mean {r.mean()*100:+.2f}% med {r.median()*100:+.1f}% win {(r>0).mean():.0%} p10 {r.quantile(.1)*100:+.1f}% | train {a.mean()*100:+.2f}% ({len(a)}) test {b.mean()*100:+.2f}% ({len(b)}) | book {cg:+.0%}/yr DD {(s/s.cummax()-1).min():.0%}', flush=True)
print('== the stack (B = $1-2, flip vol >= 1.5x)'); rep('B', T); rep('B + daily vector bull (D)', T[T.ad]); rep('B + Kalman up (K)', T[T.kal]); rep('B + D + K  (A+)', T[T.ad & T.kal]); rep('B + D + K + low-vol day', T[T.ad & T.kal & T.lv.fillna(False).astype(bool)])
print('== EMA alignment by timeframe, on B (above vs not above):')
for col, nm in (('a200', '1h EMA200'), ('4h50', '4h EMA50'), ('4h200', '4h EMA200'), ('d50', 'daily EMA50'), ('d200', 'daily EMA200')):
    Z = T[T[col].notna()]; print(f' {nm}: n with data {len(Z)}'); rep('   above', Z[Z[col] == 1]); rep('   not above', Z[Z[col] == 0])
T['cnt'] = T[['a200', '4h200', 'd200']].sum(axis=1, min_count=3); print('== count of EMA200s (1h,4h,D) the flip is above, on B (rows with all three)')
for k in (0, 1, 2, 3): rep(f'   {k} of 3', T[T.cnt == k])
print('== on A+ (D + K): add EMA alignment')
P = T[T.ad & T.kal]; rep('A+ base', P)
for col, nm in (('a200', '1h EMA200'), ('4h200', '4h EMA200'), ('4h50', '4h EMA50'), ('d50', 'daily EMA50')):
    rep(f'  & above {nm}', P[P[col] == 1]); rep(f'  & NOT above {nm}', P[P[col] == 0])
