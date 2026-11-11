"""AMT / volume-profile concepts on the vector legs (V15a, Rule A), full history 2014-26, orderflow as support where it exists (2024-12..2025-11 only). Nov 11 2026.
Profiles from NQ 5m RTH volume spread uniformly over each bar's range in 1-pt bins (approximation of a volume profile; POC, 70% value area).
Features at entry (closed bars only): open vs prior value (above VAH / inside / below VAL), price vs prior value, 80%-rule re-entry, value migration (POC1 vs POC2), price vs developing POC / developing value, initial balance (9:30-10:30) range and break, open drive.
Aligned: +1 = supports the trade direction. Both eras; PASS = same sign, |t|>=2, n>=30 per era."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
O, N = pickle.load(open(SC + 's6.pkl', 'rb')); L = dict(zip(N, O['V2']))
X = pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet'); X = X[X.rth].copy(); X['day'] = pd.to_datetime(X.day).dt.normalize()
DAY = {d: g.sort_values('m') for d, g in X.groupby('day')}; DAYS = sorted(DAY)
def profile(g):
    hi, lo, v = g.high.values, g.low.values, g.volume.values.astype(float)
    if len(g) == 0: return None
    lb, hb = np.floor(lo).astype(int), np.floor(hi).astype(int); base = lb.min(); n = hb.max() - base + 2; d = np.zeros(n); w = v / (hb - lb + 1)
    np.add.at(d, lb - base, w); np.add.at(d, hb - base + 1, -w); a = np.cumsum(d)[:-1]; tot = a.sum()
    if tot <= 0: return None
    ip = int(np.argmax(a)); lo_i = hi_i = ip; acc = a[ip]
    while acc < .7 * tot and (lo_i > 0 or hi_i < len(a) - 1):
        up = a[hi_i + 1:hi_i + 3].sum() if hi_i < len(a) - 1 else -1; dn = a[max(lo_i - 2, 0):lo_i].sum() if lo_i > 0 else -1
        if up >= dn: take = a[hi_i + 1:hi_i + 3]; hi_i = min(hi_i + len(take), len(a) - 1); acc += take.sum()
        else: take = a[max(lo_i - 2, 0):lo_i]; lo_i = max(lo_i - len(take), 0); acc += take.sum()
    return base + ip + .5, base + lo_i, base + hi_i + 1      # poc, val, vah
PR = {d: profile(DAY[d]) for d in DAYS}; rng = pd.Series({d: DAY[d].high.max() - DAY[d].low.min() for d in DAYS}); ADR = rng.rolling(14).mean().shift(1)
IDX = {d: i for i, d in enumerate(DAYS)}
def feats(D, m, dr):
    i = IDX.get(D)
    if i is None or i < 3 or PR[DAYS[i - 1]] is None or PR[DAYS[i - 2]] is None: return None
    g = DAY[D]; te = m; gb = g[g.m <= te - 5]
    if len(gb) < 3: return None
    adr = ADR[D]; poc1, val1, vah1 = PR[DAYS[i - 1]]; poc2 = PR[DAYS[i - 2]][0]; px = gb.close.iloc[-1]; op = g.open.iloc[0]; f = {}
    st = lambda x: 1 if x > vah1 else (-1 if x < val1 else 0)
    f['open_val'] = dr * st(op); f['px_val'] = dr * st(px)
    r80 = -st(op) if (st(op) != 0 and st(px) == 0) else 0; f['rule80'] = dr * r80       # opened outside, back inside value -> target = other side
    f['migr'] = dr * np.sign(poc1 - poc2) if abs(poc1 - poc2) > .1 * adr else 0
    dp = profile(gb); f['dev_poc'] = dr * np.sign(px - dp[0]) if dp and abs(px - dp[0]) > .1 * adr else 0
    f['dev_va'] = dr * (1 if px > dp[2] else (-1 if px < dp[1] else 0)) if dp else 0
    ib = gb[gb.m < 630]; ibh, ibl = ib.high.max(), ib.low.min(); f['ib_w'] = (ibh - ibl) / adr
    if te >= 630:
        f['ib_break'] = dr * (1 if px > ibh else (-1 if px < ibl else 0))
    else: f['ib_break'] = 0
    e = g[g.m < 600]
    if te >= 600 and len(e) >= 4:
        up = (e.close.values > op).all(); dn = (e.close.values < op).all(); mv = abs(e.close.iloc[-1] - op) > .15 * adr
        f['open_drive'] = dr * (1 if up and mv else (-1 if dn and mv else 0))
    else: f['open_drive'] = 0
    f['va_w'] = (vah1 - val1) / adr; f['px_poc1'] = dr * np.sign(px - poc1)
    return f
def tstat(x): x = np.asarray(x, float); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
def line(lab, s, flags):
    a, b = s[s.era == 'A'].pts, s[s.era == 'B'].pts
    ok = len(a) >= 30 and len(b) >= 30 and np.sign(a.mean()) == np.sign(b.mean()) and abs(tstat(a)) >= 2 and abs(tstat(b)) >= 2
    print(f' {lab:34} n {len(s):4d} win {np.mean(s.pts>0) if len(s) else 0:3.0%} mean {s.pts.mean() if len(s) else 0:+6.1f} | A n{len(a):3d} {a.mean() if len(a) else 0:+6.1f} ({tstat(a):+.1f}) | B n{len(b):3d} {b.mean() if len(b) else 0:+6.1f} ({tstat(b):+.1f}) {"PASS" if ok else ""}')
    if ok: flags.append(lab)
for nm in ('V15a', 'RA'):
    T = L[nm][0]; T = T[T.sz > 0].copy(); T['day'] = pd.to_datetime(T.day); rows = []
    for r in T.itertuples():
        D = pd.Timestamp(r.day).normalize(); f = feats(D, int(r.m), int(r.dir))
        if f: f.update(day=D, m=int(r.m), pts=r.pts, dir=int(r.dir)); rows.append(f)
    R = pd.DataFrame(rows); R['era'] = np.where(R.day < '2022-01-01', 'A', 'B'); flags = []
    print(f'\n######## {nm}: {len(R)} trades, mean {R.pts.mean():+.1f}, win {np.mean(R.pts>0):.0%}')
    for col, lab in (('open_val', 'open vs prior value (+1 = open beyond value in trade dir)'), ('px_val', 'price vs prior value (+1 = beyond in trade dir)'), ('rule80', '80% rule: opened outside, back in value, toward far side (+1)'),
                     ('migr', 'value migration (+1 = POC moved with trade)'), ('dev_poc', 'price vs developing POC (+1 = beyond in trade dir)'), ('dev_va', 'price vs developing value'), ('ib_break', 'initial balance break (+1 = with trade)'),
                     ('open_drive', 'open drive (+1 = with trade)'), ('px_poc1', 'price vs prior POC (+1 = beyond in trade dir)')):
        print(f'- {lab}'); [line(f'   {v:+d}', R[R[col] == v], flags) for v in (1, 0, -1)]
    for col, lab in (('ib_w', 'IB width / ADR'), ('va_w', 'prior value-area width / ADR')):
        q = pd.qcut(R[col], 3, labels=['narrow', 'mid', 'wide']); print(f'- {lab} terciles'); [line(f'   {v}', R[q == v], flags) for v in q.cat.categories]
    comp = pd.DataFrame({'open_beyond_value': R.open_val == 1, 'px_beyond_value': R.px_val == 1, 'migration_with': R.migr == 1, 'beyond_dev_POC': R.dev_poc == 1, 'IB_break_with': R.ib_break == 1, 'open_drive_with': R.open_drive == 1})
    R['amt'] = comp.sum(axis=1); print('- AMT count (open beyond value, price beyond value, migration with, beyond dev POC, IB break with, open drive with):')
    for s in sorted(R.amt.unique()): line(f'   amt {int(s)}', R[R.amt == s], flags)
    print(f'>> PASS buckets: {flags}'); R.to_pickle(SC + f'amt_{nm}.pkl')
