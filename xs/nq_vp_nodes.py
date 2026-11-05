"""Nov 4 2026: do vector flips work better at LVNs / HVNs of the volume profile, and when they leave value (searching for a new value area)? DECLARED BEFORE RESULTS.
Trades: the base NQ trade (QQQ 15m vector flip, signal bars closing 09:30-11:30, both ways, <= 2 a day, 1 NQ 75/50, flat 12:00, 1 pt cost, QQQ proxy -> NQ points), 2018 - Oct 2026.
PROFILES from QQQ 5m RTH bars of the PRIOR sessions only (each bar's volume spread evenly over its high-low range, log-price bins of 0.05%):
  D prior day | W prior 5 sessions | M prior 21 sessions | 60D prior 60 sessions.  Value area = 70% of volume grown out from the POC.
LABELS per lookback, at the entry price:
  NODE   the entry bin's volume percentile among the profile's bins: HVN >= 75th | LVN <= 25th | mid | outside the profile's range (fresh price)
  AHEAD  mean profile volume on the path from entry to the 75-pt target, divided by the profile's mean bin volume (outside the range counts as 0), terciles fixed on 2018-21:
         LVN path (thin, should travel fast) / mid / HVN path (thick, should stall)
  BEHIND the same over the 50-pt stop zone: thick behind = support / resistance behind the trade
  VALUE  entry outside value and trading AWAY from it (searching for new value) | outside and trading BACK toward value | entry inside value
HYPOTHESES (Rey): flips that leave value through thin profile do best -> COMBO = VALUE away AND AHEAD LVN path;  flips entered inside an HVN chop.
A label level counts if it beats its period's average in 2018-21, 2022-23 AND 2024-26 with n >= 30 in each.
Then realistic months 2022 - Oct 2026 (fresh 50K eval each month: -$1,000 day floor, $2,000 trailing = lost, +$3,000 with best day <= 50% = passed) for:
  base | skip HVN entries | only VALUE away | only COMBO | size 1 NQ on COMBO / 0.5 NQ otherwise, per lookback."""
_s = open('xs/nq_size_tp.py').read(); exec(_s[:_s.index("print('\\nby context")])
C = C.reset_index(drop=True); BW = .0005; lh5, ll5 = np.log(h5), np.log(l5)
DP = {}
for d, idx in pd.Series(np.arange(len(H5))[rthm]).groupby(d5[rthm]):
    ii = idx.values; a = np.floor(ll5[ii] / BW).astype(int); b = np.floor(lh5[ii] / BW).astype(int); lo = a.min(); v = np.zeros(b.max() - lo + 1)
    for x, y, vv in zip(a - lo, b - lo, vol5[ii]): v[x:y + 1] += vv / (y - x + 1)
    DP[pd.Timestamp(d)] = (lo, v)
DLI = pd.DatetimeIndex(sorted(DP))
def prof(day, n):
    i = DLI.searchsorted(day); sel = DLI[max(i - n, 0):i]
    if len(sel) < n: return None
    lo = min(DP[d][0] for d in sel); hi = max(DP[d][0] + len(DP[d][1]) for d in sel); v = np.zeros(hi - lo)
    for d in sel: a, x = DP[d]; v[a - lo:a - lo + len(x)] += x
    p = int(v.argmax()); tot = v.sum(); a = b = p; acc = v[p]
    while acc < .7 * tot and (a > 0 or b < len(v) - 1):
        up = v[b + 1] if b < len(v) - 1 else -1; dn = v[a - 1] if a > 0 else -1
        if up >= dn: b += 1; acc += v[b]
        else: a -= 1; acc += v[a]
    return lo, v, lo + a, lo + b                                                       # VAL bin, VAH bin
def zone(lo, v, b0, b1):
    bb = np.arange(min(b0, b1), max(b0, b1) + 1) - lo; ok = (bb >= 0) & (bb < len(v)); x = np.zeros(len(bb)); x[ok] = v[bb[ok]]; return x.mean() / v.mean()
LB = {'D': 1, 'W': 5, 'M': 21, '60D': 60}; cache = {}; rows = []
for r in C.itertuples():
    e = o5[r.j]; eb = int(np.floor(np.log(e) / BW)); tb = int(np.floor(np.log(e + r.dir * 75 / r.k) / BW)); sb = int(np.floor(np.log(e - r.dir * 50 / r.k) / BW)); out = {}
    for L, n in LB.items():
        key = (r.day, n)
        if key not in cache: cache[key] = prof(r.day, n)
        P_ = cache[key]
        if P_ is None: out.update({f'NODE_{L}': np.nan, f'ahead_{L}': np.nan, f'behind_{L}': np.nan, f'VALUE_{L}': np.nan}); continue
        lo, v, vl, vh = P_; rel = eb - lo
        if 0 <= rel < len(v): pc = (v < v[rel]).mean(); node = 'HVN' if pc >= .75 else ('LVN' if pc <= .25 else 'mid node')
        else: node = 'outside profile'
        side = 1 if eb > vh else (-1 if eb < vl else 0)
        val = 'inside value' if side == 0 else ('away from value' if side == r.dir else 'back toward value')
        out.update({f'NODE_{L}': node, f'ahead_{L}': zone(lo, v, eb + r.dir, tb), f'behind_{L}': zone(lo, v, eb - r.dir, sb), f'VALUE_{L}': val})
    rows.append(out)
C = pd.concat([C, pd.DataFrame(rows)], axis=1)
def ter(x, a, b, labs): y = pd.Series(np.where(x < a, labs[0], np.where(x > b, labs[2], labs[1])), index=x.index, dtype=object); y[x.isna()] = None; return y
for L in LB:
    a, b = C.loc[C.day < P0, f'ahead_{L}'].quantile([1 / 3, 2 / 3]).values; C[f'AHEAD_{L}'] = ter(C[f'ahead_{L}'], a, b, ['LVN path ahead', 'mid path', 'HVN path ahead'])
    a, b = C.loc[C.day < P0, f'behind_{L}'].quantile([1 / 3, 2 / 3]).values; C[f'BEHIND_{L}'] = ter(C[f'behind_{L}'], a, b, ['thin behind', 'mid behind', 'thick behind'])
    y = pd.Series(np.where((C[f'VALUE_{L}'] == 'away from value') & (C[f'AHEAD_{L}'] == 'LVN path ahead'), 'COMBO away + LVN ahead', 'not combo'), index=C.index, dtype=object); y[C[f'VALUE_{L}'].isna()] = None; C[f'COMBO_{L}'] = y
def run_l(Cc, T=75, S=50, per_day=2, flat_m=720, cost=1.0):
    out = []
    for day, g in Cc.sort_values('j').groupby('day'):
        busy = -1; n = 0
        for idx, j, d, kk in zip(g.index, g.j.values, g.dir.values, g.k.values):
            if n >= per_day or j <= busy: continue
            e = o5[j]; res = None
            for q in range(j, len(H5)):
                if d5[q] != d5[j] or m5[q] >= flat_m: res = ((c5[q - 1] - e) * d * kk, q - 1); break
                lo_ = ((l5[q] if d > 0 else h5[q]) - e) * d * kk; hi_ = ((h5[q] if d > 0 else l5[q]) - e) * d * kk
                if lo_ <= -S: res = (-S, q); break
                if hi_ >= T: res = (T, q); break
            if res: out.append((idx, day, res[0] - cost)); busy = res[1]; n += 1
    return pd.DataFrame(out, columns=['idx', 'day', 'net']).set_index('idx')
R = run_l(C); Z = C.join(R[['net']], how='inner'); Z['per'] = np.where(Z.day < P0, '18-21', np.where(Z.day < '2024-01-01', '22-23', '24-26'))
base = Z.groupby('per').net.mean(); PERS = ('18-21', '22-23', '24-26')
print(f'trades {len(Z)}; avg pts 18-21 {base["18-21"]:+.1f} | 22-23 {base["22-23"]:+.1f} | 24-26 {base["24-26"]:+.1f}', flush=True)
print('\nBY LABEL: avg pts per trade (n, win) in 2018-21 | 2022-23 | 2024-26', flush=True)
for L in LB:
    print(f'\n  -- profile {L}', flush=True)
    for fam in ('NODE', 'AHEAD', 'BEHIND', 'VALUE', 'COMBO'):
        col = f'{fam}_{L}'
        for v_, g in Z.groupby(col):
            p = {k: g[g.per == k] for k in PERS}; ok = all(len(p[k]) >= 30 and p[k].net.mean() > base[k] for k in PERS)
            cells = ' | '.join(f'{k} {p[k].net.mean():+6.1f} (n{len(p[k]):>3} w{p[k].net.gt(0).mean():.0%})' if len(p[k]) else f'{k}    n/a' for k in PERS)
            print(f'  {fam:<6} {str(v_):<24} {cells}' + ('   <- beats all three' if ok else ''), flush=True)
def mline(nm, dv):
    M_ = months(dv); M6 = M_[M_.month >= '2026-01']
    print(f'  {nm:<46} passed {M_.res.eq("passed").mean():4.0%} lost {M_.res.eq("lost").mean():4.0%} neither {M_.res.eq("neither").mean():4.0%} | month $ mean {M_.pnl.mean():+6,.0f} worst {M_.pnl.min():+7,.0f} | 2026 passed {M6.res.eq("passed").sum()}/{len(M6)} lost {M6.res.eq("lost").sum()}', flush=True)
print('\nREALISTIC MONTHS 2022 - Oct 2026 (filters re-run, so a skipped signal frees the slot for the next flip)', flush=True)
mline('base', (R.net * 20).groupby(R.day).sum())
for L in LB:
    for nm, msk in ((f'{L}: skip HVN entries', C[f'NODE_{L}'] != 'HVN'), (f'{L}: only away from value', C[f'VALUE_{L}'] == 'away from value'),
                    (f'{L}: only LVN path ahead', C[f'AHEAD_{L}'] == 'LVN path ahead'), (f'{L}: only COMBO', C[f'COMBO_{L}'] == 'COMBO away + LVN ahead')):
        Rr = run_l(C[msk]); post = Rr[Rr.day >= P0]; mline(f'{nm} (n{len(post)}, avg {post.net.mean():+.1f})', (Rr.net * 20).groupby(Rr.day).sum())
    w = np.where(C.loc[R.index, f'COMBO_{L}'] == 'COMBO away + LVN ahead', 20, 10); mline(f'{L}: 1 NQ on COMBO / 0.5 NQ otherwise', (R.net * w).groupby(R.day).sum())
