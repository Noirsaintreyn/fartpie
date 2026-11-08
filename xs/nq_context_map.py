"""Nov 8 2026: CONTEXT MAP - what to expect from a flip given where it is and what regime it is in, LONGS AND SHORTS SEPARATELY (no mirroring). DECLARED BEFORE RESULTS.
Every RTH 15m vector flip 2018 - Oct 2026; outcomes per flip in NQ points after cost: TH (held to the 15m vector exit, stop 50) and F75 (75 / 50, flat 15:55).
Context at entry (known then):
  VA     entry vs the PRIOR DAY's value area (5m bars spread over 0.05% bins, 70%): below VAL | VAL-POC | POC-VAH | above VAH
  BROKE  what the 15m flip bar moved through (open -> close): VAL | POC | VAH | none
  NODE   the entry price's volume in the 20-DAY composite profile (log 0.05% bins, 3-bin smoothing), as a percentile of that profile: HVN (top 30%) | middle | LVN
         (bottom 30%) | outside the 20-day range
  GAMMA  QQQ net GEX at the prior close: negative | positive          VOL  5-day realized vol percentile: high (top third of its year) | normal
Report: one-way tables by direction (n, win, TH, F75; TH by period 18-21 / 22-23 / 24-25 / 2026), then GAMMA x VOL x NODE and VA x NODE cells (n >= 15), ranked,
with whether the cell is positive in both halves (2018-21 and 2022-26)."""
_t = open('xs/nq_composite_th.py').read(); exec(_t[:_t.rindex("if __name__ == '__main__':")])
LB = np.log(1.0005); days_ = sorted(H.day.unique()); gi = H.groupby('day').indices; PROF = {}; LVd = {}
for d in days_:
    ix = gi[d]; lo, hi, v = l5[ix], h5[ix], H.volume.values[ix].astype(float); a = np.floor(np.log(lo) / LB).astype(int); b = np.floor(np.log(hi) / LB).astype(int); b0 = a.min()
    prof = np.zeros(b.max() - b0 + 1)
    for x, y, w in zip(a - b0, b - b0, v): prof[x:y + 1] += w / (y - x + 1)
    PROF[d] = (b0, prof); p = int(prof.argmax()); u = dn = p; tot = prof[p]; tgt = .7 * prof.sum()
    while tot < tgt and (u < len(prof) - 1 or dn > 0):
        up = prof[u + 1] if u < len(prof) - 1 else -1; down = prof[dn - 1] if dn > 0 else -1
        if up >= down: u += 1; tot += up
        else: dn -= 1; tot += down
    LVd[d] = (np.exp((b0 + p + .5) * LB), np.exp((b0 + u + 1) * LB), np.exp((b0 + dn) * LB))   # POC, VAH, VAL
dix = {d: i for i, d in enumerate(days_)}
def composite(d, n=20):
    i = dix[d]; prev = [PROF[days_[k]] for k in range(max(0, i - n), i)]
    if len(prev) < n: return None
    b0 = min(p[0] for p in prev); b1 = max(p[0] + len(p[1]) for p in prev); c = np.zeros(b1 - b0)
    for s, p in prev: c[s - b0:s - b0 + len(p)] += p
    return b0, np.convolve(c, np.ones(3) / 3, mode='same')
VA, BR, ND = [], [], []
for r in X.itertuples():
    j, dr = int(r.j), int(r.dir); d = pd.Timestamp(d5[j]); i = dix.get(d)
    if i is None or i == 0: VA.append(None); BR.append(None); ND.append(None); continue
    poc, vah, val = LVd[days_[i - 1]]; e = o5[j]
    VA.append('below VAL' if e < val else ('VAL-POC' if e < poc else ('POC-VAH' if e <= vah else 'above VAH')))
    fo, fc = o5[j - 3], c5[j - 1]; cr = lambda L: min(fo, fc) < L <= max(fo, fc)
    BR.append('VAH' if cr(vah) else ('POC' if cr(poc) else ('VAL' if cr(val) else 'none')))
    cp = composite(d)
    if cp is None: ND.append(None); continue
    b0, c = cp; k = int(np.floor(np.log(e) / LB)) - b0
    if k < 0 or k >= len(c) or c[k] <= 0: ND.append('outside range'); continue
    pct = (c[c > 0] < c[k]).mean(); ND.append('HVN' if pct >= .7 else ('LVN' if pct <= .3 else 'middle'))
X['VA'], X['BR'], X['ND'] = VA, BR, ND; X['GAM'] = np.where(X.gex < 0, 'neg gamma', 'pos gamma'); X['VOL'] = np.where(X.rv5 >= cut['rv5'], 'high vol', 'normal vol')
X['DIR'] = np.where(X.dir > 0, 'LONG', 'SHORT'); X['per'] = np.select([X.day < '2022-01-01', X.day < '2024-01-01', X.day < '2026-01-01'], ['18-21', '22-23', '24-25'], '2026')
X['half'] = np.where(X.day < '2022-01-01', 'a', 'b'); PER = ['18-21', '22-23', '24-25', '2026']
def one(Z, c, order):
    print(f'  {c}:', flush=True)
    for v in order:
        g = Z[Z[c] == v]
        if len(g) < 5: continue
        print(f'    {v:<14} n{len(g):4d} win {np.mean(g.th > 0):3.0%} TH {g.th.mean():+6.1f}  F75 {g.f75.mean():+6.1f} | TH ' + ' '.join(f'{g[g.per == p].th.mean():+6.1f}' if (g.per == p).sum() >= 6 else '     .' for p in PER), flush=True)
def cells(Z, cols, minn=15):
    rows = []
    for k, g in Z.groupby(cols):
        if len(g) < minn: continue
        a, b = g[g.half == 'a'], g[g.half == 'b']
        rows.append((' + '.join(k), len(g), g.th.mean(), g.f75.mean(), np.mean(g.th > 0), a.th.mean() if len(a) >= 5 else np.nan, b.th.mean() if len(b) >= 5 else np.nan))
    for k, n, th, f, w, a, b in sorted(rows, key=lambda r: -r[2]):
        both = 'both halves +' if (a > 0 and b > 0) else ('both halves -' if (a < 0 and b < 0) else '')
        print(f'    {k:<46} n{n:4d} win {w:3.0%} TH {th:+6.1f}  F75 {f:+6.1f} | 2018-21 {a:+6.1f} / 2022-26 {b:+6.1f} {both}', flush=True)
if __name__ == '__main__':
    print(f'flips {len(X)}; with VA {X.VA.notna().sum()}, with node {X.ND.notna().sum()}', flush=True)
    for dname in ('LONG', 'SHORT'):
        Z = X[X.DIR == dname]; print(f'\n=== {dname} flips ({len(Z)}) | periods ' + ' '.join(f'{p:>6}' for p in PER), flush=True)
        one(Z, 'VA', ['below VAL', 'VAL-POC', 'POC-VAH', 'above VAH']); one(Z, 'BR', ['VAL', 'POC', 'VAH', 'none'])
        one(Z, 'ND', ['LVN', 'middle', 'HVN', 'outside range']); one(Z, 'GAM', ['neg gamma', 'pos gamma']); one(Z, 'VOL', ['high vol', 'normal vol'])
        print(f'  GAMMA x VOL x NODE cells (n >= 15), best first:', flush=True); cells(Z, ['GAM', 'VOL', 'ND'])
        print(f'  VA x NODE cells (n >= 15), best first:', flush=True); cells(Z, ['VA', 'ND'])
    X[['day', 'j', 'dir', 'VA', 'BR', 'ND', 'GAM', 'VOL', 'th', 'f75']].to_parquet('/tmp/claude-0/ml/nq_context_map.parquet')
