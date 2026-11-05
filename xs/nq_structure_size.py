"""Nov 4 2026, POST-HOC (labels picked from nq_break_types.log): size UP on good structure and DOWN / CUT on bad, through the firm sims (nq_firms.py rules, real-time drawdown).
Trades: base 15m flip (live 0.35 buffer). Flags per trade:
  risk (already in the stack): THIN 60-day path thin | HARH HAR high | VWAPX extended past VWAP
  bad structure: HARD hard break | STRONG strong close on the flip bar | LVN entry at a monthly LVN
  good structure: HVN entry at a monthly HVN | MID20 middle of the 20-day range | EXTD breakout at today's high / low | WEAK weak break
SIZING (fraction of 1 NQ), used in the eval AND funded:
  S0    current stack: 1 / 0.5 / 0.3 by THIN + HARH + VWAPX
  CUT-A skip THIN, HARD and LVN trades (slot freed for the next flip), the rest 1 / 0.5 / 0.3 by HARH + VWAPX
  CUT-B CUT-A + skip STRONG
  DOWN  1 / 0.5 / 0.3 by THIN + HARH + VWAPX + HARD + STRONG + LVN
  UP    S0, but 1.5 NQ when no risk flag and >= 2 good
  UPDN  DOWN, but 1.5 NQ when no risk / bad flag and >= 2 good
  UPDN2 UPDN, and 2 NQ when >= 3 good (Topstep's $1,000 daily limit cuts a 2 NQ stop at ~25 pts)"""
_f2 = open('xs/nq_firms2.py').read(); exec(_f2[:_f2.rindex("if __name__ == '__main__':")])
L = pd.read_parquet('/tmp/claude-0/ml/nq_flip_types_C.parquet')[['day', 'j', 'disp', 'vol5']]
X = C35.copy(); X['NODE_M'] = C.NODE_M.values
X = X.rename_axis('ix').reset_index().merge(L, on=['day', 'j'], how='left').set_index('ix')
kk = end_i.reindex(pd.DatetimeIndex(t5[X.j.values])).values.astype(int) - 1; hh, ll, cc_ = H15.high.values, H15.low.values, H15.close.values
clv = np.where(hh[kk] > ll[kk], (cc_[kk] - ll[kk]) / (hh[kk] - ll[kk]), .5); X['clv'] = np.where(X.dir > 0, clv, 1 - clv)
dq_ = X.loc[X.day < P0, 'disp'].quantile([1 / 3, 2 / 3]).values
X['HARD'] = (X.disp >= dq_[1]) & (X.clv >= .75) & (X.vol5 > 1); X['WEAK'] = ~X.HARD & ((X.disp < dq_[0]) | (X.clv < .5)); X['STRONG'] = X.clv >= .75
X['LVN'] = X.NODE_M == 'LVN'; X['HVN'] = X.NODE_M == 'HVN'
RT = pd.DataFrame({'h': h5, 'l': l5, 'd': d5})[rthm].groupby('d').agg(h=('h', 'max'), l=('l', 'min')); RT.index = pd.DatetimeIndex(RT.index)
H20, L20 = RT.h.rolling(20).max().shift(1), RT.l.rolling(20).min().shift(1); rd, r20 = [], []
for j, d_, day in zip(X.j.values, X.dir.values, X.day.values):
    e = o5[j]; i0 = j
    while i0 > 0 and d5[i0 - 1] == d5[j] and m5[i0 - 1] >= 570: i0 -= 1
    hi_, lo_ = (h5[i0:j].max(), l5[i0:j].min()) if j > i0 else (np.nan, np.nan); p = (e - lo_) / (hi_ - lo_) if hi_ > lo_ else np.nan; rd.append(p if d_ > 0 else 1 - p)
    day = pd.Timestamp(day); a, b = H20.get(day, np.nan), L20.get(day, np.nan); z = (e - b) / (a - b) if a > b else np.nan; r20.append(z if d_ > 0 else 1 - z)
X['EXTD'] = np.array(rd) >= .8; X['MID20'] = (np.array(r20) >= .2) & (np.array(r20) < .8)
FL = ['THIN', 'HARH', 'VWAPX', 'HARD', 'STRONG', 'LVN', 'HVN', 'MID20', 'EXTD', 'WEAK']
for c in FL: X[c] = X[c].fillna(False).astype(bool)
def build_dayX(Cc):
    T_ = run_p(Cc); D = [[] for _ in D_]
    for day, idx, net, mae, p in T_:
        if day in dpos: D[dpos[day]].append((tuple(bool(X.at[idx, c]) for c in FL), net, mae, p))
    return D
R = run_l(X); Z = X.join(R[['net']], how='inner'); Z['per'] = np.where(Z.day < P0, '18-21', np.where(Z.day < '2024-01-01', '22-23', '24-26'))
Z['good'] = Z[['HVN', 'MID20', 'EXTD', 'WEAK']].sum(1); Z['bad'] = Z[['THIN', 'HARD', 'STRONG', 'LVN']].sum(1); Z['risk'] = Z[['HARH', 'VWAPX']].sum(1)
print('avg pts per trade by count (post-hoc labels, so in-sample by construction)', flush=True)
for col in ('good', 'bad'):
    for v_, g in Z.groupby(col):
        print(f'  {col} {v_}: ' + ' | '.join(f'{p} {g[g.per == p].net.mean():+6.1f} (n{(g.per == p).sum()})' for p in ('18-21', '22-23', '24-26')), flush=True)
cutA = X.THIN | X.HARD | X.LVN; cutB = cutA | X.STRONG
SYSX = {'all': build_dayX(X), 'cutA': build_dayX(X[~cutA]), 'cutB': build_dayX(X[~cutB])}
def S0(th): return (1.0, .5, .3)[min(th[0] + th[1] + th[2], 2)]
def CUT(th): return (1.0, .5, .3)[min(th[1] + th[2], 2)]
def DOWN(th): return (1.0, .5, .3)[min(sum(th[:6]), 2)]
def UP(th): return 1.5 if th[0] + th[1] + th[2] == 0 and sum(th[6:]) >= 2 else S0(th)
def UPDN(th): return 1.5 if sum(th[:6]) == 0 and sum(th[6:]) >= 2 else DOWN(th)
def UPDN2(th): return 2.0 if sum(th[:6]) == 0 and sum(th[6:]) >= 3 else UPDN(th)
VARS = {'S0 current stack': ('all', S0), 'CUT-A skip thin / hard / LVN': ('cutA', CUT), 'CUT-B + skip strong close': ('cutB', CUT), 'DOWN size down all bad': ('all', DOWN),
        'UP 1.5 NQ on clean + good': ('all', UP), 'UPDN up on good, down on bad': ('all', UPDN), 'UPDN2 + 2 NQ on 3+ good': ('all', UPDN2)}
for nm, (sysn, fz) in VARS.items():
    sz = np.array([fz(tuple(bool(X.at[i, c]) for c in FL)) for i in R.index]); rr = R.assign(sz=sz)
    if sysn != 'all': rr = rr[~(cutA if sysn == 'cutA' else cutB).loc[rr.index]]
    post = rr[rr.day >= P0]; print(f'  {nm:<30} 2022-26: trades {int((post.sz > 0).sum())}, avg size {post.sz.mean():.2f} NQ, avg $ per trade {(post.net * post.sz * 20).mean():+6.0f}', flush=True)
def job3(a):
    global DAY
    nm, fn = a; sysn, fz = VARS[nm]; DAY = SYSX[sysn]; SZE2['V'] = fz; SZ2['V'] = fz; return dict(job2((fn, 'V', 'V')), var=nm)
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in VARS for fn in ('TOPSTEP', 'LUCID FLEX')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job3, jobs)
    print('\nFIRM SIMS, sizing in eval and funded (real-time drawdown; starts 2022 - 2025)', flush=True)
    for r in RES:
        print(f'  {r["var"]:<30} {r["firm"]:<11} | eval passed {r["passed"]:4.0%} | funded blown before 1st payout {r["blown0"]:4.0%} | evals per 1st payout {r["per_pay"]:5.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m paid {r["k_paid"]:4.0%} median {r["k_med"]:+6,.0f} 10th {r["k_p10"]:+6,.0f}', flush=True)
