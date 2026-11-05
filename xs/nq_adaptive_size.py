"""Nov 4 2026: ADAPTIVE sizing after losses - how much to cut, graded cuts, cushion-based sizing - on the final system. DECLARED BEFORE RESULTS.
Base = final system (Rule A, Yang-Zhang + VWAP ladder 10 / 5 / 3 MNQ, 50 / 75, flat 12:00, gap-gated eval). Sizes are rounded to whole micros (min 1). Lucid 50K ($88) + Topstep.
  normal | EQ20 x0.5 (last 20 trades' 1-lot net < 0) | EQ20 x0.7 | EQ20 x0.3
  GRADED   last-20 net in [-100, 0) pts -> x0.7, below -100 -> x0.3
  GRADED3  [-50, 0) x0.7, [-150, -50) x0.5, below -150 x0.3
  CUSHION  size x min(1, room to the max-loss floor / $2,000) at the start of each day (account-based, no trade history)
  EQ20 x0.5 + CUSHION | EQ10 x0.5 | EQ40 x0.5 (window neighbours)"""
_ll = open('xs/nq_learn_losses.py').read(); exec(_ll[:_ll.rindex("if __name__ == '__main__':")])
def eqmult(N, rule):
    out = pd.Series(1.0, index=XA.index); hist = []
    for idx in RL.index:
        s_ = sum(hist[-N:]) if len(hist) >= N else 0.0
        out.loc[idx] = rule(s_) if len(hist) >= N else 1.0; hist.append(RL.net.loc[idx])
    return out
VARS2 = {'normal': (eqmult(20, lambda s: 1.0), False), 'EQ20 x0.5': (eqmult(20, lambda s: .5 if s < 0 else 1.0), False), 'EQ20 x0.7': (eqmult(20, lambda s: .7 if s < 0 else 1.0), False),
         'EQ20 x0.3': (eqmult(20, lambda s: .3 if s < 0 else 1.0), False), 'GRADED 0.7 / 0.3': (eqmult(20, lambda s: 1.0 if s >= 0 else (.7 if s >= -100 else .3)), False),
         'GRADED3 0.7 / 0.5 / 0.3': (eqmult(20, lambda s: 1.0 if s >= 0 else (.7 if s >= -50 else (.5 if s >= -150 else .3))), False),
         'CUSHION': (eqmult(20, lambda s: 1.0), True), 'EQ20 x0.5 + CUSHION': (eqmult(20, lambda s: .5 if s < 0 else 1.0), True),
         'EQ10 x0.5': (eqmult(10, lambda s: .5 if s < 0 else 1.0), False), 'EQ40 x0.5': (eqmult(40, lambda s: .5 if s < 0 else 1.0), False)}
CUSH = False
def dayq(k, st, f):
    cum = 0.0; bal = st['bal']; cf = min(1.0, max(bal - st['floor'], 0.0) / 2000.0) if CUSH else 1.0
    for th, net, mae, p in DAY[k]:
        if cum <= -f['pstop']: break
        mic = max(1, int(round(10 * th[0] * cf))); m = 2.0 * mic
        if bal + cum + mae * m <= st['floor']: return st['floor'] - bal, True
        if f['dll'] and cum + mae * m <= -f['dll']: cum = -float(f['dll']); break
        cum += net * m
    return cum, False
VAR = {}
for nm, (m, cush) in VARS2.items(): DS[f'e_{nm}'] = build(~AG['GAP'], mult=m); DS[f'f_{nm}'] = build(mult=m); VAR[nm] = (f'e_{nm}', f'f_{nm}', f'f_{nm}')
def job11(a):
    global CUSH
    CUSH = VARS2[a[0]][1]; return job10(a)
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in VAR for fn in ('LUCID 50K', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job11, jobs)
    print('\nADAPTIVE SIZING (whole micros; gap-gated eval; same sizing in eval and funded)', flush=True)
    for r in RES:
        print(f'  {r["nm"]:<24} {r["fn"]:<9} | eval pass {r["passed"]:4.0%} | funded paid {r["paid_f"]:4.0%} | evals -> payout {r["paid_e"]:4.0%} | payouts/funded {r["npay_f"]:3.1f} | avg payout ${r["avg_pay"]:,.0f} | net/eval {r["net_eval"]:+6,.0f} '
              f'| 12m median {r["k_med"]:+6,.0f} (10th {r["k_p10"]:+5,.0f}) positive {r["k_pos"]:4.0%} | 2022-Oct26 {r["full"]:+7,.0f} | 2026 {r["y26"]:+5,.0f}', flush=True)
