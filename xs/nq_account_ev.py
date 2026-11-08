"""Nov 8 2026: ONE LUCID 50K ACCOUNT - what each version returns per account (scale comes from more accounts). DECLARED BEFORE RESULTS.
One account bought every 5th session Jan 2022 - Jun 2025 (each runs to its end or Oct 2026). Per version:
  pass %, paid % (evals that ever pay), payouts per funded account and average size, % of funded accounts blown before 5 payouts,
  NET PER EVAL (payouts - $88 fee, points; mean and 10th / 50th / 90th percentile), months the account lives (eval + funded) and NET PER ACCOUNT-MONTH,
  median months to the first payout (paid accounts), trading days between payouts.
Versions (all Rule A, flat 12:00, <= 2 a day, Yang-Zhang + VWAP ladder; 'cold' = last 20 trades net negative):
  C0 current: 75 / 50, eval x0.3 cold, funded x0.7 cold          C1 C0 + skip the POC-to-VAH zone (S1)
  C2 C1, size 0.5x when the 4-pillar score disagrees (Q 0-2)    C3 C1, 0.5x at Q 0-1 / 1.5x at Q 3-4
  C4 C1, 1.5x at Q 3-4                                           C5 C1, eval x1.5 with no cold cut
  C6 C1, eval x2 with no cold cut                                C7 C6 + funded 0.5x / 1.5x by score (C3 sizing once funded)
  C8 C1, funded x1.5 on every trade                              C9 C1, funded x2 on every trade (20 micros, Lucid's first funded tier)
  C10 C1 kept trading after 5 payouts (if the live account kept the same rules)"""
_s = open('xs/nq_signal_size.py').read(); exec(_s[:_s.rindex("if __name__ == '__main__':")])
def get_one(A, k, maxpay):
    res, k2 = ev(A, k, NS)
    if res != 'pass': return res, k2 if res == 'fail' else NS, [], 'none', k2 if res == 'fail' else NS
    fr, k3, pays = fu(A, k2 + 1, NS, maxpay); return res, k2, pays, fr, (NS if fr == 'open' else k3)
q2 = pd.Series(np.where(Q >= 3, 1.0, .5), index=XA.index); qud = cap(ud(qg)); q15 = pd.Series(np.where(Q >= 3, 1.5, 1.0), index=XA.index)
V = {'C0 current': (one, one, True, 5, False), 'C1 + skip POC-to-VAH (S1)': (skip2, skip2, True, 5, True), 'C2 C1, 0.5x when score disagrees': (skip2 * q2, skip2 * q2, True, 5, True),
     'C3 C1, 0.5x / 1.5x by score': (skip2 * qud, skip2 * qud, True, 5, True), 'C4 C1, 1.5x at Q 3-4': (skip2 * q15, skip2 * q15, True, 5, True),
     'C5 C1, eval x1.5 no cold cut': (skip2 * 1.5, skip2, False, 5, True), 'C6 C1, eval x2 no cold cut': (skip2 * 2, skip2, False, 5, True),
     'C7 C6 + funded 0.5x / 1.5x by score': (skip2 * 2, skip2 * qud, False, 5, True), 'C8 C1, funded x1.5': (skip2, skip2 * 1.5, True, 5, True),
     'C9 C1, funded x2': (skip2, skip2 * 2, True, 5, True), 'C10 C1 kept after 5 payouts': (skip2, skip2, True, 99, True)}
if __name__ == '__main__':
    A = ACC['50K']; starts = list(range(s0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5))
    print(f'Lucid 50K, {len(starts)} accounts (one bought every 5th session Jan 2022 - Jun 2025); points = $ / 20; fee = {A["fee"] / 20:.1f} pts', flush=True)
    print(f'  {"version":<38} pass  paid  payouts x size  blown | net per eval: mean   p10   p50   p90 | months alive  NET/acct-month | 1st payout  days apart', flush=True)
    for nm, (me_m, mf_m, evcut, maxpay, _) in V.items():
        DE = to_D2(ev_res, (eqm(.3) if evcut else 1.0) * me_m, 'YZV'); DF = to_D2(full, eqm(.7) * mf_m, 'YZV')
        R = [get_one(A, k, maxpay) for k in starts]
        net = np.array([sum(a for _, a in r[2]) / 20 - A['fee'] / 20 for r in R]); life = np.array([(r[4] - k + 1) / 21 for r, k in zip(R, starts)])
        fund = [r for r in R if r[0] == 'pass']; npay = np.mean([len(r[2]) for r in fund]); avgp = np.mean([a for r in fund for _, a in r[2]]) / 20
        blown = np.mean([r[3] == 'lost' for r in fund]); paid = np.mean([len(r[2]) > 0 for r in R])
        fp = [(D_[min(r[2][0][0], NS - 1)] - D_[k]).days / 30.44 for r, k in zip(R, starts) if r[2]]; gaps = [b[0] - a[0] for r in fund for a, b in zip(r[2][:-1], r[2][1:])]
        print(f'  {nm:<38} {np.mean([r[0] == "pass" for r in R]):4.0%} {paid:5.0%}  {npay:4.1f} x {avgp:3.0f} pts {blown:5.0%} | {net.mean():+16.0f} {np.percentile(net, 10):+5.0f} {np.percentile(net, 50):+5.0f} {np.percentile(net, 90):+5.0f} | '
              f'{life.mean():11.1f} {net.sum() / life.sum():+14.1f} | {np.median(fp):6.1f} mo {np.median(gaps):7.0f}', flush=True)
