"""Nov 8 2026: STAT CARD - current prop system (S1) alone vs S1 + noise-area momentum at 2 micros (4 at 150K) with a 40-point stop, Lucid 50K / 150K.
One account bought every 5th session Jan 2022 - Jun 2025 (each runs to its end or Oct 2026); a 'slot' that rebuys after every failed or blown account for the time to the
first payout. Payout points = $ / 20 (after the 90% split)."""
_np = open('xs/nq_noise_prop.py').read(); exec(_np[_np.index('_e0 = open'):_np.rindex("if __name__ == '__main__':")])
NT = pd.read_parquet('/tmp/claude-0/ml/noise_stop_40.parquet')
s1e, s1f = to_D2(ev_res, eqm(.3) * skip2, 'YZV'), to_D2(full, eqm(.7) * skip2, 'YZV')
def full_one(A, k):
    res, k2 = ev(A, k, NS)
    if res != 'pass': return res, k2, [], 'none', (k2 if res == 'fail' else NS)
    fr, k3, pays = fu(A, k2 + 1, NS, 5); return res, k2, pays, fr, (NS if fr == 'open' else k3)
if __name__ == '__main__':
    starts = list(range(s0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5))
    for nm, (de, df) in (('CURRENT SYSTEM (S1)', (s1e, s1f)), ('CURRENT + NOISE 2 micros, 40-pt stop', (merge(s1e, nlists(.2)), merge(s1f, nlists(.2))))):
        DE, DF = de, df; print(f'\n{nm}', flush=True)
        for a in ('50K', '150K'):
            A = ACC[a]; C_ = {}
            get = lambda k: C_.setdefault(k, full_one(A, k))
            R = [get(k) for k in starts]; fund = [(r, k) for r, k in zip(R, starts) if r[0] == 'pass']
            dpass = [(D_[r[1]] - D_[k]).days for r, k in fund]; pays = np.array([x for r, _ in fund for _, x in r[2]]) / 20
            gaps = [b[0] - a_[0] for r, _ in fund for a_, b in zip(r[2][:-1], r[2][1:])]
            net = np.array([sum(x for _, x in r[2]) / 20 - A['fee'] / 20 for r in R]); life = np.array([(r[4] - k + 1) / 21 for r, k in zip(R, starts)])
            fp = []
            for s in starts:
                k = s; nb = 0
                while k < NS:
                    while k < NS and not GREEN[k]: k += 1
                    if k >= NS: break
                    r = get(k); nb += 1
                    if r[0] == 'open': break
                    if r[2]: fp.append((nb, (D_[r[2][0][0]] - D_[s]).days / 30.44)); break
                    k = r[4] + 1
            fp = np.array(fp)
            print(f'  {a}: eval pass {np.mean([r[0] == "pass" for r in R]):.0%} (median {np.median(dpass) / 30.44:.1f} months to pass) | payout rate {np.mean([len(r[2]) > 0 for r in R]):.0%} of evals | '
                  f'funded accounts: {np.mean([len(r[2]) for r, _ in fund]):.1f} payouts, {np.mean([len(r[2]) >= 5 for r, _ in fund]):.0%} reach all 5, {np.mean([r[3] == "lost" for r, _ in fund]):.0%} blown', flush=True)
            print(f'        PAYOUT SIZE median {np.median(pays):.0f} pts (10th-90th pct {np.percentile(pays, 10):.0f}-{np.percentile(pays, 90):.0f}), max {pays.max():.0f} | '
                  f'TIME BETWEEN PAYOUTS median {np.median(gaps):.0f} trading days ({np.percentile(gaps, 25):.0f}-{np.percentile(gaps, 75):.0f})', flush=True)
            print(f'        FIRST PAYOUT from the first eval bought: median {np.median(fp[:, 1]):.1f} months ({np.percentile(fp[:, 1], 25):.1f}-{np.percentile(fp[:, 1], 75):.1f}), '
                  f'evals bought until then median {np.median(fp[:, 0]):.0f} (mean {fp[:, 0].mean():.1f}, 80th pct {np.percentile(fp[:, 0], 80):.0f}) | '
                  f'net per eval {net.mean():+.0f} pts, per account-month {net.sum() / life.sum():+.1f} pts', flush=True)
