"""Nov 4 2026: exit when the LOWER-TIMEFRAME (5m) vector closes against the trade, on the final system. DECLARED BEFORE RESULTS. (No multi-year 1m data, so 1m is not testable.)
Exit at the next 5m open after a completed 5m bar leaves the 5m vector against the trade:  VX5 always | VX5-PROFIT only once open profit has reached +25 | VX5-LOSS only while the trade is losing.
Everything else as the final system (75 / 50, flat 12:00, Rule A, Yang-Zhang + VWAP ladder, eval gap-gated x0.3 cold streaks, funded x0.7)."""
_te = open('xs/nq_trail_exits.py').read(); exec(_te[:_te.rindex("if __name__ == '__main__':")])
reg5 = np.asarray(vec(H5[['open', 'high', 'low', 'close']].reset_index(drop=True)))
def simx(j, d, kk, mode, fbh, fbl, T=75.0, S=50.0):
    e = o5[j]; path = []; best = 0.0
    for q in range(j, len(H5)):
        if d5[q] != d5[j] or m5[q] >= 720: path.append((c5[q - 1] - e) * d * kk); return path[-1], q - 1, path, 'noon'
        lo_ = ((l5[q] if d > 0 else h5[q]) - e) * d * kk; hi_ = ((h5[q] if d > 0 else l5[q]) - e) * d * kk; cl_ = (c5[q] - e) * d * kk
        if lo_ <= -S: path += [min(hi_, T), -S]; return -S, q, path, 'stop'
        if hi_ >= T: path += [lo_, T]; return T, q, path, 'target'
        path += [hi_, lo_]; best = max(best, hi_)
        if not (q + 1 < len(H5) and d5[q + 1] == d5[j] and m5[q + 1] < 720) or mode == 'BASE': continue
        against = reg5[q] == -d
        ex = against and (mode == 'VX5' or (mode == 'VX5-PROFIT' and best >= 25) or (mode == 'VX5-LOSS' and cl_ < 0))
        if ex: x = (o5[q + 1] - e) * d * kk; path.append(x); return x, q, path, 'vector'
    return None
VAR = {}; print('Rule A trades by exit rule (1 lot, NQ pts)', flush=True)
for mode in ('BASE', 'VX5', 'VX5-PROFIT', 'VX5-LOSS'):
    full = seqx(mode); nets = pd.Series({i: v[0] - 1.0 for i, v in full.items()}); hows = pd.Series({i: v[3] for i, v in full.items()}); per = PER(XA.loc[nets.index].day)
    print(f'  {mode:<10} n{len(nets)} win {nets.gt(0).mean():.0%} avg loss {nets[nets <= 0].mean():+6.1f} avg win {nets[nets > 0].mean():+5.1f} | exits ' + ', '.join(f'{k} {v:.0%}' for k, v in hows.value_counts(normalize=True).items()) + ' | ' +
          ' / '.join(f'{p} {nets[per == p].mean():+5.1f}' for p in ('18-21', '22-23', '24-25', '2026')), flush=True)
    DS[f'e_{mode}'] = to_D(seqx(mode, ~AG['GAP']), eqm(nets, .3)); DS[f'f_{mode}'] = to_D(full, eqm(nets, .7)); VAR[mode] = (f'e_{mode}', f'f_{mode}', f'f_{mode}')
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in VAR for fn in ('LUCID 50K', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job10, jobs)
    print('\nFINAL SYSTEM WITH 5m-VECTOR EXITS', flush=True)
    for r in RES:
        print(f'  {r["nm"]:<10} {r["fn"]:<9} | eval pass {r["passed"]:4.0%} | funded paid {r["paid_f"]:4.0%} | evals -> payout {r["paid_e"]:4.0%} | payouts/funded {r["npay_f"]:3.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m median {r["k_med"]:+6,.0f} positive {r["k_pos"]:4.0%} | 2022-Oct26 {r["full"]:+7,.0f} | 2026 {r["y26"]:+5,.0f}', flush=True)
