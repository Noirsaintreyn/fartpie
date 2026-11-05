"""Nov 4 2026: (1) 2026 trade by trade through the Lucid 50K eval, final pairing (eval: Rule A, gap-gated, Yang-Zhang + VWAP ladder, x0.3 in cold streaks; funded: x0.7, no gap gate).
(2) how long accounts last: evals and funded accounts started on every session 2022-2025 (same pairing): eval length, funded life, payouts, share still alive at the data end."""
_rt = open('xs/nq_reach_tp.py').read(); exec(_rt[:_rt.rindex("if __name__ == '__main__':")])
m03 = VARS2['EQ20 x0.3'][0]; m07 = VARS2['EQ20 x0.7'][0]
ev = seq('T75', ~AG['GAP']); s26 = int(np.searchsorted(D_, pd.Timestamp('2026-01-01')))
rows = []
for idx, (net, qe, p, how) in ev.items():
    d = pd.Timestamp(XA.at[idx, 'day'])
    if d < pd.Timestamp('2026-01-01') or d not in dpos: continue
    k = dpos[d]; yzh = bool(YZF.iloc[k]); vx = bool(X.at[idx, 'VWAPX']); lad = (1.0, .5, .3)[min(int(yzh) + int(vx), 2)]
    mic = max(1, int(round(10 * lad * m03.loc[idx]))); j = int(XA.at[idx, 'j']); pa = np.array(p) - 1.0
    rows.append(dict(day=d, time=f'{m5[j] // 60:02d}:{m5[j] % 60:02d}', side='long' if XA.at[idx, 'dir'] > 0 else 'short', mnq=mic, cold=m03.loc[idx] < 1, yz_high=yzh, vwap_stretched=vx,
                     exit=how, pts=round(net - 1.0, 1), usd=round((net - 1.0) * 2 * mic), mae_usd=round(pa.min() * 2 * mic)))
T = pd.DataFrame(rows).sort_values(['day', 'time']).reset_index(drop=True)
print('2026 EVAL TRADES (Lucid 50K rules replayed: $2,000 trailing EOD max loss locking at +$100, +$3,000 target, best day <= 50%, own stop after -$1,000 on the day)', flush=True)
bal = peak = 0.0; floor = -2000.0; best = 0.0; ev_n = 1; start = T.day.iloc[0] if len(T) else None; out = []
for d, g in T.groupby('day'):
    cum = 0.0; dead = False
    for _, r in g.iterrows():
        if cum <= -1000: break
        if bal + cum + r.mae_usd <= floor: out.append((d, r, bal + cum + r.mae_usd, 'BREACH')); cum = floor - bal; dead = True; break
        cum += r.usd; out.append((d, r, bal + cum, ''))
    for (dd, r, b, tag) in [x for x in out if x[0] == d]:
        print(f'  {dd:%b %d} {r.time} {r.side:<5} {r.mnq:>2} MNQ{" (cold)" if r.cold else "       "} {r.exit:<6} {r.pts:+6.1f} pts {r.usd:+6,.0f} | eval #{ev_n} balance {b:+6,.0f} floor {floor:+6,.0f}{"  " + tag if tag else ""}', flush=True)
    if dead:
        print(f'   -> eval #{ev_n} FAILED on {d:%b %d} (max loss hit intraday); next eval starts the next session', flush=True); ev_n += 1; bal = peak = best = 0.0; floor = -2000.0; continue
    bal += cum; best = max(best, cum)
    if bal >= 3000 and best <= .5 * bal: print(f'   -> eval #{ev_n} PASSED on {d:%b %d}', flush=True); ev_n += 1; bal = peak = best = 0.0; floor = -2000.0; continue
    peak = max(peak, bal); floor = min(max(floor, peak - 2000), 100.0)
print(f'  -> eval #{ev_n} still open at Oct 8: balance {bal:+,.0f}, needs {3000 - bal:+,.0f} more', flush=True)
print(f'\n2026 summary: {len(T)} eval trades, win {np.mean(T.pts > 0):.0%}, avg {T.pts.mean():+.1f} pts, {T.usd.sum():+,.0f} $ at eval size; by exit ' + str(T.exit.value_counts().to_dict()) +
      f'; longs {T[T.side == "long"].pts.mean():+.1f} pts (n{(T.side == "long").sum()}), shorts {T[T.side == "short"].pts.mean():+.1f} (n{(T.side == "short").sum()}); cold-streak size on {T.cold.mean():.0%} of trades', flush=True)
# ---- (2) account life
f = LUC; last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))); L = []
for s in range(last):
    res, k, fees = evS(f, s, NS, DS['e_EQ20 x0.3'])
    r = dict(start=D_[s], res=res, ev_len=k - s + 1)
    if res == 'pass':
        fr, k2, pays = fuS(f, k + 1, NS, DS['f_T75'], DS['f_T75']); r.update(fres=fr, life=k2 - k, npay=len(pays), got=sum(a for _, a in pays), first=(pays[0][0] - k) if pays else np.nan)
    L.append(r)
L = pd.DataFrame(L); P_ = L[L.res == 'pass']
print('\nACCOUNT LIFE (evals started on every session 2022-2025, Lucid 50K, final pairing)', flush=True)
print(f'  eval: passed {np.mean(L.res == "pass"):.0%}, failed {np.mean(L.res == "fail"):.0%} | median length {L.ev_len.median():.0f} sessions (passes {L[L.res == "pass"].ev_len.median():.0f}, fails {L[L.res == "fail"].ev_len.median():.0f})', flush=True)
print(f'  funded: blown {np.mean(P_.fres == "lost"):.0%}, still alive at Oct 8 2026 {np.mean(P_.fres == "open"):.0%} | median life {P_.life.median():.0f} sessions (blown ones {P_[P_.fres == "lost"].life.median():.0f}, alive ones {P_[P_.fres == "open"].life.median():.0f}) '
      f'| payouts per funded: median {P_.npay.median():.0f}, mean {P_.npay.mean():.1f}, 25% get >= {P_.npay.quantile(.75):.0f} | $ per funded: median {P_.got.median():+,.0f}, mean {P_.got.mean():+,.0f} | first payout after median {P_.first.median():.0f} sessions', flush=True)
print('  payouts per funded account: ' + str(P_.npay.value_counts().sort_index().to_dict()), flush=True)
