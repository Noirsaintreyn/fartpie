"""Nov 4 2026: can a SECOND, uncorrelated prop account stop whole negative years? DECLARED BEFORE RESULTS.
Book F = final flip system (Rule A, Yang-Zhang + VWAP ladder in whole micros, gap-gated eval, last-20-trades cold-streak cut: x0.7 or x0.3).
Book O = opening-range breakout on NQ (QQQ proxy): first 5m close beyond the 09:30-10:00 range between 10:00 and 11:30, 75 / 50, one a day, flat 12:00, 1 pt cost,
         10 MNQ (5 MNQ when Yang-Zhang is in its top third) - no other filters.
Each book runs its own Lucid 50K ($88), one account at a time, rebuying after fails / blown accounts. Combined = both accounts at once (two evals' fees).
Report: calendar-year runs (start the first session of each year, stop at year end / Oct 8 2026): net per book and combined;
        12-month runs (start each month 2022-01 .. 2025-10): median, % negative, worst, per book and combined; trade-level correlation of the two books' daily P&L."""
_ad = open('xs/nq_adaptive_size.py').read(); exec(_ad[:_ad.rindex("if __name__ == '__main__':")])
rthm_ = (m5 >= 570) & (m5 < 960); OR_ = []
for d, ix in pd.Series(np.arange(len(H5))[rthm_]).groupby(d5[rthm_]):
    ii = ix.values; orb = ii[m5[ii] < 600]
    if len(orb) < 6: continue
    hi_, lo_ = h5[orb].max(), l5[orb].min()
    for i in ii[(m5[ii] >= 600) & (m5[ii] <= 685)]:
        if c5[i] > hi_ or c5[i] < lo_:
            if i + 1 < len(H5) and d5[i + 1] == d5[i]: OR_.append((pd.Timestamp(d), int(i + 1), 1 if c5[i] > hi_ else -1))
            break
ORC = pd.DataFrame(OR_, columns=['day', 'j', 'dir']); ORC['k'] = NQs.reindex(pd.DatetimeIndex(ORC.day) - pd.Timedelta(days=1), method='ffill').values / o5[ORC.j.values]; ORC = ORC[np.isfinite(ORC.k)].reset_index(drop=True)
TO = run_p(ORC, 75, 50, 1); DO = [[] for _ in D_]
for day, idx, net, mae, p in TO:
    if day in dpos: k = dpos[day]; DO[k].append((((.5 if bool(YZF.iloc[k]) else 1.0),), net, mae, p))
DS['ORB'] = DO
BOOKS = {'F x0.7': ('e_EQ20 x0.7', 'f_EQ20 x0.7', 'f_EQ20 x0.7'), 'F x0.3': ('e_EQ20 x0.3', 'f_EQ20 x0.3', 'f_EQ20 x0.3'), 'ORB': ('ORB', 'ORB', 'ORB')}
def dayP(D): return pd.Series([sum(net * 20 * th[0] for th, net, mae, p in x) for x in D], index=D_)
pf, po = dayP(DS['f_EQ20 x0.7']), dayP(DO); both = (pf != 0) & (po != 0)
print(f'ORB trades {len(TO)}; daily P&L correlation flip vs ORB (days both traded) {np.corrcoef(pf[both], po[both])[0, 1]:+.2f}; monthly {np.corrcoef(pf.groupby(D_.to_period("M")).sum(), po.groupby(D_.to_period("M")).sum())[0, 1]:+.2f}', flush=True)
yrs_ = [(y, int(np.searchsorted(D_, pd.Timestamp(f'{y}-01-01'))), int(np.searchsorted(D_, pd.Timestamp(f'{y + 1}-01-01')))) for y in range(2022, 2027)]
starts = [int(np.searchsorted(D_, pd.Timestamp(f'{y}-{m:02d}-01'))) for y in range(2022, 2026) for m in range(1, 13) if (y, m) <= (2025, 10)]
def job12(nm):
    v = BOOKS[nm]; f = LUC
    return nm, {y: carS(f, s, e, v)['net'] for y, s, e in yrs_}, [carS(f, s, min(s + 252, NS), v)['net'] for s in starts]
if __name__ == '__main__':
    with mp.get_context('fork').Pool(2) as pool: RES = dict((nm, (y, k)) for nm, y, k in pool.map(job12, list(BOOKS)))
    print('\nCALENDAR-YEAR RUNS, Lucid 50K (net after fees)', flush=True)
    for nm, (y, k) in RES.items(): print(f'  {nm:<8} ' + ' | '.join(f'{yy} {v:+7,.0f}' for yy, v in y.items()) + f' | negative years {sum(v < 0 for v in y.values())}/5', flush=True)
    for fn in ('F x0.7', 'F x0.3'):
        y = {yy: RES[fn][0][yy] + RES['ORB'][0][yy] for yy in RES[fn][0]}
        print(f'  {fn} + ORB ' + ' | '.join(f'{yy} {v:+7,.0f}' for yy, v in y.items()) + f' | negative years {sum(v < 0 for v in y.values())}/5', flush=True)
    print('\n12-MONTH RUNS (46 starts)', flush=True)
    for nm, (y, k) in RES.items(): k = np.array(k); print(f'  {nm:<14} median {np.median(k):+7,.0f} | negative {np.mean(k < 0):4.0%} | worst {k.min():+7,.0f} | best {k.max():+7,.0f}', flush=True)
    for fn in ('F x0.7', 'F x0.3'):
        k = np.array(RES[fn][1]) + np.array(RES['ORB'][1]); print(f'  {fn + " + ORB":<14} median {np.median(k):+7,.0f} | negative {np.mean(k < 0):4.0%} | worst {k.min():+7,.0f} | best {k.max():+7,.0f}', flush=True)
