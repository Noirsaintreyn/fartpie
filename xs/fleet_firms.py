"""Nov 9 2026: 5 ACCOUNTS, EVERY MONTH, ALL YEARS - what the system pays. DECLARED BEFORE RESULTS.
Trades: current setup (profit algo + rules algo C + TPO, FOMC rule; xs/fleet.py). Firm rules: xs/firms.py, withdrawing as soon as allowed.
Fleet: 5 active accounts at most; buy ONE eval a day while under 5, only while the system's trailing 20-day P&L is positive, from a cash pile that starts at 20 eval fees
and is refilled by payouts (xs/fleet4.py policy). Feb 2018 - Jan 2026. NOTE: the rules algo was trained 2014-19 and tuned 2020-21 (in-sample before 2022);
the profit algo's engines were picked with 2018-25 in view - treat every year as optimistic, 2022+ least so.
Configs: Lucid Flex 50K ($88), Lucid Flex 150K ($244), Lucid Pro 50K ($172 list), Lucid Daily 150K (price not published - $250 assumed).
Report per year: payouts, net $ a month (payouts - eval fees), worst / best month, months >= $7.5k and >= $10k, losing months; the system's NQ points a month at
1 contract (one 50K account's size); and the whole run: average, median, share of months >= $7.5k, longest run of months under $7.5k."""
_fm = open('/home/claude/fartpie/xs/firms.py').read(); exec(_fm[:_fm.rindex("if __name__ == '__main__':")])
import heapq
FIRM['Lucid Daily']['fee'] = (108, None, 250)
CFGS = [('Lucid Flex', '50K'), ('Lucid Flex', '150K'), ('Lucid Pro', '50K'), ('Lucid Daily', '150K')]
pts_day = pd.Series([sum(net * th[0] for th, net, mae, p in z) for z in LF], index=D_)
if __name__ == '__main__':
    LOCK['on'] = False
    pm = pts_day.groupby(pts_day.index.to_period('M')).sum()
    print('system NQ points a month at 1 contract (one 50K account): by year avg / worst month: ' + ' | '.join(
        f'{y}: {pm[pm.index.year == y].mean():+.0f} / {pm[pm.index.year == y].min():+.0f}' for y in sorted(set(pm.index.year))), flush=True)
    for fn, zn in CFGS:
        R = FIRM[fn]; z = SZ[zn]; i = z['i']; fee = R['fee'][i]; TRJ = {}
        pnl = pd.Series([sum(net * 2 * max(1, round(10 * th[0] * z['scale'])) for th, net, mae, p in zz) for zz in LE]); hot = (pnl.rolling(20).sum().shift(1) > 0).values
        def traj(s):
            if s not in TRJ:
                r, k2 = eval_run(R, z, s, NS)
                if r != 'pass': TRJ[s] = (k2 if r == 'fail' else NS, [])
                else: fr, k3, pays = fund_run(R, z, k2 + 1, NS); TRJ[s] = (NS if fr == 'open' else k3, pays)
            return TRJ[s]
        cash = 20 * fee; act = []; pq = []; P = {}; Cn = {}; Fe = {}; bought = 0
        for k in range(0, NS):
            mo = D_[k].to_period('M')
            while pq and pq[0][0] == k:
                kk, x = heapq.heappop(pq); cash += x; P[mo] = P.get(mo, 0) + x; Cn[mo] = Cn.get(mo, 0) + 1
            act = [e for e in act if e >= k]
            if len(act) < 5 and cash >= fee and hot[k]:
                end, pays = traj(k); cash -= fee; bought += 1; act.append(end); Fe[mo] = Fe.get(mo, 0) + fee
                for kk, x in pays: heapq.heappush(pq, (kk, x))
        months = pd.period_range(D_[0].to_period('M'), D_[NS - 1].to_period('M'), freq='M')
        Ps = pd.Series({mo: P.get(mo, 0.0) for mo in months}); Fs = pd.Series({mo: Fe.get(mo, 0.0) for mo in months}); Cs = pd.Series({mo: Cn.get(mo, 0) for mo in months}); N = Ps - Fs
        print(f'\n=== {fn} {zn} x 5 accounts (fee ${fee}, start cash ${20 * fee:,}) - {bought} evals bought ===', flush=True)
        print(f'  {"year":<5} payouts  net $/mo   worst mo   best mo  mo>=7.5k mo>=10k  losing mo', flush=True)
        for y in sorted(set(months.year)):
            n_ = N[N.index.year == y]; c_ = Cs[Cs.index.year == y]
            print(f'  {y:<5} {c_.sum():7d} {n_.mean():9,.0f} {n_.min():10,.0f} {n_.max():9,.0f} {int((n_ >= 7500).sum()):6d}/{len(n_):<2d} {int((n_ >= 10000).sum()):5d} {int((n_ < 0).sum()):8d}', flush=True)
        run = mx = 0
        for v in N.values: run = run + 1 if v < 7500 else 0; mx = max(mx, run)
        for lab, nn in (('all years', N), ('2022+', N[N.index.year >= 2022])):
            print(f'  {lab:<9}: average ${nn.mean():,.0f}/mo, median ${nn.median():,.0f}, months >= $7.5k {np.mean(nn >= 7500):.0%}, >= $10k {np.mean(nn >= 10000):.0%}, '
                  f'losing {np.mean(nn < 0):.0%}', flush=True)
        print(f'  longest run of months under $7.5k: {mx}; total paid ${Ps.sum():,.0f}, fees ${Fs.sum():,.0f}', flush=True)
