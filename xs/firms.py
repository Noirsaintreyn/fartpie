"""Nov 9 2026: WHICH PROP FIRM'S RULES FIT OUR SYSTEM? DECLARED BEFORE RESULTS.
Trades: the current setup - profit algo (U3 + YZ) + rules algo C + TPO, no rules entries after 13:30 on FOMC days (xs/fleet.py), same trades in every firm.
Rules (third-party summaries checked against each firm's help center Oct - Nov 2026: copilink.com Lucid 4 Nov 2026; proptradingvibes.com Lucid payouts 19 Oct 2026;
tradervana.com Apex 22 Oct 2026, Take Profit Trader 13 Oct 2026, MyFundedFutures 13 Oct 2026; tradecovex.com MFFU prices). 50K / 100K / 150K:
  LUCID FLEX (now)  eval EOD trailing, 50% consistency (add-on); funded EOD, floor locks at +100; payout = 50% of profit up to 2000/2500/3000 after 5 days >= 150/200/250; 90%; 5 payouts
  LUCID PRO         eval EOD, NO consistency; funded EOD, lock +100; payout when cycle profit >= 500/750/1000, best day <= 40% of cycle profit, from balance above buffer +2100/+3100/+4600;
                    caps 2000 then 2500 (50K) / 2500 then 3000 / 3000 then 3500; 90%; 5 payouts
  LUCID DAILY       eval EOD, 50% consistency; funded INTRADAY trailing, lock +100; payout any profit above buffer +2100/+3100/+4600 (>= $500, net positive since last), no cap; 90%
  APEX (EOD)        eval EOD trailing 2000/3000/4000, NO consistency, 30-calendar-day limit, daily loss 1000/1500/2000; PA EOD, lock +100; payout after 5 days >= 250/300/350,
                    largest day <= 50% of profit since last payout, keep safety net +2100/+3100/+4100, weekly at most; caps 1500,1500,2000,2500,2500,3000 (50K) /
                    2000,2500,2500,3000,4000,4000 / 2500,3000,3000,3000,4000,5000; 100% split; 6 payouts then the PA closes; up to 20 PAs
  TAKE PROFIT (PRO) test EOD trailing, 50% consistency, 3 days min; PRO INTRADAY trailing until the floor reaches the start balance; withdraw anything above buffer +2000/+3000/+4500
                    from day one; no consistency; 80%; fee $170/$330/$360 a month while testing + $130 activation; up to 5 funded
  MFFU RAPID        eval EOD, 50% consistency, 2 days min; funded INTRADAY trailing, lock +100; payouts daily above buffer +2100/+3100/+4600; no consistency; 90%; one-time fee
Intraday trailing is modeled on closed-trade equity (open-trade peaks unknown) - optimistic for the intraday firms. Payout policy for every firm (Rey, Nov 9 run 2): withdraw AS SOON AS ALLOWED - the minimum request ($500) or more, the first
day each firm's rules allow (run 1 waited for >= $1,350 to you; log kept as xs/firms_1350.log). Our own daily stop (-$1,000 x size) runs in every firm.
Fees (list, before promo codes): Lucid Flex $88 50K / $244 150K (Rey's cost with his code), Lucid Pro $172 50K, Lucid Daily ~$108 50K (Phidias, Aug - Oct 2026),
Apex EOD $490 / $790 / $1,490 + $129 activation on a 50K pass (proptradingvibes 10 Aug 2026; Apex runs deep promos), TPT $170/$330/$360 a month + $130, MFFU Rapid $125.
Accounts started every 5th trading day Jan 2022 - Jun 2025 (the rules algo never saw 2022+). Report: eval pass rate and days, funded payouts per month, size, blown,
$ to you per funded month, expected $ paid per eval bought (= the most an eval can cost and break even)."""
_f = open('/home/claude/fartpie/xs/fleet.py').read(); exec(_f[:_f.rindex("if __name__ == '__main__':")])
LE, LF = lists(spec(1.0, 1.0), 'e'), lists(spec(1.0, 1.0), 'f')
SZ = {'50K': dict(scale=1.0, i=0), '100K': dict(scale=1.5, i=1), '150K': dict(scale=2.25, i=2)}
def F(base, **kw): d = dict(base); d.update(kw); return d
FIRM = {
 'Lucid Flex': dict(target=(3000, 6000, 9000), mll=(2000, 3000, 4500), lock=100, econs=.5, emin=0, tlim=None, e_intra=False, f_intra=False, pay='flex', qday=(150, 200, 250), nq=5,
                    caps=((2000,) * 5, (2500,) * 5, (3000,) * 5), split=.9, maxpay=5, fee=(88, None, 244), monthly=None),
 'Lucid Pro': dict(target=(3000, 6000, 9000), mll=(2000, 3000, 4500), lock=100, econs=None, emin=0, tlim=None, e_intra=False, f_intra=False, pay='pro', cyc=(500, 750, 1000),
                   buf=(2100, 3100, 4600), fcons=.4, caps=((2000, 2500, 2500, 2500, 2500), (2500, 3000, 3000, 3000, 3000), (3000, 3500, 3500, 3500, 3500)), split=.9, maxpay=5, fee=(172, None, None), monthly=None),
 'Lucid Daily': dict(target=(3000, 6000, 9000), mll=(2000, 3000, 4500), lock=100, econs=.5, emin=0, tlim=None, e_intra=False, f_intra=True, pay='buffer', buf=(2100, 3100, 4600),
                     split=.9, maxpay=99, fee=(108, None, None), monthly=None),
 'Apex EOD': dict(target=(3000, 6000, 9000), mll=(2000, 3000, 4000), lock=100, econs=None, emin=0, tlim=30, e_intra=False, f_intra=False, pay='apex', qday=(250, 300, 350), nq=5,
                  buf=(2100, 3100, 4100), caps=((1500, 1500, 2000, 2500, 2500, 3000), (2000, 2500, 2500, 3000, 4000, 4000), (2500, 3000, 3000, 3000, 4000, 5000)), split=1.0, maxpay=6,
                  fee=(490, 790, 1490), act=(129, None, None), monthly=None),
 'Take Profit PRO': dict(target=(3000, 6000, 9000), mll=(2000, 3000, 4500), lock=0, econs=.5, emin=3, tlim=None, e_intra=False, f_intra=True, pay='buffer', buf=(2000, 3000, 4500),
                         split=.8, maxpay=99, fee=(130, 130, 130), monthly=(170, 330, 360)),
 'MFFU Rapid': dict(target=(3000, 6000, 9000), mll=(2000, 3000, 4500), lock=100, econs=.5, emin=2, tlim=None, e_intra=False, f_intra=True, pay='buffer', buf=(2100, 3100, 4600),
                    split=.9, maxpay=99, fee=(125, None, None), monthly=None)}
def run_day(L, k, st, mll, lock, scale, intra):
    cum = 0.0; bal = st['bal']
    for th, net, mae, p in L[k]:
        if cum <= -1000 * scale: break
        mm = 2.0 * max(1, int(round(10 * th[0] * scale)))
        if bal + cum + mae * mm <= st['floor']: return st['floor'] - bal, True
        cum += net * mm
        if intra and st['floor'] < lock:
            st['peak'] = max(st['peak'], bal + cum); st['floor'] = max(st['floor'], min(st['peak'] - mll, lock))
    return cum, False
def eod(st, mll, lock):
    if st['floor'] < lock: st['peak'] = max(st['peak'], st['bal']); st['floor'] = max(st['floor'], min(st['peak'] - mll, lock))
def eval_run(R, z, s, end):
    i = z['i']; mll = R['mll'][i]; st = dict(bal=0.0, floor=-mll, peak=0.0); best = 0.0; nd = 0
    for k in range(s, end):
        if R['tlim'] and (D_[k] - D_[s]).days > R['tlim']: return 'fail', k
        pnl, br = run_day(LE, k, st, mll, R['lock'], z['scale'], R['e_intra'])
        if br: return 'fail', k
        st['bal'] += pnl; best = max(best, pnl); nd += len(LE[k]) > 0
        if st['bal'] >= R['target'][i] and (R['econs'] is None or best <= R['econs'] * st['bal']) and nd >= R['emin']: return 'pass', k
        eod(st, mll, R['lock'])
    return 'open', end
def fund_run(R, z, s, end):
    i = z['i']; mll = R['mll'][i]; st = dict(bal=0.0, floor=-mll, peak=0.0); pays = []; since = 0.0; qd = 0; bestc = 0.0; lastk = -99; want = 500.0
    for k in range(s, end):
        pnl, br = run_day(LF, k, st, mll, R['lock'], z['scale'], R['f_intra'])
        if br: return 'lost', k, pays
        st['bal'] += pnl; since += pnl; bestc = max(bestc, pnl); eod(st, mll, R['lock']); b = st['bal']; amt = 0.0; n = len(pays)
        if R['pay'] == 'flex':
            qd += pnl >= R['qday'][i]
            if qd >= R['nq'] and b > 0 and (not pays or since > 0):
                amt = min(.5 * b, R['caps'][i][min(n, len(R['caps'][i]) - 1)]); amt = amt if amt >= 500 else 0.0
        elif R['pay'] == 'pro':
            cap = R['caps'][i][min(n, 4)]; av = b - R['buf'][i]
            if since >= R['cyc'][i] and bestc <= R['fcons'] * since and av >= min(want, cap) and av >= 500: amt = min(cap, av)
        elif R['pay'] == 'apex':
            qd += pnl >= R['qday'][i]; cap = R['caps'][i][min(n, 5)]; av = b - R['buf'][i]
            if qd >= R['nq'] and since > 0 and bestc <= .5 * since and av >= 500 and av >= min(want, cap) and k - lastk >= 5: amt = min(cap, av)
        else:
            av = b - R['buf'][i]
            if av >= want and (not pays or since > 0): amt = av
        if amt > 0:
            pays.append((k, amt * R['split'])); st['bal'] -= amt; since = 0.0; qd = 0; bestc = 0.0; lastk = k
            if R['pay'] == 'flex': st['floor'] = max(st['floor'], R['lock'])
            if len(pays) >= R['maxpay']: return 'graduated', k, pays
    return 'open', end, pays
if __name__ == '__main__':
    LOCK['on'] = False; starts = list(range(k22, k25, 5))
    print(f'accounts started every 5th trading day, {D_[starts[0]].date()} - {D_[starts[-1]].date()} ({len(starts)} starts), data to {D_[NS - 1].date()}', flush=True)
    print(f'\n  {"firm":<16} {"size":<5} pass  days | funded: payouts/mo x avg to you, blown, $/funded mo | $ paid per eval bought | fee (list) -> net per eval, per account-month', flush=True)
    for fn, R in FIRM.items():
        for zn, z in SZ.items():
            i = z['i']; Rr = []
            for s in starts:
                r, k2 = eval_run(R, z, s, NS)
                if r != 'pass': Rr.append((r, s, k2, [], 'none', k2 if r == 'fail' else NS)); continue
                fr, k3, pays = fund_run(R, z, k2 + 1, NS); Rr.append((r, s, k2, pays, fr, NS if fr == 'open' else k3))
            fund = [x for x in Rr if x[0] == 'pass']; pays = [p for x in fund for _, p in x[3]]; flife = sum((x[5] - x[2]) / 21 for x in fund)
            dpass = np.median([D_[x[2]].toordinal() - D_[x[1]].toordinal() for x in fund]) if fund else np.nan
            per_eval = sum(pays) / len(Rr)
            fee = R['fee'][i]
            if R['monthly'] is not None:
                fees = [R['monthly'][i] * (1 + (D_[x[2]] - D_[x[1]]).days // 30) + (R['fee'][i] if x[0] == 'pass' else 0) for x in Rr]; fee_avg = np.mean(fees)
            else: fee_avg = None if fee is None else fee + (R.get('act', (0, 0, 0))[i] or 0) * len(fund) / len(Rr)
            life = sum((x[5] - x[1] + 1) / 21 for x in Rr)
            net_txt = (f'${fee_avg:5.0f} -> ${per_eval - fee_avg:+6.0f}, ${(sum(pays) - fee_avg * len(Rr)) / life:+5.0f}/mo') if fee_avg is not None else 'fee n/a'
            print(f'  {fn:<16} {zn:<5} {len(fund) / len(Rr):4.0%} {dpass:4.0f}d | {len(pays) / flife if flife else 0:4.2f} x ${np.mean(pays) if pays else 0:5.0f}, '
                  f'{np.mean([x[4] == "lost" for x in fund]) if fund else 0:4.0%}, ${sum(pays) / flife if flife else 0:5.0f} | ${per_eval:6.0f} | {net_txt}', flush=True)
