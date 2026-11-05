"""Nov 4 2026: the whole prop lifecycle, not just the eval: eval -> funded (XFA) -> payouts or blown, and how many evals until the first dollar. DECLARED BEFORE RESULTS.
Trade: the base NQ flip (QQQ 15m vector flip 09:30-11:30, both ways, <= 2 a day, 75/50, flat 12:00, 1 pt cost), daily $ from the QQQ proxy, 2022 - Oct 2026.
RULES (Topstep-style 50K, Nov 2026 help pages; fees assumed, check your plan):
  EVAL   +$3,000 target, $2,000 trailing max loss (end of day, stops trailing at the start balance), $1,000 day floor, best day <= 50% of profit to pass,
         no time limit but $49 per 21 sessions while it runs. Fail -> buy a new one the next session.
  FUNDED $149 activation. Balance starts at $0, $2,000 trailing max loss (end of day) that stops trailing at $0, $1,000 day floor.
         STANDARD payout path: 5 winning days of >= $150 since the last payout (+ net profit since the last payout > 0 after the first),
         request 50% of the balance up to $2,000 (min $125), trader keeps 90% minus a $30 fee; after ANY payout the max loss locks at $0 for good
         (the cushion is whatever balance is left). CONSISTENCY path: >= 3 trading days and best day <= 40% of profit since the last payout, cap $3,000.
         Requested at the end of the first eligible day. Blown -> buy a new eval the next session.
VARIANTS  A eval 1 NQ, funded 1 NQ | B eval 1 NQ, funded 0.5 NQ when the 60-day profile path to target is thin (else 1 NQ) | C eval 1 NQ, funded 0.5 NQ flat
          A-cons = A on the consistency payout path | COIN = A with random direction (10 draws, medians).
OUTPUT  per eval bought (a start on every session 2022 - 2025): pass / fail, sessions, fees;  per funded account: blown before the first payout, >= 1 payout, payouts, $ received;
        evals per first payout; $ net per eval bought;  12-MONTH CAREERS starting each month 2022-01 .. 2025-10 (one account at a time, rebuy on every fail / blow):
        % paid at least once, evals bought before the first payout, net $ after 12 months (received - fees);  and one career Jan 2022 -> Oct 2026."""
_s = open('xs/nq_vp_nodes.py').read(); exec(_s[:_s.index("R = run_l(C); Z = ")])
R = run_l(C); D_ = days[days >= P0]; thin = (C['AHEAD_60D'] == 'LVN path ahead').loc[R.index].values
dvA = (R.net * 20).groupby(R.day).sum().reindex(D_).fillna(0.0).values
dvB = (R.net * np.where(thin, 10, 20)).groupby(R.day).sum().reindex(D_).fillna(0.0).values
dvC = dvA / 2
def eval_phase(v, s, end):
    bal = peak = best = 0.0; floor = -2000.0; fees = 49.0
    for k in range(s, end):
        if k > s and (k - s) % 21 == 0: fees += 49.0
        day = max(v[k], -1000.0); bal += day; best = max(best, day)
        if bal <= floor: return 'fail', k, fees
        if bal >= 3000 and best <= .5 * bal: return 'pass', k, fees
        peak = max(peak, bal); floor = min(max(floor, peak - 2000), 0.0)
    return 'open', end, fees
def funded(v, s, end, path='std'):
    bal = peak = since = best = 0.0; floor = -2000.0; locked = False; wins = tdays = 0; pays = []
    for k in range(s, end):
        day = max(v[k], -1000.0); bal += day; since += day
        if bal <= floor: return 'lost', k, pays
        if not locked: peak = max(peak, bal); floor = min(max(floor, peak - 2000), 0.0)
        wins += day >= 150; tdays += v[k] != 0; best = max(best, day)
        if path == 'std': ok = wins >= 5 and bal > 0 and (not pays or since > 0); cap = 2000.0
        else: ok = tdays >= 3 and since > 0 and bal > 0 and best <= .4 * since; cap = 3000.0
        amt = min(.5 * bal, cap)
        if ok and amt >= 125:
            pays.append((k, amt)); bal -= amt; floor = 0.0; locked = True; wins = tdays = 0; since = best = 0.0
    return 'open', end, pays
def attempts(v, w, path='std'):
    n = len(v); last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))); out = []
    for s in range(last):
        res, k, fees = eval_phase(v, s, n); rec = dict(res=res, ev_days=k - s + 1, fees=fees, fres=None, npay=0, got=0.0, f_days=0)
        if res == 'pass':
            rec['fees'] += 149.0; fr, k2, pays = funded(w, k + 1, n, path)
            rec.update(fres=fr, npay=len(pays), got=sum(.9 * a - 30 for _, a in pays), f_days=k2 - k)
        out.append(rec)
    return pd.DataFrame(out)
def career(v, w, s, end, path='std'):
    k = s; fees = got = 0.0; evals = blown = 0; first = None; ev_first = fees_first = None
    while k < end:
        evals += 1; res, k2, f = eval_phase(v, k, end); fees += f
        if res == 'fail': k = k2 + 1; continue
        if res == 'open': break
        fees += 149.0; fr, k3, pays = funded(w, k2 + 1, end, path)
        for kp, a in pays:
            got += .9 * a - 30
            if first is None: first, ev_first, fees_first = kp, evals, fees
        if fr == 'lost': blown += 1; k = k3 + 1; continue
        break
    return dict(evals=evals, blown=blown, fees=fees, got=got, net=got - fees, paid=first is not None, ev_first=ev_first, fees_first=fees_first,
                days_first=(first - s + 1) if first is not None else None)
def summarize(v, w=None, path='std'):
    w = v if w is None else w; A = attempts(v, w, path); F = A[A.res == 'pass']; p_pay = (A.npay > 0).mean()
    starts = [int(np.searchsorted(D_, pd.Timestamp(f'{y}-{m:02d}-01'))) for y in range(2022, 2026) for m in range(1, 13) if (y, m) <= (2025, 10)]
    K = pd.DataFrame([career(v, w, s, min(s + 252, len(v)), path) for s in starts]); full = career(v, w, 0, len(v), path)
    return dict(passed=(A.res == 'pass').mean(), failed=(A.res == 'fail').mean(), ev_days=A.ev_days.median(), fee=A.fees.mean(),
                f_blown0=((F.fres == 'lost') & (F.npay == 0)).mean(), f_paid=(F.npay > 0).mean(), f_open0=((F.fres == 'open') & (F.npay == 0)).mean(),
                f_npay=F.npay.mean(), f_got=F.got.mean(), f_got_paid=F[F.npay > 0].got.mean(), f_days=F.f_days.median(),
                p_pay=p_pay, ev_per_pay=1 / p_pay if p_pay else np.inf, net_attempt=(A.got - A.fees).mean(),
                k_paid=K.paid.mean(), k_evfirst=K.ev_first.median(), k_daysfirst=K.days_first.median(), k_net_med=K.net.median(), k_net_p10=K.net.quantile(.1),
                k_net_p90=K.net.quantile(.9), k_net_mean=K.net.mean(), k_blown=K.blown.mean(), k_evals=K.evals.mean(),
                full_net=full['net'], full_evals=full['evals'], full_blown=full['blown'], full_got=full['got'], full_fees=full['fees'], full_evfirst=full['ev_first'])
rows = {'A  eval 1 NQ, funded 1 NQ': summarize(dvA), 'B  funded 0.5 NQ on thin 60D path': summarize(dvA, dvB), 'C  funded 0.5 NQ flat': summarize(dvA, dvC),
        'A-cons  consistency payout path': summarize(dvA, path='cons'), 'B-all  0.5 NQ on thin path in eval too': summarize(dvB, dvB)}
rng = np.random.default_rng(3); coin = []
for _ in range(10):
    Rz = run_l(C.assign(dir=rng.choice([-1, 1], len(C)))); coin.append(summarize((Rz.net * 20).groupby(Rz.day).sum().reindex(D_).fillna(0.0).values))
rows['COIN  random direction (median of 10)'] = pd.DataFrame(coin).median().to_dict()
T = pd.DataFrame(rows).T
def show(title, cols):
    print(f'\n{title}', flush=True)
    for nm, r in T.iterrows(): print(f'  {nm:<40} ' + ' | '.join(f(r) for f in cols), flush=True)
show('PER EVAL BOUGHT (starts on every session 2022 - 2025)', [lambda r: f'passed {r.passed:4.0%} failed {r.failed:4.0%}', lambda r: f'median {r.ev_days:3.0f} sessions', lambda r: f'avg fees {r.fee:4.0f}'])
show('PER FUNDED ACCOUNT (after a pass)', [lambda r: f'blown before 1st payout {r.f_blown0:4.0%}', lambda r: f'>= 1 payout {r.f_paid:4.0%}', lambda r: f'still open, unpaid {r.f_open0:3.0%}',
     lambda r: f'avg payouts {r.f_npay:3.1f}', lambda r: f'avg $ to you {r.f_got:+6,.0f} (paid ones {r.f_got_paid:+6,.0f})', lambda r: f'median life {r.f_days:3.0f} sessions'])
show('ECONOMICS PER EVAL BOUGHT', [lambda r: f'evals bought -> paid {r.p_pay:4.1%}', lambda r: f'evals per first payout {r.ev_per_pay:4.1f}', lambda r: f'net $ per eval bought {r.net_attempt:+5,.0f}'])
show('12-MONTH CAREERS (start each month 2022-01 .. 2025-10, rebuy on every fail / blow)', [lambda r: f'paid at least once {r.k_paid:4.0%}', lambda r: f'evals before 1st payout (median) {r.k_evfirst:3.0f}',
     lambda r: f'sessions to 1st payout {r.k_daysfirst:3.0f}', lambda r: f'evals bought {r.k_evals:4.1f}, funded blown {r.k_blown:3.1f}',
     lambda r: f'net $ median {r.k_net_med:+6,.0f} (10th {r.k_net_p10:+6,.0f} / 90th {r.k_net_p90:+6,.0f}, mean {r.k_net_mean:+6,.0f})'])
show('ONE CAREER Jan 2022 -> Oct 2026', [lambda r: f'evals bought {r.full_evals:3.0f}, funded blown {r.full_blown:3.0f}', lambda r: f'evals before 1st payout {r.full_evfirst}',
     lambda r: f'fees {r.full_fees:6,.0f}, received {r.full_got:7,.0f}, net {r.full_net:+7,.0f}'])
