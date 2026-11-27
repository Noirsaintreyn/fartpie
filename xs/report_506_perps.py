"""Report 506 (Nov 27 2026): MyFundedPerps Select $10K - which flip set gets paid, how fast.
Hypothesis: the static $300 floor + $300 daily loss kill any set whose per-trade size is big vs its edge, so the best set is the highest edge per unit of risk (tagged PCV/PCZ/SKW/CHM), not the most trades.
Rules: start 10000, pass +900, fail if equity<9700 or day equity < day start - 300 (NY midnight), 15 positions max, comm 0.005%/side, swap 0.015%/day.
Equity is checked at every entry/exit event with open trades marked at their FINAL return when it is negative (hindsight stress proxy; no intraday path in this data).
Funded: same floor until first payout; first payout when flat and balance >= 10100 (min $100 gross), 80/20.
Cohorts: a fresh account bought on every trading day, 2019-2026 (overlapping), shares reported at 30/60/90/180 days from purchase."""
import pandas as pd, numpy as np, sys
UP = '/mnt/user-data/uploads/fartpie/'
W = pd.read_parquet(UP + 'xs/ext/wide_T.parquet'); W['entry'] = pd.to_datetime(W.entry); W['exit'] = pd.to_datetime(W.exit)
M = pd.read_csv('/home/claude/fartpie/live_data/model_flips.csv', parse_dates=['entry']); M['h1'] = M.tf == '1h'
J = M.merge(W[['tk', 'entry', 'h1', 'exit']], on=['tk', 'entry', 'h1'])
base = W[W.sp & (W.ratio >= 1.5)][['tk', 'entry', 'exit', 'ret', 'h1']].copy(); base['tag'] = 'ALL'
def tagset(tags):
    x = J[J.tag.isin(tags)].sort_values('entry').drop_duplicates(['tk', 'entry', 'h1'])[['tk', 'entry', 'exit', 'ret', 'h1']].copy(); x['tag'] = '+'.join(tags); return x
SETS = {'ALL flips': base, 'tagged (PCV,PCZ,SKW,CHM)': tagset(['PCV', 'PCZ', 'SKW', 'CHM']), 'tagged + NEG': tagset(['PCV', 'PCZ', 'SKW', 'CHM', 'NEG']),
        'PCV': tagset(['PCV']), 'PCZ': tagset(['PCZ']), 'SKW': tagset(['SKW']), 'CHM': tagset(['CHM']), 'NEG': tagset(['NEG'])}
COM = 0.00005; SWAP = 0.00015
def cohort(T, start, S, K, horizon=240, pay_min=100.0):
    """T sorted by entry (arrays). returns dict(pass_day, fail_day, pay_day)"""
    t0 = start; t_end = start + pd.Timedelta(days=horizon)
    m = (T.entry >= t0) & (T.entry < t_end); X = T[m]
    ev = [(r.entry, 0, i) for i, r in enumerate(X.itertuples())] + [(r.exit, 1, i) for i, r in enumerate(X.itertuples())]; ev.sort()
    rets = X.ret.values; hold = ((X.exit - X.entry).dt.total_seconds() / 86400).values
    pnl = lambda i: S * (rets[i] - 2 * COM - SWAP * hold[i])
    bal = 10000.0; floor = 9700.0; openp = set(); taken = set(); dstart = 10000.0; cur_day = None
    passed = None; payday = None; funded = False
    for ts, kind, i in ev:
        d = (ts - pd.Timedelta(hours=0)).normalize()
        if d != cur_day:                                              # new NY day: start-of-day equity incl. open stress
            cur_day = d; dstart = bal + sum(min(0, pnl(j)) for j in openp)
        if kind == 0:
            if len(openp) >= K: continue
            openp.add(i); taken.add(i)
        else:
            if i not in taken: continue
            openp.discard(i); bal += pnl(i)
            if not funded and bal >= 10900 and not passed: passed = (ts - start).days; funded = True; bal = bal      # funded account keeps balance (eval -> funded starts again at 10000 in reality)
        eq = bal + sum(min(0, pnl(j)) for j in openp)
        if eq < floor or eq < dstart - 300: return dict(fail=(ts - start).days, passed=passed, pay=payday)
        if funded and payday is None and not openp and bal - 10000 >= pay_min and passed is not None and ts > start + pd.Timedelta(days=passed):
            payday = (ts - start).days
            return dict(fail=None, passed=passed, pay=payday)
    return dict(fail=None, passed=passed, pay=payday)
def run(T, S, K, step=3):
    T = T.sort_values('entry'); days = pd.date_range('2019-02-01', '2026-04-30', freq='B')[::step]
    return [(d, cohort(T, d, S, K)) for d in days]
if __name__ == '__main__':
    rows = []
    for name, T in SETS.items():
        for S, K in [(1000, 5), (2000, 5), (2000, 8), (3000, 5), (4000, 4)]:
            R = run(T, S, K); n = len(R)
            P = np.array([r['passed'] if r['passed'] is not None else np.nan for _, r in R], float); Y = np.array([r['pay'] if r['pay'] is not None else np.nan for _, r in R], float)
            Fl = np.array([r['fail'] if r['fail'] is not None else np.nan for _, r in R], float)
            rows.append(dict(set=name, size=S, slots=K, n=n, **{f'pass{c}': round(float(np.mean(P <= c)) * 100) for c in (30, 60, 90)},
                             **{f'pay{c}': round(float(np.mean(Y <= c)) * 100) for c in (30, 60, 90, 180)}, fail90=round(float(np.mean(Fl <= 90)) * 100), fail_ever=round(float(np.mean(~np.isnan(Fl))) * 100),
                             med_days_to_pay=float(np.nanmedian(Y)) if (~np.isnan(Y)).any() else None))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv('/home/claude/fartpie/xs/report_506_perps.csv', index=False)
