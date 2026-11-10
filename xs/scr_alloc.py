"""Nov 9 2026: SLOT ALLOCATION TESTS FROM REY'S ATTACHED REVIEW (allocation, not another ranking indicator). DECLARED BEFORE RESULTS - rules frozen here, every trial logged.
Trade-level decisions are exact (who is taken, when slots free, breaker trips on REALIZED P&L at exits, as the live book); only the reported daily equity path
is the stand-in (each trade's return spread over its holding days). 6 slots, 8% breaker, 2 bp. 2019-22 / 2023-26 (Sep 25) / all. 2023-26 has informed earlier
choices, so treat everything here as exploratory unless it is clearly and consistently better.
A) OCCUPANCY: taken trades by track (15m / 1h), entry time, holding time: share of trades, of slot-days, of P&L, of the top-decile winners.
B) ROBUSTNESS of the ratio order: 60 paths shuffling only NEAR-TIES (same bar, ratio within 10% of each other) and 60 fully random same-bar paths.
C) NEAR-TIE DIVERSIFICATION: among same-bar candidates within 10% of the best remaining ratio, take the one least correlated (60-day daily returns, prior
   data) with current holdings | control: prefer a sector not already held. Ratio order otherwise; no holding is ever sold.
D) SECTOR CAP: at most 2 (and, separately, 3) open names per sector; ratio order otherwise.
E) CASH ACCOUNT SETTLEMENT ($700, no margin): T+2 before May 28 2024, T+1 after. Count good-faith-violation risks (a position bought with unsettled sale
   proceeds and sold before they settle), and the variant that only buys with SETTLED cash (a slot it cannot fund is skipped).
(Reserve-the-6th-slot, opening batch and the valid-signal queue enter at LATER prices -> need the 15m bars: added to xs/scr_intraday.py for the Mac run.)"""
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
SEC = json.load(open('/home/claude/fartpie/data/yahoo/sectors.json')); T0['sec'] = T0.tk.map(SEC).fillna('?')
RET = {}
for tk in T0.tk.unique():
    try: Y = pd.read_parquet(f'/home/claude/fartpie/data/yahoo/daily_v/{tk}.parquet'); RET[tk] = Y.close.pct_change()
    except Exception: pass
R = pd.DataFrame(RET)
from collections import namedtuple
Row = namedtuple('Row', ['tk', 'exit', 'ret', 'ratio', 'h1', 'sec'])
GROUPS = [(e, [Row(*v) for v in g[['tk', 'exit', 'ret', 'ratio', 'h1', 'sec']].itertuples(index=False, name=None)]) for e, g in T0.groupby('entry', sort=True)]
def corr_to(tk, held, day):
    if not held or tk not in R: return 0.0
    w = R.loc[(R.index < day)].tail(60); cs = [w[tk].corr(w[h]) for h in held if h in w and h != tk]
    return float(np.nanmean(cs)) if cs else 0.0
def run(rule=None, cap=None, rng=None, band=.10, settle=None, div=6):
    """settle: None | 'count' | 'only' (cash account ledger)"""
    N = 6; op = []; pend = []; real = 0.0; win = deque(); hist = []; lvl0 = 0.0; out = []
    cash_s = 1.0; unset = []; gfv = 0; skipped_cash = 0                       # settled cash, [(settle_time, amt)]
    def lag(t): return pd.Timestamp(t).normalize() + pd.offsets.BDay(2 if pd.Timestamp(t) < pd.Timestamp('2024-05-28') else 1)
    for e, grp in GROUPS:
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr, h in done:
            real += np.log1p(rr / N); hist.append((xt, real)); win.append((xt, real))
            if settle:
                proceeds = h['amt'] * (1 + rr)
                if h['fund_t'] is not None and xt < h['fund_t']: gfv += 1
                unset.append((lag(xt), proceeds))
        op = [h for h in op if h['x'] > e]
        if settle:
            keep = []
            for t_, a_ in unset:
                if t_ <= e: cash_s += a_
                else: keep.append((t_, a_))
            unset = keep
        rows = list(grp)
        if rng is not None and len(rows) > 1:
            if rule == 'near': rows = sorted(rows, key=lambda r_: (-round(np.log(r_.ratio) / np.log(1 + band)), rng.random()))
            else: rows = [rows[i] for i in rng.permutation(len(rows))]
        while rows:
            if len(op) >= N: break
            if rule in ('corr', 'sector') and rng is None:
                top = max(r_.ratio for r_ in rows); pool = [r_ for r_ in rows if r_.ratio >= top * (1 - band)]; held = [h['tk'] for h in op]
                if rule == 'corr': r = min(pool, key=lambda r_: (corr_to(r_.tk, held, e.normalize()), -r_.ratio))
                else: hs = {h['sec'] for h in op}; r = min(pool, key=lambda r_: (r_.sec in hs, -r_.ratio))
            else: r = rows[0]
            rows.remove(r)
            if cap and sum(h['sec'] == r.sec for h in op) >= cap: continue
            if any(h['tk'] == r.tk and h['h1'] == r.h1 for h in op): continue
            if hist:
                t0 = e - pd.Timedelta(days=28)
                while win and win[0][0] < t0: lvl0 = win.popleft()[1]
                if real - max([lvl0] + [hh for _, hh in win]) < np.log1p(-.08): break
            amt = 1.0 / N; fund_t = None
            if settle:
                equity = cash_s + sum(a for _, a in unset) + sum(h['amt'] for h in op); amt = equity / div
                if cash_s >= amt - 1e-12: cash_s -= amt
                elif settle == 'only': skipped_cash += 1; continue
                else:
                    need = amt - max(cash_s, 0); cash_s = 0.0; unset.sort(); used = []
                    for k_, (t_, a_) in enumerate(unset):
                        take = min(a_, need); need -= take; used.append(t_); unset[k_] = (t_, a_ - take)
                        if need <= 1e-12: break
                    unset = [(t_, a_) for t_, a_ in unset if a_ > 1e-12]; fund_t = max(used) if used else None
            h = dict(tk=r.tk, h1=r.h1, sec=r.sec, x=r.exit, amt=amt, fund_t=fund_t); op.append(h); pend.append((r.exit, r.ret, h))
            out.append((r.tk, pd.Timestamp(e), pd.Timestamp(r.exit), r.ret, r.ratio, r.h1))
    return out, dict(gfv=gfv, skipped_cash=skipped_cash)
def curve(TK): return daily([(a, b, c, d, e_) for a, b, c, d, e_, _ in TK], 6)[0]
def st4(s):
    out = []
    for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')):
        z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25
        out.append(f'{e.iloc[-1] ** (1 / y) - 1:6.1%} / DD {(e / e.cummax() - 1).min():6.1%}')
    e = (1 + s).cumprod(); return ' | '.join(out) + f' | all {e.iloc[-1] ** (1 / ((s.index[-1] - s.index[0]).days / 365.25)) - 1:6.1%}'
def cg(s, a='2019-01-01', b='2027-01-01'):
    z = s[(s.index >= a) & (s.index < b)]; return (1 + z).prod() ** (365.25 / (z.index[-1] - z.index[0]).days) - 1
if __name__ == '__main__':
    LV, _ = run(); s0 = curve(LV); print(f'live (re-run with grouped bars): {st4(s0)} | trades {len(LV):,}', flush=True)
    Z = pd.DataFrame(LV, columns=['tk', 'e', 'x', 'ret', 'ratio', 'h1']); Z['days'] = [max(1, np.busday_count(a.date(), b.date()) + 1) for a, b in zip(Z.e, Z.x)]
    Z['lp'] = np.log1p(Z.ret / 6); Z['top'] = Z.ret >= Z.ret.quantile(.9); Z['tm'] = pd.cut(Z.e.dt.hour * 60 + Z.e.dt.minute, [0, 585, 630, 780, 900], labels=['09:30-09:45', '10:00-10:30', '10:45-13:00', '13:15+'])
    Z['hold'] = pd.cut(Z.days, [0, 1, 3, 10, 999], labels=['same day', '2-3 days', '4-10 days', '> 10 days'])
    print('\n=== A) OCCUPANCY: share of trades / slot-days / P&L / top-decile winners | avg return ===', flush=True)
    for col in ('h1', 'tm', 'hold'):
        for k, g_ in Z.groupby(col, observed=True):
            print(f'  {col:<5} {str(k):<12} trades {len(g_) / len(Z):4.0%} | slot-days {g_.days.sum() / Z.days.sum():4.0%} | P&L {g_.lp.sum() / Z.lp.sum():5.0%} | '
                  f'top winners {g_.top.sum() / Z.top.sum():4.0%} | avg {g_.ret.mean():+.2%}, win {np.mean(g_.ret > 0):.0%}', flush=True)
    print('\n=== B) ROBUSTNESS: CAGR 2019-22 / 2023-26 / all, distribution over 60 paths (5th / 50th / 95th pct) and where live sits ===', flush=True)
    rng = np.random.default_rng(0)
    for lab, rl in (('shuffle near-ties only (within 10%)', 'near'), ('fully random within the bar', 'rand')):
        V = []
        for _ in range(60):
            tk_, _ = run(rule=rl, rng=rng); s_ = curve(tk_); V.append((cg(s_, '2019-01-01', '2023-01-01'), cg(s_, '2023-01-01'), cg(s_)))
        V = np.array(V); lv = (cg(s0, '2019-01-01', '2023-01-01'), cg(s0, '2023-01-01'), cg(s0))
        cells = [f'{np.percentile(V[:, j], 5):5.1%} / {np.percentile(V[:, j], 50):5.1%} / {np.percentile(V[:, j], 95):5.1%} (live {lv[j]:5.1%}, beats {np.mean(V[:, j] < lv[j]):4.0%})' for j in range(3)]
        print(f'  {lab:<36} ' + ' | '.join(cells), flush=True)
    print('\n=== C / D) near-tie diversification and sector caps | CAGR / DD 2019-22 | 2023-26 | all ===', flush=True)
    for lab, kw in (('live', {}), ('C near-tie: least correlated', dict(rule='corr')), ('C control: new sector first', dict(rule='sector')), ('D cap 2 per sector', dict(cap=2)), ('D cap 3 per sector', dict(cap=3))):
        tk_, _ = run(**kw); print(f'  {lab:<30} {st4(curve(tk_))}', flush=True)
    print('\n=== E) $700 CASH ACCOUNT, settlement T+2 -> T+1 (May 28 2024) ===', flush=True)
    tk_, inf = run(settle='count'); Zs = pd.DataFrame(tk_, columns=['tk', 'e', 'x', 'ret', 'ratio', 'h1'])
    print(f'  same trades as live, ledger kept: good-faith-violation risks {inf["gfv"]:,} of {len(tk_):,} trades ({inf["gfv"] / len(tk_):.0%})', flush=True)
    tk_, inf = run(settle='only'); print(f'  buy only with SETTLED cash: {st4(curve(tk_))} | trades {len(tk_):,}, signals skipped for unsettled cash {inf["skipped_cash"]:,}', flush=True)
    # follow-up (EXPLORATORY, after seeing E): settled-cash-only with a cash buffer - still at most 6 names, each sized at equity / 8, 9 or 12;
    # real compounding of the cash ledger (not the stand-in curve, whose 1/6 weights do not apply here)
    for dv in (6, 8, 9, 12):
        tk_, inf = run(settle='only', div=dv); Zs = pd.DataFrame(tk_, columns=['tk', 'e', 'x', 'ret', 'ratio', 'h1'])
        lp = np.log1p(Zs.ret * (1 - 2e-4) / dv); yrs = (pd.Timestamp('2026-09-25') - pd.Timestamp('2019-01-02')).days / 365.25
        print(f'  settled-only, each name = equity / {dv:<2}: realized ~{np.exp(lp.sum()) ** (1 / yrs) - 1:6.1%} a year | trades {len(tk_):,}, skipped for unsettled cash {inf["skipped_cash"]:,}', flush=True)
    LV2, _ = run(); lp = np.log1p(pd.DataFrame(LV2, columns=['tk', 'e', 'x', 'ret', 'ratio', 'h1']).ret * (1 - 2e-4) / 6); yrs = (pd.Timestamp('2026-09-25') - pd.Timestamp('2019-01-02')).days / 365.25
    print(f'  reference, live with margin / no settlement limit, same realized method: ~{np.exp(lp.sum()) ** (1 / yrs) - 1:6.1%} a year', flush=True)
