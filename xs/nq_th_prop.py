"""Nov 8 2026, step 2: the TREND-HOLD versions through the Lucid engine. DECLARED BEFORE RESULTS (FIX_A's cut points are post-hoc, so FIX_A is an upper bound).
Engine as nq_account_sizes: real-time trailing drawdown on each trade's worst open loss, locks at start + $100, 90% of min(50% of profit, cap), 5 qualifying days,
5 payouts per funded account, day stop -$1,000 x size, 5 accounts, a slot buys an eval only on GREEN days, copied trades. Eval x0.3 / funded x0.7 while the last 20 trades are net negative.
  P0     the current prop system: Rule A, 75 / 50, flat 12:00, eval gap-gated (reference)
  RA_TH  Rule A entries held to the 15m vector exit, same 10 / 5 / 3 micro ladder
  WF_A   every RTH flip held to the vector exit, walk-forward alignment size 0.5 / 1 / 2 units, 5 micros a unit (max 10 micros = 1 NQ)
  WF_A7  same at 7 micros a unit
  FIX_A  every flip, fixed A+ count: skip 0-2, 1 unit at 3, 2 at 4-5, 5 micros a unit (post-hoc)
Account sizes 50K and 150K (150K = 2.25x micros). Points = dollars / $20 (NQ points of payouts). Single evals: one started every 5th session 2022-2025."""
_as = open('xs/nq_account_sizes.py').read(); exec(_as[:_as.rindex("if __name__ == '__main__':")])
TT = pd.read_parquet('/tmp/claude-0/ml/nq_th_trades.parquet'); dposm = {pd.Timestamp(d): i for i, d in enumerate(D_)}
def lists(var, per_unit):
    g = TT[TT['var'] == var].sort_values(['day', 'j']); E = [[] for _ in D_]; F = [[] for _ in D_]; hist = []
    for r in g.itertuples():
        cold = len(hist) >= 20 and sum(hist[-20:]) < 0; k = dposm.get(pd.Timestamp(r.day)); hist.append(r.net)
        if k is None: continue
        u = r.size * per_unit / 10.0                                                           # day() turns units into round(10 x u x scale) micros
        E[k].append(((u * (.3 if cold else 1.0),), r.net, r.mae, None)); F[k].append(((u * (.7 if cold else 1.0),), r.net, r.mae, None))
    return E, F
V = {'P0 current prop system (Rule A 75/50)': (DS['e_BASE'], DS['f_BASE']), 'RA_TH Rule A, vector exit, 10/5/3 micros': lists('RA_TH', 10),
     'WF_A every flip, walk-forward size, 5 micros a unit': lists('WF_A', 5), 'WF_A7 same, 7 micros a unit': lists('WF_A', 7), 'FIX_A fixed A+ count (post-hoc), 5 a unit': lists('FIX_A', 5)}
def single(A, s_list):
    out = []
    for s in s_list:
        res, k = ev(A, s, NS); paid = 0.0; npay = 0
        if res == 'pass': fr, k2, pays = fu(A, k + 1, NS, 5); paid = sum(a for _, a in pays); npay = len(pays)
        out.append((res == 'pass', npay > 0, paid - A['fee']))
    o = np.array(out, dtype=float); return o[:, 0].mean(), o[:, 1].mean(), o[:, 2].mean()
if __name__ == '__main__':
    s0 = int(np.searchsorted(D_, pd.Timestamp('2022-01-01'))); last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01')))
    starts = [int(np.searchsorted(D_, pd.Timestamp(x))) for x in ('2022-01-01', '2022-07-01', '2023-01-01', '2023-07-01', '2024-01-01', '2024-07-01')]
    print(f'NQ prop test, Jan 2022 -> {D_[NS - 1]:%b %d %Y}; 5 Lucid accounts on GREEN days, copied trades; points = $ / 20', flush=True)
    for nm, (de, df) in V.items():
        DE, DF = de, df
        print(f'\n  {nm}', flush=True)
        for a in ('50K', '150K'):
            A = ACC[a]; E, r = book(A, s0, NS, 5); W = [book(A, s, min(s + 504, NS), 5)[1] for s in starts]; pe, pp, ne = single(A, list(range(s0, last, 5)))
            yr = []
            for y in (2022, 2023, 2024, 2025, 2026):
                lo, hi = int(np.searchsorted(D_, pd.Timestamp(f'{y}-01-01'))), int(np.searchsorted(D_, pd.Timestamp(f'{y + 1}-01-01'))); g = E[(E.k >= lo) & (E.k < min(hi, NS))]
                yr.append(f'{y} {g.amt.sum() / 20:+,.0f}')
            print(f'    {a:<5} single eval: pass {pe:.0%}, payout {pp:.0%}, net {ne / 20:+,.0f} pts | 5 accounts full run: {r["net"] / 20:+,.0f} pts ({r["mo"] / 20:+.0f} a month), evals {r["evals"]}, '
                  f'payouts {r["npay"]}, dry spell {r["gap"]} months | 24-month median {np.median([w["net"] for w in W]) / 20:+,.0f} pts, worst {min(w["net"] for w in W) / 20:+,.0f}', flush=True)
            print(f'          by year (pts): ' + ' | '.join(yr), flush=True)
