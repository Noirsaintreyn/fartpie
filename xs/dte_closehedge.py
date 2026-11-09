"""Nov 8 2026: DEALER HEDGING INTO THE CLOSE (Baltussen, Da, Lammers & Martens 2021: intraday momentum comes from short-gamma hedging). DECLARED BEFORE RESULTS.
Unconditional last-half-hour momentum lost on NQ (xs/nq_public_strats.py). Here it is conditioned on dealer gamma: short-gamma dealers must buy into strength / sell into
weakness late in the day. Real NQ futures 5m, 2014 - Jan 2026; GEX at the prior close (own file 2018+; greeks-file percentile 2014+); 1 contract, 1 pt cost.
  C1 direction = sign of the move from the prior RTH close to the 15:25 bar close; enter the 15:30 open, exit the 15:55 bar close; stop 0.10 ADR
  C2 same with the move measured from today's open                        C3 C1 but enter 15:00 (move to the 14:55 close), exit 15:55
  C4 C1 only when the move is beyond 0.25 ADR (a real trend day)
Each by regime: negative / positive GEX (2018+), GEX bottom / top third (2014+), and DEX bottom / top third; by period 2014-17 / 2018-21 / 2022-25."""
_y = open('/home/claude/fartpie/xs/dte_yz.py').read(); exec(_y[:_y.rindex("if __name__ == '__main__':")])
def close_trades(start_m, ref='pc', min_mv=0.0):
    rows = []
    for k, d in enumerate(days):
        if k == 0 or not ok(d) or DY.roll[d]: continue
        ix = gi[d]; mm_ = m[ix]; a = np.where(mm_ == start_m - 5)[0]; b = np.where(mm_ == start_m)[0]
        if not len(a) or not len(b): continue
        refp = c[gi[days[k - 1]][-1]] if ref == 'pc' else o[ix[0]]; mv = (c[ix[a[0]]] - refp) / DY.adr[d]
        if abs(mv) < max(min_mv, 1e-9): continue
        dr = int(np.sign(mv)); e = o[ix[b[0]]]; S = e - dr * .10 * DY.adr[d]; j = b[0]
        pts = walk(ix, j, dr, e, e + dr * 1e9, S, until_m=960); rows.append((d, dr, pts - 1.0))
    return pd.DataFrame(rows, columns=['day', 'dir', 'pts'])
if __name__ == '__main__':
    PER = [(2014, 2017), (2018, 2021), (2022, 2026)]
    for nm, T in (('C1 prior close -> 15:25, trade 15:30-16:00', close_trades(930)), ('C2 open -> 15:25, trade 15:30-16:00', close_trades(930, 'open')),
                  ('C3 prior close -> 14:55, trade 15:00-16:00', close_trades(900)), ('C4 C1 with a move > 0.25 ADR', close_trades(930, 'pc', .25))):
        T = T[T.day.isin(DY.index)]; yr = T.day.dt.year
        per = lambda msk: ' / '.join(f'{T.pts[msk & (yr >= a) & (yr <= b)].mean():+.2f}' if (msk & (yr >= a) & (yr <= b)).sum() >= 15 else '   .' for a, b in PER)
        gx = T.day.map(DY.gex); g2 = T.day.map(DY.g2p if 'g2p' in DY else pd.Series(dtype=float)); dx = np.array([fget(d, 'dexp') for d in T.day], dtype=float)
        print(f'\n{nm}: {len(T)} trades', flush=True)
        for lab, msk in (('all', np.ones(len(T), bool)), ('negative GEX (2018+)', (gx < 0).values), ('positive GEX (2018+)', (gx > 0).values),
                         ('GEX bottom third (2014+)', (np.array([fget(d, 'gex2p') for d in T.day], dtype=float) <= 1 / 3)),
                         ('GEX top third (2014+)', (np.array([fget(d, 'gex2p') for d in T.day], dtype=float) >= 2 / 3)), ('DEX bottom third', dx <= 1 / 3), ('DEX top third', dx >= 2 / 3)):
            t = T.pts[msk]; print(f'  {lab:<26} n{len(t):5d} win {np.mean(t > 0):3.0%} avg {t.mean():+6.2f} | 14-17 / 18-21 / 22-25: {per(msk)}', flush=True)
