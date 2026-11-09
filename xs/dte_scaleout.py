"""Nov 8 2026: THE BASIC THING - Lucid pays for GREEN DAYS (5 days of +$150 / +$250 per payout), but our edges are CONVEX (30-50% winners, profit in a few big
days). Convert convexity into green days: on every trade, bank part of the position at a small fixed gain and run the rest with the stop at breakeven, so most trades
that ever go our way leave the day green. DECLARED BEFORE RESULTS.
Engines as xs/dte_prop.py on real NQ futures 5m (Feb 2018 - Jan 2026): RA (Rule A, 75 / 50, flat 12:00), M1g (noise momentum, negative-gamma days, hard stop 0.15 ADR),
R1g (80% rule, positive-gamma days). Scale-out: when the trade is +X points (bar high / low), close fraction f at +X, move the rest's stop to entry; if a bar touches both the
stop and +X, the stop is assumed first. Extra 0.5 pt cost when the partial fills. Per trade: P(reaching +X), points per trade, share of trade days that end >= the
qualifying amount at the given size. Lucid as xs/dte_prop.py.
  B0  V4 sizes (RA 10, M1g 3, R1g 5 micros at 50K), no scale-out
  S1  V4 sizes, X = 15, f = 1/2
  S2  RA 10, M1g 6, R1g 6 micros, X = 20, f = 2/3 (the banked part alone is a qualifying day at 50K)
  S3  RA 10, M1g 6, R1g 6 micros, X = 0.06 ADR (~15 now), f = 1/2
  S4  RA 10 alone, X = 15, f = 1/2"""
_p = open('/home/claude/fartpie/xs/dte_prop.py').read(); exec(_p[:_p.rindex("if __name__ == '__main__':")])
def path_trade(ix, j, d_, e, stop, target, X, f, until_m=955, check=None):
    """bar-by-bar; returns (pts per contract-unit, mae, hitX). check(i, p) -> exit price or None at trailing-check bars (noise)."""
    hit = False; worst = 0.0; banked = 0.0; stp = stop; frac = 1.0
    for q in range(j, len(ix)):
        i = ix[q]
        if m[i] >= until_m: px = c[ix[q - 1]] if q > j else e; return banked + frac * (px - e) * d_, worst, hit
        if (l[i] <= stp) if d_ > 0 else (h[i] >= stp):
            if not hit: worst = min(worst, (stp - e) * d_)
            return banked + frac * (stp - e) * d_, worst, hit
        if not hit: worst = min(worst, ((l[i] if d_ > 0 else h[i]) - e) * d_)
        if X is not None and not hit and ((h[i] - e) * d_ >= X if d_ > 0 else (e - l[i]) >= X):
            hit = True; banked = f * X; frac = 1 - f; stp = e
        if target is not None and ((h[i] >= target) if d_ > 0 else (l[i] <= target)): return banked + frac * (target - e) * d_, worst, hit
        if check is not None:
            px = check(i, q)
            if px is not None: return banked + frac * (px - e) * d_, worst, hit
    return banked + frac * (c[ix[-1]] - e) * d_, worst, hit
def cost(hit): return 1.0 + (.5 if hit else 0.0)
def ra_p(X, f):
    R = pd.read_csv('/home/claude/fartpie/live_data/nq_rule_a_trades.csv'); rows = []
    for r_ in R.itertuples():
        i = tpos.get(np.datetime64(pd.Timestamp(r_.entry), 'ns'))
        if i is None: continue
        d = pd.Timestamp(dd[i]); ix = gi[d]; j = int(np.where(ix == i)[0][0]); e = o[i]; dr = int(r_.dir); XX = X(d) if callable(X) else X
        pts, mae, hit = path_trade(ix, j, dr, e, e - 50 * dr, e + 75 * dr, XX, f, until_m=720); rows.append((d, m[i], pts - cost(hit), mae - 1.0, hit))
    return pd.DataFrame(rows, columns=['day', 'm', 'pts', 'mae', 'hit'])
def m1_p(X, f, gate):
    rows = []
    for k, d in enumerate(days):
        if k == 0 or not ok(d) or d not in SIG.index or not gate(d): continue
        ix = gi[d]; o0 = o[ix[0]]; pc = c[gi[days[k - 1]][-1]]; S = .15 * DY.adr[d]; XX = X(d) if callable(X) else X; pos_i = 0
        for p, i in enumerate(ix[:-1]):
            if p < pos_i or (m[i] + 5) % 30 != 0 or m[i] + 5 < 600 or m[i] + 5 > 930: continue
            s = SIG.loc[d].get(p, np.nan)
            if not np.isfinite(s): continue
            ub, lb = max(o0, pc) * (1 + s), min(o0, pc) * (1 - s); d_ = 1 if c[i] > ub else (-1 if c[i] < lb else 0)
            if d_ == 0: continue
            e = o[ix[p + 1]]
            def chk(ii, qq, d_=d_):
                if qq + 1 >= len(ix) or (m[ii] + 5) % 30 != 0 or m[ii] + 5 < 600 or m[ii] + 5 > 930: return None
                s2 = SIG.loc[d].get(qq, np.nan)
                if not np.isfinite(s2): return None
                u2, l2 = max(o0, pc) * (1 + s2), min(o0, pc) * (1 - s2)
                if (d_ > 0 and c[ii] < max(u2, vw[ii])) or (d_ < 0 and c[ii] > min(l2, vw[ii])): return o[ix[qq + 1]]
                return None
            pts, mae, hit = path_trade(ix, p + 1, d_, e, e - d_ * S, None, XX, f, check=chk)
            rows.append((d, m[ix[p + 1]], pts - cost(hit), mae - 1.0, hit))
            # resume scanning after this trade's exit: find exit bar roughly by re-walking (one position at a time)
            q = p + 1; stp = e - d_ * S
            while q < len(ix) - 1:
                i2 = ix[q]
                if ((l[i2] <= stp) if d_ > 0 else (h[i2] >= stp)) or chk(i2, q) is not None: break
                q += 1
            pos_i = q + 1
    return pd.DataFrame(rows, columns=['day', 'm', 'pts', 'mae', 'hit'])
def r1_p(X, f, gate):
    rows = []
    for k, d in enumerate(days):
        if k == 0 or not ok(d) or DY.roll[d] or not gate(d): continue
        val, poc, vah = VA[days[k - 1]]; ix = gi[d]; o0 = o[ix[0]]; side = 1 if o0 > vah else (-1 if o0 < val else 0); XX = X(d) if callable(X) else X
        if side == 0: continue
        run = 0
        for p, i in enumerate(ix[:-1]):
            if m[i] >= 780: break
            run = run + 1 if val <= c[i] <= vah else 0
            if run >= 6:
                d_ = -side; e = o[ix[p + 1]]; T = val if d_ < 0 else vah; S = (vah + .08 * DY.adr[d]) if d_ < 0 else (val - .08 * DY.adr[d])
                if (T - e) * d_ <= 0 or (e - S) * d_ <= 0: break
                pts, mae, hit = path_trade(ix, p + 1, d_, e, S, T, XX, f); rows.append((d, m[ix[p + 1]], pts - cost(hit), mae - 1.0, hit)); break
    return pd.DataFrame(rows, columns=['day', 'm', 'pts', 'mae', 'hit'])
if __name__ == '__main__':
    gneg = (DY.gex < 0); ng = lambda d: bool(gneg.get(d, False)); pg = lambda d: (d in gneg.index) and (gneg[d] == False) and np.isfinite(DY.gex.get(d, np.nan))
    D_ = pd.DatetimeIndex([d for d in days if d >= pd.Timestamp('2018-02-01') and ok(d)]); NS = len(D_); kpos = {d: i for i, d in enumerate(D_)}
    adrX = lambda d: .06 * DY.adr[d]
    CFG = {'B0': (None, 1.0, (10, 3, 5)), 'S1': (15, .5, (10, 3, 5)), 'S2': (20, 2 / 3, (10, 6, 6)), 'S3': (adrX, .5, (10, 6, 6)), 'S4': (15, .5, (10, 0, 0))}
    print(f'{"cfg":<4} {"engine":<5} trades  reach+X  win   pts/trade', flush=True)
    LIST = {}
    for nm, (X, f, sz) in CFG.items():
        T = {'RA': ra_p(X, f), 'M1g': m1_p(X, f, ng), 'R1g': r1_p(X, f, pg)}
        for en, t in T.items():
            t = t[t.day.isin(D_)]; T[en] = t
            print(f'{nm:<4} {en:<5} {len(t):6d} {np.mean(t.hit):7.0%} {np.mean(t.pts > 0):5.0%} {t.pts.mean():+9.2f}', flush=True)
        L = [[] for _ in range(NS)]; dayusd = np.zeros(NS)
        for (en, t), mic in zip(T.items(), sz):
            if mic <= 0: continue
            for r_ in t.itertuples():
                k = kpos.get(r_.day)
                if k is not None: L[k].append((r_.m, ((mic / 10,), r_.pts, r_.mae, None))); dayusd[k] += r_.pts * mic * 2
        LIST[nm] = [[x for _, x in sorted(z, key=lambda t_: t_[0])] for z in L]
        tdays = np.array([len(z) > 0 for z in L])
        print(f'     -> 50K day P&L at these sizes: trade days {tdays.mean() * 21:.1f}/mo, QUALIFYING days (>= $150) {np.mean(dayusd >= 150) * 21:.1f}/mo '
              f'({np.mean(dayusd[tdays] >= 150):.0%} of trade days), losing days {np.mean(dayusd < 0) * 21:.1f}/mo, avg ${dayusd.mean():+.0f}/day', flush=True)
    starts = list(range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5)); LOCK['on'] = False
    print(f'\n  {"cfg":<4} {"acct":<5} pass  paid  payouts x size  blown | per acct-month | 1st payout  every  payouts/mo funded', flush=True)
    for nm in CFG:
        DE = DF = LIST[nm]
        for a in ('50K', '150K'):
            A = ACC[a]; R = []
            for s in starts:
                res, k2 = ev(A, s, NS)
                if res != 'pass': R.append((res, k2, [], 'none', k2 if res == 'fail' else NS)); continue
                fr, k3, pays = fu(A, k2 + 1, NS, 5); R.append((res, k2, pays, fr, NS if fr == 'open' else k3))
            net = np.array([sum(x for _, x in r[2]) / 20 - A['fee'] / 20 for r in R]); life = np.array([(r[4] - s + 1) / 21 for r, s in zip(R, starts)])
            fund = [r for r in R if r[0] == 'pass']; pays = [x for r in fund for _, x in r[2]]; gaps = [b[0] - a_[0] for r in fund for a_, b in zip(r[2][:-1], r[2][1:])]
            fp = [(D_[min(r[2][0][0], NS - 1)] - D_[s]).days / 30.44 for r, s in zip(R, starts) if r[2]]; flife = sum((r[4] - r[1]) / 21 for r in fund)
            print(f'  {nm:<4} {a:<5} {np.mean([r[0] == "pass" for r in R]):4.0%} {np.mean([len(r[2]) > 0 for r in R]):5.0%}  {np.mean([len(r[2]) for r in fund]) if fund else 0:4.1f} x {np.mean(pays) / 20 if pays else 0:3.0f} pts '
                  f'{np.mean([r[3] == "lost" for r in fund]) if fund else 0:5.0%} | {net.sum() / life.sum():+14.1f} | {np.median(fp) if fp else np.nan:6.1f} mo {np.median(gaps) if gaps else np.nan:4.0f} d {len(pays) / flife if flife else 0:10.2f}', flush=True)
