"""Nov 9 2026: A MEAN-REVERSION ENGINE: PRIOR-DAY HVNs PAIRED WITH THOD/TLOD, THE OPTIONS REGIME AND REY'S ALGO LEVELS. DECLARED BEFORE RESULTS.
Levels: prior RTH session volume profile from NQ 5m bars (each bar's volume spread over its range, 5-pt bins, 3-bin smoothing): POC + the 2 biggest other peaks >= 0.05 ADR apart.
Event: FIRST touch of each level 09:45-15:30 with the bar before fully on one side. From above = buy the HVN, from below = sell it. Limit fill at the level.
Trade: stop 0.05 ADR (about 15-20 NQ pts) | targets 0.05 ADR (1:1) and 0.10 ADR (2:1); stop and target in the same 5m bar = stop; stop inside the touch bar = stop; out 15:55; 1 pt cost.
Pairings at the touch (each also measured on the placebo level):
  GAMMA+   QQQ net GEX > 0 at the prior close (dealers long gamma = the mean-reverting regime) | GAMMA- the rest
  THOD     the HVN within 0.25 sigma of a THOD/TLOD line on the trade's side (open -/+ 1, 2, 3 x the 1h Garman-Klass x Hurst sigma at 09:30, Rey's bot port)
  EXH      the day's range already used: buy only after a drop of >= 0.8 x the expected range from the high so far (sell: rise >= 0.8 x from the low so far);
           expected range = THOD 1h sigma x k (xs/anchored_range.py)
  WALL     put wall (buy) / call wall (sell) within 0.05 ADR (QQQ walls mapped to NQ, to Dec 2025)
  ALGO     Rey's point-in-time QQQ 1h / 4h algo levels (OPTICS / KDE / ... ; levels_hist2) mapped to NQ, within 0.03 ADR - only Oct 2024 - Jan 2026 exists
PLACEBO: the same level moved 0.03-0.12 ADR to a random side (3 draws), pairings recomputed there. Judged 2018-21 / 2022-25.
PASS = positive in both periods AND beats its placebo by >= 2 pts a trade in both AND >= 1 trade a month -> added to the best 50K setup (V2) as a mean-reversion engine
(weight 0.5 and 1.0 NQ), Lucid Flex 50K: per eval and fleet, as xs/flex50.py."""
_ft = open('/home/claude/fartpie/xs/flex50_thod.py').read(); exec(_ft[:_ft.rindex("if __name__ == '__main__':")])
import json
U_ = '/mnt/user-data/uploads/fartpie'
vol = (H['volume'] if 'volume' in H else H['v']).values.astype(float)
rr_ = (DY.h - DY.l); kk = (rr_ / SG.h1_rth.reindex(DY.index))[(DY.index >= '2018-01-01') & (DY.index < '2022-01-01')].median(); RX = SG.h1_rth.reindex(DY.index) * kk
WL = pd.read_csv(f'{U_}/gex_walls_nq.csv', parse_dates=['date']).set_index('date').sort_index()
ALG = {}
for tf in ('1h', '4h'):
    for x in open(f'{U_}/xs/ext/levels_hist2/QQQ_{tf}.jsonl'):
        j_ = json.loads(x); ALG.setdefault(pd.Timestamp(j_['day']), []).extend([y['price'] for y in j_['levels']] and [(y['price'], j_['price']) for y in j_['levels']])
def hvns(d0):
    ix = gi[d0]; lo0 = np.floor(l[ix].min() / 5); nb = int(np.floor(h[ix].max() / 5) - lo0) + 1; vp = np.zeros(nb)
    for i in ix:
        a, b = int(np.floor(l[i] / 5) - lo0), int(np.floor(h[i] / 5) - lo0); vp[a:b + 1] += vol[i] / (b - a + 1)
    sm = np.convolve(vp, np.ones(3) / 3, 'same'); px = (lo0 + np.arange(nb) + .5) * 5; adr = DY.adr[d0]
    pk = [k for k in range(nb) if sm[k] == sm[max(0, k - 3):k + 4].max() and sm[k] > 0]; pk = sorted(pk, key=lambda k: -sm[k]); out = []
    for k in pk:
        if all(abs(px[k] - q) >= .05 * adr for q in out): out.append(px[k])
        if len(out) == 3: break
    return out
def flags(d, L, side, p, ix):
    sg = SG.h1_rth.get(d, np.nan); o0 = o[ix[0]]; adr = DY.adr[d]; R = RX.get(d, np.nan); seg = ix[:p + 1]
    lines = [o0 - side * k * sg for k in (1, 2, 3)] if np.isfinite(sg) else []
    thod = bool(len(lines)) and min(abs(L - x) for x in lines) <= .25 * sg
    exh = np.isfinite(R) and ((h[seg].max() - L) >= .8 * R if side > 0 else (L - l[seg].min()) >= .8 * R)
    wl = WL[WL.index < d]; wall = np.nan
    if len(wl) and (d - wl.index[-1]).days < 6: wall = float(abs(L - (wl.put_wall_nq.iloc[-1] if side > 0 else wl.call_wall_nq.iloc[-1])) <= .05 * adr)
    alg = np.nan
    if d in ALG: alg = float(any(abs(L - lp * o0 / qp) <= .03 * adr for lp, qp in ALG[d]))
    return dict(gpos=bool(DY.gex.get(d, np.nan) > 0), thod=bool(thod), exh=bool(exh), wall=wall, alg=alg)
def fade(d, L, side, j, ix, sd, td):
    S, T = L - side * sd, L + side * td; e = L; mae = 0.0
    if (l[ix[j]] <= S) if side > 0 else (h[ix[j]] >= S): return -sd - 1.0, -sd - 1.0
    for q in range(j + 1, len(ix)):
        i = ix[q]
        if m[i] >= 955: return (c[ix[q - 1]] - e) * side - 1.0, mae - 1.0
        st = (l[i] <= S) if side > 0 else (h[i] >= S); tg = (h[i] >= T) if side > 0 else (l[i] <= T)
        if st: return -sd - 1.0, -sd - 1.0
        mae = min(mae, ((l[i] - e) if side > 0 else (e - h[i])))
        if tg: return td - 1.0, mae - 1.0
    return (c[ix[-1]] - e) * side - 1.0, mae - 1.0
def events(plc, rng):
    rows = []
    for k_, d in enumerate(days):
        if k_ == 0 or d < pd.Timestamp('2018-02-01') or not ok(d) or DY.roll[d]: continue
        d0 = days[k_ - 1]
        if not ok(d0) or DY.roll[d0] or (d - d0).days > 4: continue
        ix = gi[d]; adr = DY.adr[d]; levs = hvns(d0)
        if plc: levs = [L + rng.choice([-1, 1]) * rng.uniform(.03, .12) * adr for L in levs]
        for L in levs:
            for p in range(1, len(ix)):
                i, ip = ix[p], ix[p - 1]
                if not (l[i] <= L <= h[i]): continue
                if m[i] < 585 or m[i] > 925: break
                side = 1 if l[ip] > L else (-1 if h[ip] < L else 0)
                if side == 0: break
                f_ = flags(d, L, side, p - 1, ix); r1 = fade(d, L, side, p, ix, .05 * adr, .05 * adr); r2 = fade(d, L, side, p, ix, .05 * adr, .10 * adr)
                rows.append(dict(day=d, m=m[i], dir=side, plc=plc, p11=r1[0], mae11=r1[1], p21=r2[0], mae21=r2[1], **f_)); break
    return pd.DataFrame(rows)
if __name__ == '__main__':
    rng = np.random.default_rng(11); E = pd.concat([events(0, rng)] + [events(1, rng) for _ in range(3)], ignore_index=True)
    E['per'] = np.where(E.day.dt.year >= 2022, 'b', 'a'); nmo = {'a': 47, 'b': 49}
    print(f'walls cover {WL.index.min().date()} - {WL.index.max().date()} | algo levels {min(ALG).date()} - {max(ALG).date()} | k = {kk:.2f}', flush=True)
    SUBS = {'all HVN touches': lambda z: z.index == z.index, 'GAMMA+': lambda z: z.gpos, 'GAMMA-': lambda z: ~z.gpos, 'THOD line': lambda z: z.thod, 'EXH (range used)': lambda z: z.exh,
            'WALL': lambda z: z.wall == 1, 'GAMMA+ & THOD': lambda z: z.gpos & z.thod, 'GAMMA+ & EXH': lambda z: z.gpos & z.exh, 'THOD & EXH': lambda z: z.thod & z.exh,
            'GAMMA+ & THOD & EXH': lambda z: z.gpos & z.thod & z.exh, 'GAMMA- & EXH': lambda z: ~z.gpos & z.exh}
    PASS = []
    for col, lab in (('p11', 'stop 0.05 / target 0.05 ADR'), ('p21', 'stop 0.05 / target 0.10 ADR')):
        print(f'\n=== {lab} | trades a month, HOLD (target first), NQ pts per trade: 2018-21 / 2022-25 | placebo pts per trade ===', flush=True)
        for nm, f in SUBS.items():
            R_, P_ = E[(E.plc == 0)], E[(E.plc == 1)]; R_, P_ = R_[f(R_)], P_[f(P_)]; cells = []; ok_ = True
            for per in 'ab':
                a, b = R_[R_.per == per], P_[P_.per == per]
                cells.append(f'{len(a) / nmo[per]:4.1f}/mo {np.mean(a[col] > 0):3.0%} {a[col].mean():+6.2f} (plc {b[col].mean():+6.2f})')
                ok_ &= bool(len(a) >= nmo[per] and a[col].mean() > 0 and a[col].mean() - b[col].mean() >= 2)
            print(f'  {nm:<22} ' + '  |  '.join(cells) + ('   <- PASS' if ok_ else ''), flush=True)
            if ok_: PASS.append((nm, col, R_))
    W = E[(E.day >= '2024-10-14')]
    print('\n=== Oct 2024 - Jan 2026 only: paired with Rey\'s QQQ 1h / 4h algo levels | n, hold, pts per trade (2:1 / 1:1) | placebo ===', flush=True)
    for nm, f in (('all HVN touches (window)', lambda z: z.index == z.index), ('HVN + algo level', lambda z: z.alg == 1), ('HVN, no algo level', lambda z: z.alg == 0),
                  ('HVN + algo level + GAMMA+', lambda z: (z.alg == 1) & z.gpos)):
        a, b = W[(W.plc == 0)], W[(W.plc == 1)]; a, b = a[f(a)], b[f(b)]
        print(f'  {nm:<28} n{len(a):4d} {np.mean(a.p21 > 0):3.0%} {a.p21.mean():+6.2f} / {a.p11.mean():+6.2f} | placebo n{len(b) / 3:5.0f} {b.p21.mean():+6.2f} / {b.p11.mean():+6.2f}', flush=True)
    if not PASS: print('\n  nothing passed - no fleet test', flush=True)
    else:
        nm, col, R_ = max(PASS, key=lambda t: t[2][t[1]].mean()); mcol = 'mae11' if col == 'p11' else 'mae21'
        T_ = R_.assign(pts=R_[col], mae=R_[mcol], sz=1.0, ce=1.0, cf=1.0)[['day', 'm', 'dir', 'pts', 'mae', 'sz', 'ce', 'cf']].sort_values(['day', 'm'])
        print(f'\n=== fleet: best pass = {nm}, {col} ===', flush=True); LOCK['on'] = False; A = ACCX['50K']; fee = A['fee']
        base = variant(True, relm)
        for lab, sp in (('V2 as is', base), ('V2 + HVN mean reversion x0.5', base + [(T_, .5, None)]), ('V2 + HVN mean reversion x1.0', base + [(T_, 1.0, None)])):
            LE_, LF_ = lists(sp, 'e'), lists(sp, 'f'); TRJ = {}
            def traj(k):
                if k not in TRJ:
                    r, k2 = eval_x(LE_, A, k, NS, 1000)
                    if r != 'pass': TRJ[k] = (r, k2 if r == 'fail' else NS, [])
                    else: fr, k3, pays = fund_l(LF_, A, k2 + 1, NS, None); TRJ[k] = (r, NS if fr == 'open' else k3, pays)
                return TRJ[k]
            tr_ = [traj(s) for s in range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 2)]
            ppe = np.mean([len(t[2]) for t in tr_]); s = 0; chain = []
            while s < NS - 1:
                r, end, pays = traj(s); chain.append(pays); s = end + 1
            run = mx = 0
            for p_ in chain: run = run + 1 if not p_ else 0; mx = max(mx, run)
            fl = multi_x({'x': sp}, '50K')
            print(f'  {lab:<32} pass {np.mean([t[0] == "pass" for t in tr_]):3.0%}, evals per payout {1 / ppe if ppe else np.inf:4.1f}, longest no-payout run {mx} | fleet 2022-23: '
                  f'avg ${fl["avg"]:,.0f} / median ${fl["med"]:,.0f}, losing {fl["lose"]:.0%}, dry {fl["dry"]:.1f} mo', flush=True)
