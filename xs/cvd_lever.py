"""Nov 9 2026: CVD AS A LEVER ON THE 50K LUCID FLEX SYSTEM (Rey: add the CVD trades as extra leverage). DECLARED BEFORE RESULTS.
Order flow: MNQ TBBO 1-minute file with the buy / sell sign FIXED (true delta = -signed_vol; xs/flow_flips.py). Only 18 Dec 2024 - 18 Nov 2025 exists, so everything here
is inside that window (about 11 months) - a small sample, read as a direction, not a verdict.
CVD AGREES = session CVD (reset 09:30) above / below its 20-bar EMA on 5m at the bar before entry, on the trade's side (xs/flow_flips.py feats).
1) Per RTH engine of the best 50K setup V2 (S6 + IV-jump skip + HAR relative sizing, xs/flex50.py): points per trade when CVD agrees vs disagrees, 1st half (to May 2025) / 2nd half.
2) Lucid Flex 50K inside the window (an eval started every trading day with >= 40 days left; funded to the window end; one account at a time from 18 Dec 2024):
   V2 as is | skip V15a when CVD disagrees | skip every RTH engine when CVD disagrees | that + 1.5x size when CVD agrees |
   V2 + CVD flips as an extra engine (own entries: CVD crosses its EMA, exit on the opposite cross, stop 0.15 ADR, <= 4 a day, weight 0.3 like V15a)."""
_f5 = open('/home/claude/fartpie/xs/flex50.py').read(); exec(_f5[:_f5.rindex("if __name__ == '__main__':")])
_ff = open('/home/claude/fartpie/xs/flow_flips.py').read(); exec(_ff[_ff.index("OF = pd.read_parquet"):_ff.index("def cvd_flips")])
W0, W1 = pd.Timestamp('2024-12-18'), pd.Timestamp('2025-11-18')
def cvd_engine():
    rows = []
    for d in days:
        if not (W0 <= d <= W1) or not ok(d): continue
        ix = gi[d]; busy = -1; cnt = 0
        for p in range(2, len(ix) - 1):
            i = ix[p]; t = m[i] + 5
            if t < 585 or t > 930 or cnt >= 4 or p + 1 <= busy: continue
            if not (np.isfinite(cst[i]) and np.isfinite(cst[i - 1])) or cst[i] == cst[i - 1] or cst[i] == 0: continue
            dr = int(cst[i]); j = p + 1; e = o[ix[j]]; S = e - dr * .15 * DY.adr[d]; res = None; mae = 0.0
            for q in range(j, len(ix)):
                ii = ix[q]
                if m[ii] >= 955: res = ((c[ix[q - 1]] - e) * dr, q); break
                mae = min(mae, ((l[ii] - e) if dr > 0 else (e - h[ii])))
                if (l[ii] <= S) if dr > 0 else (h[ii] >= S): res = ((S - e) * dr, q); mae = (S - e) * dr; break
                if q > j and cst[ii] == -dr and q + 1 < len(ix): res = ((o[ix[q + 1]] - e) * dr, q + 1); break
            if res is None: res = ((c[ix[-1]] - e) * dr, len(ix) - 1)
            rows.append((d, m[ix[j]], dr, res[0] - 1.0, max(mae, (S - e) * dr) - 1.0)); busy = res[1]; cnt += 1
    T_ = pd.DataFrame(rows, columns=['day', 'm', 'dir', 'pts', 'mae']); T_['sz'], T_['ce'], T_['cf'] = 1.0, 1.0, 1.0; return T_
base = variant(True, relm); CA = {}
for nm, (T_, u, yz) in zip(NAMES, base):
    if nm not in RTH: continue
    dd = pd.to_datetime(T_.day); w = ((dd >= W0) & (dd <= W1)).values; ca = np.full(len(T_), np.nan)
    if w.any(): ca[np.where(w)[0]] = [f_[1] for f_ in feats(T_[w])]
    CA[nm] = ca
def lever(engines, skip=True, up=1.0, extra=False):
    out = []
    for nm, (T_, u, yz) in zip(NAMES, base):
        T_ = T_.copy()
        if nm in engines:
            ca = CA[nm]
            if skip: T_.loc[ca < 0, 'sz'] = 0.0
            T_.loc[ca > 0, 'sz'] = T_.loc[ca > 0, 'sz'] * up
        out.append((T_, u, yz))
    if extra: out.append((cvd_engine(), .3, None))
    return out
if __name__ == '__main__':
    half = lambda dd: np.where(pd.to_datetime(dd) < pd.Timestamp('2025-06-01'), '1st', '2nd')
    print('1) points per trade, CVD agrees vs disagrees | 1st half (n) / 2nd half (n) | V2 engines, IV-skipped trades left out', flush=True)
    for nm, (T_, u, yz) in zip(NAMES, base):
        if nm not in RTH: continue
        T = T_.assign(ca=CA[nm]); T = T[T.ca.notna() & (T.sz > 0)]; T['h'] = half(T.day); cells = []
        for lab, msk in (('agrees', T.ca > 0), ('disagrees', T.ca < 0)):
            a, b = T[msk & (T.h == '1st')].pts, T[msk & (T.h == '2nd')].pts; cells.append(f'{lab} {a.mean():+6.2f} ({len(a):3d}) / {b.mean():+6.2f} ({len(b):3d})')
        print(f'  {nm:<6} ' + '   '.join(cells), flush=True)
    A = ACCX['50K']; fee = A['fee']; LOCK['on'] = False; k0 = int(np.searchsorted(D_, W0)); k1 = int(np.searchsorted(D_, W1, 'right'))
    VAR = {'V2 as is': base, 'skip V15a when CVD disagrees': lever({'V15a'}), 'skip all RTH engines when CVD disagrees': lever(RTH),
           'skip disagree + 1.5x when CVD agrees': lever(RTH, True, 1.5), 'V2 + CVD flips as an extra engine': lever(set(), False, 1.0, True)}
    print(f'\n2) Lucid Flex 50K inside the order-flow window ({D_[k0].date()} - {D_[k1 - 1].date()}) | per eval: pass, paid, payouts/eval, evals per payout, net $/eval after the ${fee} fee'
          f' | one account at a time: evals bought, payouts, net $', flush=True)
    for nm, sp in VAR.items():
        LE_, LF_ = lists(sp, 'e'), lists(sp, 'f'); TRJ = {}
        def traj(s):
            if s not in TRJ:
                r, k2 = eval_x(LE_, A, s, k1, 1000)
                if r != 'pass': TRJ[s] = (r, k2 if r == 'fail' else k1, [])
                else: fr, k3, pays = fund_l(LF_, A, k2 + 1, k1, None); TRJ[s] = (r, k1 if fr == 'open' else k3, pays)
            return TRJ[s]
        tr_ = [traj(s) for s in range(k0, k1 - 40)]
        pas = np.mean([t[0] == 'pass' for t in tr_]); paid = np.mean([len(t[2]) > 0 for t in tr_]); ppe = np.mean([len(t[2]) for t in tr_])
        net = np.mean([sum(x for _, x in t[2]) - fee for t in tr_]); s = k0; ne = npay = 0; cash = 0.0
        while s < k1 - 1:
            r, end, pays = traj(s); ne += 1; npay += len(pays); cash += sum(x for _, x in pays) - fee; s = end + 1
        print(f'  {nm:<42} {pas:4.0%}, {paid:4.0%}, {ppe:.2f}, {1 / ppe if ppe else np.inf:4.1f}, ${net:6,.0f} | {ne:2d} evals, {npay} payouts, ${cash:6,.0f}', flush=True)
