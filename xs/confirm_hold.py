"""Nov 9 2026: DOES THE BREAK HOLD? CONFIRMATION ONE 15m BAR AFTER THE VECTOR FLIP - PRICE ACCEPTANCE + VOLUME FOLLOW-THROUGH (Rey). DECLARED BEFORE RESULTS.
Rey: a break only works if it keeps working - if volume dies it won't sustain the new value area; check whether the next candle holds after the break.
NQ futures 2018-25, the current 50K setup (V2 + skip slow flips): V15a (15m flips, out on the opposite flip, stop 0.15 ADR) and Rule A (morning flips, 75 / 50, flat 12:00).
Flip bar = the 15m bar that flipped; NEXT bar = the one after it (closes 15 min after the normal entry). Broken level = the vector level before the flip.
  HOLD  next bar closes beyond the broken level (on the trade's side)          VOL  next bar's volume >= 0.7 x the flip bar's (volume did not die)
Variants: C0 enter at the flip (now) | C1 wait: enter after the next bar only if HOLD | C2 wait: only if HOLD and VOL | C3 wait: only if VOL |
          C4 enter at the flip as now, but get out after the next bar if HOLD and VOL both fail | C5 wait, enter after the next bar regardless (pure delay control)
Delayed entries: next 5m open after the confirming bar, same exit rules measured from the new entry (stop 0.15 ADR / 50 pts, target 75 for Rule A).
NQ points per SIGNAL (skipped signals count 0), per trade taken, trades; 2018-21 / 2022-25. PASS: more points per signal in BOTH -> Lucid Flex 50K."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_u = open('/home/claude/fartpie/xs/updown_vol.py').read(); exec(_u[:_u.rindex("if __name__ == '__main__':")])
B15v_ = B15v.values
def setup(T_):
    out = []
    for r_ in T_.itertuples():
        d = pd.Timestamp(r_.day); q = endmap.get(np.datetime64(d + pd.Timedelta(minutes=int(r_.m)), 'ns')); dr = r_.dir
        if q is None or q + 1 >= len(B) or Bday[q + 1] != Bday[q] or not np.isfinite(lv[q - 1]): out.append((np.nan,) * 4); continue
        hold = float((Bc[q + 1] - lv[q - 1]) * dr > 0); vol_ = float(B15v_[q + 1] >= .7 * B15v_[q]) if np.isfinite(B15v_[q + 1]) and np.isfinite(B15v_[q]) and B15v_[q] > 0 else np.nan
        t2 = np.datetime64(B.end.values[q + 1], 'ns'); ix = gi[d]; w = np.where(H.dt.values[ix] == t2)[0]
        out.append((hold, vol_, ix[w[0]] if len(w) else -1, q))
    return pd.DataFrame(out, columns=['hold', 'vol', 'i2', 'q'], index=T_.index)
def run_var(T_, en, var):
    res = []
    for r_ in T_.itertuples():
        d = pd.Timestamp(r_.day); ix = gi[d]; w = np.where(m[ix] == r_.m)[0]
        if not len(w) or not np.isfinite(r_.hold): res.append((np.nan, np.nan, 0)); continue
        i1 = ix[w[0]]; a_ = DY.adr[d]; ra = en == 'RA'
        kw = dict(stop=50 if ra else None, until=720 if ra else 955, flip=not ra, T=75 if ra else None)
        ok_ = {'C1': r_.hold == 1, 'C2': r_.hold == 1 and r_.vol == 1, 'C3': r_.vol == 1, 'C5': True}.get(var)
        if var == 'C0': p_, mae = sim_x(d, i1, r_.dir, int(r_.q), None, **kw); res.append((p_, mae, 1)); continue
        if var == 'C4':
            fail = not (r_.hold == 1 or r_.vol == 1)
            rule = (lambda bq, dr_, e_: True) if fail else None
            p_, mae = sim_x(d, i1, r_.dir, int(r_.q), rule, **kw); res.append((p_, mae, 1)); continue
        if not ok_ or r_.i2 < 0: res.append((0.0, 0.0, 0)); continue
        if ra and m[int(r_.i2)] >= 720: res.append((0.0, 0.0, 0)); continue
        p_, mae = sim_x(d, int(r_.i2), r_.dir, int(r_.q) + 1, None, **kw); res.append((p_, mae, 1))
    return np.array(res, dtype=float)
if __name__ == '__main__':
    V = run_sig2('s_vector'); Vg = V[~hig(V)].copy(); RA_ = RAs[(RAs.sz > 0)].copy()
    REF = V[V.day >= '2018-02-01'].join(feats(V[V.day >= '2018-02-01'])); cut = REF[pd.to_datetime(REF.day).dt.year < 2022].mv30.quantile(1 / 3); SL = {}
    for key, T_ in (('V15a', Vg), ('RA', RA_)):
        F = T_.join(feats(T_)); SL[key] = set(zip(pd.to_datetime(F[F.mv30 < cut].day), F[F.mv30 < cut].m))
    base = []
    for en, (T_, u, yz) in zip(NAMES, variant(True, relm)):
        T_ = T_.copy()
        if en in SL: T_.loc[np.array([(pd.Timestamp(d), mm_) in SL[en] for d, mm_ in zip(T_.day, T_.m)]), 'sz'] = 0.0
        base.append((T_, u, yz))
    BE = dict(zip(NAMES, base)); RES = {}
    for en in ('V15a', 'RA'):
        T_ = BE[en][0]; T_ = T_[(T_.sz > 0) & (pd.to_datetime(T_.day) >= '2018-02-01')].copy(); T_ = T_.join(setup(T_)); T_['per'] = np.where(pd.to_datetime(T_.day).dt.year >= 2022, 'b', 'a')
        print(f'\n=== {en}: {len(T_)} signals | next bar HOLDS {T_.hold.mean():.0%}, VOLUME kept {T_.vol.mean():.0%} | NQ pts per SIGNAL (per trade taken, trades) 2018-21 ; 2022-25 ===', flush=True)
        for hv, g_ in T_.groupby([T_.hold, T_.vol]):
            print(f'    outcome of trades entered now when next bar hold={int(hv[0])} vol={int(hv[1])}: {g_[g_.per == "a"].pts.mean():+6.2f} (n {(g_.per == "a").sum()}) ; '
                  f'{g_[g_.per == "b"].pts.mean():+6.2f} (n {(g_.per == "b").sum()})', flush=True)
        for var, lab in (('C0', 'enter at the flip (now)'), ('C1', 'wait, enter if HOLD'), ('C2', 'wait, enter if HOLD + VOL'), ('C3', 'wait, enter if VOL'),
                         ('C4', 'enter now, cut if both fail'), ('C5', 'wait 15 min regardless')):
            A_ = run_var(T_, en, var); X_ = T_.assign(p2=A_[:, 0], mae2=A_[:, 1], took=A_[:, 2]); RES[(en, var)] = X_; cells = []
            for p_ in 'ab':
                z = X_[X_.per == p_]; tk_ = z[z.took == 1]
                cells.append(f'{z.p2.fillna(0).mean():+6.2f} per signal ({tk_.p2.mean():+6.2f} per trade, {len(tk_)})')
            print(f'  {var} {lab:<30} ' + ' ; '.join(cells), flush=True)
    PASS = [(en, v) for en in ('V15a', 'RA') for v in ('C1', 'C2', 'C3', 'C4') if all(RES[(en, v)][RES[(en, v)].per == p_].p2.fillna(0).mean() > RES[(en, 'C0')][RES[(en, 'C0')].per == p_].p2.fillna(0).mean() for p_ in 'ab')]
    print(f'\npasses: {PASS or "none"}', flush=True)
    if PASS:
        LOCK['on'] = False; A = ACCX['50K']; fee = A['fee']
        def apply(en, X_):
            out = []
            for e_, (T_, u, yz) in zip(NAMES, base):
                T_ = T_.copy()
                if e_ == en:
                    took = X_.took.reindex(T_.index); p2 = X_.p2.reindex(T_.index); m2 = X_.mae2.reindex(T_.index)
                    has = took.notna(); T_.loc[has & (took == 0), 'sz'] = 0.0; sel = has & (took == 1) & p2.notna()
                    T_.loc[sel, 'pts'] = p2[sel]; T_.loc[sel, 'mae'] = m2[sel]
                out.append((T_, u, yz))
            return out
        print('\n=== Lucid Flex 50K ===', flush=True)
        for lab, sp in [('V2 + skip slow flips (now)', base)] + [(f'{en} {v}', apply(en, RES[(en, v)])) for en, v in PASS[:4]]:
            LE_, LF_ = lists(sp, 'e'), lists(sp, 'f'); TRJ = {}
            def traj(k):
                if k not in TRJ:
                    r, k2 = eval_x(LE_, A, k, NS, 1000)
                    if r != 'pass': TRJ[k] = (r, k2 if r == 'fail' else NS, [])
                    else: fr, k3, pays = fund_l(LF_, A, k2 + 1, NS, None); TRJ[k] = (r, NS if fr == 'open' else k3, pays)
                return TRJ[k]
            tr_ = [traj(s) for s in range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 2)]; ppe = np.mean([len(t[2]) for t in tr_]); fl = multi_x({'x': sp}, '50K')
            print(f'  {lab:<30} pass {np.mean([t[0] == "pass" for t in tr_]):3.0%}, paid {np.mean([len(t[2]) > 0 for t in tr_]):3.0%}, evals/payout {1 / ppe:4.2f} | '
                  f'24 starts 2022-23: avg ${fl["avg"]:,.0f} / med ${fl["med"]:,.0f}, losing {fl["lose"]:.0%}', flush=True)
# --- follow-up (EXPLORATORY, after seeing that waiting costs more than it saves): keep entering at the flip, and ADD a second V15a unit after the next bar
#     only if it HOLDS with VOLUME (the C2 entries as extra trades). Per signal this is C0 + C2. Lucid Flex 50K.
if __name__ == '__main__':
    X_ = RES[('V15a', 'C2')]; add = X_[(X_.took == 1) & X_.p2.notna()].copy(); add['pts'], add['mae'] = add.p2, add.mae2; add['m'] = add['m'] + 15
    iv = NAMES.index('V15a'); T_, u, yz = base[iv]; add = add[T_.columns.intersection(add.columns)]
    LOCK['on'] = False; A = ACCX['50K']
    for lab, sp in (('V2 + skip slow flips (now)', base), ('+ add a V15a unit on HOLD + VOL', base + [(add, u, yz)])):
        LE_, LF_ = lists(sp, 'e'), lists(sp, 'f'); TRJ = {}
        def traj(k):
            if k not in TRJ:
                r, k2 = eval_x(LE_, A, k, NS, 1000)
                if r != 'pass': TRJ[k] = (r, k2 if r == 'fail' else NS, [])
                else: fr, k3, pays = fund_l(LF_, A, k2 + 1, NS, None); TRJ[k] = (r, NS if fr == 'open' else k3, pays)
            return TRJ[k]
        tr_ = [traj(s) for s in range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 2)]; ppe = np.mean([len(t[2]) for t in tr_]); fl = multi_x({'x': sp}, '50K')
        print(f'  {lab:<34} pass {np.mean([t[0] == "pass" for t in tr_]):3.0%}, paid {np.mean([len(t[2]) > 0 for t in tr_]):3.0%}, evals/payout {1 / ppe:4.2f} | '
              f'24 starts 2022-23: avg ${fl["avg"]:,.0f} / med ${fl["med"]:,.0f}, losing {fl["lose"]:.0%}', flush=True)
