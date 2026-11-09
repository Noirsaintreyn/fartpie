"""Nov 9 2026: FLEET TEST FOR 'SPEED INTO THE FLIP' (follow-up to xs/flip_char.py). DECLARED BEFORE RESULTS.
flip_char found the 30-minute move into the flip close (ADR, on the trade's side) ordered low < mid < high in BOTH periods for every 15m flip, V15a and Rule A.
Declared pass (docstring rule, a third that loses in both periods): Rule A when that move is < 0.23 ADR (-12.1 / -16.4 pts, n 19 / 22; the code's extra n >= 20 guard
missed it by one trade). Exploratory, NOT a declared pass: V15a with the same cut (-1.9 / +0.3), and both together.
Lucid Flex 50K on the best setup V2: per eval 2018-25 (pass, evals per payout, longest run without a payout) and the 5-account fleet (24 monthly starts 2022-23)."""
_fc = open('/home/claude/fartpie/xs/flip_char.py').read(); exec(_fc[:_fc.rindex("if __name__ == '__main__':")])
if __name__ == '__main__':
    V = run_sig2('s_vector'); V = V[(V.day >= '2018-02-01')].copy(); Vg = V[~hig(V)].copy(); RA_ = RAs[(RAs.sz > 0) & (RAs.day >= '2018-02-01')].copy()
    REF = V.join(feats(V)); cut = REF[pd.to_datetime(REF.day).dt.year < 2022].mv30.quantile(1 / 3)
    SL = {}
    for key, T_ in (('V15a', Vg), ('RA', RA_)):
        F = T_.join(feats(T_)); SL[key] = set(zip(pd.to_datetime(F.day), F.m)) & set(zip(pd.to_datetime(F[F.mv30 < cut].day), F[F.mv30 < cut].m))
        print(f'  {key}: {len(SL[key])} slow-flip trades skipped of {len(T_)} (cut {cut:.3f} ADR)', flush=True)
    LOCK['on'] = False; A = ACCX['50K']; base = variant(True, relm)
    def modify(keys):
        out = []
        for en, (T_, u, yz) in zip(NAMES, base):
            T_ = T_.copy()
            if en in keys: T_.loc[np.array([(pd.Timestamp(d), mm_) in SL[en] for d, mm_ in zip(T_.day, T_.m)]), 'sz'] = 0.0
            out.append((T_, u, yz))
        return out
    for lab, sp in (('V2 as is', base), ('V2, Rule A skips slow flips (declared pass)', modify({'RA'})), ('V2, V15a skips slow flips (exploratory)', modify({'V15a'})),
                    ('V2, both skip slow flips (exploratory)', modify({'RA', 'V15a'}))):
        LE_, LF_ = lists(sp, 'e'), lists(sp, 'f'); TRJ = {}
        def traj(k):
            if k not in TRJ:
                r, k2 = eval_x(LE_, A, k, NS, 1000)
                if r != 'pass': TRJ[k] = (r, k2 if r == 'fail' else NS, [])
                else: fr, k3, pays = fund_l(LF_, A, k2 + 1, NS, None); TRJ[k] = (r, NS if fr == 'open' else k3, pays)
            return TRJ[k]
        tr_ = [traj(s) for s in range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 2)]; ppe = np.mean([len(t[2]) for t in tr_]); s = 0; chain = []
        while s < NS - 1:
            r, end, pays = traj(s); chain.append(pays); s = end + 1
        run = mx = 0
        for p_ in chain: run = run + 1 if not p_ else 0; mx = max(mx, run)
        fl = multi_x({'x': sp}, '50K')
        print(f'  {lab:<46} pass {np.mean([t[0] == "pass" for t in tr_]):3.0%}, evals/payout {1 / ppe if ppe else np.inf:4.2f}, longest dry run {mx:2d} | fleet avg ${fl["avg"]:,.0f}'
              f' / median ${fl["med"]:,.0f}, >= $2.5k {fl["ge25"]:.0%}, losing {fl["lose"]:.0%}, dry {fl["dry"]:.1f} mo', flush=True)
