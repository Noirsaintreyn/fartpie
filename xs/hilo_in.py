"""Nov 9 2026: REY'S HIGHS / LOWS MODEL ON NQ FUTURES (screener's high-in / low-in, xs/lod_hod.py + xs/range_models.py). DECLARED BEFORE RESULTS.
Signal (15m checkpoints 10:30 - 15:00, RTH 5m bars): HIGH LOOKS IN = the day's high so far was set >= 30 min earlier and price is >= 0.5 sigma below it (mirror: low in).
Sigma, two anchors: HTF = HAR forecast of the day's high/low range / 1.665 (range-based, as range_models.py, fit on NQ 2014-19) | 4H = Garman-Klass sigma of the last 60
completed 4h bars at 09:30 (24h NQ bars), scaled to a day by calibration. Both known before the signal.
Calibration on 2018-21 only: P(the 'in' extreme holds to the close) by checkpoint; the distance from the in-extreme to the day's final other extreme in sigma units ->
projected other extreme = median (q50) and the level reached 80% of the time (q20).
TRADE: first signal of the day per side; short at the next 5m open after 'high in' (long after 'low in'); stop = the in-extreme + 0.1 sigma; target = projected other extreme
(q50 or q20) if it is at least 0.05 sigma beyond the entry; out at 15:55. 1 pt cost. Judged 2022-25 vs 2018-21.
FILTER: the system's RTH trades (noise momentum, V15a, rules algo, Rule A) taken AGAINST an extreme that looks in at entry (long while the high looks in) vs the rest."""
_ft = open('/home/claude/fartpie/xs/flex50_thod.py').read(); exec(_ft[:_ft.rindex("if __name__ == '__main__':")])
def gk4(t):
    k = X4.index.searchsorted(t, 'left'); r = X4.iloc[max(0, k - 60):k]
    if len(r) < 40: return np.nan
    v_ = np.maximum(.5 * np.log(r.high / r.low) ** 2 - (2 * np.log(2) - 1) * np.log(r.close / r.open) ** 2, 1e-9); return np.sqrt(v_.mean()) * r.close.values[-1]
S4 = pd.Series({d: gk4(d + pd.Timedelta(hours=9, minutes=30)) for d in DY.index if d >= pd.Timestamp('2014-03-01')})
SH = FC / 1.665
def events():
    rows = []
    for d in days:
        if not ok(d) or DY.roll[d] or d not in S4.index: continue
        ix = gi[d]; endp = np.where(m[ix] < 955)[0]
        if not len(endp): continue
        endp = endp[-1]; fh, fl = h[ix].max(), l[ix].min(); sh_, s4_ = SH.get(d, np.nan), S4.get(d, np.nan); hi_ = -np.inf; lo_ = np.inf; th = tl = 570; done = {1: False, -1: False}
        for p in range(len(ix)):
            i = ix[p]; t = m[i] + 5
            if h[i] > hi_: hi_, th = h[i], t
            if l[i] < lo_: lo_, tl = l[i], t
            if t % 15 or t < 630 or t > 900 or p + 1 > endp: continue
            for src, sg in (('HTF', sh_), ('4H', s4_)):
                if not np.isfinite(sg): continue
                if t - th >= 30 and c[i] <= hi_ - .5 * sg and not done.get((src, -1)):
                    rows.append(dict(day=d, src=src, side=-1, t=t, p=p, anchor=hi_, sg=sg, held=fh <= hi_, dist=(hi_ - fl) / sg, e=o[ix[p + 1]])); done[(src, -1)] = True
                if t - tl >= 30 and c[i] >= lo_ + .5 * sg and not done.get((src, 1)):
                    rows.append(dict(day=d, src=src, side=1, t=t, p=p, anchor=lo_, sg=sg, held=fl >= lo_, dist=(fh - lo_) / sg, e=o[ix[p + 1]])); done[(src, 1)] = True
    return pd.DataFrame(rows)
def trade(r_, q):
    d = r_.day; ix = gi[d]; endp = np.where(m[ix] < 955)[0][-1]; j = r_.p + 1; dr = -r_.side if False else (1 if r_.side == 1 else -1)
    e = r_.e; S = r_.anchor - dr * .1 * r_.sg; T = r_.anchor + dr * q * r_.sg
    if (T - e) * dr < .05 * r_.sg or (e - S) * dr <= 0: return np.nan, np.nan
    hi2, lo2, op2, cl2 = h[ix[j:endp + 1]], l[ix[j:endp + 1]], o[ix[j:endp + 1]], c[ix[j:endp + 1]]
    adv = (lo2 <= S) if dr > 0 else (hi2 >= S); fav = (hi2 >= T) if dr > 0 else (lo2 <= T)
    ia = int(np.argmax(adv)) if adv.any() else 10 ** 9; ib = int(np.argmax(fav)) if fav.any() else 10 ** 9
    if ia == ib == 10 ** 9: return (cl2[-1] - e) * dr - 1.0, 0
    if ia < ib or (ia == ib and (cl2[ia] >= op2[ia]) == (dr > 0)): return (S - e) * dr - 1.0, 0
    return (T - e) * dr - 1.0, 1
if __name__ == '__main__':
    EV = events(); EV = EV[EV.day >= '2018-02-01']; EV['per'] = np.where(EV.day.dt.year >= 2022, 'b', 'a')
    nd = {p_: sum(1 for d in days if ok(d) and ((d.year >= 2022) == (p_ == 'b')) and d >= pd.Timestamp('2018-02-01')) for p_ in 'ab'}
    for src in ('HTF', '4H'):
        E = EV[EV.src == src].copy(); a = E[E.per == 'a']
        print(f'\n=== sigma anchor {src}: {len(E)} signals ({len(E[E.per == "a"]) / nd["a"] * 21:.1f} / {len(E[E.per == "b"]) / nd["b"] * 21:.1f} a month) ===', flush=True)
        for lab, lo_t, hi_t in (('10:30-11:59', 630, 719), ('12:00-13:59', 720, 839), ('14:00-15:00', 840, 900)):
            ga, gb = E[(E.per == 'a') & E.t.between(lo_t, hi_t)], E[(E.per == 'b') & E.t.between(lo_t, hi_t)]
            print(f'  P(the in-extreme holds) {lab}: {ga.held.mean():.0%} (n{len(ga)}) / {gb.held.mean():.0%} (n{len(gb)})', flush=True)
        q50, q20 = a.dist.quantile(.5), a.dist.quantile(.2); E['bt'] = E.dist >= q50
        print(f'  projected other extreme (2018-21): median {q50:.2f} sigma from the in-extreme, 80%-reached {q20:.2f} sigma | 2022-25 share reaching them: '
              f'{np.mean(E[E.per == "b"].dist >= q50):.0%} / {np.mean(E[E.per == "b"].dist >= q20):.0%}', flush=True)
        for qn, q in (('target = projected (q50)', q50), ('target = 80% level (q20)', q20)):
            R = np.array([trade(r_, q) for r_ in E.itertuples()], dtype=float); E['pts'], E['hit'] = R[:, 0], R[:, 1]
            for per in ('a', 'b'):
                g = E[(E.per == per) & E.pts.notna()]
                print(f'  {qn} {"2018-21" if per == "a" else "2022-25"}: {len(g) / nd[per] * 21:4.1f} trades/mo, target hit {g.hit.mean():.0%}, {g.pts.mean():+6.2f} pts/trade '
                      f'(shorts {g[g.side == -1].pts.mean():+.2f}, longs {g[g.side == 1].pts.mean():+.2f}), avg risk {((g.e - g.anchor).abs() + .1 * g.sg).mean():.0f} pts', flush=True)
    # filter on the system's RTH trades
    print('\n=== as a filter: system trades taken AGAINST an extreme that looks in at entry (HTF sigma) ===', flush=True)
    def against(d, mm_, dr):
        d = pd.Timestamp(d); ix = gi[d]; w = np.where(m[ix] == mm_)[0]
        if not len(w) or d not in SH.index or not np.isfinite(SH[d]): return np.nan
        p = w[0] - 1
        if p < 0: return np.nan
        sg = SH[d]; seg = ix[:p + 1]; hi_ = h[seg].max(); lo_ = l[seg].min(); th = m[seg][np.argmax(h[seg])] + 5; tl = m[seg][np.argmin(l[seg])] + 5; t = m[ix[p]] + 5; px = c[ix[p]]
        hin = t - th >= 30 and px <= hi_ - .5 * sg; lin = t - tl >= 30 and px >= lo_ + .5 * sg
        return float((dr > 0 and hin) or (dr < 0 and lin))
    for nm, T_ in (('noise momentum', EZ['M1'] if 'EZ' in globals() else None), ('V15a', V15a), ('rules C+TPO', rules_take('C', GR['TPO'])), ('Rule A', RAs[RAs.sz > 0])):
        if T_ is None: continue
        T_ = T_[T_.day >= '2018-02-01'].copy(); T_['ag'] = [against(d, mm_, dr) for d, mm_, dr in zip(T_.day, T_.m, T_.dir)]; T_['per'] = np.where(T_.day.dt.year >= 2022, 'b', 'a')
        out = []
        for lab, msk in (('against an in-extreme', T_.ag == 1), ('the rest', T_.ag == 0)):
            out.append(f'{lab}: {T_[msk & (T_.per == "a")].pts.mean():+6.2f} (n{(msk & (T_.per == "a")).sum()}) / {T_[msk & (T_.per == "b")].pts.mean():+6.2f} (n{(msk & (T_.per == "b")).sum()})')
        print(f'  {nm:<15} ' + ' | '.join(out), flush=True)
