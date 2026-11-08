"""Nov 8 2026: 5-MINUTE FLIPS WITH TIGHTER BRACKETS, and only the best top % by a rule score picked on 2018-21. DECLARED BEFORE RESULTS.
Candidates: every QQQ 5m vector flip 09:35 - 15:00 (xs/nq_meta5.py), entry at the next 5m open, 1 pt cost, stop checked first inside a bar.
Brackets (target / stop as a fraction of the 20-day ADR; ~264 NQ pts in 2022-26): 0.04/0.04 (~10/10), 0.06/0.06, 0.06/0.04, 0.10/0.05, 0.12/0.06, 0.16/0.08; time limit 30 or 60 min;
also 'reverse' = exit early when the 5m vector flips back.
Score: each 0/1 feature whose 2018-21 'with' minus 'without' average is positive for the chosen bracket counts 1 (features: negative gamma, 5-day vol top third, RVOL top 30%,
not the bad value-area side (long below VAL / short below VAL), not at a 20-day LVN, 15m vector agrees, 1h vector agrees, first 90 minutes, flip-bar range top 30%);
top 10 / 20 / 30% by score (ties broken by RVOL) judged on 2022 - Oct 2026 only. Points per trade, trades a day, points a day and +7.5-point days a month at 1 contract."""
_m = open('xs/nq_meta5.py').read(); exec(_m[:_m.rindex("if __name__ == '__main__':")])
def relabel(tf, sf, maxb, rev):
    out = []
    for r in C5.itertuples():
        i, dr = int(r.i), int(r.dir); e = o5[i + 1]; kk = k5[i]; T = tf * r.adr; S = sf * r.adr; res = None
        for q in range(i + 1, min(i + 1 + maxb, n)):
            if d5[q] != d5[i] or m5[q] >= 955: res = (c5[q - 1] - e) * dr * kk; break
            lo_ = ((l5[q] if dr > 0 else h5[q]) - e) * dr * kk; hi_ = ((h5[q] if dr > 0 else l5[q]) - e) * dr * kk
            if lo_ <= -S: res = -S; break
            if hi_ >= T: res = T; break
            if rev and r5v[q] == -dr and q + 1 < n and d5[q + 1] == d5[i]: res = (o5[q + 1] - e) * dr * kk; break
        if res is None: res = (c5[min(i + maxb, n - 1)] - e) * dr * kk
        out.append(res - 1.0)
    return np.array(out)
tr = (C5.yr < 2022).values; te = (C5.yr >= 2022).values
bad_va = ((C5.dir > 0) & (C5.va == 0)) | ((C5.dir < 0) & (C5.va == 3))
FB = {'neg gamma': C5.gexneg == 1, 'vol top third': C5.rv5p >= 2 / 3, 'RVOL top 30%': C5.rvol >= C5.rvol[tr].quantile(.7), 'not bad VA side': ~bad_va, 'not LVN': C5.node > .3,
      '15m vector agrees': C5.a15 > 0, '1h vector agrees': C5.a1h > 0, 'first 90 min': C5.minute <= 660, 'flip bar range top 30%': C5.rng >= C5.rng[tr].quantile(.7)}
AD = pd.DatetimeIndex(sorted(C5.day.unique())); AD22 = AD[AD >= '2022-01-01']
def book(pts, mask):
    G = C5[mask].assign(pts=pts[mask]); rows = []
    for d, g in G.groupby('day', sort=True):
        busy = -1; k_ = 0
        for r in g.itertuples():
            if r.i <= busy or k_ >= 3: continue
            rows.append((d, r.pts)); busy = r.i + 6; k_ += 1
    T_ = pd.DataFrame(rows, columns=['day', 'pts']); P = T_.groupby('day').pts.sum().reindex(AD22).fillna(0.0)
    return len(T_) / len(AD22), T_.pts.mean(), P.mean(), (P >= 7.5).mean() * 21, P.mean() / P.std() * np.sqrt(252) if P.std() > 0 else np.nan
if __name__ == '__main__':
    print(f'candidates {len(C5)}: 2018-21 {tr.sum()}, 2022-26 {te.sum()}', flush=True)
    print('\nBRACKETS, every 5m flip: points per trade 2018-21 | 2022-26 | win 2022-26', flush=True)
    best = None
    for tf, sf in ((.04, .04), (.06, .06), (.06, .04), (.10, .05), (.12, .06), (.16, .08)):
        for maxb, rev in ((6, False), (12, False), (12, True)):
            p = relabel(tf, sf, maxb, rev); a, b = p[tr].mean(), p[te].mean()
            print(f'  {tf:.2f} / {sf:.2f}  {maxb * 5:>2} min {"+ reverse" if rev else "         "}: {a:+6.2f} | {b:+6.2f} | {np.mean(p[te] > 0):3.0%}', flush=True)
            if best is None or a > best[0]: best = (a, tf, sf, maxb, rev, p)
    a, tf, sf, maxb, rev, p = best
    print(f'\nbest bracket on 2018-21: {tf:.2f} / {sf:.2f}, {maxb * 5} min{" + reverse" if rev else ""} ({a:+.2f} per trade in 2018-21, {p[te].mean():+.2f} in 2022-26)', flush=True)
    print('feature spreads (with minus without, points per trade) 2018-21 -> 2022-26:', flush=True)
    keep = []
    for k, m in FB.items():
        m = m.values; s_a = p[tr & m].mean() - p[tr & ~m].mean(); s_b = p[te & m].mean() - p[te & ~m].mean()
        if s_a > 0: keep.append(k)
        print(f'  {k:<24} {s_a:+6.2f} -> {s_b:+6.2f} {"(in score)" if s_a > 0 else ""}', flush=True)
    sc = sum(FB[k].astype(int) for k in keep).values + C5.rvol.rank(pct=True).values * 1e-3
    print(f'\nTOP % by score ({len(keep)} features), judged 2022-26 at 1 contract, <= 3 trades a day, one at a time:', flush=True)
    print('  selection         trades/day  pts/trade  pts/day  +7.5-pt days/month  Sharpe', flush=True)
    for lab, mask in [('every flip', np.ones(len(C5), bool))] + [(f'top {int(x * 100)}%', sc >= np.quantile(sc[tr], 1 - x)) for x in (.3, .2, .1, .05)]:
        tpd, ppt, ppd, q75, sh = book(p, mask & te)
        print(f'  {lab:<16} {tpd:11.2f} {ppt:+10.2f} {ppd:+8.2f} {q75:19.1f} {sh:7.2f}', flush=True)
