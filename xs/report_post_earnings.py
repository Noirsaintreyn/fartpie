"""BUY AFTER the earnings beat has shown itself (not before, not waiting). Event study on the 753-name 1h universe (daily bars built from the 1h RTH bars; Yahoo earnings dates/EPS surprise). Reaction session rd = next session if reported after 16:00 else same day. All entry info is known before entry.
 ENTRY A: open of rd+1 (after the whole reaction day is visible).   ENTRY B: close of rd (assumes a market-on-close fill).  Exit: open of session entry+N (N=5,10,20), with an optional close-basis stop below the reaction-day low (STOP).
 SIGNALS (DECLARED before results; mechanism = post-earnings drift + 'reason behind the move', Bernard-Thomas 1989; Frazzini-Lamont):
  S0 any event (control) | S1 beat (surprise>0) | S2 beat & gap>=+3% | S3 beat & gap>=+5% & close>=open | S4 S3 & close in top half of range & day volume>=2x its 20d mean | S5 = S4 with surprise>=20%
 Costs: 40bp round trip when entry<$5, 20bp otherwise. Compared with the control S0 and with random same-size draws from S0 (p). H1 = events before 2025-04-01, H2 after."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb')); EARN = pickle.load(open(SC + 'earn_uni.pkl', 'rb'))
import midsmall_screener as M
R = []
for tk, E in EARN.items():
    if tk not in data: continue
    try: Yc = M.closed_only(data[tk])
    except Exception: continue
    d = Yc.index.normalize(); D = pd.DataFrame({'o': Yc.open.groupby(d).first(), 'h': Yc.high.groupby(d).max(), 'l': Yc.low.groupby(d).min(), 'c': Yc.close.groupby(d).last(), 'v': Yc.volume.groupby(d).sum()}); D['vm'] = D.v.rolling(20).mean().shift(1)
    if len(D) < 60: continue
    E = E[E['Reported EPS'].notna()].copy(); E.index = pd.DatetimeIndex(E.index).tz_localize(None)
    for ts, row in E.iterrows():
        rd = (ts.normalize() + pd.Timedelta(days=1)) if ts.hour >= 16 else ts.normalize(); p = D.index.searchsorted(rd)
        if p < 1 or p + 22 >= len(D) or p >= len(D) or abs((D.index[p] - rd).days) > 4: continue
        pc = D.c.iat[p - 1]; o, h, l, c, v, vm = (D[k].iat[p] for k in ('o', 'h', 'l', 'c', 'v', 'vm'))
        if not (pc > 0 and h > l): continue
        rec = dict(tk=tk, rd=D.index[p], surp=row['Surprise(%)'], gap=o / pc - 1, react=c / pc - 1, hold=c >= o, upper=(c - l) / (h - l) >= .5, vol2=(v >= 2 * vm) if vm == vm else False, low=l)
        for N in (5, 10, 20):
            for nm, pe, pi in (('A', D.o.iat[p + 1], p + 1), ('B', c, p)):
                px = D.o.iat[pi + N] if pi + N < len(D) else np.nan; rec[f'{nm}{N}'] = px / pe - 1
                # stop: first close below reaction-day low during the hold -> exit next open
                path = D.c.iloc[pi:pi + N].values; hit = np.where(path < l)[0]
                if len(hit) and pi + hit[0] + 1 < len(D): rec[f'{nm}{N}s'] = D.o.iat[pi + hit[0] + 1] / pe - 1
                else: rec[f'{nm}{N}s'] = rec[f'{nm}{N}']
        rec['entry_px'] = D.o.iat[p + 1]; R.append(rec)
X = pd.DataFrame(R); X = X[(X.rd >= '2023-12-01')].reset_index(drop=True); X['sub5'] = X.entry_px < 5; print('events', len(X), 'sub5', X.sub5.sum())
X['S0'] = True; X['S1'] = X.surp > 0; X['S2'] = X.S1 & (X.gap >= .03); X['S3'] = X.S1 & (X.gap >= .05) & X.hold; X['S4'] = X.S3 & X.upper & X.vol2; X['S5'] = X.S4 & (X.surp >= 20)
rng = np.random.default_rng(4); X.to_pickle(SC + 'post_earn_X.pkl')
for pn, pm, cost in (('SUB-$5', X.sub5, .004), ('>= $5', ~X.sub5, .002)):
    Xp = X[pm]
    for col in ('A10', 'A10s', 'B10', 'A5', 'A20'):
        print(f'\n=== {pn} | exit rule {col}  (N sessions{", stop = close below reaction-day low" if col.endswith("s") else ""}) ===')
        r0 = Xp[col].dropna() - cost; print(f"   {'signal':32s}{'n':>6s}{'mean':>8s}{'median':>8s}{'win%':>6s}{'t':>6s}{'H1 mean (n)':>16s}{'H2 mean (n)':>16s}{'p(rand)':>8s}")
        for s, lab in (('S0', 'control: any earnings event'), ('S1', 'beat'), ('S2', 'beat & gap>=3%'), ('S3', 'beat & gap>=5% & close>=open'), ('S4', 'S3 & upper half & vol>=2x'), ('S5', 'S4 & surprise>=20%')):
            x = Xp[Xp[s]]; r = (x[col] - cost).dropna()
            if len(r) < 8: print(f'   {lab:32s}{len(r):6d} (too few)'); continue
            h1 = (x[x.rd < '2025-04-01'][col] - cost).dropna(); h2 = (x[x.rd >= '2025-04-01'][col] - cost).dropna(); rm = np.array([r0.sample(len(r), random_state=int(k)).mean() for k in rng.integers(0, 1e9, 1500)]); p = (rm >= r.mean()).mean()
            print(f'   {lab:32s}{len(r):6d}{r.mean() * 100:+7.2f}%{r.median() * 100:+7.2f}%{(r > 0).mean() * 100:6.0f}{r.mean() / (r.std() / np.sqrt(len(r)) + 1e-9):6.1f}{h1.mean() * 100:+10.2f}% ({len(h1):3d}){h2.mean() * 100:+10.2f}% ({len(h2):3d}){p:8.3f}')
