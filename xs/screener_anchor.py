"""Nov 5 2026: SCREENER - anchored-volatility room as an entry filter, and 'most room' as the rank. DECLARED BEFORE RESULTS.
Live screener model (premarket+RTH 15m vector, 15m track + 1h track, sticky 1h, +5% / keep-half lock, entries and exits at RTH opens), long only, on the 42 names
with 15m history 2018-2026. At each entry:
  sigma_d  20-day stdev of daily close-to-close returns (prior close);  day low = RTH low so far today (incl. the entry open);  week low = RTH low so far this week
  usedD = (entry - day low) / (entry x sigma_d)            share of a 1-sigma day already travelled up from the day's low (the site's anchored projection: low + 1 sigma)
  usedW = (entry - week low) / (entry x sigma_d x sqrt 5)  same from the week's low with the weekly sigma
Checks: (1) is the low already in when we enter? (share of trades whose day / week low never breaks later) (2) return per trade by usedD / usedW bucket by period
(3) slot book (6 slots, equal weight, ties by options ratio where the ratio file has the name, else alphabetical):
    BASE | SKIP-D skip usedD > 1 | SKIP-W skip usedW > 1 | SKIP-DW both | ROOM ties broken by most weekly room (lowest usedW) | SKIP-W+ROOM.
Periods 2019-21 / 2022-25 / 2026 YTD: CAGR and max drawdown of the book."""
import sys, glob, os, numpy as np, pandas as pd, warnings, multiprocessing as mp; warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/claude/fartpie/xs')
exec(open('/home/claude/fartpie/xs/ah_sticky.py').read().split('def run(')[0])
ROOT = '/mnt/user-data/uploads/fartpie'; OUT = '/tmp/claude-0/ml/scr_anchor'; os.makedirs(OUT, exist_ok=True)
fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob(f'{ROOT}/lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
def run_e(H, h1track):
    reg, s1, o, c, hi, d, m = H.reg.values, H.s1.values, H.open.values, H.close.values, H.high.values, H.dt.values, H.m.values
    g = [j for j in range(1, len(H)) if 570 <= m[j] < 960]; T = []; pos = False
    KR = [reg[j - 1] for j in g]; KS = [s1[j - 1] for j in g]
    for p in range(1, len(g)):
        j = g[p]
        if not pos:
            sig = (KS[p] == 1 and KS[p - 1] != 1) if h1track else (KR[p] == 1 and KR[p - 1] != 1)
            if sig: pos, e, px, b1, peak, h1 = True, j, o[j], False, o[j], False
            continue
        if j == e: continue
        k = g[p - 1]
        if k >= e: peak = max(peak, hi[k])
        b1 = b1 or KS[p] == 1; h1 = h1 or (c[j - 1] / px - 1 >= .02 and b1)
        ex = KS[p] == -1 if h1track else ((KS[p] == -1) if h1 else (KR[p] == -1))
        pg = peak / px - 1
        if pg >= .05 and (c[j - 1] / px - 1) <= .5 * pg: ex = True
        if ex: T.append((e, d[e], d[j], o[j] / px - 1 - 2e-4)); pos = False
    return T
def job(tk):
    H = bars(fs[tk], 960); R = H[(H.m >= 570) & (H.m < 960)]; dd = R.dt.dt.normalize()
    D = R.groupby(dd).agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last')); sig = np.log(D.c).diff().rolling(20).std().shift(1)
    wk = D.index.to_period('W-FRI'); dlow_cum = R.groupby(dd).low.cummin(); rows = []
    for h1t in (False, True):
        for e, te, tx, ret in run_e(H, h1t):
            day = pd.Timestamp(te).normalize(); px = H.open.values[e]
            if day not in D.index or not np.isfinite(sig.get(day, np.nan)): continue
            before = R[(dd == day) & (R.index < e)]; lowD = min(before.low.min() if len(before) else px, px)
            wdays = D.index[(wk == day.to_period('W-FRI')) & (D.index < day)]; lowW = min([lowD] + list(D.l.loc[wdays].values))
            s = sig.loc[day]; usedD = (px - lowD) / (px * s); usedW = (px - lowW) / (px * s * np.sqrt(5))
            after = R[(dd == day) & (R.index >= e)]; dset = after.low.min() >= lowD if len(after) else True
            wafter = D.l.loc[D.index[(wk == day.to_period('W-FRI')) & (D.index > day)]]; wset = dset and (wafter.min() >= lowW if len(wafter) else True)
            rows.append((tk, h1t, te, tx, ret, usedD, usedW, dset, wset, s))
    Z = pd.DataFrame(rows, columns=['tk', 'h1', 'entry', 'exit', 'ret', 'usedD', 'usedW', 'dset', 'wset', 'sig']); Z.to_parquet(f'{OUT}/{tk}.parquet'); return tk, len(Z)
if __name__ == '__main__':
    todo = [t for t in fs if not os.path.exists(f'{OUT}/{t}.parquet') and t not in ('QQQ', 'SPY', 'SMH', 'XLE', 'XLF')]
    with mp.get_context('fork').Pool(2) as pool:
        for tk, n in pool.imap_unordered(job, todo): print(tk, n, end=' | ', flush=True)
    T = pd.concat(pd.read_parquet(f) for f in glob.glob(f'{OUT}/*.parquet')); T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit); T = T[T.entry >= '2019-01-01']
    O = pd.read_csv(f'{ROOT}/xs/ext/optvol_ratio.csv', parse_dates=['date']); O = O[O.sh > 0]; O['r'] = O.opt_contracts * 100 / O.sh
    RAT = O.pivot_table(index='date', columns='tk', values='r').sort_index().rolling(252, min_periods=60).median().shift(1)
    T['ratio'] = [RAT[t].asof(d.normalize()) if t in RAT else 0.0 for t, d in zip(T.tk, T.entry)]; T['ratio'] = T.ratio.fillna(0.0)
    T['per'] = np.select([T.entry < '2022-01-01', T.entry < '2026-01-01'], ['2019-21', '2022-25'], '2026')
    print(f'\n\n{len(T)} screener trades 2019-2026 ({T.tk.nunique()} names), avg {T.ret.mean():+.2%} per trade')
    print('(1) LOW ALREADY IN at entry (post-hoc check of the premise): ' + ' | '.join(f'{p}: day low held {g.dset.mean():.0%}, week low held {g.wset.mean():.0%}' for p, g in T.groupby('per')))
    print('    usedD median %.2f, usedW median %.2f; share with usedD > 1: %.0f%%, usedW > 1: %.0f%%' % (T.usedD.median(), T.usedW.median(), 100 * (T.usedD > 1).mean(), 100 * (T.usedW > 1).mean()))
    for col in ('usedD', 'usedW'):
        b = pd.cut(T[col], [-1, .33, .67, 1.0, 1.5, 99], labels=['<0.33', '0.33-0.67', '0.67-1', '1-1.5', '>1.5']); print(f'(2) return per trade by {col}')
        for v_ in b.cat.categories:
            g = T[b == v_]; print(f'    {v_:<10} n{len(g):>5} avg {g.ret.mean():+.2%} win {g.ret.gt(0).mean():.0%} | ' + ' | '.join(f'{p} {g[g.per == p].ret.mean():+.2%} (n{(g.per == p).sum()})' for p in ('2019-21', '2022-25', '2026')))
    def sim(Z, key='ratio', n=6):
        Z = Z.assign(_k=-Z.usedW if key == 'room' else Z.ratio).sort_values(['entry', '_k', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable'); op, ev = [], []
        for e, x, r in zip(Z.entry.values, Z.exit.values, Z.ret.values):
            op = [o for o in op if o > e]
            if len(op) < n: op.append(x); ev.append((x, r / n))
        return pd.Series([v for _, v in ev], index=pd.DatetimeIndex([t for t, _ in ev])).sort_index()
    def st(s):
        e = (1 + s).groupby(s.index.normalize()).prod().cumprod(); y = max((e.index[-1] - e.index[0]).days / 365.25, .25)
        return e.iloc[-1] ** (1 / y) - 1 if y >= 1 else e.iloc[-1] - 1, (e / e.cummax() - 1).min()
    V = {'BASE': (T, 'ratio'), 'SKIP-D usedD > 1': (T[T.usedD <= 1], 'ratio'), 'SKIP-W usedW > 1': (T[T.usedW <= 1], 'ratio'), 'SKIP-DW': (T[(T.usedD <= 1) & (T.usedW <= 1)], 'ratio'),
         'ROOM most weekly room first': (T, 'room'), 'SKIP-W + ROOM': (T[T.usedW <= 1], 'room')}
    print(f'\n(3) SLOT BOOK, 6 slots      {"2019-21":>16}{"2022-25":>16}{"2026 YTD":>16}   trades taken')
    for nm, (Z, key) in V.items():
        res = [st(sim(Z[Z.per == p], key)) for p in ('2019-21', '2022-25', '2026')]
        print(f'  {nm:<28}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in res) + f'{len(sim(Z, key)):>10,}', flush=True)
