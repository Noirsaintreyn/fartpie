"""Nov 2 2026: three untested structural questions (rules fixed in advance; 40 matched runs vs live, return per drawdown).
 A  overnight vs intraday: split every booked trade's P&L into overnight gaps vs regular-session moves; then allow entries after 14:00
    (live blocks them) so late flips are held overnight
 B  the big day: while held, the first day the stock closes > 2 sigma up (20d) - B1 take profit at that close; B2 keep it but exit at the
    first later close below that big day's low; also how early the move is visible (time it first exceeded 1 sigma from the prior close)
 C  correlation-aware slots: a new flip whose 60-day daily-return correlation with any held name is > 0.7 is ranked after all others
    (C1) or skipped when 2+ held names are that correlated with it (C2)"""
import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
RAW = pd.concat(pd.read_parquet(f) for f in glob.glob('tm/*.parquet')); RAW = RAW[RAW['var'] == 'vec']
Ls = Ls.reset_index(drop=True)
DAY, BAR = {}, {}
for tk in Ls.tk.unique():
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; R = H[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt')
    D = R.groupby(R.dt.dt.normalize()).agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last'))
    D['s'] = D.c.pct_change().rolling(20).std().shift(1); DAY[tk] = D; BAR[tk] = R.set_index('dt')[['open', 'close']]
rng0 = lambda: np.random.default_rng(97)
def runs(Z, simf=None, n=41):
    rng = rng0(); out = []
    for i in range(n):
        dl = rng.random(len(Z)) < .05 if i else np.zeros(len(Z), bool)
        eq = (simf or simw)(Z[~dl]); out.append([v for cd in stat(dret(eq).cumsum()) for v in (cd[0], cd[1], cd[0] / -cd[1])])
    return np.array(out)
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<46} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}  (return {(a[1:, 0] > b[1:, 0]).mean():.0%} / {(a[1:, 3] > b[1:, 3]).mean():.0%})'
    print(s, flush=True)
B0 = runs(Ls); print('baseline'); line('live', B0)
# ---------------- A
TK = taken(Ls); on = intra = 0.0; rows = []
for tk, e, x, r in TK.itertuples(index=False):
    if tk not in DAY: continue
    D = DAY[tk]; b = BAR[tk]; e, x = pd.Timestamp(e), pd.Timestamp(x); px = b.open.get(e, np.nan); xp = px * (1 + r)
    if px != px: continue
    days = D[(D.index >= e.normalize()) & (D.index <= x.normalize())]; o_, i_ = 0.0, 0.0; prev = px
    for k, (d_, row) in enumerate(days.iterrows()):
        if k > 0: o_ += np.log(row.o / prev); prev = row.o
        end = xp if d_ == x.normalize() else row.c; i_ += np.log(end / prev); prev = row.c
    rows.append((e, o_, i_))
A = pd.DataFrame(rows, columns=['e', 'on', 'intra']); A['h'] = np.where(A.e.dt.year <= 2021, '2019-21', '2022-26')
print('\nA) booked trades, sum of log returns:'); print(A.groupby('h')[['on', 'intra']].sum().round(2).to_string())
LATE = prep(glob.glob('tm/*.parquet')) if False else None
src = open('/home/claude/fartpie/xs/book_mtm.py').read()
def prep_all(fs):                                                                       # prep without the 14:00 entry cutoff
    exec(src.split('def prep(fs):')[0]) if False else None
    S_ = pd.concat(pd.read_parquet(f) for f in fs); S_ = S_[S_['var'] == 'vec']; S_['d'] = S_.entry.dt.normalize()
    S_ = S_[S_.entry >= '2019-01-01'].sort_values('d')
    S_ = pd.merge_asof(S_, RT, on='d', by='tk', direction='nearest', tolerance=pd.Timedelta(days=5)); S_ = S_[(S_.sp == True) & (S_.ratio >= 1.5)]; S_['w'] = 1.0; return S_.reset_index(drop=True)
_W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); _W['d'] = _W.entry.dt.normalize()
RT = _W.groupby(['tk', 'd']).agg(ratio=('ratio', 'first'), sp=('sp', 'first')).reset_index().sort_values('d')
LA = prep_all(glob.glob('tm/*.parquet')); late = (LA.entry.dt.hour * 60 + LA.entry.dt.minute) >= 840
print(f'  entries at/after 14:00 available: {late.mean():.0%} of flips; their avg trade {LA[late].ret.mean():+.2%} vs earlier {LA[~late].ret.mean():+.2%}')
line('A: allow entries after 14:00', runs(LA), B0)
# ---------------- B
def bigday(mode):
    ex, rt, hit = Ls.exit.copy(), Ls.ret.copy(), 0
    for i, (tk, e, x, r) in enumerate(zip(Ls.tk, Ls.entry, Ls.exit, Ls.ret)):
        if tk not in DAY: continue
        D = DAY[tk]; px = BAR[tk].open.get(e, np.nan)
        if px != px: continue
        days = D[(D.index >= e.normalize()) & (D.index < x.normalize())]
        for k, (d_, row) in enumerate(days.iterrows()):
            if k == 0 or row.s != row.s: continue
            prevc = D.c.shift().get(d_)
            if row.c / prevc - 1 > 2 * row.s:
                if mode == 'B1': ex[i], rt[i] = d_ + pd.Timedelta('15h59min'), row.c / px - 1 - 2e-4; hit += 1
                else:
                    later = D[(D.index > d_) & (D.index < x.normalize())]; brk = later[later.c < row.l]
                    if len(brk): ex[i], rt[i] = brk.index[0] + pd.Timedelta('15h59min'), brk.c.iloc[0] / px - 1 - 2e-4; hit += 1
                break
    Z = Ls.copy(); Z['exit'], Z['ret'] = ex, rt; return Z, hit
for mode, nm in (('B1', 'B1: take profit at the close of the first >2-sigma up day'), ('B2', 'B2: after it, exit on a close below that day\'s low')):
    Z, hit = bigday(mode); print(f'  {mode}: changes {hit} trades ({hit / len(Ls):.1%})'); line(nm, runs(Z), B0)
tm_ = []
for tk, e, x, r in TK.itertuples(index=False):                                       # how early is the big day visible?
    if tk not in DAY: continue
    D = DAY[tk]; b = BAR[tk]
    for d_, row in D[(D.index > pd.Timestamp(e).normalize()) & (D.index < pd.Timestamp(x).normalize())].iterrows():
        prevc = D.c.shift().get(d_)
        if row.s == row.s and row.c / prevc - 1 > 2 * row.s:
            bb = b[(b.index >= d_) & (b.index < d_ + pd.Timedelta(days=1))]; up = bb.close / prevc - 1 > row.s
            tm_.append(bb.index[up.values.argmax()].strftime('%H:%M') if up.any() else 'never'); break
tm_ = pd.Series(tm_); print('  big up days: time the move first exceeded 1 sigma (bar close):', (tm_.value_counts(normalize=True).sort_index().cumsum().round(2)).to_dict())
# ---------------- C
CL = pd.DataFrame({tk: DAY[tk].c for tk in DAY}).sort_index(); RET = CL.pct_change()
def simc(Z, mode, n=6, brk=.08, thr=.7):
    Z = Z.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0
    rows_ = list(zip(Z.tk.values, Z.entry.values, Z.exit.values, Z.ret.values)); i = 0
    while i < len(rows_):
        e = rows_[i][1]; grp = []
        while i < len(rows_) and rows_[i][1] == e: grp.append(rows_[i]); i += 1
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / n); hist.append((xt, real)); win.append((xt, real))
        op = [q for q in op if q[1] > e]
        d0 = pd.Timestamp(e).normalize(); W_ = RET[(RET.index < d0)].tail(60)
        def ncorr(tk):
            if tk not in W_ or not op: return 0
            return sum(1 for h_, _ in op if h_ in W_ and W_[tk].corr(W_[h_]) > thr)
        if mode == 'C1': grp = sorted(grp, key=lambda g_: ncorr(g_[0]) > 0)                # stable: correlated flips go last
        for tk, e_, x, r_ in grp:
            if len(op) >= n: break
            if mode == 'C2' and ncorr(tk) >= 2: continue
            if brk and hist:
                t0 = e_ - np.timedelta64(28, 'D')
                while win and win[0][0] < t0: lvl0 = win.popleft()[1]
                if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
            op.append((tk, x)); pend.append((x, r_))
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr / n); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index(); return s.groupby(s.index.normalize()).last()
line('C1: correlated flips ranked last', runs(Ls, lambda Z: simc(Z, 'C1')), B0)
line('C2: skip if 2+ held names correlated > 0.7', runs(Ls, lambda Z: simc(Z, 'C2')), B0)
