"""Nov 18 2026 report 327. PAPER CLAIMS ON SMALL CAPS (events): PEAD / quality of a beat (Harvest 'The Event, the Expectation and the Quality of a Beat'; desk D1b PEAD; Bernard-Thomas; Chan-Jegadeesh-Lakonishok EAR).  Declared BEFORE running.
THEORY: investors under-react to earnings news (limited attention, anchoring, analyst herding); small caps have fewer analysts and higher frictions, so drift should be LARGER than in large caps (where report 315/319 found none).  Direction fixed: higher surprise / higher announcement return -> higher drift.
Why it would fail: after-announcement liquidity is thin (the reaction day already prices the news), post-2010 decay (Martineau 2021 'Rest in peace PEAD' says it is gone for big names), survivorship (names that blew up after a miss are gone), EPS 'beats' are managed (guidance walk-down: ~70% beat), so a beat is only information relative to the history of beats.
EVENTS: yfinance earnings dates 2005+ (EPS estimate, reported, surprise %), price data adjusted daily closes.  Reaction day r0 = same day if announced before 15:30 ET else the next trading day.  Entry = CLOSE of r0 (reaction already seen -> tradable).  Abnormal = stock minus the equal-weight universe daily mean.  Price >= $1, 20d $ vol >= $300k.
SIGNALS: (a) surprise % (clipped +-100; needs |estimate| > 0.02); (b) EAR = abnormal return on r0 and the 3 days ending r0 (no overlap with the drift window); (c) beat AND positive reaction ('confirmed beat') vs beat with NEGATIVE reaction ('sold the beat': the market read the quality as poor -> Harvest says the reaction is the quality verdict).
OUTPUT: abnormal drift r0+1..r0+20 and +60 trading days by quintile of each signal, train (2019-21) | test (2018, 2022-26), mean (t, event-clustered by month), n; plus long-top-quintile-minus-universe."""
import glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
C = pd.read_parquet(SP + 'sm_close.parquet'); V = pd.read_parquet(SP + 'sm_vol.parquet'); C = C.dropna(how='all', axis=1); idx = C.index
R = C.pct_change().replace([np.inf, -np.inf], np.nan).clip(-.9, 3); EW = R.mean(axis=1); AR = R.sub(EW, axis=0); CA = AR.fillna(0).cumsum(); DV = (C * V).rolling(20, min_periods=15).median()
rows = []
for f in glob.glob('data/fund/yf_small/*_earn.parquet'):
    tk = os.path.basename(f)[:-13]
    if tk not in C.columns: continue
    e = pd.read_parquet(f)
    if e.empty or 'Reported EPS' not in e: continue
    e = e.dropna(subset=['Reported EPS', 'EPS Estimate']); ts = e.index.tz_convert('America/New_York')
    for t, (est, rep, sp) in zip(ts, e[['EPS Estimate', 'Reported EPS', 'Surprise(%)']].values):
        if t < pd.Timestamp('2016-09-01', tz='America/New_York') or t > pd.Timestamp('2026-08-15', tz='America/New_York'): continue
        d0 = pd.Timestamp(t.date()); late = (t.hour * 60 + t.minute) >= 15 * 60 + 30; i = idx.searchsorted(d0)
        if i >= len(idx): continue
        if idx[i] == d0 and late: i += 1
        elif idx[i] != d0: pass
        if i < 2 or i + 62 >= len(idx): continue
        px = C[tk].iloc[i]; dv = DV[tk].iloc[i]
        if not (px >= 1 and dv >= 3e5): continue
        ca = CA[tk].values; r0 = ca[i] - ca[i - 1]; ear3 = ca[i] - ca[i - 3]
        rows.append(dict(tk=tk, day=idx[i], est=est, rep=rep, sp=sp, r0=r0, ear3=ear3, d20=ca[i + 20] - ca[i], d60=ca[i + 60] - ca[i], px=px))
E = pd.DataFrame(rows).dropna(subset=['d20', 'r0']); E['surp'] = np.where(E.est.abs() > .02, (E.rep - E.est) / E.est.abs(), np.nan).clip(-1, 1); E['m'] = E.day.dt.to_period('M')
E['tr'] = (E.day >= '2019-01-01') & (E.day < '2022-01-01'); E['te'] = ~E.tr & (E.day >= '2018-01-01'); E['lo5'] = E.px < 5
E['beat'] = (E.rep > E.est); E['conf'] = E.beat & (E.r0 > 0); E['sold'] = E.beat & (E.r0 <= 0)
print(f'events {len(E)} (train {int(E.tr.sum())}, test {int(E.te.sum())}); sub-$5 {E.lo5.mean():.0%}; beat rate {E.beat.mean():.0%}; mean r0 {E.r0.mean() * 100:+.2f}%')
def tt(s, m):
    s = s.dropna(); mm = m.loc[s.index]; g = s.groupby(mm).mean()
    return f'{s.mean() * 100:+5.2f}% (t{g.mean() / (g.std() / np.sqrt(len(g))) if len(g) > 5 else np.nan:+4.1f}, n{len(s):4d})'
def quint(sig, lab, sub=None):
    D = E if sub is None else E[sub]
    D = D.dropna(subset=[sig]).copy(); D['q'] = D.groupby(D.day.dt.to_period('Q'))[sig].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) if len(s) >= 25 else np.nan)
    print(f'\n{lab}  (drift = abnormal, from the close of the reaction day)')
    for w in ('d20', 'd60'):
        for nm, m in (('train', D.tr), ('test ', D.te)):
            x = D[m]; a, b = x[x.q == 4], x[x.q == 0]
            print(f'   {w} {nm}: Q5 {tt(a[w], a.m)} | Q1 {tt(b[w], b.m)} | Q5-Q1 {(a[w].mean() - b[w].mean()) * 100:+5.2f}%')
quint('surp', 'SURPRISE % quintiles (all)'); quint('surp', 'SURPRISE % quintiles, price < $5', E.lo5); quint('surp', 'SURPRISE % quintiles, price >= $5', ~E.lo5)
quint('r0', 'REACTION-DAY abnormal return quintiles (EAR)'); quint('ear3', '3-day EAR quintiles')
print('\nbeat quality: drift after a beat by what the market did on the reaction day')
for nm, m in (('train', E.tr), ('test ', E.te)):
    for w in ('d20', 'd60'):
        a, b, c = E[m & E.conf], E[m & E.sold], E[m & ~E.beat]
        print(f'  {w} {nm}: confirmed beat {tt(a[w], a.m)} | beat but sold {tt(b[w], b.m)} | miss {tt(c[w], c.m)}')
print('\nlong-only view: confirmed beat AND surprise in top half, sub-$5 / >=$5')
for nm, m in (('train', E.tr), ('test ', E.te)):
    for lab, s in (('<$5', E.lo5), ('>=$5', ~E.lo5)):
        a = E[m & s & E.conf & (E.surp > E.surp.median())]; print(f'  {nm} {lab:5s} d20 {tt(a.d20, a.m)}  d60 {tt(a.d60, a.m)}')
E.to_parquet(SP + 'e327.parquet')
