"""Nov 23 2026 report 412. Rey: 'check outlets for QUARTERLY (yfinance / SEC / FRED / LSE)'. Re-run the fundamentals layer with SEC QUARTERLY data (10-Q + 10-K, point in time: usable 3 days after the filing date; discrete quarters from cumulative YTD differences; Q4 = FY - 9M).
FEATURES (declared directions, percentile within the PIT S&P eligible universe each month-end): Q1 revenue YoY growth (+), Q2 growth acceleration = YoY_now - YoY_prior quarter (+), Q3 operating-margin change vs the year-ago quarter (+), Q4 ROE ttm = NI_ttm / equity (+), Q5 earnings yield ttm (+), Q6 accruals = (NI_ttm - CFO_ttm)/assets (LOW better, Sloan 1996), Q7 gross profitability = GP_ttm / assets (+, Novy-Marx).  Composite Fq = mean percentile (direction-adjusted), needs >= 4 of 7.
TESTS: (1) monthly Spearman IC vs 21-day forward excess return (report 407 y), per feature and Fq, train 2019-21 | test 2018H2+2022-26 | pre/post 0DTE; (2) 63-day fixed-hold arms (report 406 sample): top quintile Fq, with gate; (3) analysis-then-flip: Fq >= 0.6 as a gate on the live book's flips (trade level + 6-slot book).  Pass: same sign both halves and |pooled t| >= 2 (IC), or Calmar and MDD better in both halves (book)."""
import numpy as np
_t = open('xs/report_408_analysis_then_flip.py').read(); exec(_t[:_t.index("# 15m: trades + book")])
Qr = pd.read_parquet(SP + 'sec_q.parquet')
for c in ('start', 'end', 'filed'): Qr[c] = pd.to_datetime(Qr[c], errors='coerce')
Qr['dur'] = (Qr.end - Qr.start).dt.days; DQ = {}; INST = {}
for (tk, kind), g in Qr.groupby(['tk', 'kind']):
    if kind in ('rev', 'ni', 'oi', 'gp', 'cfo'):
        g = g[g.dur.between(80, 100) | g.dur.between(170, 195) | g.dur.between(260, 285) | g.dur.between(350, 380)].sort_values('filed').drop_duplicates(['start', 'end'], keep='first'); rows = []
        for s0, h in g.groupby('start'):
            cum = {}
            for _, r in h.iterrows(): cum[1 if r.dur <= 100 else 2 if r.dur <= 195 else 3 if r.dur <= 285 else 4] = (r.val, r.filed, r.end)
            for b in (1, 2, 3, 4):
                if b in cum and (b == 1 or (b - 1) in cum): rows.append((cum[b][2], cum[b][1], cum[b][0] - (cum[b - 1][0] if b > 1 else 0)))
        d = pd.DataFrame(rows, columns=['end', 'filed', 'val']).sort_values('filed').drop_duplicates('end', keep='first').sort_values('end'); DQ[(tk, kind)] = d
    elif kind == 'sh':
        d = g[g.dur.between(80, 100) | g.dur.between(350, 380)].sort_values('filed').drop_duplicates('end', keep='first')[['end', 'filed', 'val']].sort_values('end'); DQ[(tk, kind)] = d
    else:
        d = g[g.start.isna()].sort_values('filed').drop_duplicates('end', keep='first')[['end', 'filed', 'val']].sort_values('end'); INST[(tk, kind)] = d
def avail(d, t): return d[d.filed + pd.Timedelta(days=3) <= t]
def near(d, e, days, tol=25):
    x = d[(d.end - (e - pd.Timedelta(days=days))).abs() <= pd.Timedelta(days=tol)]; return x.val.iloc[-1] if len(x) else np.nan
def qfeat(tk, t, price):
    out = {}; R = DQ.get((tk, 'rev')); 
    if R is None: return None
    R = avail(R, t)
    if len(R) < 6: return None
    e0 = R.end.iloc[-1]
    if (t - e0).days > 200: return None
    yoy = lambda e: (near(R, e, 0, 5) / near(R, e, 365) - 1) if near(R, e, 365) > 0 else np.nan
    out['Q1'] = yoy(e0); out['Q2'] = out['Q1'] - yoy(e0 - pd.Timedelta(days=91)) if np.isfinite(yoy(e0 - pd.Timedelta(days=91))) else np.nan
    O = DQ.get((tk, 'oi')); N = DQ.get((tk, 'ni')); C = DQ.get((tk, 'cfo')); GP = DQ.get((tk, 'gp'))
    if O is not None:
        O = avail(O, t); a, b = near(O, e0, 0, 5), near(O, e0, 365); ra, rb = near(R, e0, 0, 5), near(R, e0, 365)
        out['Q3'] = a / ra - b / rb if all(np.isfinite([a, b, ra, rb])) and ra > 0 and rb > 0 else np.nan
    def ttm(D):
        if D is None: return np.nan
        D = avail(D, t); D = D[D.end <= e0].tail(4)
        return D.val.sum() if len(D) == 4 and (D.end.iloc[-1] - D.end.iloc[0]).days < 300 else np.nan
    ni, cfo, gp = ttm(N), ttm(C), ttm(GP)
    def inst(kd):
        d = INST.get((tk, kd))
        if d is None: return np.nan
        d = avail(d, t); return d.val.iloc[-1] if len(d) else np.nan
    eq, as_ = inst('eq'), inst('assets'); sh = np.nan
    S_ = DQ.get((tk, 'sh'))
    if S_ is not None: S_ = avail(S_, t); sh = S_.val.iloc[-1] if len(S_) else np.nan
    out['Q4'] = ni / eq if np.isfinite(ni) and eq and eq > 0 else np.nan; out['Q5'] = ni / (price * sh) if np.isfinite(ni) and sh and sh > 0 else np.nan
    out['Q6'] = (ni - cfo) / as_ if np.isfinite(ni) and np.isfinite(cfo) and as_ and as_ > 0 else np.nan; out['Q7'] = gp / as_ if np.isfinite(gp) and as_ and as_ > 0 else np.nan
    return out
T6f = pd.read_parquet(SP + 'stock406.parquet')[['tk', 't']]; CLq = {tk: g.sort_values('day').set_index('day').close for tk, g in D1.groupby('symbol')}; recs = []
for tk, t in zip(T6f.tk, T6f.t):
    if (tk, 'rev') not in DQ or tk not in CLq: continue
    px = CLq[tk].reindex([t], method='ffill').iloc[0] if t >= CLq[tk].index[0] else np.nan; f = qfeat(tk, t, px)
    if f: recs.append(dict(tk=tk, t=pd.Timestamp(t), **f))
FQ = pd.DataFrame(recs); DIRQ = {'Q1': 1, 'Q2': 1, 'Q3': 1, 'Q4': 1, 'Q5': 1, 'Q6': -1, 'Q7': 1}
for c, dr in DIRQ.items(): FQ[c + 'p'] = FQ.groupby('t')[c].rank(pct=True) if dr == 1 else 1 - FQ.groupby('t')[c].rank(pct=True)
FQ['nq'] = FQ[[c + 'p' for c in DIRQ]].notna().sum(axis=1); FQ['Fq'] = FQ[[c + 'p' for c in DIRQ]].mean(axis=1).where(FQ.nq >= 4); FQ['Fqp'] = FQ.groupby('t').Fq.rank(pct=True); FQ.to_parquet(SP + 'fq412.parquet')
print('quarterly feature rows', len(FQ), 'with Fq', FQ.Fq.notna().sum(), 'names', FQ.tk.nunique(), flush=True)
# (1) ICs
from scipy.stats import spearmanr
Y = pd.read_parquet(SP + 'opt407.parquet')[['tk', 't', 'y', 'Fp']]; Y['t'] = pd.to_datetime(Y.t); FQ['t'] = pd.to_datetime(FQ.t); Z = Y.merge(FQ, on=['tk', 't'], how='inner'); Z = Z[Z.y.notna()]
def win(t): return 'train' if pd.Timestamp('2019-01-01') <= t < pd.Timestamp('2022-01-01') else 'test'
print(f"\n21d forward-excess monthly IC (direction-adjusted: positive = declared direction works), n names ~{Z.groupby('t').size().mean():.0f}/month\n{'feature':34s} | train IC (t) | test IC (t) | pooled t | pre0DTE | post0DTE")
for c, l in (('Q1p', 'Q1 revenue growth YoY'), ('Q2p', 'Q2 growth acceleration'), ('Q3p', 'Q3 op-margin change'), ('Q4p', 'Q4 ROE ttm'), ('Q5p', 'Q5 earnings yield ttm'), ('Q6p', 'Q6 accruals (low better)'), ('Q7p', 'Q7 gross profitability'), ('Fq', 'Fq composite'), ('Fp', 'old annual F')):
    s = {t: spearmanr(g[c], g.y)[0] for t, g in Z.groupby('t') if g[c].notna().sum() >= 12 and g[c].nunique() > 3 for g in [g[g[c].notna()]]}; s = pd.Series(s).dropna(); f = lambda x: (x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 3 else np.nan)
    a, b, p = f(s[[win(t) == 'train' for t in s.index]]), f(s[[win(t) == 'test' for t in s.index]]), f(s); pre, post = f(s[s.index < pd.Timestamp('2022-05-11')]), f(s[s.index >= pd.Timestamp('2022-05-11')])
    print(f"{l:34s} | {a[0]:+.3f} ({a[1]:+.1f}) | {b[0]:+.3f} ({b[1]:+.1f}) | {p[1]:+.1f} | {pre[0]:+.3f} ({pre[1]:+.1f}) | {post[0]:+.3f} ({post[1]:+.1f})")
# (2)+(3) arms: 63d fixed hold sample and analysis-then-flip
T = pd.read_parquet(SP + 'stock406.parquet'); T['t'] = pd.to_datetime(T.t); T = T.merge(FQ[['tk', 't', 'Fqp']], on=['tk', 't'], how='left')
print('\n63d fixed-hold (E1), top quintile of Fq vs same-month universe: n, win%, median%, arm-minus-universe excess pp (t) train | test')
base = T[T.n1.notna() & (T.half != 'x') & T.Fqp.notna()]
for nm, m in (('Fq top quintile', base.Fqp >= .8), ('Fq top quintile & ratio>=1.5', (base.Fqp >= .8) & (base.G == 1)), ('Fq top 40% & G & R & V', (base.Fqp >= .6) & (base.G == 1) & (base.R == 1) & (base.V == 1)), ('Fq bottom quintile', base.Fqp <= .2)):
    d = base[m]; res = []
    for h in ('train', 'test'):
        dh = d[d.half == h]; bh = base[base.half == h]; mm = dh.groupby('t').x1.mean() - bh.groupby('t').x1.mean().reindex(dh.groupby('t').x1.mean().index); mm = mm.dropna(); res.append((mm.mean() * 100, mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm))) if len(mm) > 3 else np.nan))
    print(f"  {nm:30s} n {len(d):5d} win {(d.n1>0).mean()*100:4.1f}% med {d.n1.median()*100:5.2f}% | {res[0][0]:+5.2f} ({res[0][1]:+.1f}) | {res[1][0]:+5.2f} ({res[1][1]:+.1f})")
T6 = FQ[['tk', 't', 'Fqp']].rename(columns={'Fqp': 'Fp'}).dropna(); T6['t'] = T6.t.astype('datetime64[ns]'); T6 = T6.sort_values('t')
F15 = feats(B0); trades(F15, '15m flips with QUARTERLY Fq (F40 = Fq top 40%)')
TK0, s0 = live_fn(B0); b = score('LIVE', TK0, s0); print('\n' + line(b))
for nm in ('F40', 'F40&R'):
    Bf = B0[ARM[nm](F15).values].reset_index(drop=True); TKs, s = live_fn(Bf); d = score('book: ' + nm + ' (quarterly)', TKs, s); print(line(d) + f" | kept {len(Bf)/len(B0):.0%} | CAGR halves {d['ca'][0]:.1%} | {d['cb'][0]:.1%}", flush=True)
