"""Report 532 (Nov 28 2026): Rey: "do smaller hedge funds post reports ... backtest all we can". Testable with data in hand = SEC 13D/13G activist/ownership filings (74,673 filings 2018-2026, form type + date only; no letter text, no filer identity).
DECLARED BEFORE RESULTS. Mechanism: a 13D (>5% with INTENT to influence; filed within 10 days of crossing, amendments on any material change) is an activist/strategic signal and a liquidity-demand event; 13G is passive. Sub-$5 names are cheap for a hedge fund to take a 5% stake in.
Flags known by close (accepted <20Z same day else next day), prior windows:
  D10  initial SC 13D / SCHEDULE 13D (not /A) in prior 10 trading days;  DA10 13D/A in prior 10d;  D30 any 13D or 13D/A in prior 30d;  G30 initial 13G (not /A) in prior 30d (passive new holder).
Tests: (1) stock-day within-ATRP lift on P(y=+100% in 40d) (template 515/522: month x ATRP decile x price bucket; flag=1 minus 0; eras 2018/2019-21/2022/2023-26; PASS pooled |t|>=3 and same sign >=3/4 eras); (2) flips (1,504): reach/mean/median flag vs rest by era, test 2022+ permutation, hurdle t>=3. Trials: 4 flags x2 tests = 8 (x8 for permutation p).
Not testable (no data): activist short reports (need dataset: HBS Activist Short Campaign Data is library-gated; activistshorts.com unreachable), VIC write-ups (login-gated), fund letters.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; X = SP + 'ext/'
P = pd.read_pickle(X + 'runs515_panel2.pkl'); P['d'] = P.d.astype('datetime64[ns]'); P = P[P.y.notna()].copy(); P['yr'] = P.d.dt.year
A = pd.read_csv(X + '13d_wide.csv'); A['filed'] = pd.to_datetime(A.filed); A['acc'] = pd.to_datetime(A.accept, utc=True, errors='coerce'); A['known'] = pd.to_datetime(np.where(A.acc.dt.hour < 20, A.filed, A.filed + pd.Timedelta(days=1))).astype('datetime64[ns]'); f = A.form.str.upper()
A['is13D'] = f.str.contains('13D'); A['isA'] = f.str.contains('/A'); A['init13D'] = A.is13D & ~A.isA; A['amend13D'] = A.is13D & A.isA; A['init13G'] = f.str.contains('13G') & ~A.isA
KG = {c: {t: np.sort(g.loc[g[c], 'known'].values) for t, g in A.groupby('ticker')} for c in ('init13D', 'amend13D', 'is13D', 'init13G')}
def within(G, tk, d, days):
    a = G.get(tk)
    if a is None or len(a) == 0: return np.zeros(len(d), bool)
    j = np.searchsorted(a, d, side='right'); prev = np.where(j > 0, a[np.clip(j - 1, 0, len(a) - 1)], np.datetime64('NaT')); return (j > 0) & ((d - prev) <= np.timedelta64(days, 'D'))
def flags(df):
    out = {k: np.zeros(len(df), bool) for k in ('D10', 'DA10', 'D30', 'G30')}; ix = pd.Series(range(len(df))).groupby(df.tk.values).indices
    for tk, ii in ix.items():
        d = df.d.values[ii]; out['D10'][ii] = within(KG['init13D'], tk, d, 14); out['DA10'][ii] = within(KG['amend13D'], tk, d, 14); out['D30'][ii] = within(KG['is13D'], tk, d, 42); out['G30'][ii] = within(KG['init13G'], tk, d, 42)
    return pd.DataFrame(out, index=df.index).astype(int)
P = P.join(flags(P)); P['pb'] = np.select([P.c < 1, P.c < 2.5], [0, 1], 2); P['ab10'] = pd.qcut(P.ATRP.rank(method='first'), 10, labels=False); P['era4'] = np.select([P.yr == 2018, P.yr <= 2021, P.yr == 2022], ['2018', '2019-21', '2022'], '2023-26')
print('stock-days', len(P), 'base y', round(P.y.mean(), 4), '| flag shares', {k: round(P[k].mean() * 100, 2) for k in ('D10', 'DA10', 'D30', 'G30')})
def lift2(Gx, col):
    r = []
    for (m, b, p), gp in Gx.groupby(['mon', 'ab10', 'pb']):
        a, c = gp[gp[col] == 1].y, gp[gp[col] == 0].y
        if len(a) >= 3 and len(c) >= 3: r.append((m, a.mean() - c.mean(), len(a)))
    if len(r) < 4: return np.nan, np.nan
    o = pd.DataFrame(r, columns=['m', 'x', 'n']); mm = o.groupby('m').apply(lambda z: np.average(z.x, weights=z.n)); s = mm.std(); return mm.mean(), (mm.mean() / (s / np.sqrt(len(mm))) if s and s > 0 else np.nan)
rows = []
for c in ('D10', 'DA10', 'D30', 'G30'):
    r = dict(flag=c, n1=int(P[c].sum()))
    for e in ('2018', '2019-21', '2022', '2023-26'): d, t = lift2(P[P.era4 == e], c); r[e] = None if d != d else round(d * 100, 1)
    d, t = lift2(P, c); r['pooled_pp'] = None if d != d else round(d * 100, 1); r['t'] = None if t != t else round(t, 1); rows.append(r)
pd.set_option('display.width', 200); print('\nSTOCK-DAY lift on P(+100% in 40d), pp:'); print(pd.DataFrame(rows).to_string(index=False))
F = pd.read_pickle(X + 'flips517.pkl').reset_index(drop=True); F['d'] = pd.to_datetime(F.d).astype('datetime64[ns]'); F['yr'] = F.d.dt.year; F = F.join(flags(F)); te = F.yr >= 2022; rng = np.random.default_rng(11); hit = F[te].hit.astype(float).values
print('\nFLIPS (n=1504) base test reach %.0f%% mean %.1f med %.1f' % (hit.mean() * 100, F[te].r.mean() * 100, F[te].r.median() * 100))
for c in ('D10', 'DA10', 'D30', 'G30'):
    s = F[F[c] == 1]; t = s[te.loc[s.index]]; a = s[~te.loc[s.index]]
    if len(t) < 5: print(c, 'n', len(s), 'test n', len(t), 'too few'); continue
    d = rng.choice(len(hit), (5000, len(t))); p = (hit[d].mean(1) >= t.hit.mean()).mean(); pl = (hit[d].mean(1) <= t.hit.mean()).mean()
    print(f'{c:5} n={len(s):3d} reach {s.hit.mean()*100:3.0f}% mean {s.r.mean()*100:5.1f} med {s.r.median()*100:5.1f} | train n{len(a)} reach {a.hit.mean()*100 if len(a) else float("nan"):.0f}% | test n{len(t)} reach {t.hit.mean()*100:.0f}% mean {t.r.mean()*100:.1f} med {t.r.median()*100:.1f} | p(higher) {p:.3f} p(lower) {pl:.3f} x8 {min(1,min(p,pl)*8):.3f}')
