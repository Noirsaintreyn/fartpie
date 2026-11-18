"""Nov 18 2026 report 326. PAPER CLAIMS ON SMALL CAPS (fundamentals): accounting forensics, quality, valuation, micro/corporate-finance factors.  Declared BEFORE running.
Universe: 753-name mid/small/sub-$5 list (today's list = SURVIVORSHIP BIAS: failed names are missing; it flatters distress/leverage/low-quality longs and understates the quality spread -- read the sign and the split, not the level).
Point-in-time: annual 10-K values (original, earliest filing), usable from filed + 3 days, <= 18 months old. Monthly rebalance, month-end -> next month-end, equal weight, quintiles, 20-day median $ volume >= $300k, price >= $1.
Direction of every factor FIXED here from the papers (higher score = predicted better):
  accr   -(NI-CFO)/assets          Sloan; Harvest 'Earnings quality & forensic accounting': accruals revert, cash earnings persist.
  cfoa   CFO/assets                cash profitability (conversion of reported profit into cash).
  cashconv  CFO/|NI| capped        Harvest 'Financial statements': a profit is a claim, not a cash receipt.
  pio    Piotroski F (9 tests)     Piotroski; Harvest 'Reading filings'.
  gpa    gross profit/assets       Novy-Marx; micro: pricing power/ability to retain surplus (Harvest 'Microeconomics of the firm', 'Bottleneck').
  roce   EBIT/(assets-current liab)   corporate finance doc: ROIC vs WACC.
  gmstab -std(gross margin, 5y)    micro: stable margin = pricing power; bottleneck paper: capture of surplus.
  agrow  -asset growth             investment/overinvestment (micro; Cooper-Gulen-Schill).
  issue  -net share issuance       dilution transfers value away from holders (very common in small caps).
  lev    -liabilities/assets       distress / Merton-style leverage.
  ey, fcfy, sy, bm   NI/mcap, (CFO-capex)/mcap, sales/mcap, equity/mcap     valuation anchors (DCF/relative-valuation papers; Harvest 'regime' notes warn value can be a trap).
  sgrow  sales growth yoy          growth WITHOUT capture is predicted NOT to pay: test sgrow alone and sgrow x gpa.
  mom    12-1 momentum             price-only control (composite-Z doc uses 12-1 RS).
  qual   z(gpa)+z(roce)+z(accr)+z(-lev) pre-registered composite.
Why any could fail: survivorship; micro-cap illiquidity (returns not capturable); value trap regimes (2019-21 growth); annual data are stale up to 12 months.
Admission rule (from the large-cap work): same sign in train (2019-21) and test (2018 + 2022-26), |t| >= 2 in at least one, and a stated failure regime.  ~17 factors tested => a single |t|>2 is NOT enough alone (expect ~1 false hit); look for coherence across related factors."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
C = pd.read_parquet(SP + 'sm_close.parquet'); V = pd.read_parquet(SP + 'sm_vol.parquet'); A = pd.read_parquet(SP + 'sec_annual.parquet')
C = C.dropna(how='all', axis=1); A = A[A.ticker.isin(C.columns)].sort_values(['ticker', 'end']).copy()
g = A.groupby('ticker'); gap = (A.end - g.end.shift(1)).dt.days; ok = (gap > 330) & (gap < 400)
def lag(c): return g[c].shift(1).where(ok)
A['gp2'] = A.gp.fillna(A.rev - A.cogs); sh = A.shs.fillna(A.shs_w)
A['accr'] = -(A.ni - A.cfo) / A.assets; A['cfoa'] = A.cfo / A.assets; A['cashconv'] = (A.cfo / A.ni.abs().replace(0, np.nan)).clip(-3, 3)
A['gpa'] = A.gp2 / A.assets; A['roce'] = A.opinc / (A.assets - A.cl); A['gm'] = A.gp2 / A.rev
A['gmstab'] = -A.groupby('ticker').gm.transform(lambda s: s.rolling(5, min_periods=3).std())
A['agrow'] = -(A.assets / lag('assets') - 1); A['issue'] = -(sh / sh.groupby(A.ticker).shift(1).where(ok) - 1)
A['lev'] = -A.liab / A.assets; A['sgrow'] = A.rev / lag('rev') - 1
roa, roa0 = A.ni / A.assets, lag('ni') / lag('assets')
pio = pd.concat([(roa > 0), (A.cfo > 0), (roa > roa0), (A.cfo > A.ni), (A.ltd / A.assets < lag('ltd') / lag('assets')), (A.ca / A.cl > lag('ca') / lag('cl')),
                 (sh <= sh.groupby(A.ticker).shift(1).where(ok) * 1.005), (A.gm > lag('gm')), (A.rev / A.assets > lag('rev') / lag('assets'))], axis=1)
A['pio'] = pio.sum(axis=1).where(pio.notna().sum(axis=1) >= 6)
_sp = pd.read_parquet(SP + 'splits.parquet'); _sp['date'] = pd.to_datetime(_sp.date); _sg = {k: v for k, v in _sp.groupby('tk')}
A['sh'] = [(x * (float(_sg[t][_sg[t].date > d].ratio.prod()) if t in _sg and (_sg[t].date > d).any() else 1.0)) for t, d, x in zip(A.ticker, A.filed, sh)]   # shares on today's split-adjusted basis (fix of Nov 18: prices are split-adjusted, SEC shares are as reported)
A['avail'] = A.filed + pd.Timedelta(days=3)
FAC = ['accr', 'cfoa', 'cashconv', 'pio', 'gpa', 'roce', 'gmstab', 'agrow', 'issue', 'lev', 'ey', 'fcfy', 'sy', 'bm', 'sgrow', 'mom', 'qual', 'gpa_x_growth']
# ---- monthly panel
me = C.resample('ME').last(); ME = me.index; dv = (C * V).rolling(20, min_periods=15).median().resample('ME').last()
rows = []
Af = A.dropna(subset=['avail']).sort_values('avail')
for i in range(12, len(ME) - 1):
    t = ME[i]; px = me.iloc[i]; fwd = me.iloc[i + 1] / px - 1; mom = me.iloc[i - 1] / me.iloc[i - 12] - 1
    d = pd.DataFrame({'ticker': px.index, 'px': px.values, 'fwd': fwd.values, 'mom': mom.values, 'dv': dv.iloc[i].values}); d = d[(d.px >= 1) & (d.dv >= 3e5) & d.fwd.notna()]
    a = Af[(Af.avail <= t) & (Af.end >= t - pd.Timedelta(days=550))].groupby('ticker').tail(1); d = d.merge(a, on='ticker', how='left'); d['t'] = t
    mc = d.px * d.sh; d['ey'] = d.ni / mc; d['fcfy'] = (d.cfo - d.capex.fillna(0)) / mc; d['sy'] = d.rev / mc; d['bm'] = d['eq'] / mc
    for c in ('ey', 'fcfy', 'sy', 'bm'): d[c] = d[c].where(mc > 0)
    rows.append(d)
P = pd.concat(rows); z = lambda s: (s - s.mean()) / s.std()
P['qual'] = P.groupby('t').gpa.transform(z) + P.groupby('t').roce.transform(lambda s: z(s.clip(s.quantile(.02), s.quantile(.98)))) + P.groupby('t').accr.transform(lambda s: z(s.clip(s.quantile(.02), s.quantile(.98)))) + P.groupby('t').lev.transform(z)
P['gpa_x_growth'] = P.groupby('t').gpa.transform(lambda s: s.rank(pct=True)) * P.groupby('t').sgrow.transform(lambda s: s.rank(pct=True))   # growth that is also profitable
P['tr'] = (P.t >= '2019-01-01') & (P.t < '2022-01-01'); P['lo5'] = P.px < 5; P.to_parquet(SP + 'p326.parquet')
print(f'panel: {P.t.nunique()} months, avg names/month {len(P) / P.t.nunique():.0f}, with 10-K data {P.assets.notna().mean():.0%}; sub-$5 share {P.lo5.mean():.0%}')
def spread(d, f, q=5):
    d = d.dropna(subset=[f, 'fwd']); out = {}
    for t, x in d.groupby('t'):
        if len(x) < 40: continue
        r = x[f].rank(method='first'); b = pd.qcut(r, q, labels=False); out[t] = (x.fwd[b == q - 1].mean() - x.fwd[b == 0].mean(), x.fwd[b == q - 1].mean(), x.fwd[b == 0].mean(), x.fwd.mean())
    r = pd.DataFrame(out, index=['ls', 'top', 'bot', 'mkt']).T
    if r.empty: r = pd.DataFrame(columns=['ls', 'top', 'bot', 'mkt'], index=pd.DatetimeIndex([]), dtype=float)
    return r
def fmt(s):
    s = s.dropna()
    if len(s) < 8: return '   n/a'
    return f'{s.mean() * 100:+5.2f}% (t{s.mean() / (s.std() / np.sqrt(len(s))):+4.1f},n{len(s):3d})'
print('\nTop-minus-bottom quintile, monthly, equal-weight.  [train 2019-21 | test 2018+2022-26]   (ALL | price<$5 | price>=$5)')
rep = {}
for f in FAC:
    sp = spread(P, f); tr = sp[(sp.index >= '2019-01-01') & (sp.index < '2022-01-01')].ls; te = sp[~((sp.index >= '2019-01-01') & (sp.index < '2022-01-01'))].ls
    s5 = spread(P[P.lo5], f).ls; s6 = spread(P[~P.lo5], f).ls; sel = lambda s, m: s[m(s.index)]; isTr = lambda ix: (ix >= '2019-01-01') & (ix < '2022-01-01')
    ok_ = np.sign(tr.mean()) == np.sign(te.mean()) and (abs(tr.mean() / (tr.std() / np.sqrt(len(tr)))) >= 2 or abs(te.mean() / (te.std() / np.sqrt(len(te)))) >= 2)
    print(f'  {f:13s} train {fmt(tr)} | test {fmt(te)} {"ADMIT?" if ok_ else "      "} | <$5: tr {fmt(sel(s5, isTr))[:24]} te {fmt(sel(s5, lambda ix: ~isTr(ix)))[:24]} | >=$5: tr {fmt(sel(s6, isTr))[:24]} te {fmt(sel(s6, lambda ix: ~isTr(ix)))[:24]}')
    rep[f] = sp
# placebo: shuffled factor
rng = np.random.default_rng(3); Q = P.copy(); Q['plc'] = Q.groupby('t').gpa.transform(lambda s: rng.permutation(s.values)); sp = spread(Q, 'plc'); print('\nplacebo (shuffled factor within month):', fmt(sp.ls))
print('equal-weight universe monthly mean: train', fmt(sp.mkt[(sp.index >= '2019-01-01') & (sp.index < '2022-01-01')]), '| test', fmt(sp.mkt[~((sp.index >= '2019-01-01') & (sp.index < '2022-01-01'))]))
print('\nTop-quintile ABSOLUTE vs universe (monthly, excess) for the ADMIT? factors -- is the long leg alone useful (long-only book)?')
for f in FAC:
    sp = rep[f]; ex = sp.top - sp.mkt; a = ex[(ex.index >= '2019-01-01') & (ex.index < '2022-01-01')]; b = ex[~((ex.index >= '2019-01-01') & (ex.index < '2022-01-01'))]
    print(f'  {f:13s} top-quintile minus universe: train {fmt(a)} | test {fmt(b)}')
