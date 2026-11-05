"""Nov 4 2026: GATE or EXPLANATION? Assign a strategy to each regime, walk-forward, for the NQ prop account. DECLARED BEFORE RESULTS.
STRATEGY LIBRARY (all intraday, flat by the window end, never overnight; NQ via the QQQ proxy, 1 NQ = $20/pt, 1 pt cost; ES via SPY 15m, 1 ES = $50/pt, 0.5 pt cost):
  FLIP   15m vector flip, signal bars closing 09:30-11:30, both ways, <= 2 a day, 75/50, flat 12:00 (the base)
  LONG / SHORT  the same flips, one direction only      FADE  trade AGAINST the flip, 75/50
  TIGHT  flip at 40/25        WIDE  flip at 150/75
  PM     15m flip, signal bars closing 13:30-15:15, 75/50, flat 15:55
  RSI    RSI14 (15m) crosses back above 30 / below 70 in 09:30-11:30, 75/50 (mean reversion)
  ORB    first 5m close beyond the 09:30-10:00 range between 10:00 and 11:30, 75/50, one a day
  IMOM   intraday momentum: 15:30-16:00 in the direction of prior close -> 10:00, no stop
  ES     SPY 15m flip NY AM, 1 ES 30/20 (SPY 15m ends Aug 2026)
  CASH   no trade
REGIMES, all known at the PRIOR close: HMM bull/bear (2-state on QQQ daily returns, fit 2016-21, forward-filtered) | QZ QQQ 20d z sign | HAR forecast tercile |
  GAMMA QQQ net gamma sign | VXN vs its 20d average (falling < -10% / flat / rising > +10%) | TREND QQQ 20d return sign | HMMxQZ (4 cells) | HARxGAMMA (6 cells)
META, re-chosen at every month start from days BEFORE that month only, then traded unchanged for the month (fallback FLIP when a cell has < 40 history days):
  STATIC FLIP | STATIC best-of-2018-21 | ROLL best trailing 12m / 24m / expanding mean $/day (no regime) |
  GATE  per cell: trade FLIP only if FLIP's trailing mean $/day in that cell > 0, else cash (expanding / 24m) |
  MAP   per cell: the strategy (or cash) with the best trailing mean $/day in that cell (expanding / 24m) |
  ORACLE the map picked ON 2022-26 itself = hindsight, i.e. the 'explanation' |
  PLACEBO the expanding MAP with the regime series circularly shifted by 1-6 years (20 draws): same machinery, no real regime information.
Yardstick: realistic months 2022 - Sep 2026 (fresh 50K eval each calendar month: -$1,000 day floor, $2,000 trailing = lost, +$3,000 with best day <= 50% = passed), 2026, mean month $.
Diagnostics: strategy $ by year; does last year's strategy ranking predict next year's (rank correlation); best strategy per regime cell in 2018-21 / 22-23 / 24-26."""
_s = open('xs/nq_size_tp.py').read(); exec(_s[:_s.index("print('\\nby context")])
from hmmlearn.hmm import GaussianHMM
from scipy.stats import spearmanr
C = C.reset_index(drop=True)
print(f'gamma data {G.index.min():%Y-%m-%d} -> {G.index.max():%Y-%m-%d}; sessions {days[0]:%Y-%m-%d} -> {days[-1]:%Y-%m-%d}', flush=True)
def run_g(Cc, T, S, per_day=2, flat_m=720, cost=1.0):
    out = []
    for day, g in Cc.sort_values('j').groupby('day'):
        busy = -1; n = 0
        for j, d, kk in zip(g.j.values, g.dir.values, g.k.values):
            if n >= per_day or j <= busy or not np.isfinite(kk): continue
            e = o5[j]; res = None
            for q in range(j, len(H5)):
                if d5[q] != d5[j] or m5[q] >= flat_m: res = ((c5[q - 1] - e) * d * kk, q - 1); break
                lo_ = ((l5[q] if d > 0 else h5[q]) - e) * d * kk; hi_ = ((h5[q] if d > 0 else l5[q]) - e) * d * kk
                if lo_ <= -S: res = (-S, q); break
                if hi_ >= T: res = (T, q); break
            if res: out.append((day, res[0] - cost)); busy = res[1]; n += 1
    return pd.DataFrame(out, columns=['day', 'net']).groupby('day').net.sum()
def kcol(X): X = pd.DataFrame(X, columns=['day', 'j', 'dir']); X['k'] = NQs.reindex(pd.DatetimeIndex(X.day) - pd.Timedelta(days=1), method='ffill').values / o5[X.j.values]; return X
def cands(lo, hi):
    r = H15.reg.values; me = H15.me.values; ends = H15.end.values; dts = H15.dt.values; out = []
    for k in range(2, len(H15)):
        if r[k] == r[k - 1] or r[k - 1] == 0 or not (lo < me[k] <= hi): continue
        day = pd.Timestamp(dts[k]).normalize(); j = np.searchsorted(t5, np.datetime64(ends[k]))
        if j < len(H5) and d5[j] == np.datetime64(day): out.append((day, int(j), int(r[k])))
    return kcol(out)
# ---------------------------------------------------------------- RSI, ORB, IMOM, ES candidates
cl = pd.Series(H15.close.values); dlt = cl.diff(); up = dlt.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); dn = (-dlt.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
rs = (100 - 100 / (1 + up / dn)).values; MR = []
for k in range(1, len(H15)):
    if not (570 < H15.me.iloc[k] <= 690): continue
    d_ = 1 if (rs[k - 1] < 30 <= rs[k]) else (-1 if (rs[k - 1] > 70 >= rs[k]) else 0)
    if not d_: continue
    day = H15.dt.iloc[k].normalize(); j = np.searchsorted(t5, np.datetime64(H15.end.iloc[k]))
    if j < len(H5) and d5[j] == np.datetime64(day): MR.append((day, int(j), d_))
MR = kcol(MR)
rth = (m5 >= 570) & (m5 < 960); OR = []
for d, idx in pd.Series(np.arange(len(H5))[rth]).groupby(d5[rth]):
    ii = idx.values; orb = ii[m5[ii] < 600]
    if len(orb) < 6: continue
    hi_, lo_ = h5[orb].max(), l5[orb].min()
    for i in ii[(m5[ii] >= 600) & (m5[ii] <= 685)]:
        if c5[i] > hi_ or c5[i] < lo_:
            if i + 1 < len(H5) and d5[i + 1] == d5[i]: OR.append((pd.Timestamp(d), int(i + 1), 1 if c5[i] > hi_ else -1))
            break
OR = kcol(OR)
dd = pd.DataFrame({'d': d5, 'm': m5, 'o': o5, 'c': c5})[rth]
IM = pd.DataFrame({'c16': dd.groupby('d').c.last(), 'c10': dd[dd.m < 600].groupby('d').c.last(), 'o': dd[dd.m >= 930].groupby('d').o.first()}).dropna(); IM.index = pd.DatetimeIndex(IM.index)
IM['pc'] = IM.c16.shift(1); IM['lvl'] = NQs.reindex(IM.index - pd.Timedelta(days=1), method='ffill').values
IM['pts'] = np.sign(IM.c10 / IM.pc - 1) * (IM.c16 / IM.o - 1) * IM.lvl - 1.0
S15 = load(glob.glob(f'{ROOT}/lse_ohlcv_cache/SPY_15m_*.parquet')[0]).reset_index(drop=True); S15['reg'] = np.asarray(vec(S15[['open', 'high', 'low', 'close']]))
S15['end'] = S15.dt + pd.Timedelta('15min'); S15['me'] = S15.end.dt.hour * 60 + S15.end.dt.minute
ESd = yf.download('ES=F', start='2017-12-01', interval='1d', progress=False, auto_adjust=False)['Close'].squeeze(); ESd.index = ESd.index.normalize(); ESd = ESd.sort_index()
so, sh, sl, sc, sm, sd = S15.open.values, S15.high.values, S15.low.values, S15.close.values, S15.m.values, S15.dt.dt.normalize().values
def es_run(T=30, S=20, per_day=2, cost=.5):
    rg = S15.reg.values; me = S15.me.values; out = []; cur_day = None; n = 0; busy = -1
    for k in range(2, len(S15) - 1):
        if rg[k] == rg[k - 1] or rg[k - 1] == 0 or not (570 < me[k] <= 690): continue
        day = pd.Timestamp(sd[k]); j = k + 1
        if sd[j] != sd[k]: continue
        if day != cur_day: cur_day, n, busy = day, 0, -1
        if n >= per_day or j <= busy: continue
        lvl = ESd.reindex([day - pd.Timedelta(days=1)], method='ffill').values[0]; kk = lvl / so[j]; e = so[j]; d = int(rg[k]); res = None
        for q in range(j, len(S15)):
            if sd[q] != sd[j] or sm[q] >= 720: res = ((sc[q - 1] - e) * d * kk, q - 1); break
            lo_ = ((sl[q] if d > 0 else sh[q]) - e) * d * kk; hi_ = ((sh[q] if d > 0 else sl[q]) - e) * d * kk
            if lo_ <= -S: res = (-S, q); break
            if hi_ >= T: res = (T, q); break
        if res: out.append((day, res[0] - cost)); busy = res[1]; n += 1
    return pd.DataFrame(out, columns=['day', 'net']).groupby('day').net.sum()
# ---------------------------------------------------------------- daily $ per strategy
AMC = C[['day', 'j', 'dir', 'k']].copy(); S = {}
S['FLIP'] = run_g(AMC, 75, 50) * 20; S['LONG'] = run_g(AMC[AMC.dir > 0], 75, 50) * 20; S['SHORT'] = run_g(AMC[AMC.dir < 0], 75, 50) * 20
S['FADE'] = run_g(AMC.assign(dir=-AMC.dir), 75, 50) * 20; S['TIGHT'] = run_g(AMC, 40, 25) * 20; S['WIDE'] = run_g(AMC, 150, 75) * 20
S['PM'] = run_g(cands(810, 915), 75, 50, 2, 955) * 20; S['RSI'] = run_g(MR, 75, 50) * 20; S['ORB'] = run_g(OR, 75, 50, 1) * 20
S['IMOM'] = IM.pts.dropna() * 20; S['ES'] = es_run() * 50
P = pd.DataFrame(S).reindex(days).fillna(0.0); P['CASH'] = 0.0; STR = list(P.columns)
print('strategies built: ' + ', '.join(f'{k} {int((P[k] != 0).sum())} days' for k in STR if k != 'CASH'), flush=True)
# ---------------------------------------------------------------- regimes at the prior close
dq = Dq.copy(); dq['ret'] = np.log(dq['Adj Close']).diff(); dq = dq.dropna(subset=['ret']); tr_ = (dq.index >= '2016-01-01') & (dq.index < '2022-01-01')
hm = GaussianHMM(2, covariance_type='diag', n_iter=200, random_state=0).fit(dq.ret.values[tr_].reshape(-1, 1))
bear = int(np.argmin(hm.means_[:, 0])); A_ = hm.transmat_; mu = hm.means_[:, 0]; sdv = np.sqrt(hm.covars_.reshape(2, -1)[:, 0]); alpha = hm.startprob_.copy(); pb = []
for x in dq.ret.values:
    lik = np.exp(-.5 * ((x - mu) / sdv) ** 2) / sdv; alpha = (alpha @ A_) * lik; alpha /= alpha.sum(); pb.append(alpha[bear])
hmm_bear = pd.Series(pb, index=dq.index)
pr = lambda s: pd.Series(prior(s, days), index=days)
REG = pd.DataFrame(index=days)
REG['HMM'] = np.where(pr(hmm_bear) > .5, 'bear', 'bull'); REG['QZ'] = np.where(pr(XQ.qz) < 0, 'z<0', 'z>=0')
hp_ = pr(HP); REG['HAR'] = np.where(hp_ < 1 / 3, 'low', np.where(hp_ > 2 / 3, 'high', 'mid'))
REG['GAMMA'] = np.where(pr(G) < 0, 'neg', 'pos'); vr = pr(XQ.vrel); REG['VXN'] = np.where(vr > .10, 'rising', np.where(vr < -.10, 'falling', 'flat'))
REG['TREND'] = np.where(pr(XQ.qr20) > 0, 'up', 'down'); REG['HMMxQZ'] = REG.HMM + '/' + REG.QZ; REG['HARxGAMMA'] = REG.HAR + '/' + REG.GAMMA
FAM = ['HMM', 'QZ', 'HAR', 'GAMMA', 'VXN', 'TREND', 'HMMxQZ', 'HARxGAMMA']
print('regime mix 2022-26: ' + ' | '.join(f'{f} ' + str(REG[REG.index >= P0][f].value_counts(normalize=True).round(2).to_dict()) for f in FAM[:6]), flush=True)
# ---------------------------------------------------------------- yardstick
def mline(nm, dv, extra=''):
    M_ = months(dv); M6 = M_[M_.month >= '2026-01']; v = dv.reindex(days).fillna(0.0); v = v[v.index >= P0]
    print(f'  {nm:<42} passed {M_.res.eq("passed").mean():4.0%} lost {M_.res.eq("lost").mean():4.0%} neither {M_.res.eq("neither").mean():4.0%} | month $ mean {M_.pnl.mean():+6,.0f} worst {M_.pnl.min():+7,.0f} | $/day {v.mean():+5.0f} (22-23 {v[v.index < "2024"].mean():+4.0f} / 24-26 {v[v.index >= "2024"].mean():+4.0f}) | 2026 passed {M6.res.eq("passed").sum()}/{len(M6)} lost {M6.res.eq("lost").sum()}{extra}', flush=True)
    return M_
per = np.where(days < P0, '18-21', np.where(days < pd.Timestamp('2024-01-01'), '22-23', '24-26'))
print('\nSTRATEGY LIBRARY alone ($/day by period; realistic months 2022-26)', flush=True)
for k in STR[:-1]:
    mline(f'{k} (18-21 {P[k][per == "18-21"].mean():+4.0f}/day)', P[k])
Y = P.drop(columns='CASH').groupby(days.year).sum()
print('\n$ by year (1 contract)\n' + Y.round(0).astype(int).to_string(), flush=True)
print('\nwinner each year: ' + ', '.join(f'{y} {Y.loc[y].idxmax()}' for y in Y.index), flush=True)
rc = [spearmanr(Y.loc[y - 1], Y.loc[y])[0] for y in Y.index[1:]]
print('rank correlation, last year -> this year: ' + ', '.join(f'{y} {c:+.2f}' for y, c in zip(Y.index[1:], rc)) + f' | mean {np.mean(rc):+.2f}', flush=True)
Mo = P.drop(columns='CASH').groupby(days.to_period('M')).sum(); ic = []
for i in range(12, len(Mo)): ic.append(spearmanr(Mo.iloc[i - 12:i].sum(), Mo.iloc[i])[0])
print(f'rank correlation, trailing 12 months -> next month: mean {np.nanmean(ic):+.3f} (2022-26 {np.nanmean([c for c, m in zip(ic, Mo.index[12:]) if m >= pd.Period("2022-01", "M")]):+.3f})', flush=True)
# ---------------------------------------------------------------- per-cell winners by period (explanation side)
print('\nBEST STRATEGY PER REGIME CELL by period (mean $/day; does the winner repeat?)', flush=True)
for f in FAM[:6]:
    for cell in sorted(REG[f].unique()):
        row = []
        for p_ in ('18-21', '22-23', '24-26'):
            m = (REG[f].values == cell) & (per == p_); mm = P[m].drop(columns='CASH').mean().sort_values(ascending=False)
            row.append(f'{p_}: {mm.index[0]} {mm.iloc[0]:+.0f}, {mm.index[1]} {mm.iloc[1]:+.0f} (FLIP {mm["FLIP"]:+.0f}, n{m.sum()})')
        print(f'  {f:<6} {cell:<8} ' + ' | '.join(row), flush=True)
# ---------------------------------------------------------------- walk-forward meta
MON = pd.period_range(P0, days[-1], freq='M'); dm = days.to_period('M'); Pv = P.values; si = {k: i for i, k in enumerate(STR)}
def walk(lab, mode, window=None, min_n=40):
    out = np.zeros(len(days)); ch = []
    for mo in MON:
        cur = dm == mo; st = mo.start_time; hist = days < st
        if window: hist = hist & (days >= st - pd.DateOffset(months=window))
        for cell in np.unique(lab[cur]):
            mc = cur & (lab == cell); mh = hist & (lab == cell)
            if mh.sum() < min_n: s = 'FLIP'
            elif mode == 'gate': s = 'FLIP' if Pv[mh, si['FLIP']].mean() > 0 else 'CASH'
            else: s = STR[int(np.argmax(Pv[mh].mean(0)))]
            out[mc] = Pv[mc, si[s]]; ch.append((str(mo), cell, s, int(mc.sum())))
    return pd.Series(out, index=days), pd.DataFrame(ch, columns=['mo', 'cell', 's', 'n'])
def oracle(lab):
    out = np.zeros(len(days)); post = days >= P0; pick = {}
    for cell in np.unique(lab[post]):
        m = post & (lab == cell); s = STR[int(np.argmax(Pv[m].mean(0)))]; out[m] = Pv[m, si[s]]; pick[cell] = s
    return pd.Series(out, index=days), pick
def mix(ch): x = ch.groupby('s').n.sum(); return ' | picks ' + ', '.join(f'{k} {v / x.sum():.0%}' for k, v in x.sort_values(ascending=False).head(4).items())
print('\nMETA, realistic months 2022 - Sep 2026', flush=True)
mline('STATIC FLIP', P.FLIP)
bis = P[days < P0].drop(columns='CASH').mean().idxmax(); mline(f'STATIC best of 2018-21 ({bis})', P[bis])
one = np.array(['all'] * len(days))
for w, nm in ((12, 'ROLL best trailing 12m'), (24, 'ROLL best trailing 24m'), (None, 'ROLL best expanding')):
    dv, ch = walk(one, 'map', w, 20); mline(nm, dv, mix(ch))
rng = np.random.default_rng(7); SUM = []
for f in FAM:
    lab = REG[f].values; print(f'\n  == {f}', flush=True)
    for mode, w in (('gate', None), ('gate', 24), ('map', None), ('map', 24)):
        dv, ch = walk(lab, mode, w); M_ = mline(f'{mode.upper()} {"expanding" if w is None else "24m"}', dv, mix(ch) if mode == 'map' else '')
        SUM.append((f, mode, w, M_.res.eq('passed').mean(), M_.res.eq('lost').mean(), M_.pnl.mean()))
    dv, pick = oracle(lab); mline('ORACLE (hindsight map, 2022-26)', dv, ' | ' + ', '.join(f'{c}->{s}' for c, s in pick.items()))
    pp, pl, pm = [], [], []
    for _ in range(20):
        off = int(rng.integers(252, 6 * 252)); dv, _c = walk(np.roll(lab, off), 'map', None); M_ = months(dv); pp.append(M_.res.eq('passed').mean()); pl.append(M_.res.eq('lost').mean()); pm.append(M_.pnl.mean())
    print(f'  {"PLACEBO map (shifted regimes, 20 draws)":<42} passed {np.median(pp):4.0%} lost {np.median(pl):4.0%} | month $ mean {np.median(pm):+6,.0f} (range {np.min(pm):+,.0f} to {np.max(pm):+,.0f})', flush=True)
S_ = pd.DataFrame(SUM, columns=['fam', 'mode', 'win', 'passed', 'lost', 'mean'])
print('\nSUMMARY walk-forward (2022-26): ' + S_.groupby('mode')[['passed', 'lost', 'mean']].mean().round(3).to_string().replace('\n', ' | '), flush=True)
P.assign(**{f'REG_{f}': REG[f] for f in FAM}).to_parquet('/tmp/claude-0/ml/nq_regime_switch_P.parquet')
