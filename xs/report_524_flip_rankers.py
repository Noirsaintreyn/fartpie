"""Report 524 (Nov 27 2026): Rey's attachment #4 (Consensus): rank flips by net issuance, illiquidity, short-interest surprise (not sector rules).
DECLARED BEFORE RESULTS. Universe: all 1,504 sub-$5 flips (flip_history, 2019-01..2026-02), all sectors. Outcome: reach (hit) and r (flip-exit return) -- flip's own exit, no time hold.
Rankers + declared direction (Fama-French 2007, Liu 2026, Hanauer 2020, Boehmer 2021):
 ISS  net share issuance: shares_pit yoy change (higher -> WORSE)      ILLIQ Amihud 20d |ret|/$vol (higher -> BETTER)   TURN 20d vol/shares (higher -> WORSE per Liu z(TURN) sign)
 SIS  short-interest surprise: DTC at d minus mean of prior 6 reports (higher -> WORSE)     DTC days-to-cover level (higher -> WORSE)
 COMP 0.5z(ILLIQ) -0.5z(TURN) -z(ISS) -z(SIS)  (z by train 2019-21 stats)
Trials=6 (Bonferroni x6). Test: Spearman IC(feature, r) by era and pooled; train 2019-21, test 2022+; top-vs-bottom tercile (train cut points) reach/mean/median; permutation p (5000 shuffles of feature) for pooled-test IC.
PASS: sign as declared in train AND test, test permutation p*6<0.05.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; X = SP + 'ext/'
F = pd.read_pickle(X + 'flips517.pkl'); F['d'] = pd.to_datetime(F.d).astype('datetime64[ns]'); F = F.reset_index(drop=True)
D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl')
sh = pd.read_pickle(X + 'shares_pit.pkl'); sh['filed'] = pd.to_datetime(sh.filed).astype('datetime64[ns]'); sh = sh.sort_values('filed')
SI = pd.read_pickle(X + 'shortint.pkl'); SI['use'] = pd.to_datetime(SI.settlementDate).astype('datetime64[ns]') + pd.Timedelta(days=9); SI['dtc'] = SI.currentShortPositionQuantity / SI.averageDailyVolumeQuantity.replace(0, np.nan)
SIG = {t: g.sort_values('use') for t, g in SI.groupby('symbolCode')}; SHG = {t: g for t, g in sh.groupby('tk')}
rows = []
for i, f in F.iterrows():
    tk, d = f.tk, f.d; o = dict(i=i)
    q = D.get(tk)
    if q is not None:
        q = q.dropna(subset=['close']); q.index = pd.DatetimeIndex(q.index).astype('datetime64[ns]'); q = q[q.index <= d].tail(21)
        if len(q) >= 15:
            ret = q.close.pct_change().abs(); dv = (q.close * q.volume).replace(0, np.nan); o['ILLIQ'] = (ret / dv).mean()
    g = SHG.get(tk)
    if g is not None:
        a = g[g.filed <= d]; b = g[g.filed <= d - pd.Timedelta(days=365)]
        if len(a) and len(b) and (d - a.filed.iloc[-1]).days < 600 and b.val.iloc[-1] > 0: o['ISS'] = a.val.iloc[-1] / b.val.iloc[-1] - 1
        if len(a) and q is not None and len(q): o['TURN'] = q.volume.mean() / a.val.iloc[-1]
    s = SIG.get(tk)
    if s is not None:
        s = s[s.use <= d]
        if len(s) >= 4 and (d - s.use.iloc[-1]).days <= 30:
            o['DTC'] = s.dtc.iloc[-1]; prev = s.dtc.iloc[-7:-1]; o['SIS'] = s.dtc.iloc[-1] - prev.mean() if prev.notna().sum() >= 3 else np.nan
    rows.append(o)
R = F.join(pd.DataFrame(rows).set_index('i')); R['yr'] = R.d.dt.year; R['era4'] = np.select([R.yr <= 2021, R.yr == 2022], ['2019-21', '2022'], '2023-26'); tr = R.yr.between(2019, 2021)
for c in ('ILLIQ', 'ISS', 'TURN', 'SIS', 'DTC'): R[c] = R[c].replace([np.inf, -np.inf], np.nan); lo, hi = R[c].quantile([.01, .99]); R[c] = R[c].clip(lo, hi)
Z = lambda c: (R[c] - R.loc[tr, c].mean()) / R.loc[tr, c].std()
R['LILL'] = np.log(R.ILLIQ.clip(lower=1e-12)); R['LTURN'] = np.log(R.TURN.clip(lower=1e-9))
zs = {'ILLIQ': (R.LILL - R.loc[tr, 'LILL'].mean()) / R.loc[tr, 'LILL'].std(), 'TURN': (R.LTURN - R.loc[tr, 'LTURN'].mean()) / R.loc[tr, 'LTURN'].std(), 'ISS': Z('ISS'), 'SIS': Z('SIS')}
R['COMP'] = (.5 * zs['ILLIQ'] - .5 * zs['TURN'] - zs['ISS'] - zs['SIS']); R['ILLIQ'] = R.LILL; R['TURN'] = R.LTURN
print('coverage', {c: round(R[c].notna().mean(), 2) for c in ('ILLIQ', 'ISS', 'TURN', 'SIS', 'DTC', 'COMP')})
DIR = dict(ISS=-1, ILLIQ=1, TURN=-1, SIS=-1, DTC=-1, COMP=1); rng = np.random.default_rng(1); out = []
for c, sgn in DIR.items():
    r = dict(feat=c, declared=('+' if sgn > 0 else '-'))
    for e, m in (('train19-21', tr), ('2022', R.era4 == '2022'), ('2023-26', R.era4 == '2023-26'), ('TEST22+', ~tr)):
        x = R[m & R[c].notna() & R.r.notna()]; r['IC_' + e] = round(spearmanr(x[c], x.r)[0], 3) if len(x) > 20 else np.nan; r['n_' + e] = len(x)
    x = R[~tr & R[c].notna() & R.r.notna()]; ic = spearmanr(x[c], x.r)[0]
    nul = np.array([spearmanr(x[c].values, rng.permutation(x.r.values))[0] for _ in range(2000)]); p = (nul * sgn >= ic * sgn).mean(); r['p_test'] = round(p, 4); r['p_x6'] = round(min(1, p * 6), 3)
    lo, hi = R.loc[tr, c].quantile([.33, .67]); T = R[~tr]; top, bot = T[(T[c] > hi)], T[(T[c] <= lo)]
    if sgn < 0: top, bot = bot, top
    for nm, s in (('fav', top), ('unfav', bot)): r[nm] = f"n{len(s)} reach{round(s.hit.mean()*100)}% mean{round(s.r.mean()*100,1)} med{round(s.r.median()*100,1)}"
    out.append(r)
pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 60); print(pd.DataFrame(out).to_string(index=False))
print('base test: reach', round(R[~tr].hit.mean() * 100), 'mean', round(R[~tr].r.mean() * 100, 1), 'med', round(R[~tr].r.median() * 100, 1), 'n', (~tr).sum())
