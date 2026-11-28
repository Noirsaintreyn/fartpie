"""Report 523 (Nov 27 2026): stacked HEALTH rule on live-style flips. Rey's attachment #3 (Consensus): gate stacked rule with walk-forward + multiple-testing count.
DECLARED BEFORE RESULTS. Universe: 360 HEALTH sub-$5 flips (flip_history via flips517.pkl). Stack = RND_hi (train-tertile R&D/assets) & small cap (below train median MCAP) & high B/M (above train median) & no DEAL/PR/EARN 8-K in prior 7d.
Thresholds fixed from 2019-21 (train) ONLY; test = 2018 + 2022+. Variants tried (counted for multiple testing): V0 all health, V1 RND_hi, V2 RND_hi&noNews, V3 V2&small, V4 V2&hiBM, V5 V3&hiBM. N=6 variants.
Report: n, reach(hit), mean r, median r, trades/yr, by era; permutation null: random subsets of same size from the 360 health flips (5000 draws) -> p for mean r and reach; Bonferroni x6."""
import numpy as np, pandas as pd, json, io, contextlib, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_516_sector.py').read().split("TRm = (P.yr")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src)
F = pd.read_pickle(X + 'flips517.pkl'); H = F[F.sector == 'HEALTH'].copy(); H['d'] = pd.to_datetime(H.d).astype('datetime64[ns]')
sh = pd.read_pickle(X + 'shares_pit.pkl'); sh['filed'] = pd.to_datetime(sh.filed).astype('datetime64[ns]'); sh = sh.sort_values('filed')
H = pd.merge_asof(H.sort_values('d'), sh[['tk', 'filed', 'val']].rename(columns={'filed': 'fs', 'val': 'SH'}), left_on='d', right_on='fs', by='tk', direction='backward', tolerance=pd.Timedelta(days=730)); H['MCAP'] = H.SH * H.px
eq = []
for tk in H.tk.unique():
    try: gg = json.load(open(f'data/fund/sec_small/{tk}.json'))['facts']['us-gaap']
    except Exception: continue
    a = ann(gg, ['StockholdersEquity'], 1)
    if a is None: continue
    e = pd.DataFrame({'EQ': a.val, 'filed': a.filed}); e['tk'] = tk; eq.append(e.dropna(subset=['filed']))
EQ = pd.concat(eq).sort_values('filed'); EQ['filed'] = EQ.filed.astype('datetime64[ns]')
H = pd.merge_asof(H.sort_values('d'), EQ.rename(columns={'filed': 'fe'}), left_on='d', right_on='fe', by='tk', direction='backward', tolerance=pd.Timedelta(days=540))
H['BM'] = H.EQ / H.MCAP; H.loc[H.BM <= 0, 'BM'] = np.nan
H['yr'] = H.d.dt.year; H['era4'] = np.select([H.yr == 2018, H.yr <= 2021, H.yr == 2022], ['2018', '2019-21', '2022'], '2023-26'); tr = H.yr.between(2019, 2021)
q = H.loc[tr, 'RND'].quantile(.67); mc = H.loc[tr, 'MCAP'].median(); bm = H.loc[tr, 'BM'].median()
print('train thresholds RND>', round(q, 4), 'MCAP<', round(mc / 1e6), 'M BM>', round(bm, 2), '| coverage MCAP', round(H.MCAP.notna().mean(), 2), 'BM', round(H.BM.notna().mean(), 2), 'RND', round(H.RND.notna().mean(), 2))
nonews = (H.DEAL7 == 0) & (H.PR7 == 0) & (H.EARN7 == 0)
V = {'V0 all health': H.r.notna(), 'V1 RND_hi': H.RND > q, 'V2 RND_hi&noNews': (H.RND > q) & nonews, 'V3 V2&small': (H.RND > q) & nonews & (H.MCAP < mc), 'V4 V2&hiBM': (H.RND > q) & nonews & (H.BM > bm), 'V5 V3&hiBM': (H.RND > q) & nonews & (H.MCAP < mc) & (H.BM > bm)}
yrs = (H.d.max() - H.d.min()).days / 365.25; rng = np.random.default_rng(0); base_r = H.r.dropna().values; base_h = H.hit.astype(float).values
rows = []
for k, m in V.items():
    s = H[m.fillna(False)]; n = len(s)
    if n == 0: continue
    r = dict(variant=k, n=n, per_yr=round(n / yrs, 1), reach=round(s.hit.mean() * 100), mean=round(s.r.mean() * 100, 1), med=round(s.r.median() * 100, 1))
    for e in ('2018', '2019-21', '2022', '2023-26'):
        t = s[s.era4 == e]; r[e] = f"{len(t)}:{round(t.hit.mean()*100) if len(t) else '-'}%/{round(t.r.mean()*100,1) if len(t) else '-'}"
    te = s[~s.era4.eq('2019-21')]; r['test_n'] = len(te); r['test_reach'] = round(te.hit.mean() * 100) if len(te) else None; r['test_mean'] = round(te.r.mean() * 100, 1) if len(te) else None
    if k != 'V0 all health' and n >= 5:
        d = rng.choice(len(H), (5000, n)); rm = np.nanmean(H.r.values[d], 1); hm = H.hit.astype(float).values[d].mean(1)
        r['p_mean'] = round((rm >= s.r.mean()).mean(), 4); r['p_reach'] = round((hm >= s.hit.mean()).mean(), 4); r['p_mean_bonf6'] = round(min(1, r['p_mean'] * 6), 3)
    rows.append(r)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); print(pd.DataFrame(rows).to_string(index=False))
print('legend: era cells = n:reach%/mean r% ; test = 2018+2022+ (2023-26 incl.)')
