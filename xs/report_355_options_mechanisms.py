"""Nov 22 2026 report 355. Rey: 'it's not just the options ratio that holds an edge - what is the REASON it works, and can we improve it?'  Declared before running.
WHAT WE ALREADY KNOW (reports 294, 338): within the ratio>=1.5 universe the ratio is a threshold/liquidity effect (corr 0.62 with log dollar volume; zero edge after controls); the flip beats random timing in the same names (p=0.015) but over half of the mean trade is the names' own drift.
LITERATURE-LED MECHANISMS (prior-close, lagged one day, own-history z-scores over the trailing 126 sessions so a name's level cannot stand in for the signal; terciles cut on 2019-21; flips of the 23 research names with option chains):
 H1 SKEW (Xing-Zhang-Zhao 2010): steeper put skew (IV of the ~0.90-moneyness put minus ATM call IV, ~30 DTE) -> WORSE flips.
 H2 CALL-PUT IV SPREAD (Cremers-Weinbaum 2010): ATM call IV minus ATM put IV (same strike, ~30 DTE) higher (calls bid) -> BETTER.
 H3 ABNORMAL O/S (Johnson-So 2012: high option/stock volume = informed BEARISH flow): 5d option-to-stock volume vs its own 252d median higher -> WORSE.
 H4 CALL SHARE (Pan-Poteshman 2006): calls / all option contracts, 5d, higher -> BETTER.
 H5 DEALER GAMMA (Pearson-Poteshman-White; Baltussen et al.): the name's own net GEX lower (dealers short gamma, amplifying) -> BETTER mean and LARGER |move|.
Pass rule (strict): same sign in both halves in the declared direction and |t| >= 2 (cluster-robust by week) in at least one half; 5 hypotheses so |t| >= 2.6 is the stricter bar to call it solid.
Outputs: top-minus-bottom tercile mean trade return, win rate, >=+5% rate, train | test.  Then a decomposition of what the ratio proxies for (corr with log dollar volume, beta, vol, momentum)."""
import glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
B = pd.read_parquet(SP + 'b302.parquet'); B['entry'] = pd.to_datetime(B.entry); B['day'] = pd.to_datetime(B.day)
names = sorted(os.path.basename(f)[:-8] for f in glob.glob(U + 'data/theta_compact/*.parquet')); names = [n for n in names if n in set(B.tk)]
def bs_iv(price, S, K, T, cp):
    lo = np.full(len(price), 1e-3); hi = np.full(len(price), 4.0)
    for _ in range(40):
        m = (lo + hi) / 2; sd = m * np.sqrt(T); d1 = (np.log(S / K) + .5 * sd * sd) / sd; d2 = d1 - sd
        from scipy.special import ndtr
        pr = np.where(cp == 1, S * ndtr(d1) - K * ndtr(d2), K * ndtr(-d2) - S * ndtr(-d1)); hi = np.where(pr > price, m, hi); lo = np.where(pr > price, lo, m)
    return (lo + hi) / 2
rows = []
for tk in names:
    t = pd.read_parquet(U + f'data/theta_compact/{tk}.parquet'); t = t[(t.dte >= 20) & (t.dte <= 45) & (t.bid > 0) & (t.ask > t.bid)].copy(); t['mid'] = (t.bid + t.ask) / 2; t = t[(t.ask - t.bid) / t.mid < .5]
    t['gap'] = (t.dte - 30).abs(); ex = t.loc[t.groupby('d').gap.idxmin(), ['d', 'e']]; t = t.merge(ex, on=['d', 'e'])
    t['mn'] = t.K / t.S; c = t[t.r == 1]; p = t[t.r == -1]
    m = c.merge(p, on=['d', 'K'], suffixes=('_c', '_p')); m['dist'] = (m.mn_c - 1).abs(); a = m.loc[m.groupby('d').dist.idxmin()]
    p9 = p.assign(dist=(p.mn - .9).abs()); p9 = p9.loc[p9.groupby('d').dist.idxmin()]; p9 = p9[p9.dist < .06]
    T_ = a.dte_c.values / 365.; ivc = bs_iv(a.mid_c.values, a.S_c.values, a.K.values, T_, np.ones(len(a))); ivp = bs_iv(a.mid_p.values, a.S_c.values, a.K.values, T_, -np.ones(len(a)))
    q = pd.DataFrame({'d': a.d.values, 'ivc': ivc, 'ivp': ivp}).set_index('d'); iv9 = bs_iv(p9.mid.values, p9.S.values, p9.K.values, p9.dte.values / 365., -np.ones(len(p9))); q['iv9'] = pd.Series(iv9, index=p9.d.values)
    q['skew'] = q.iv9 - q.ivc; q['cps'] = q.ivc - q.ivp; q['tk'] = tk; rows.append(q.reset_index().rename(columns={'d': 'date'}))
CH = pd.concat(rows); CH['date'] = pd.to_datetime(CH.date)
O = pd.read_csv(U + 'xs/ext/optvol_ratio.csv', parse_dates=['date']); O = O[O.tk.isin(names)].sort_values(['tk', 'date']); O['cs'] = O.call_c / O.opt_contracts
G = O.groupby('tk'); O['os5'] = G.opt_ratio.transform(lambda s: s.rolling(5).mean()); O['os252'] = G.opt_ratio.transform(lambda s: s.rolling(252, min_periods=120).median()); O['abn'] = O.os5 / O.os252; O['cs5'] = G.cs.transform(lambda s: s.rolling(5).mean())
GX = pd.concat(pd.read_csv(f).assign(tk=os.path.basename(f)[:-4]) for f in glob.glob(U + 'xs/ext/gex/*.csv')); GX['date'] = pd.to_datetime(GX.date); GX = GX[GX.tk.isin(names)]
F = CH.merge(O[['tk', 'date', 'abn', 'cs5']], on=['tk', 'date'], how='outer').merge(GX[['tk', 'date', 'net_gex']], on=['tk', 'date'], how='outer').sort_values(['tk', 'date'])
def z(s): m = s.rolling(126, min_periods=60).mean(); sd = s.rolling(126, min_periods=60).std(); return (s - m) / sd
for c in ('skew', 'cps', 'abn', 'cs5', 'net_gex'): F[c + '_z'] = F.groupby('tk')[c].transform(z)
F = F[['tk', 'date', 'skew_z', 'cps_z', 'abn_z', 'cs5_z', 'net_gex_z']].dropna(how='all', subset=['skew_z', 'cps_z', 'abn_z', 'cs5_z', 'net_gex_z'])
X = B[B.tk.isin(names)].sort_values('entry'); X['dte_'] = X.entry.dt.normalize(); F = F.sort_values('date')
X = pd.merge_asof(X, F, left_on='dte_', right_on='date', by='tk', allow_exact_matches=False, tolerance=pd.Timedelta(days=5)); X['tr'] = (X.entry >= '2019-01-01') & (X.entry < '2022-01-01'); X = X[X.entry >= '2019-01-01']
print(f'names with chains {len(names)} | flips {len(X)} (train {X.tr.sum()} | test {(~X.tr).sum()}) | feature coverage: ' + ', '.join(f'{c} {X[c].notna().mean():.0%}' for c in ('skew_z', 'cps_z', 'abn_z', 'cs5_z', 'net_gex_z')))
print(f'baseline mean trade {X.ret[X.tr].mean():+.3%} | {X.ret[~X.tr].mean():+.3%}; win {np.mean(X.ret[X.tr] > 0):.0%} | {np.mean(X.ret[~X.tr] > 0):.0%}; >=+5% {np.mean(X.ret[X.tr] >= .05):.1%} | {np.mean(X.ret[~X.tr] >= .05):.1%}')
def tert(col, sign):
    d = X.dropna(subset=[col]); lo, hi = d[col][d.tr].quantile([1 / 3, 2 / 3]); out = []
    for lab, m in (('train', d.tr), ('test', ~d.tr)):
        e = d[m]; top = e[e[col] >= hi]; bot = e[e[col] <= lo]; y = np.r_[top.ret.values, bot.ret.values]; g = np.r_[np.ones(len(top)), np.zeros(len(bot))]; cl = np.r_[top.wk.values, bot.wk.values]
        r = sm.OLS(y, sm.add_constant(g)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(cl)[0]}); diff = r.params[1]
        out.append((diff, r.tvalues[1], top.ret.mean(), bot.ret.mean(), (top.ret >= .05).mean(), (bot.ret >= .05).mean(), len(top), len(bot), top.ret.abs().mean(), bot.ret.abs().mean()))
    ok = np.sign(out[0][0]) == np.sign(out[1][0]) == sign and max(abs(out[0][1]), abs(out[1][1])) >= 2
    print(f'  {col:10s} top-bot {out[0][0]:+.3%} (t{out[0][1]:+.1f}) | {out[1][0]:+.3%} (t{out[1][1]:+.1f})   top {out[0][2]:+.2%}|{out[1][2]:+.2%} bot {out[0][3]:+.2%}|{out[1][3]:+.2%}  >=5%: top {out[0][4]:.1%}|{out[1][4]:.1%} bot {out[0][5]:.1%}|{out[1][5]:.1%}  mean|ret| top {out[0][8]:.2%}|{out[1][8]:.2%} bot {out[0][9]:.2%}|{out[1][9]:.2%}  n {out[0][6]}/{out[0][7]} | {out[1][6]}/{out[1][7]}  declared sign {sign:+d}: {"PASS" if ok else "no"}')
print('\nTERCILES of the own-history z-score (top minus bottom):')
for col, sg in (('skew_z', -1), ('cps_z', 1), ('abn_z', -1), ('cs5_z', 1), ('net_gex_z', -1)): tert(col, sg)
print('\nWHAT THE RATIO PROXIES FOR (flip-level, 2019-26, all 54 names): Spearman corr of log ratio with')
for c in ('ldv', 'beta', 'vol20', 'mom', 'mom20', 'hi52', 'rvol'): print(f'  {c:6s} {B.ratio.apply(np.log).corr(B[c], method="spearman"):+.2f}', end='')
print()
