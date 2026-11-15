"""Nov 14 2026 PARTICIPANT SIGNAL NOT IN OHLCV: FINRA DAILY SHORT-SALE VOLUME AT THE FLIP (declared before results). Data: Reg SHO consolidated daily short volume / total volume (SVR) per name-day, 2018-12 to 2026-11, wide universe. Mechanism: SVR is the share of the day's volume executed as short sales (mostly market makers and shorts selling into strength). High SVR during a flip = supply is being met by sellers who fade it (distribution / hedging); low SVR = buying is not being absorbed by short sellers (real demand, or shorts already covered). The live screener already half-sizes names with a 20d SVR above 53% (SVR_CUT) using only 2023-11+ data, so this is the first strict-split test. To remove the long-run drift in SVR (shorts' share rose over the years), use the name's CROSS-SECTIONAL PERCENTILE RANK among all names that day: R20 = rank of the 20-session mean SVR ending at the flip bar, R0 = rank of the flip-bar SVR, RD = rank of (5-session mean SVR minus 20-session mean SVR) = recent change. Terciles of each rank (fixed 33/67 percentiles), per era (2019-21 / 2022-26Q1), all flips and screener rule (BV>=1.5): n, mean net, median, HUGE per 100, lift, ex-top-3 mean. Pass = same tercile has HUGE lift >=1.3 and mean above base in both eras, or the opposite tercile has lift <=0.75 and mean below base in both (a CUT rule). Then the existing live rule: R20 raw level <= 53% vs above."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; S = pd.read_pickle(X + 'finra_sv_wide.pkl'); P = pd.read_pickle(X + 'wide_flips_feat.pkl'); P['rn'] = P.ret - 0.003; P['d0'] = pd.to_datetime(P.d0)
S = S[S.TotalVolume > 0].copy(); S['svr'] = S.ShortVolume / S.TotalVolume; W = S.pivot_table(index='Date', columns='Symbol', values='svr'); TV = S.pivot_table(index='Date', columns='Symbol', values='TotalVolume'); SV = S.pivot_table(index='Date', columns='Symbol', values='ShortVolume')
m20 = SV.rolling(20, min_periods=15).sum() / TV.rolling(20, min_periods=15).sum(); m5 = SV.rolling(5, min_periods=4).sum() / TV.rolling(5, min_periods=4).sum()
rk = lambda df: df.rank(axis=1, pct=True)
R0, R20, RD = rk(W), rk(m20), rk(m5 - m20)
def look(df, name):
    return np.array([df.at[d, tk] if (d in df.index and tk in df.columns) else np.nan for d, tk in zip(P.d0, P.tk)])
P['R0'] = look(R0, 'R0'); P['R20'] = look(R20, 'R20'); P['RD'] = look(RD, 'RD'); P['SVR20'] = look(m20, 'svr20'); print('flips with SVR data:', P.R20.notna().sum(), 'of', len(P), '| first flip date with data', P[P.R20.notna()].t.min().date())
def tert(z, f):
    q = z[f]; return pd.cut(q, [-.01, 1 / 3, 2 / 3, 1.01], labels=['LOW third', 'MID', 'HIGH third'])
res = {}
for lab, S0 in (('ALL FLIPS', P[P.R20.notna()]), ('SCREENER RULE (BV>=1.5)', P[P.R20.notna() & (P.BV >= 1.5)])):
    print(f'\n######## {lab}')
    for f, nm in (('R20', '20-session SVR rank'), ('R0', 'flip-bar SVR rank'), ('RD', 'SVR change rank (5d minus 20d)')):
        for en, z in (('2019-21', S0[S0.t < '2022']), ('2022-26Q1', S0[S0.t >= '2022'])):
            z = z[z[f].notna()].copy(); z['q'] = tert(z, f); base = z.HUGE.mean(); line = []
            for k, v in z.groupby('q'):
                res[(lab, f, en, k)] = (len(v), v.rn.mean() * 100, v.HUGE.mean() / base, z.rn.mean() * 100); line.append(f'{k[:4]} n {len(v)} mean {v.rn.mean()*100:+6.2f}% med {v.rn.median()*100:+5.2f} HUGE/100 {v.HUGE.mean()*100:4.2f} lift {v.HUGE.mean()/base:4.2f} ex3 {np.sort(v.rn.values)[:-3].mean()*100:+6.2f}')
            print(f'  {nm:30s} {en:9s} base {z.rn.mean()*100:+.2f}% HUGE/100 {base*100:.2f} | ' + ' | '.join(line))
    print('  VERDICT: ' + ('; '.join(f'{f} {k}: {"BOOST" if kind == "b" else "CUT"}' for f in ('R20', 'R0', 'RD') for k in ('LOW third', 'HIGH third') for kind in ('b', 'c') if all(((res[(lab, f, e, k)][2] >= 1.3 and res[(lab, f, e, k)][1] > res[(lab, f, e, k)][3]) if kind == 'b' else (res[(lab, f, e, k)][2] <= .75 and res[(lab, f, e, k)][1] < res[(lab, f, e, k)][3])) for e in ('2019-21', '2022-26Q1'))) or 'none passes'))
print('\nLIVE RULE CHECK: 20-session SVR level <= 53% (screener flag) vs above, screener-rule flips'); 
for en, z in (('2019-21', P[(P.t < '2022') & P.SVR20.notna() & (P.BV >= 1.5)]), ('2022-26Q1', P[(P.t >= '2022') & P.SVR20.notna() & (P.BV >= 1.5)])):
    a = z[z.SVR20 <= .53]; b = z[z.SVR20 > .53]; print(f'  {en}: <=53% n {len(a)} mean {a.rn.mean()*100:+.2f}% HUGE/100 {a.HUGE.mean()*100:.2f} | >53% n {len(b)} mean {b.rn.mean()*100:+.2f}% HUGE/100 {b.HUGE.mean()*100:.2f}')
