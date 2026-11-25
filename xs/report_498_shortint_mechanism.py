"""Report 498 (Nov 25 2026): follow-up on 497's one survivor, SI_PCT (FINRA short interest >= 15% of XBRL shares outstanding before the flip): adj reach +7 to +10 pts at px<5/<10/<20 (t 2.1-2.6), same sign train & test.
DECLARED BEFORE RESULTS. Is it real or a proxy? Mechanism = squeeze fuel: shorts must cover into the flip, so reach should rise with the short % (dose-response monotone over bins <5%, 5-15%, 15-25%, 25%+ of shares), be present in both eras (2022 | 2023-26) and not hang on a few tickers (leave-top-ticker-out; share of events in top-5 tickers), and survive a harder control (ATRP and log-mcap as continuous terms + LTURN turnover, since a heavily shorted name is also thin/volatile). Also: does it add inside the 30% LIVE group (the alert universe)? Test = does the effect on reach stay >= +5 pts at px<10 after all controls with |t|>=2. Also report SI_DTC alone and SI_PCT x DTC (shares short vs liquidity) to see which carries it. If dose-response is not monotone or one era carries it -> proxy/noise, no ship. Pass -> candidate factor (still needs the live forward test) with SI% shown on the card, not a grade point until it has forward evidence."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
src = open('xs/report_497_titles_dilution_shortint.py').read(); exec(src[:src.index("flags = ['BUY_TOP'")])
O['sip'] = sip; O['dtc'] = dtc
def reg(D, extra=''):
    return smf.ols(f'hit ~ F + C(a3) + C(c3){extra}', D.assign(F=D.F.astype(float))).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.mo_)[0]})
for CAP in (10, 20):
    S = O[(O.px < CAP) & O.ATRP.notna() & O.LMC.notna() & O.sip.notna()].copy(); tr = S.tr
    S['a3'] = pd.cut(S.ATRP, [-np.inf, *np.quantile(S.ATRP[tr], [1/3, 2/3]), np.inf], labels=False); S['c3'] = pd.cut(S.LMC, [-np.inf, *np.quantile(S.LMC[tr], [1/3, 2/3]), np.inf], labels=False)
    S['bin'] = pd.cut(S.sip, [-1, .05, .15, .25, 99], labels=['<5%', '5-15%', '15-25%', '25%+'])
    print(f'\n######## px<{CAP} (with SI match): n {len(S)}')
    print('dose-response (raw | adj vs <5%):')
    m = smf.ols('hit ~ C(bin) + C(a3) + C(c3)', S).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(S.mo_)[0]})
    for b in ['<5%', '5-15%', '15-25%', '25%+']:
        x = S[S.bin == b]; k = f"C(bin)[T.{b}]"; print(f"  {b:7s} n {len(x):4d}  reach {x.hit.mean():.0%}  PEAK med {x.PEAK.median():+.1%}  mean rw {x.rw.mean():+.1%}  adj vs <5%: " + (f"{m.params[k]:+.3f} (t {m.tvalues[k]:+.1f})" if k in m.params else 'ref'))
    S['F'] = S.SI_PCT
    for lab, D in (('train', S[S.tr]), ('2022', S[S.d.dt.year == 2022]), ('2023-26', S[S.d.dt.year >= 2023]), ('ALL', S)):
        r = reg(D); print(f"  era {lab:8s} n flag {int(D.F.sum()):3d}: adj reach {r.params['F']:+.3f} (t {r.tvalues['F']:+.1f})")
    ex = ' + ATRP + LMC + LTURN'; r = reg(S.dropna(subset=['LTURN']), ex); print(f"  harder control (continuous ATRP, LMC, LTURN): adj reach {r.params['F']:+.3f} (t {r.tvalues['F']:+.1f})")
    X = S[S.F]; tks = X.tk.value_counts(); print(f"  flagged events {len(X)}, tickers {X.tk.nunique()}, top-5 share {tks.head(5).sum() / len(X):.0%} (" + ', '.join(f'{t} {n}' for t, n in tks.head(5).items()) + ')')
    print('  leave-top-ticker-out: ' + ', '.join(f"{t}: {reg(S[S.tk != t]).params['F']:+.3f}" for t in tks.head(5).index))
    print(f"  ATRP med flag {X.ATRP.median():.3f} vs {S[~S.F].ATRP.median():.3f}; LMC {X.LMC.median():.2f} vs {S[~S.F].LMC.median():.2f}; PEAK med {X.PEAK.median():+.1%} vs {S[~S.F].PEAK.median():+.1%}; mean rw {X.rw.mean():+.1%} vs {S[~S.F].rw.mean():+.1%}; median r {X.r.median():+.1%} vs {S[~S.F].r.median():+.1%}")
    for nm, F_ in (('SI_DTC alone', S.SI_DTC), ('SI_PCT & DTC>=5', S.SI_PCT & S.SI_DTC), ('SI_PCT & DTC<5', S.SI_PCT & ~S.SI_DTC)):
        S['F'] = F_; r = reg(S); print(f"  {nm}: n {int(F_.sum())} adj reach {r.params['F']:+.3f} (t {r.tvalues['F']:+.1f})")
    S['F'] = S.SI_PCT
    if CAP == 10:
        live = S[(S.score >= 0)] if 'score' in S else S
        G = S[S.ATRP >= np.quantile(S[S.tr].ATRP, .7)]; print(f"  high-ATRP slice (top 30% ATRP, proxy for live group): n {len(G)}, flag n {int(G.F.sum())}, reach flag {G[G.F].hit.mean():.0%} vs rest {G[~G.F].hit.mean():.0%}; adj {reg(G).params['F']:+.3f} (t {reg(G).tvalues['F']:+.1f})")
print('\n######## decisive check: inside the high-ATRP slice (top 30% ATRP, px<10 and px<20) with CONTINUOUS ATRP + LMC + LTURN (+ SI_DTC) controls')
for CAP in (10, 20):
    S = O[(O.px < CAP) & O.ATRP.notna() & O.LMC.notna() & O.LTURN.notna() & O.sip.notna()].copy(); tr = S.tr
    S['a3'] = 0; S['c3'] = 0; G = S[S.ATRP >= np.quantile(S[S.tr].ATRP, .7)].copy(); G['F'] = G.SI_PCT
    for lab, D in (('ALL', G), ('train', G[G.tr]), ('test', G[~G.tr])):
        r = reg(D, ' + ATRP + LMC + LTURN'); print(f"  px<{CAP} {lab:5s} n {len(D)} flag {int(D.F.sum())}: adj reach {r.params['F']:+.3f} (t {r.tvalues['F']:+.1f}); raw {D[D.F].hit.mean():.0%} vs {D[~D.F].hit.mean():.0%}")
