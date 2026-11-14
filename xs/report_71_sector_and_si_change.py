"""Nov 14 2026 TWO LEADS FROM THE REGIME FINDING (declared before results; Rey: explain, don't just trash). 
LEAD 1 SECTOR LEADERSHIP: same mechanism as the high-short basket, by industry. For each SIC group (report_66 grouping) build the daily equal-weight basket; GLEAD = group's 40d compounded return minus the whole universe's > 0 at the flip bar; GTOP3 = group is among the 3 best groups by 40d return. Hypothesis: flips inside a leading industry (participants rotating into it) pay more than flips in lagging ones. Primary = GLEAD, secondary = GTOP3; nothing else tried. 
LEAD 2 SHORT-INTEREST CHANGE: FINRA latest published report (+8d lag) vs the previous one: SI_UP = short position up >=20%, SI_DN = down >=20% (cover). Hypothesis: shorts piling in before a flip = squeeze fuel; shorts already covering = move is spent. Tested alone, with DTC>=10, and inside the high-short-basket regime LEAD (report_69). 
Eras 2019-21 / 2022-24 / 2025-26Q1 (the three regimes found in report_69), flips from report_64 (31.8k, survivorship-biased), 30bp, month-clustered t vs complement, ex-top-3 mean. A lead counts only if it has HUGE lift>=1.3 and mean above base in the era where its mechanism is ON and does NOT invert in the others."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('report_69_short_regime.py').read(); exec(src.split("print('PART A")[0])
ret = cl.pct_change(fill_method=None).clip(-0.5, 1.0); hi = (dtc >= 10).shift(1).fillna(False); bk = ret.where(hi).mean(axis=1); un = ret.mean(axis=1)
leadSQ = ((1 + bk.fillna(0)).rolling(40).apply(np.prod, raw=True) - (1 + un.fillna(0)).rolling(40).apply(np.prod, raw=True)); P['LEAD'] = leadSQ.reindex(P.d0).values > 0
def grp(s):
    s = int(s) if pd.notna(s) else -1
    if 2833 <= s <= 2836 or s in (8731, 2834): return 'BIOPHARMA'
    if s in (3841, 3842, 3844, 3845, 3826, 3829, 8071, 8090, 8093, 8000, 8011, 8050, 8060): return 'MEDDEV/HEALTHSVC'
    if 7370 <= s <= 7379: return 'SOFTWARE/IT'
    if 3570 <= s <= 3579 or 3660 <= s <= 3679 or s in (3690, 3695, 3559): return 'HARDWARE/SEMIS'
    if 1300 <= s <= 1389 or 2900 <= s <= 2999 or s in (4922, 4923, 4924, 5171, 5172): return 'ENERGY'
    if 1000 <= s <= 1499: return 'MINING/METALS'
    if 3711 <= s <= 3716 or 3790 <= s <= 3799 or s in (5500, 5511): return 'AUTOS/EV'
    if 6000 <= s <= 6799: return 'FINANCIAL/REIT'
    if 5200 <= s <= 5999 or 7000 <= s <= 7099 or 5800 <= s <= 5899: return 'RETAIL/CONSUMER'
    if 4400 <= s <= 4799: return 'TRANSPORT/AIR'
    if 2000 <= s <= 3999: return 'INDUSTRIAL/MFG'
    return 'OTHER/SERVICES'
sg = sic.set_index('tk').sic.map(grp); G = {g: [t for t in cl.columns if sg.get(t) == g] for g in sg.unique()}; G = {g: v for g, v in G.items() if len(v) >= 25 and g != 'OTHER/SERVICES'}
r40u = (1 + un.fillna(0)).rolling(40).apply(np.prod, raw=True); gl = pd.DataFrame({g: (1 + ret[v].mean(axis=1).fillna(0)).rolling(40).apply(np.prod, raw=True) - r40u for g, v in G.items()}); grk = gl.rank(axis=1, ascending=False)
P['G'] = P.tk.map(sg); ok = P.G.isin(gl.columns)
P['GLEAD'] = False; P['GTOP3'] = False
for g in gl.columns:
    m = (P.G == g).values; P.loc[m, 'GLEAD'] = gl[g].reindex(P.d0[m]).values > 0; P.loc[m, 'GTOP3'] = grk[g].reindex(P.d0[m]).values <= 3
P['GVAL'] = [gl.at[d, g] if (g in gl.columns and d in gl.index) else np.nan for d, g in zip(P.d0, P.G)]
# short-interest change at the latest published report
si2 = si.copy(); si2['sd'] = pd.to_datetime(si2.settlementDate); si2 = si2.sort_values('avail'); si2['chg'] = si2.currentShortPositionQuantity / si2.previousShortPositionQuantity.replace(0, np.nan) - 1
chg = pd.DataFrame(np.nan, index=cl.index, columns=cl.columns)
for tk, g in si2.groupby('symbolCode'):
    s = g.drop_duplicates('avail').set_index('avail').chg; s = s[~s.index.duplicated()]; chg[tk] = s.reindex(s.index.union(cl.index)).ffill(limit=20).reindex(cl.index)
P['SICHG'] = [chg.at[d, tk] if (d in chg.index and tk in chg.columns) else np.nan for d, tk in zip(P.d0, P.tk)]; P['SI_UP'] = P.SICHG >= 0.20; P['SI_DN'] = P.SICHG <= -0.20
def mc(S, m, base):
    s = S[m & base]; r = S[~m & base]; a = s.groupby(s.t.dt.to_period('M')).rn.mean(); b = r.groupby(r.t.dt.to_period('M')).rn.mean(); dm = (a - b).dropna(); return dm.mean() / (dm.std() / np.sqrt(len(dm))) if len(dm) > 5 else np.nan
ERAS = (('2019-21', '2019', '2022'), ('2022-24', '2022', '2025'), ('2025-26Q1', '2025', '2026-04'))
def show(title, flags, need):
    print(f'\n######## {title}')
    for en, a, b in ERAS:
        S = P[(P.t >= a) & (P.t < b) & need(P)].reset_index(drop=True); base = S.HUGE.mean(); print(f'--- {en}: n {len(S)} base mean {S.rn.mean()*100:+.2f}% med {S.rn.median()*100:+.2f} HUGE/100 {base*100:.2f}')
        for lab, f in flags:
            m = f(S)
            for sub, mm in ((lab, m), (lab + ' & BV>=1.5', m & (S.BV >= 1.5))):
                s = S[mm]
                if len(s) < 25: print(f'    {sub:34s} n {len(s)}'); continue
                print(f'    {sub:34s} n {len(s):5d} mean {s.rn.mean()*100:+7.2f}% med {s.rn.median()*100:+6.2f} HUGE/100 {s.HUGE.mean()*100:5.2f} lift {s.HUGE.mean()/base:4.2f} ex-top3 {np.sort(s.rn.values)[:-3].mean()*100:+6.2f} t {mc(S, mm, pd.Series(True, index=S.index)):+.1f}')
show('LEAD 1: SECTOR LEADERSHIP (flips in groups with >=25 names)', [('GLEAD (group beats universe 40d)', lambda S: S.GLEAD), ('group LAGS', lambda S: ~S.GLEAD), ('GTOP3 (top-3 group)', lambda S: S.GTOP3), ('GLEAD>+5% (strong lead)', lambda S: S.GVAL > 0.05)], lambda P: ok)
show('LEAD 2: SHORT-INTEREST CHANGE (flips with a published report)', [('SI_UP (>=+20%)', lambda S: S.SI_UP), ('SI_DN (<=-20%)', lambda S: S.SI_DN), ('DTC>=10 & SI_UP', lambda S: S.SQ10 & S.SI_UP), ('DTC>=10 & SI_DN', lambda S: S.SQ10 & S.SI_DN), ('DTC>=10 & SI_UP & basket LEAD', lambda S: S.SQ10 & S.SI_UP & S.LEAD), ('DTC>=10 & SI_DN & basket LEAD', lambda S: S.SQ10 & S.SI_DN & S.LEAD)], lambda P: P.SICHG.notna() & P.DTC.notna())
print('\n######## by industry group per era (flips in group: mean%, n | share of HUGE) to see what GLEAD is picking up')
for en, a, b in ERAS:
    S = P[(P.t >= a) & (P.t < b) & ok]; print(f'--- {en}: ' + ' | '.join(f'{g[:9]} {x.rn.mean()*100:+.1f}/{len(x)}' for g, x in S.groupby('G')))
