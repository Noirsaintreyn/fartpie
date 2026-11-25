"""Report 469 (Nov 24 2026): Rey - base = volatility/ATRP group, then test REAL-WORLD reasons INSIDE it, for px<20, px<10, px<5.
DECLARED BEFORE RESULTS. Unit = daily bull vector flips 2019-2026Q1 (structure_small), entry known at flip. Base group = top 30% of the base5 rank score (ATRP, F11, VOLX, LTURN, -LMC; train-ECDF, cut from train rows of the same price bucket). Train = 2019-21, test = 2022+.
Targets: REACH (PEAK>=+10%, catching the move) and REAL (r>=+10%).
Reasons (each with a mechanism, all point-in-time: SEC facts filed before flip date, 8-Ks known before the entry):
 1 RUNWAY: runway_m = latest cash / monthly operating cash burn (NetCashProvidedByUsedInOperatingActivities / duration; 60 if cash-generative). Mechanism: short runway => a financing/deal catalyst is coming, price discovers it. Declared sign: SHORTER = more REACH. SQ = burning and runway<12m.
 2 RAISE90: item 3.02 8-K in prior 90d (just funded => runway pressure relieved / momentum from the raise). Sign undeclared (two-sided, Bonferroni).
 3 SECMAC: the sector's macro driver moved in its favour (report 464 TAIL mapping: oil->energy, USD down/gold->miners, curve->financials, XLV rel->health/biotech, XLK rel->tech, yields down->utilities). Sign: favourable = more REACH.
 4 DEAL (8-K 1.01/2.01, 7d), INS (open-market insider buy 30d), EV5 (any 8-K in prior 5d). Sign: present = more REACH.
 5 GATE2 = at least one of {DEAL, INS, SECMAC favourable, SQ}: reason vs none.
PASS per feature/target = AUC (or flag) same side train and test, |AUC-.5|>=.04 both, test z>=3.3 (7 features x 2 targets x 3 buckets). Flag lift tables also shown by era 2019-21 / 2022 / 2023-26."""
import numpy as np, pandas as pd, json, os, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_464_catalyst_flips.py').read(); exec(src[:src.index('def mp_(')])
O['tr'] = (O.d.dt.year <= 2021).values
# ---- runway from SEC facts ----
def runway_table(tk):
    f = f'data/fund/sec_small/{tk}.json'
    if not os.path.exists(f): return None
    us = json.load(open(f))['facts'].get('us-gaap', {})
    def ent(tags):
        for t in tags:
            if t in us and 'USD' in us[t]['units']: return us[t]['units']['USD']
        return []
    ocf = ent(['NetCashProvidedByUsedInOperatingActivities']); cash = ent(['CashAndCashEquivalentsAtCarryingValue', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'])
    if not ocf or not cash: return None
    C = pd.DataFrame([(pd.Timestamp(e['end']), pd.Timestamp(e['filed']), e['val']) for e in cash if 'filed' in e], columns=['end', 'filed', 'val']).sort_values(['end', 'filed']).drop_duplicates('end', keep='first').set_index('end')
    rows = []
    for e in ocf:
        if 'start' not in e or 'filed' not in e: continue
        s, en = pd.Timestamp(e['start']), pd.Timestamp(e['end']); n = (en - s).days
        if not 80 <= n <= 380: continue
        c = C.val[C.index <= en]
        if not len(c): continue
        rows.append((en, pd.Timestamp(e['filed']), e['val'] / n * 30.4, c.iloc[-1]))
    if not rows: return None
    R = pd.DataFrame(rows, columns=['end', 'filed', 'burn_m', 'cash']).sort_values(['end', 'filed']).drop_duplicates(['end'], keep='first')
    R['runway_m'] = np.where(R.burn_m < 0, np.clip(R.cash / -R.burn_m, 0, 60), 60.0); R['burning'] = R.burn_m < 0; R['filed'] = R.filed.astype('datetime64[ns]')
    return R.sort_values('filed')[['filed', 'runway_m', 'burning']]
parts = []
for tk, g in O.groupby('tk'):
    R = runway_table(tk)
    if R is None or R.empty: parts.append(g.assign(runway_m=np.nan, burning=np.nan, stale=np.nan)); continue
    m = pd.merge_asof(g.sort_values('d'), R, left_on='d', right_on='filed', direction='backward', allow_exact_matches=False); m['stale'] = (m.d - m.filed).dt.days; parts.append(m.drop(columns='filed'))
O = pd.concat(parts).reset_index(drop=True); ok = O.stale.notna() & (O.stale <= 200); O.loc[~ok, ['runway_m', 'burning']] = np.nan
O['SQ'] = np.where(O.runway_m.isna(), np.nan, ((O.runway_m < 12) & (O.burning == True)).astype(float))
def flagwin(mask, days):
    k = K[mask][['ticker', 'known']].drop_duplicates(); g = {t: x.known.values for t, x in k.groupby('ticker')}; out = np.zeros(len(O), bool)
    for i, (tk, d) in enumerate(zip(O.tk.values, O.d.values)):
        a = g.get(tk)
        if a is not None: out[i] = ((a <= d) & (a > d - np.timedelta64(days, 'D'))).any()
    return out
it_ = K['items'].astype(str).str.split(','); O['RAISE90'] = flagwin(it_.apply(lambda l: '3.02' in l), 90).astype(float); O['EV5'] = flagwin(pd.Series(True, index=K.index), 5).astype(float)
O['SECMAC'] = O.TAIL; O['DEALf'] = O.DEAL.astype(float); O['INSf'] = O.INS.astype(float)
O['GATE2'] = np.where(O.runway_m.isna() & O.SECMAC.isna() & (O.DEALf == 0) & (O.INSf == 0), np.nan, ((O.DEALf == 1) | (O.INSf == 1) | (O.SECMAC == 1) | (O.SQ == 1)).astype(float))
# ---- base score ----
B = {'ATRP': 1, 'F11': 1, 'VOLX': 1, 'LTURN': 1, 'LMC': -1}
def score(D):
    R = pd.DataFrame(index=D.index)
    for f, s in B.items():
        x = D[f].astype(float); ref = np.sort(x[D.tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); R[f] = q if s > 0 else 1 - q; R.loc[x.isna(), f] = np.nan
    return R.mean(axis=1)
def auc(x, y):
    m = ~np.isnan(x); x, y = x[m], y[m]; p, n = y.sum(), (1 - y).sum()
    if p < 10 or n < 10: return np.nan, np.nan
    r = pd.Series(x).rank().values; a = (r[y == 1].sum() - p * (p + 1) / 2) / (p * n); se = np.sqrt((p + n + 1) / (12 * p * n)); return a, (a - .5) / se
FE = [('RUNWAY_SHORT', -O.runway_m), ('SQ', O.SQ), ('RAISE90', O.RAISE90), ('SECMAC', O.SECMAC), ('DEAL', O.DEALf), ('INS', O.INSf), ('EV5', O.EV5), ('GATE2', O.GATE2)]
for c, v in FE: O['f_' + c] = v.astype(float)
for lab, cap in (('px<20', 20), ('px<10', 10), ('px<5', 5)):
    S = O[O.px < cap].copy(); S['s_'] = score(S); cut = np.quantile(S[S.tr].s_.dropna(), .7); G = S[S.s_ >= cut]
    print(f'\n################ {lab}: all flips {len(S)}, base group (top 30%) {len(G)} (train {G.tr.sum()}, test {(~G.tr).sum()}) | fundamentals coverage in group {G.runway_m.notna().mean():.0%} | REACH base train {(G[G.tr].PEAK >= .1).mean():.1%} test {(G[~G.tr].PEAK >= .1).mean():.1%} | REAL test {(G[~G.tr].r >= .1).mean():.1%}')
    for tgt, y in (('REACH', (G.PEAK >= .1).values.astype(float)), ('REAL', (G.r >= .1).values.astype(float))):
        print(f'  == {tgt}: feature | AUC train / test (z) | n test | PASS')
        for c, _ in FE:
            x = G['f_' + c].values.astype(float); t = G.tr.values; a1, z1 = auc(x[t], y[t]); a2, z2 = auc(x[~t], y[~t]); n2 = int((~np.isnan(x[~t])).sum())
            ps = a1 == a1 and a2 == a2 and (a1 - .5) * (a2 - .5) > 0 and abs(a1 - .5) >= .04 and abs(a2 - .5) >= .04 and abs(z2) >= 3.3
            print(f'    {c:13s} {a1:.3f} / {a2:.3f} (z {z2:+.1f}) n {n2:5d} {"PASS" if ps else ""}')
    print('  flag lift on REACH (flag vs no flag) by era 2019-21 / 2022 / 2023-26 [n flagged]:')
    for c in ('SQ', 'RAISE90', 'SECMAC', 'DEAL', 'INS', 'EV5', 'GATE2'):
        f = G['f_' + c]; s = ''
        for e in ('2019-21', '2022', '2023-26'):
            E_ = G[G.era3 == e]; fe = E_['f_' + c]; a = E_[fe == 1]; b = E_[fe == 0]; s += f' {(a.PEAK >= .1).mean() * 100:4.0f}% v {(b.PEAK >= .1).mean() * 100:3.0f}% [{len(a)}] /'
        a = G[f == 1]; b = G[f == 0]; print(f'    {c:8s}{s}  ALL: REACH {(a.PEAK >= .1).mean():.0%} v {(b.PEAK >= .1).mean():.0%} | REAL {(a.r >= .1).mean():.0%} v {(b.r >= .1).mean():.0%} | median r {a.r.median() * 100:+.1f}% v {b.r.median() * 100:+.1f}%')
