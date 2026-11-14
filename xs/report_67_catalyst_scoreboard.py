"""Nov 14 2026 CATALYST SCOREBOARD: every item on the pasted small/mid catalyst list, tested on the same vector flips (declared before results). Universe: wide daily long flips (report_64, 31,814 flips, 2019-2026Q1, survivorship-biased). Every flag is known before the entry open (flag window ends on the flip-bar date d0). Items: RVOL spike (flip-bar volume vs prior-20d mean: >=1.5x, >=2x, >=3x); short squeeze (FINRA days-to-cover >=5 / >=10, latest report available with +8d lag; data exists only 2025-01 onward, so NO strict split); 13D (new / any, last 30d) with 13G as control; binary-event proxies from 8-K items in the 7 days to d0 (1.01 material agreement = contract/partnership proxy, 7.01/8.01 event press release, 5.02 management change, 3.02 unregistered equity sale = financing) alone and inside biotech/pharma SIC (2833-2836, 8731) as the FDA/trial proxy (no free PDUFA calendar exists, so this is a proxy); index inclusion (S&P 400/600 addition effective in prior 20 days; Russell reconstitution window = flip bar May 20-Jun 30 seasonal proxy since membership lists are not free); insider open-market buy in prior 30d (benchmark). NOT TESTABLE with data we can get: options sweeps (no OPRA data), 13F (needs CUSIP maps and quarter-lagged bulk files), real PDUFA calendar. 
Metrics per flag, each era (2019-21 / 2022-26Q1) on all flips AND on top of BV>=1.5: n, HUGE(>=+100%) per 100 flips (precision) and lift vs the same base, share of all HUGE captured (recall), BIG50 per 100, loser share (net<0), mean/median net (30bp), mean excluding the top 3 winners, month-clustered t vs complement. PASS = n>=30 in both eras, HUGE lift >=1.3 in both eras, and mean net above the base in both eras."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; P = pd.read_pickle(SC + 'wide_flips_feat.pkl'); P['rn'] = P.ret - 0.003; P['d0'] = pd.to_datetime(P.d0)
K = pd.read_csv(SC + '8k_wide.csv'); K['fd'] = pd.to_datetime(K.filed); K['it'] = K['items'].astype(str).str.split(','); F13 = pd.read_csv(SC + '13d_wide.csv'); F13['fd'] = pd.to_datetime(F13.filed)
sic = pd.read_csv(SC + 'sic.csv'); P = P.merge(sic[['tk', 'sic']], on='tk', how='left'); P['BIO'] = ((P.sic >= 2833) & (P.sic <= 2836)) | (P.sic == 8731)
def recent(ev, days, col='fd', key='ticker'):
    per = {tk: g[col].sort_values().values for tk, g in ev.groupby(key)}; out = np.zeros(len(P), bool)
    for tk, ix in P.groupby('tk').groups.items():
        a = per.get(tk)
        if a is None: continue
        t = P.loc[ix, 'd0'].values; pos = np.searchsorted(a, t, side='right') - 1; ok = pos >= 0; out[P.index.get_indexer(ix)] = ok & (a[np.clip(pos, 0, None)] > t - np.timedelta64(days, 'D'))
    return out
def item_ev(items): return K[K.it.map(lambda l: any(i in items for i in l))]
P['RV15'] = P.BV >= 1.5; P['RV2'] = P.BV >= 2; P['RV3'] = P.BV >= 3
P['D13NEW'] = recent(F13[F13.form.str.contains('13D') & ~F13.form.str.contains('/A')], 30); P['D13ANY'] = recent(F13[F13.form.str.contains('13D')], 30); P['G13CTRL'] = recent(F13[F13.form.str.contains('13G')], 30)
P['K101'] = recent(item_ev({'1.01'}), 7); P['K701_801'] = recent(item_ev({'7.01', '8.01'}), 7); P['K502'] = recent(item_ev({'5.02'}), 7); P['K302'] = recent(item_ev({'3.02'}), 7)
P['BIO'] = P.BIO.fillna(False); P['BIO_EVT'] = P.BIO & P.K701_801; P['BIO_ONLY'] = P.BIO
IC = pd.read_csv(SC + 'index_changes.csv'); IC['date'] = pd.to_datetime(IC.date); IC = IC[IC.added.notna()]; P['SPADD'] = recent(IC.rename(columns={'added': 'tk2'}), 20, col='date', key='tk2')
P['RUSSELL_WIN'] = ((P.d0.dt.month == 5) & (P.d0.dt.day >= 20)) | (P.d0.dt.month == 6); P['INS'] = P.NI >= 1
# short interest (2025+ only)
si = pd.read_pickle(SC + 'shortint.pkl'); si['avail'] = pd.to_datetime(si.settlementDate) + pd.Timedelta(days=8); si = si.sort_values('avail'); P['DTC'] = np.nan
for tk, ix in P.groupby('tk').groups.items():
    g = si[si.symbolCode == tk]
    if g.empty: continue
    t = P.loc[ix, 'd0'].values; a = g.avail.values; pos = np.searchsorted(a, t, side='right') - 1; ok = (pos >= 0) & (np.where(pos >= 0, t - a[np.clip(pos, 0, None)], np.timedelta64(99, 'D')) < np.timedelta64(20, 'D')); v = g.daysToCoverQuantity.values[np.clip(pos, 0, None)]; P.loc[ix, 'DTC'] = np.where(ok, v, np.nan)
P['SQ5'] = P.DTC >= 5; P['SQ10'] = P.DTC >= 10; P['SI_COVER'] = P.t >= '2025-02-15'
E1 = (P.t >= '2019') & (P.t < '2022'); E2 = P.t >= '2022'
def mc(S, m, base):
    s = S[m & base]; r = S[~m & base]; a = s.groupby(s.t.dt.to_period('M')).rn.mean(); b = r.groupby(r.t.dt.to_period('M')).rn.mean(); dm = (a - b).dropna(); return dm.mean() / (dm.std() / np.sqrt(len(dm))) if len(dm) > 5 else np.nan
FL = [('RV15', 'RVOL: flip-bar volume >=1.5x'), ('RV2', 'RVOL >=2x'), ('RV3', 'RVOL >=3x'), ('INS', 'insider open-mkt buy 30d (benchmark)'), ('D13NEW', 'new 13D 30d'), ('D13ANY', 'any 13D 30d'), ('G13CTRL', '13G 30d (control)'), ('K101', '8-K 1.01 agreement 7d (contract proxy)'), ('K701_801', '8-K 7.01/8.01 event PR 7d'), ('K502', '8-K 5.02 mgmt change 7d'), ('K302', '8-K 3.02 equity sale 7d'), ('BIO_ONLY', 'biotech/pharma SIC (FDA proxy pool)'), ('BIO_EVT', 'biotech + event 8-K 7d'), ('SPADD', 'S&P 400/600 addition 20d'), ('RUSSELL_WIN', 'Russell recon window May20-Jun'), ('SQ5', 'short squeeze DTC>=5 (2025+)'), ('SQ10', 'short squeeze DTC>=10 (2025+)')]
def stat(S, m, base):
    z = S[base]; s = S[m & base]; n = len(s)
    if n < 15: return None
    rate = s.HUGE.mean(); br = z.HUGE.mean(); top = np.sort(s.rn.values)[:-3].mean() if n > 20 else np.nan
    return dict(n=n, per100=rate * 100, lift=rate / br if br > 0 else np.nan, recall=s.HUGE.sum() / max(z.HUGE.sum(), 1) * 100, b50=s.BIG50.mean() * 100, lose=(s.rn < 0).mean() * 100, mean=s.rn.mean() * 100, med=s.rn.median() * 100, ex3=top * 100, t=mc(S, m, base), basemean=z.rn.mean() * 100)
res = {}
for bn, bsel in (('ALL FLIPS', lambda S: pd.Series(True, index=S.index)), ('ON TOP OF BV>=1.5', lambda S: S.BV >= 1.5)):
    print(f'\n######## BASE: {bn}')
    for en, e in (('2019-21', E1), ('2022-26Q1', E2)):
        S = P[e].reset_index(drop=True); base = bsel(S); z = S[base]
        print(f'\n--- {en}: base n {len(z)} | HUGE {z.HUGE.mean()*100:.2f}/100 | BIG50 {z.BIG50.mean()*100:.2f}/100 | losers {(z.rn<0).mean()*100:.0f}% | mean {z.rn.mean()*100:+.2f}% med {z.rn.median()*100:+.2f}%')
        print(f'  {"flag":42s} {"n":>6s} {"HUGE/100":>8s} {"lift":>5s} {"recall%":>7s} {"BIG50/100":>9s} {"lose%":>5s} {"mean%":>7s} {"med%":>6s} {"ex-top3":>7s} {"t":>5s}')
        for k, lab in FL:
            if k in ('SQ5', 'SQ10') and en == '2019-21': continue
            m = S[k].fillna(False).astype(bool) if k not in ('SQ5', 'SQ10') else S[k]
            if k in ('RV15', 'RV2', 'RV3') and bn.startswith('ON TOP') and k == 'RV15': continue
            r = stat(S, m, base)
            if r is None: print(f'  {lab:42s} n<15'); continue
            res[(bn, en, k)] = r; print(f'  {lab:42s} {r["n"]:6d} {r["per100"]:8.2f} {r["lift"]:5.2f} {r["recall"]:7.1f} {r["b50"]:9.2f} {r["lose"]:5.0f} {r["mean"]:+7.2f} {r["med"]:+6.2f} {r["ex3"]:+7.2f} {r["t"]:+5.1f}')
print('\n######## VERDICT (PASS = n>=30 both eras, HUGE lift>=1.3 both eras, mean above base both eras)')
for bn in ('ALL FLIPS', 'ON TOP OF BV>=1.5'):
    print(f'-- base {bn}')
    for k, lab in FL:
        a = res.get((bn, '2019-21', k)); b = res.get((bn, '2022-26Q1', k))
        if k in ('SQ5', 'SQ10'):
            if b: print(f'  {lab:42s} ONE WINDOW ONLY (2025-26): n {b["n"]} lift {b["lift"]:.2f} mean {b["mean"]:+.2f}% vs base {b["basemean"]:+.2f}% -> no strict split'); continue
        if not a or not b: continue
        ok = a['n'] >= 30 and b['n'] >= 30 and a['lift'] >= 1.3 and b['lift'] >= 1.3 and a['mean'] > a['basemean'] and b['mean'] > b['basemean']
        print(f'  {lab:42s} {"PASS" if ok else "fail"} | lift {a["lift"]:.2f}/{b["lift"]:.2f} | mean-vs-base {a["mean"]-a["basemean"]:+.2f}/{b["mean"]-b["basemean"]:+.2f}pp | n {a["n"]}/{b["n"]}')
