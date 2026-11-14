"""Nov 14 2026 WHY DID LEADERSHIP FLIPS FAIL IN 2022-24? PERSISTENCE OF LEADERSHIP, MEASURED DIRECTLY (declared before results). report_69/71 show the same split for two different leads (high-short basket, strong sector lead): they pay in 2019-21 and 2025-26 and invert in 2022-24. Hypothesis (Rey's persistence theory): leadership flips pay only when leadership PERSISTS (past-40d outperformance continues over the next 40d); in 2022-24 leadership mean-reverted. 
Part A (descriptive, no flips): cross-sectional rank correlation across industry groups between a group's 40d excess return at s and its excess return over (s, s+40], sampled every 20 days, mean and %>0 by era; same for the high-short basket (time-series corr of lead_s vs lead_{s+40}). 
Part B (the ex-ante version, no look-ahead): PIC_g(t) = mean group IC over sampling dates s with s+40 <= t in the trailing 250 days; PIC_sq(t) = corr(lead_s, lead_{s+40}) over s with s+40<=t in trailing 250 days. Rule fixed now: a leadership signal counts only when PIC>0. Test on flips: GLEAD>+5% & PIC_g>0 vs the same & PIC_g<=0; SQ10 & LEAD & PIC_sq>0 vs <=0; per era. A real mechanism = PIC>0 flips beat PIC<=0 flips in all three eras."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('report_71_sector_and_si_change.py').read(); exec(src.split("def mc(S, m, base):")[0].replace("src = open('report_69_short_regime.py').read(); exec(src.split(\"print('PART A\")[0])", "src0 = open('report_69_short_regime.py').read(); exec(src0.split(\"print('PART A\")[0])"))
H = 40; idxs = gl.index; fw = gl.shift(-H)
ic = pd.Series(np.nan, index=idxs)
for i in range(300, len(idxs) - H, 5):
    d = idxs[i]; a = gl.iloc[i]; b = fw.iloc[i]; m = a.notna() & b.notna()
    if m.sum() >= 8: ic.iloc[i] = a[m].rank().corr(b[m].rank())
icd = ic.dropna(); lead = leadSQ; lfw = lead.shift(-H)
print('PART A: leadership persistence by era (sampled every 5 days; overlapping windows, so read as descriptive)')
for en, a, b in (('2019-21', '2019', '2022'), ('2022-24', '2022', '2025'), ('2025-26', '2025', '2027')):
    x = icd[(icd.index >= a) & (icd.index < b)]; l1 = lead[(lead.index >= a) & (lead.index < b)]; l2 = lfw.reindex(l1.index); m = l1.notna() & l2.notna()
    print(f'  {en}: sector IC mean {x.mean():+.3f} (>0 on {(x>0).mean()*100:.0f}% of dates, n {len(x)}) | high-short lead persistence corr(lead_s, lead_s+40) {l1[m].corr(l2[m]):+.2f}, sign persists {(np.sign(l1[m])==np.sign(l2[m])).mean()*100:.0f}% of days')
# Part B: ex-ante persistence at each flip date
sq_pairs = pd.DataFrame({'a': lead, 'b': lfw})
def pic_sector(t):
    x = icd[(icd.index <= idxs[max(idxs.get_indexer([t], method='pad')[0] - H, 0)]) & (icd.index > t - pd.Timedelta(days=365))]; return x.mean() if len(x) >= 10 else np.nan
def pic_sq(t):
    j = idxs.get_indexer([t], method='pad')[0]
    if j < 0: return np.nan
    x = sq_pairs.iloc[max(j - H - 250, 0):max(j - H, 0)].dropna(); return x.a.corr(x.b) if len(x) >= 60 else np.nan
ud = P.d0.drop_duplicates().sort_values(); pg = {d: pic_sector(d) for d in ud}; ps = {d: pic_sq(d) for d in ud}; P['PICG'] = P.d0.map(pg); P['PICS'] = P.d0.map(ps)
print('\nPART B: ex-ante persistence gate (trailing 250d, only windows already completed)')
def mc(S, m, base):
    s = S[m & base]; r = S[~m & base]; a = s.groupby(s.t.dt.to_period('M')).rn.mean(); b = r.groupby(r.t.dt.to_period('M')).rn.mean(); dm = (a - b).dropna(); return dm.mean() / (dm.std() / np.sqrt(len(dm))) if len(dm) > 5 else np.nan
for en, a, b in (('2019-21', '2019', '2022'), ('2022-24', '2022', '2025'), ('2025-26Q1', '2025', '2026-04')):
    S = P[(P.t >= a) & (P.t < b)].reset_index(drop=True); base = S.HUGE.mean(); print(f'--- {en}: flips {len(S)} | share of flip days with PIC_g>0 {(S.PICG>0).mean()*100:.0f}%, PIC_sq>0 {(S.PICS>0).mean()*100:.0f}% | base mean {S.rn.mean()*100:+.2f}% HUGE/100 {base*100:.2f}')
    T = S[S.G.isin(gl.columns)]; 
    for lab, m, D_ in (('GLEAD>+5% & PIC_g>0', (T.GVAL > .05) & (T.PICG > 0), T), ('GLEAD>+5% & PIC_g<=0', (T.GVAL > .05) & (T.PICG <= 0), T), ('GLEAD (any) & PIC_g>0', T.GLEAD & (T.PICG > 0), T), ('GLEAD (any) & PIC_g<=0', T.GLEAD & (T.PICG <= 0), T)):
        s = D_[m]; print(f'    {lab:28s} n {len(s):5d} mean {s.rn.mean()*100 if len(s) else np.nan:+7.2f}% HUGE/100 {s.HUGE.mean()*100 if len(s) else np.nan:5.2f} lift {s.HUGE.mean()/base if len(s) else np.nan:4.2f}')
    W = S[S.DTC.notna()]
    for lab, m in (('SQ10 & LEAD & PIC_sq>0', W.SQ10 & W.LEAD & (W.PICS > 0)), ('SQ10 & LEAD & PIC_sq<=0', W.SQ10 & W.LEAD & (W.PICS <= 0))):
        s = W[m]; print(f'    {lab:28s} n {len(s):5d} mean {s.rn.mean()*100 if len(s) else np.nan:+7.2f}% HUGE/100 {s.HUGE.mean()*100 if len(s) else np.nan:5.2f} lift {s.HUGE.mean()/max(W.HUGE.mean(),1e-9) if len(s) else np.nan:4.2f}')
    for lab, col in (('ALL flips PIC_g>0 vs <=0', 'PICG'), ('ALL flips PIC_sq>0 vs <=0', 'PICS')):
        a_ = S[S[col] > 0]; b_ = S[S[col] <= 0]; print(f'    {lab:28s} mean {a_.rn.mean()*100 if len(a_) else np.nan:+6.2f}% (n {len(a_)}) vs {b_.rn.mean()*100 if len(b_) else np.nan:+6.2f}% (n {len(b_)})')
