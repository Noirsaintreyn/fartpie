"""Nov 15 2026 ADD VOLUME OR INSIDER BUYING TO THE SCORE BOOK, WITH FEWER SLOTS (Rey: 20 positions is a lot; add vol or insider). Declared before results. Same simulator and frozen score cutoff as report_100/101, TEST 2022-01..2026-02, entry next open, exit open after bear flip, 30bp. Flags: VOL = flip-bar volume >= 1.5x prior-20d mean; INS = open-market insider PURCHASE (Form 4 code P) by an officer or director totaling >= $20k filed in the 30 days before the entry date (insider_P_hist, 2018+). Scenarios: SCORE (frozen top 20%), SCORE&VOL, SCORE&INS, SCORE&(VOL|INS), VOL only, INS only, VOL&INS. Slot caps N in (5, 10, 20), sizing FULL (1/N each) and FIXED5 (5% each). Random-order books averaged over 20 tie-break draws. Declared expectation: extra filters shrink the signal count (fewer than N slots filled), cut variance of what is taken, and the earlier insider+volume tilt (report_77: top 3 winners were 62-68% of its return) should again be lumpy; a filter only helps if it improves Sharpe and maxDD together at the same N, not just total return. Survivorship-flattered; small-cap costs understated."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('report_100_portfolio_dd.py').read(); exec(src[:src.index("S = {'CURRENT-LIKE")])
I = pd.read_pickle(X + 'insider_P_hist.pkl'); I = I[I.TRANS_CODE == 'P'].copy(); I['fd'] = pd.to_datetime(I.fd); I = I[I.RPTOWNER_RELATIONSHIP.astype(str).str.contains('Officer|Director', na=False)]; I['val'] = pd.to_numeric(I.TRANS_SHARES, errors='coerce') * pd.to_numeric(I.TRANS_PRICEPERSHARE, errors='coerce'); ins = {k: z.sort_values('fd')[['fd', 'val']].values for k, z in I.dropna(subset=['val']).groupby('ISSUERTRADINGSYMBOL')}
def insflag(p):
    a = ins.get(p[7]); 
    if a is None: return False
    d = np.datetime64(p[0]); m = (a[:, 0] <= d) & (a[:, 0] > d - np.timedelta64(30, 'D')); return float(a[m, 1].sum()) >= 2e4
flag = {id(p): insflag(p) for p in paths}; V = lambda p: p[5] >= 1.5; Z = lambda p: p[3] >= cut; Ii = lambda p: flag[id(p)]
S = {'SCORE': (lambda p: Z(p), 'score'), 'SCORE & VOL': (lambda p: Z(p) and V(p), 'score'), 'SCORE & INS': (lambda p: Z(p) and Ii(p), 'score'), 'SCORE & (VOL or INS)': (lambda p: Z(p) and (V(p) or Ii(p)), 'score'), 'VOL only': (lambda p: V(p), 'random'), 'INS only': (lambda p: Ii(p), 'random'), 'VOL & INS': (lambda p: V(p) and Ii(p), 'random')}
a, b = pd.Timestamp('2022-01-01'), pd.Timestamp('2026-03-01'); T = [p for p in paths if a <= p[0] < b]
print('test flips', len(T), '| flagged: score', sum(Z(p) for p in T), 'vol', sum(V(p) for p in T), 'ins', sum(Ii(p) for p in T), 'score&vol', sum(Z(p) and V(p) for p in T), 'score&ins', sum(Z(p) and Ii(p) for p in T), 'vol&ins', sum(V(p) and Ii(p) for p in T))
for rule in ('FULL (1/N each)', 'FIXED5 (5% each)'):
    print(f'\n== SIZING {rule} == TEST 2022-01..2026-02')
    for nm, (sel, order) in S.items():
        print(f' {nm}')
        for N in (5, 10, 20):
            f = 1.0 / N if rule.startswith('FULL') else .05; seeds = range(20) if order == 'random' else range(1); R = [sim(sel, a, b, np.random.default_rng(sd), order, f=f, N=N) for sd in seeds]; g = lambda k: np.array([r[k] for r in R]); t = g('total')
            print(f'   N={N:2d}: total {t.mean():+7.1f}%' + (f' ({t.min():+.0f}..{t.max():+.0f})' if len(t) > 1 else '') + f' maxDD {g("mdd").mean():+6.1f}pts sharpe {np.nanmean(g("sharpe")):+.2f} calmar {g("annual").mean()/abs(g("mdd").mean()):+.2f} trades {g("trades").mean():4.0f} avgpos {g("avgpos").mean():4.1f}')
