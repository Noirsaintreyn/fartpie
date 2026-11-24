"""Nov 24 2026 report 438 = Rey's mid/small-cap idea: buybacks + fundamentals pick the names, vector flips confirm.  DECLARED BEFORE LOOKING.
Data: 102 sub-$5 names, SEC filings (xs/pull_sec_smallcap.py -> sec_small.parquet), PIT (filed + 1 day), annual (10-K / 20-F) values only; vector flips = tr400_B.parquet (small-cap flips, 108 names, base win 32%, median -1.44%, mean -0.32% net of 40bp, exits = vector).
H1 BUYBACK: latest-FY PaymentsForRepurchaseOfCommonStock / operating cash flow >= 0.10 (BUYER) vs no buyback tag or zero (NON).  Prediction: BUYER > NON.
H2 DILUTION: change in FY weighted diluted shares vs the prior FY: SHRINK <= -1% vs DILUTE >= +5%.  Prediction: SHRINK > DILUTE.
H3 PROFITABLE: latest-FY net income > 0 AND operating cash flow > 0 (GOOD) vs both <= 0 (BAD).  Prediction: GOOD > BAD.
PASS (3 hypotheses -> |t| >= 2.4, month-clustered, same sign train 2019-21 / test 2018+2022+); ALSO report absolute means: a filter only matters here if the selected group is itself profitable after cost.  Macro is not a separate arm: it was tested as a layer on the large-cap flips and added nothing; a macro regime split is shown for the winning group only if one passes."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
S = pd.read_parquet(SP + 'sec_small.parquet'); T = pd.read_parquet(SP + 'tr400_B.parquet'); T['entry'] = pd.to_datetime(T.entry); T['asof'] = (T.entry - pd.Timedelta(days=1)).dt.normalize()
S['filed'] = pd.to_datetime(S.filed); S['end'] = pd.to_datetime(S.end); S = S[S.form.isin(['10-K', '20-F', '10-K/A', '40-F'])]
def series(tag):
    X = S[S.tag == tag].sort_values(['tk', 'end', 'filed']).drop_duplicates(['tk', 'end'], keep='first')          # first-filed value for each fiscal end (PIT, no restatements)
    X = X.assign(prev=X.groupby('tk').val.shift(1)); return X[['tk', 'filed', 'val', 'prev']].sort_values('filed')
def asof(tag, name):
    X = series(tag).rename(columns={'val': name, 'prev': name + '_prev'}); return pd.merge_asof(T.sort_values('asof'), X, left_on='asof', right_on='filed', by='tk', direction='backward').drop(columns='filed')
T['ret'] = T.ret - 0.004          # tr400_B.ret is gross; 40bp round-trip cost for small caps
D = T.sort_values('asof')
for tag, nm in (('PaymentsForRepurchaseOfCommonStock', 'bb'), ('NetCashProvidedByUsedInOperatingActivities', 'ocf'), ('NetIncomeLoss', 'ni'), ('WeightedAverageNumberOfDilutedSharesOutstanding', 'sh')):
    X = series(tag).rename(columns={'val': nm, 'prev': nm + '_prev'}); D = pd.merge_asof(D, X, left_on='asof', right_on='filed', by='tk', direction='backward').drop(columns='filed')
D = D.dropna(subset=['ocf']).copy(); D['bb'] = D.bb.fillna(0)
D['buyer'] = np.where(D.ocf > 0, D.bb / D.ocf, np.where(D.bb > 0, 9, 0)); D['dsh'] = D.sh / D.sh_prev - 1; D['good'] = (D.ni > 0) & (D.ocf > 0); D['bad'] = (D.ni <= 0) & (D.ocf <= 0)
D['half'] = np.where((D.entry >= '2019-01-01') & (D.entry < '2022-01-01'), 'train', 'test'); D['era'] = pd.cut(D.entry.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26']); D['mon'] = D.entry.dt.to_period('M')
print(len(T), 'flips;', len(D), 'with a PIT annual filing;', D.entry.min(), '->', D.entry.max())
def contrast(A, B, lab):
    col = 'ret'; dd = pd.concat([A.assign(s=1), B.assign(s=-1)]); g = dd.groupby('mon').apply(lambda x: pd.Series(dict(n1=(x.s == 1).sum(), n0=(x.s == -1).sum(), s1=x[col][x.s == 1].sum(), s0=x[col][x.s == -1].sum()))); m1, m0 = A[col].mean(), B[col].mean()
    se = np.sqrt((((g.s1 - g.n1 * m1) / len(A) - (g.s0 - g.n0 * m0) / len(B)) ** 2).sum()); f = lambda x: f"n{len(x):5d} win {(x[col]>0).mean()*100:4.1f} med {x[col].median()*100:6.2f} mean {x[col].mean()*100:6.2f}"
    print(f"  {lab}\n    A {f(A)}\n    B {f(B)}\n    A-B {(m1-m0)*100:+.2f}pp  t {(m1-m0)/se:+.2f} | " + ' | '.join(f"{h} {(A[A.half==h][col].mean()-B[B.half==h][col].mean())*100:+.2f}pp (n {len(A[A.half==h])}/{len(B[B.half==h])})" for h in ('train', 'test')) + ' | ' + ' '.join(f"{e} {(A[A.era==e][col].mean()-B[B.era==e][col].mean())*100:+.2f}" for e in ('18-19', '20-21', '22', '23-26')))
print(f"\nALL flips with data: n {len(D)} win {(D.ret>0).mean()*100:.1f} med {D.ret.median()*100:.2f} mean {D.ret.mean()*100:.2f}")
contrast(D[D.buyer >= .10], D[D.buyer == 0], 'H1 BUYBACK: A = BUYER (>=10% of OCF), B = NON; predict A > B')
contrast(D[D.dsh <= -.01], D[D.dsh >= .05], 'H2 DILUTION: A = SHRINK (<=-1%), B = DILUTE (>=+5%); predict A > B')
contrast(D[D.good], D[D.bad], 'H3 PROFITABLE: A = GOOD (NI>0 & OCF>0), B = BAD (both <=0); predict A > B')
print('\n(context) buyer-years: names with any buyback', D[D.bb > 0].tk.nunique(), 'of', D.tk.nunique(), '| GOOD', int(D.good.sum()), 'BAD', int(D.bad.sum()))
D.to_parquet(SP + 'g438.parquet')
