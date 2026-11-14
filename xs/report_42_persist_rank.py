"""Nov 14 2026, THEORY-DRIVEN (declared before results): if post-flip continuation is order-flow persistence, flips in stocks whose OWN recent price path is persistent should earn more than flips in stocks whose path is mean-reverting. Features (all price-only, known at the prior close, cross-sectional percentile among the ~500 names that day): VR5 = var(5d ret)/(5 var(1d ret)) over 250d; ER20 = |20d net move| / sum|daily moves|; AC1 = 250d lag-1 autocorrelation; COMP = mean of the three ranks. Events = report_27 daily vector bull flips (12,349, S&P PIT names 2018+), outcome = fwd 21d / 63d excess vs universe (r21, r63). PASS = high tercile > low tercile in BOTH eras (2018-21, 2022+), pooled |t| >= 2, and in >= 4 of 5 regimes. Hypothesis fixed in advance: HIGH persistence is better."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
O = pickle.load(open(SC + 'flipval_ohlc.pkl', 'rb')); E = pd.read_pickle(SC + 'flipval_events.pkl')
C = pd.DataFrame({t: x.close for t, x in O.items()}).sort_index(); r1 = np.log(C).diff(); r5 = np.log(C / C.shift(5))
VR5 = r5.rolling(250, min_periods=200).var() / (5 * r1.rolling(250, min_periods=200).var())
ER = (C - C.shift(20)).abs() / C.diff().abs().rolling(20).sum()
AC = pd.DataFrame({t: r1[t].rolling(250, min_periods=200).corr(r1[t].shift(1)) for t in C.columns})
RK = {n: f.rank(axis=1, pct=True).shift(1) for n, f in (('VR5', VR5), ('ER20', ER), ('AC1', AC))}; RK['COMP'] = (RK['VR5'] + RK['ER20'] + RK['AC1']) / 3
E = E.copy(); E['d'] = pd.to_datetime(E.d)
for n, F in RK.items():
    st = F.stack(); st.index.names = ['d', 'tk']; E[n] = [st.get((d, t), np.nan) for d, t in zip(E.d, E.tk)]
E = E.dropna(subset=['COMP'])
RG = [('R1', '2018-01-01', '2020-03-01'), ('R2', '2020-03-01', '2022-01-01'), ('R3', '2022-01-01', '2023-04-01'), ('R4', '2023-04-01', '2025-04-01'), ('R5', '2025-04-01', '2027-01-01')]
ERA = {'18-21': ('2018-01-01', '2022-01-01'), '22+': ('2022-01-01', '2027-01-01'), 'pooled': ('2018-01-01', '2027-01-01')}
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 5 and len(b) > 5 else np.nan
print(f'events with a state: {len(E)}')
for n in ('VR5', 'ER20', 'AC1', 'COMP'):
    E['T'] = pd.cut(E[n], [0, 1 / 3, 2 / 3, 1.0001], labels=[0, 1, 2], include_lowest=True)
    for h in ('r21', 'r63'):
        line = f' {n:5s} {h}'
        for e, (a, b) in ERA.items():
            s = E[(E.d >= a) & (E.d < b)]; lo = s[s['T'] == 0][h]; hi = s[s['T'] == 2][h]; line += f' | {e}: low {lo.mean() * 100:+5.2f}% high {hi.mean() * 100:+5.2f}% diff {(hi.mean() - lo.mean()) * 100:+5.2f}% t {tt(hi, lo):+4.1f}'
        reg = [(E[(E.d >= a) & (E.d < b) & (E['T'] == 2)][h].mean() - E[(E.d >= a) & (E.d < b) & (E['T'] == 0)][h].mean()) * 100 for _, a, b in RG]
        print(line + ' | regimes high-low: ' + ' '.join(f'{x:+.2f}' for x in reg) + f' ({sum(x > 0 for x in reg)}/5 positive)')
