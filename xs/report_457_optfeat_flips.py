"""Report 457 (Nov 24 2026): do other greeks / option flow / vol features improve flip picks beyond the daily net-gamma sign, and short-dated (<=2 DTE) vs longer-dated gamma.
Data: data/theta_feat2/<TK>.csv built by xs/build_opt_features.py (OI(report_date d) x EOD(d)); features shifted ONE trading day (prior close) before they touch a flip. Flips = xs/ext/wide_T.parquet (live filter: S&P member, options ratio >= 1.5), names with theta data. 15m and 1h.
Declared before running (direction in brackets; |t|>=2 month-paired AND same sign in train 2019-21 and test 2018+2022+ to PASS):
 F1 net gamma < 0 beats >= 0 [baseline, reports 451-455]; F2 vanna in top third of own trailing-252d rank beats bottom third [high]; F3 charm top third beats bottom [high];
 F4 put/call VOLUME ratio top third beats bottom [high: hedging fear is contrarian]; F5 ATM IV (30d) 252d rank top third beats bottom [high]; F6 IV term inverted (short IV > long IV) beats not [inverted];
 F7 skew (put IV - call IV) top third beats bottom [high]; F8 DEX top third beats bottom [high]; F9 (0DTE vs regular) flips where the SHORT-dated (<=2 DTE) gamma is negative beat flips where only the longer-dated gamma is negative [short-dated].
Incremental: each feature inside the net-gamma strata (neg, pos) so overlap with F1 is visible. True 0DTE needs intraday data; EOD files only give <=2 DTE. PASS = |t| >= 2 in the DECLARED direction (first group better) and same sign in train and test. Run: python3 xs/report_457_optfeat_flips.py"""
import pandas as pd, numpy as np, os
UP = '/mnt/user-data/uploads/fartpie/' if os.path.exists('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet') else ''
SPd = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
W = pd.read_parquet(UP + 'xs/ext/wide_T.parquet'); W = W[W.sp & (W.ratio >= 1.5)].copy().reset_index(drop=True); W['entry'] = pd.to_datetime(W.entry); W['day'] = W.entry.dt.normalize()
TK = [f[:-4] for f in os.listdir(UP + 'data/theta_feat2')]; W = W[W.tk.isin(TK)].copy()
print('names with flips + theta:', sorted(W.tk.unique()), 'flips', len(W), 'of which 1h', int(W.h1.sum()))
H0 = {}
def prep(tk):
    X = pd.read_csv(UP + f'data/theta_feat2/{tk}.csv', parse_dates=['date']).set_index('date').sort_index()
    rk = lambda s: s.rolling(252, min_periods=60).apply(lambda a: (a[:-1] < a[-1]).mean() if len(a) > 1 else np.nan, raw=True)
    F = pd.DataFrame(index=X.index)
    F['gex'] = X.gex; F['gex_short'] = X.gex_short; F['gex_long'] = X.gex_long
    for c in ['vanna', 'charm', 'dex', 'pc_vol', 'iv30', 'skew']: F['r_' + c] = rk(X[c])
    F['term'] = X.term
    return F.shift(1)                          # prior-close known
FE = pd.concat({tk: prep(tk) for tk in TK}, names=['tk', 'day'])
W = W.merge(FE.reset_index(), on=['tk', 'day'], how='left')
y = W.entry.dt.year; W['mon'] = W.entry.dt.strftime('%Y-%m'); W['half'] = np.where(y.isin([2019, 2020, 2021]), 'train', 'test'); W['era'] = np.select([y <= 2019, y <= 2021, y == 2022], ['18-19', '20-21', '22'], '23-26')
def paired(a, b):
    d = (a.groupby('mon').ret.mean() - b.groupby('mon').ret.mean()).dropna()
    return (d.mean() * 1e4, d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else np.nan, len(d))
def groups(D, name):
    if name == 'F1': return D[D.gex < 0], D[D.gex >= 0]
    if name == 'F6': return D[D.term < 0], D[D.term >= 0]
    if name == 'F9':
        a = D[D.gex_short < 0]; b = D[(D.gex_short >= 0) & (D.gex_long < 0)]; return a, b
    col = {'F2': 'r_vanna', 'F3': 'r_charm', 'F4': 'r_pc_vol', 'F5': 'r_iv30', 'F7': 'r_skew', 'F8': 'r_dex'}[name]
    return D[D[col] >= 2 / 3], D[D[col] <= 1 / 3]
NAMES = dict(F1='net gamma < 0 vs >= 0', F2='vanna rank high vs low', F3='charm rank high vs low', F4='put/call volume rank high vs low', F5='ATM IV 30d rank high vs low', F6='term inverted vs not', F7='skew rank high vs low', F8='DEX rank high vs low', F9='short-dated gamma<0 vs only-long-dated<0')
res = []
for h1 in [False, True]:
    D = W[W.h1 == h1]; tf = '1h' if h1 else '15m'; print(f'\n=== {tf}: flips {len(D)} (with features {int(D.gex.notna().sum())}) base mean {D.ret.mean() * 1e4:+.1f}bp ===')
    for f in NAMES:
        a, b = groups(D, f); d, t, n = paired(a, b); out = f'{f} {NAMES[f]:42s} n {len(a):5d}/{len(b):5d}  {a.ret.mean() * 1e4:+6.1f} vs {b.ret.mean() * 1e4:+6.1f}  diff {d:+6.1f}bp t {t:5.2f}'
        sg = []
        for hf in ['train', 'test']:
            d2, t2, _ = paired(groups(D[D.half == hf], f)[0], groups(D[D.half == hf], f)[1]); sg.append(np.sign(d2) if not np.isnan(d2) else 0); out += f' | {hf} {d2:+.1f}/t{t2:.2f}'
        ok = t >= 2 and sg[0] == sg[1] == 1; res.append((tf, f, d, t, ok)); print(out, ' PASS' if ok else '')
        if f != 'F1':
            for nm, m in (('in neg-gamma', D.gex < 0), ('in pos-gamma', D.gex >= 0)):
                a2, b2 = groups(D[m], f); d3, t3, _ = paired(a2, b2); print(f'      {nm:13s} n {len(a2):5d}/{len(b2):5d} diff {d3:+6.1f}bp t {t3:5.2f}')
    print('  by era (bp) F1 neg-pos: ' + ' | '.join(f'{e} {paired(*groups(D[D.era == e], "F1"))[0]:+.0f}' for e in ['18-19', '20-21', '22', '23-26']))
print('\nPASS count:', sum(r[4] for r in res), 'of', len(res))
