"""Nov 12 2026: what flags a sub-$5 stock that JUMPS in the next 5 sessions? DECLARED BEFORE RESULTS. Literature-driven, hypothesis-first, no fitting (single pre-declared features, rank AUC + top-decile lift, both halves):
 H0 control  past volatility (ATR14/close) - the vol-clustering baseline Rey does not want to rely on
 H1 compression  5-day range / 20-day range LOW (energy builds before the expansion)                      [vol-of-vol / squeeze]
 H2 dry-up       3-day mean volume / 20-day mean volume LOW (supply exhausted)
 H3 attention    today's volume / 20-day mean volume HIGH (attention precedes jumps; Barber-Odean type)
 H4 distance below the 52-week high HIGH (the arXiv 2602.06198 microcap insider paper's main gradient; beaten-down names react most)
 H5 lottery      MAX = largest 1-day return of the last 21 sessions HIGH (Bali-Cakici-Whitelaw: lottery demand persists)
 H6 short pressure  5-day short-sale volume share (FINRA Reg SHO) HIGH (squeeze fuel)
 H7 trend        close above its 50-day EMA (our vector context) and 20-day return HIGH (momentum)
Universe: 753 survivors (biased), daily bars built from the 1h cache, close in [$0.5, $5), 20-day mean dollar volume >= $200k, since 2024-03.
Labels from t+1..t+5: UP = max high >= +20% over the day-t close; DOWN = min low <= -20%. AUC on UP (0.5 = no info); lift = hit rate in the top decile of the feature / base rate. Train < 2025-07-01 | test after."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import midsmall_screener as M
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb')); FN = pd.read_pickle(SC + 'finra_sv.pkl')
FN['Date'] = pd.to_datetime(FN.Date.astype(str), errors='coerce'); FN = FN.dropna(subset=['Date']); SV = (FN.pivot_table(index='Date', columns='Symbol', values='ShortVolume', aggfunc='sum') / FN.pivot_table(index='Date', columns='Symbol', values='TotalVolume', aggfunc='sum'))
rows = []
for tk, Y in data.items():
    Y = M.closed_only(Y.dropna(subset=['open', 'close']))
    if len(Y) < 400: continue
    D = Y.groupby(Y.index.normalize()).agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last'), v=('volume', 'sum'))
    if len(D) < 300: continue
    c, h, l, v = D.c, D.h, D.l, D.v; pc = c.shift(); tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1); ret = c.pct_change()
    F = pd.DataFrame({'atr': tr.rolling(14).mean() / c, 'comp': (h.rolling(5).max() - l.rolling(5).min()) / (h.rolling(20).max() - l.rolling(20).min()), 'dry': v.rolling(3).mean() / v.rolling(20).mean(), 'att': v / v.rolling(20).mean(),
                      'dist': 1 - c / h.rolling(252, min_periods=120).max(), 'maxr': ret.rolling(21).max(), 'trend': (c > c.ewm(span=50, adjust=False).mean()).astype(float), 'mom': c / c.shift(20) - 1, 'dv': (c * v).rolling(20).mean(), 'px': c})
    fh = pd.concat([h.shift(-k) for k in range(1, 6)], axis=1).max(axis=1); fl = pd.concat([l.shift(-k) for k in range(1, 6)], axis=1).min(axis=1); F['up'] = fh / c - 1 >= .20; F['dn'] = fl / c - 1 <= -.20; F['ok'] = fh.notna()
    if tk in SV.columns: F['svr'] = SV[tk].reindex(F.index).rolling(5, min_periods=3).mean()
    F['tk'] = tk; rows.append(F)
P = pd.concat(rows); P = P[P.ok & (P.px >= .5) & (P.px < 5) & (P.dv >= 2e5) & (P.index >= '2024-03-01')].dropna(subset=['atr', 'comp', 'dry', 'att', 'maxr']); P['up'] = P.up.astype(bool); P['dn'] = P.dn.astype(bool); P['tr'] = P.index < '2025-07-01'
print(f'panel {len(P):,} stock-days, {P.tk.nunique()} tickers | base rate UP>=20%/5d {P.up.mean():.2%} (train {P[P.tr].up.mean():.2%} test {P[~P.tr].up.mean():.2%}), DOWN<=-20% {P.dn.mean():.2%}')
def auc(x, y):
    d = pd.DataFrame({'x': x, 'y': y}).dropna(); r = d.x.rank(); n1 = d.y.sum(); n0 = len(d) - n1; return (r[d.y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
def lift(x, y, hi=True):
    d = pd.DataFrame({'x': x, 'y': y}).dropna(); q = d.x.quantile(.9 if hi else .1); s = d[d.x >= q] if hi else d[d.x <= q]; return s.y.mean() / d.y.mean() if d.y.mean() > 0 else np.nan
feats = [('H0 past volatility (ATR%) HIGH', 'atr', True), ('H1 compression LOW (5d/20d range)', 'comp', False), ('H2 volume dry-up LOW (3d/20d)', 'dry', False), ('H3 attention: volume today/20d HIGH', 'att', True), ('H4 below 52w high HIGH', 'dist', True), ('H5 lottery MAX21 HIGH', 'maxr', True), ('H6 short-sale vol share HIGH', 'svr', True), ('H7a close above EMA50', 'trend', True), ('H7b 20d momentum HIGH', 'mom', True)]
print(f"{'feature':<38} | UP AUC train/test  lift top-decile tr/te | DOWN AUC tr/te lift tr/te | any-big |move| AUC test")
for nm, k, hi in feats:
    sg = 1 if hi else -1; o = []
    for lab in ('up', 'dn'):
        a = [auc(sg * P[m][k], P[m][lab]) for m in (P.tr, ~P.tr)]; L = [lift(P[m][k], P[m][lab], hi) for m in (P.tr, ~P.tr)]; o.append((a, L))
    big = P.up | P.dn; ab = auc(sg * P[~P.tr][k], big[~P.tr])
    print(f"{nm:<38} | {o[0][0][0]:.3f} / {o[0][0][1]:.3f}   {o[0][1][0]:.2f}x / {o[0][1][1]:.2f}x | {o[1][0][0]:.3f} / {o[1][0][1]:.3f}  {o[1][1][0]:.2f}x / {o[1][1][1]:.2f}x | {ab:.3f}")
print('== simple combos (top-decile of an equal-weight rank score), UP hit rate vs base, train / test')
R = P.copy()
for k, hi in (('atr', True), ('comp', False), ('dry', False), ('att', True), ('dist', True), ('maxr', True), ('svr', True)): R['r_' + k] = (R[k] if hi else -R[k]).groupby(R.index).rank(pct=True)
for nm, cols in (('compression + attention + dist', ['comp', 'att', 'dist']), ('att + dist + maxr', ['att', 'dist', 'maxr']), ('compression + dry-up + dist', ['comp', 'dry', 'dist']), ('att + maxr + svr', ['att', 'maxr', 'svr']), ('vol(ATR) + att + dist', ['atr', 'att', 'dist']), ('vol(ATR) alone', ['atr'])):
    cc = ['r_' + c for c in cols]; Z = R.dropna(subset=cc); Z['s'] = Z[cc].mean(axis=1); out = []
    for m in (Z.tr, ~Z.tr):
        z = Z[m]; q = z.s.quantile(.9); t = z[z.s >= q]; out.append((t.up.mean(), z.up.mean(), t.dn.mean(), len(t)))
    print(f'  {nm:<32} UP top-decile {out[0][0]:.1%} vs {out[0][1]:.1%} base (train) | {out[1][0]:.1%} vs {out[1][1]:.1%} (test) | DOWN rate {out[1][2]:.1%} | n(test top decile) {out[1][3]:,}')
