"""Nov 18 2026 report 328. PAPER CLAIMS (macro / sector rotation) applied to the small-cap sleeve.  Declared BEFORE running.
Papers: 'The Macro Framework' (yield curve, credit, flows, sector rotation), D1a sector rotation, D5 macro/micro, Harvest 'Macroeconomic accounting and transmission', Harvest 'Does a regime label improve the model?', GS/Citi rotation notes, Moskowitz-Grinblatt industry momentum.
Harvest rule: a macro number matters only through a stated TRANSMISSION path, and a regime label must improve decisions out of sample.  So each test names its path and fixed direction:
 M1 curve steepening (10y - 13w, 3m change > 0) -> small caps UP next month (early-cycle: credit availability to bank-dependent small firms; steep curve = bank lending margin).      Path: financing cost/availability.
 M2 credit easing (HYG/LQD 3m change > 0) -> small caps up (small caps are lower-rated, floating-rate borrowers).                                                          Path: credit spread -> distress premium.
 M3 yields falling (10y 3m change < 0) -> small caps up (floating-rate debt service, discount rate on long-duration growth).                                                 Path: debt service + duration.
 M4 cyclical/defensive ratio rising (mean of XLY/XLP and XLI/XLU 3m change > 0) -> small caps up (domestic cyclical beta).                                                   Path: earnings cyclicality.
 M5 high VIX (above its trailing 252d median) -> HIGHER small-cap forward return (risk premium widens; contrarian) -- the weakest-theory one, kept because desks quote it.
 M6 sector momentum (Moskowitz-Grinblatt): 11 sector ETFs, 6m return top-3 minus bottom-3 -> next month.                                                                      Path: slow diffusion of industry news.
Tests are TIME-SERIES with ~100 months, so each has few independent observations; with 6 tests one |t|>2 is expected by chance.  Admit only if same sign in train (2019-21) and test (2018, 2022-26) AND the transmission path is also visible in the sleeve it should hit.
Target series: IWM (small caps) next-month return, and IWM minus SPY.  Signal known at month-end t, return month t+1 (no overlap)."""
import numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
T = ['IWM', 'SPY', '^TNX', '^IRX', '^VIX', 'HYG', 'LQD', 'XLY', 'XLP', 'XLI', 'XLU', 'XLK', 'XLF', 'XLE', 'XLV', 'XLB', 'XLRE', 'XLC']
D = yf.download(T, start='2015-01-01', end='2026-11-17', auto_adjust=True, progress=False)['Close']; D.to_parquet(SP + 'macro_px.parquet')
M = D.resample('ME').last(); r = M.pct_change(); fw = r.shift(-1)
y10, y3m = M['^TNX'], M['^IRX']; slope = y10 - y3m
sig = {'M1 curve steepening': (slope - slope.shift(3)) > 0, 'M2 credit easing (HYG/LQD up)': ((M.HYG / M.LQD) / (M.HYG / M.LQD).shift(3) - 1) > 0, 'M3 yields falling': (y10 - y10.shift(3)) < 0,
       'M4 cyclicals/defensives up': (((M.XLY / M.XLP) / (M.XLY / M.XLP).shift(3) - 1) + ((M.XLI / M.XLU) / (M.XLI / M.XLU).shift(3) - 1)) > 0, 'M5 high VIX': M['^VIX'] > D['^VIX'].rolling(252).median().resample('ME').last()}
tr = lambda ix: (ix >= '2019-01-01') & (ix < '2022-01-01'); ts = lambda ix: (ix >= '2018-01-01') & ~tr(ix)
def tt(s):
    s = s.dropna(); return '   n/a' if len(s) < 8 else f'{s.mean() * 100:+5.2f}% (t{s.mean() / (s.std() / np.sqrt(len(s))):+4.1f},n{len(s):3d})'
print('Next-month IWM return when signal ON minus OFF (diff of means), and IWM-SPY.   [train | test]   base IWM monthly mean: train', tt(fw.IWM[tr(fw.index)]), '| test', tt(fw.IWM[ts(fw.index)]))
for k, s in sig.items():
    s = s.reindex(fw.index); out = []
    for nm, m in (('IWM', fw.IWM), ('IWM-SPY', fw.IWM - fw.SPY)):
        for lab, f in (('tr', tr), ('te', ts)):
            ix = m.index[f(m.index) & m.notna() & s.notna()]; a, b = m[ix][s[ix]], m[ix][~s[ix]]
            if len(a) < 5 or len(b) < 5: out.append(f'{nm} {lab}  n/a'); continue
            d = a.mean() - b.mean(); se = np.sqrt(a.var() / len(a) + b.var() / len(b)); out.append(f'{nm:7s} {lab} {d * 100:+5.2f}% (t{d / se:+4.1f}, on {len(a)}/off {len(b)})')
    print(f'  {k:30s} ' + ' | '.join(out))
# M6 sector momentum
S = ['XLY', 'XLP', 'XLI', 'XLU', 'XLK', 'XLF', 'XLE', 'XLV', 'XLB']; mom = (M[S] / M[S].shift(6) - 1); lsr = []
for t in mom.index[:-1]:
    x = mom.loc[t].dropna()
    if len(x) < 8: continue
    o = x.sort_values(); lsr.append((t, fw.loc[t, o.index[-3:]].mean() - fw.loc[t, o.index[:3]].mean()))
L = pd.Series(dict(lsr)); print('\n  M6 sector momentum top3-bottom3 next month: train', tt(L[tr(L.index)]), '| test', tt(L[ts(L.index)]))
# overlay: what would timing have done?  IWM held only when the signal is on, else cash (0), vs always IWM
print('\nTiming overlay: hold IWM only when signal ON (else cash), monthly mean | vs always-IWM  [train | test]')
for k, s in sig.items():
    s = s.reindex(fw.index); ov = fw.IWM.where(s, 0); print(f'  {k:30s} train {tt(ov[tr(ov.index) & fw.IWM.notna()])} vs {tt(fw.IWM[tr(fw.index)])[:7]} | test {tt(ov[ts(ov.index) & fw.IWM.notna()])} vs {tt(fw.IWM[ts(fw.index)])[:7]}')
