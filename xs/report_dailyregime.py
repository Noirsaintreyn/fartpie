"""Are NQ-system bad days trend/vol days for other futures? yfinance =F DAILY bars (long history), 2018-2026. For each asset: trend efficiency |c-o|/(h-l), range expansion (h-l)/ATR20(prior), vs NQ system tier (1 terrible .. 5 great). Nov 11 2026."""
import numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
TD = pd.read_pickle(SC + 'tiers_days.pkl'); TD.index = pd.to_datetime(TD.index).normalize(); TD['T'] = TD.tier.str[0].astype(int)
def tstat(x): x = np.asarray(x, float); x = x[~np.isnan(x)]; return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
print('asset  | corr(NQ day pts, eff) | eff on NQ tier1-2 vs tier4-5 (t of diff) | range-expansion on tier1-2 vs tier4-5 | corr(NQ pts, |ret|)')
for t in ('ES=F', 'GC=F', 'SI=F', 'HG=F', 'CL=F', 'NG=F', 'ZN=F', '6E=F', '6J=F', 'ZC=F'):
    d = yf.download(t, start='2018-01-01', interval='1d', progress=False, auto_adjust=False)
    if d.empty: print(t, 'no data'); continue
    d.columns = [c[0] for c in d.columns]; d.index = pd.to_datetime(d.index).normalize()
    rg = d.High - d.Low; eff = (d.Close - d.Open).abs() / rg.replace(0, np.nan); rx = rg / rg.shift(1).rolling(20).mean(); ab = (d.Close / d.Close.shift(1) - 1).abs()
    J = TD.join(pd.DataFrame({'eff': eff, 'rx': rx, 'ab': ab}), how='inner').dropna(subset=['eff', 'rx'])
    lo, hi = J[J['T'] <= 2], J[J['T'] >= 4]
    dt = (lo.eff.mean() - hi.eff.mean()) / np.sqrt(lo.eff.var() / len(lo) + hi.eff.var() / len(hi)); dr = (lo.rx.mean() - hi.rx.mean()) / np.sqrt(lo.rx.var() / len(lo) + hi.rx.var() / len(hi))
    print(f'{t:5} | {J.pts.corr(J.eff):+.2f} | {lo.eff.mean():.2f} vs {hi.eff.mean():.2f} (t {dt:+.1f}) | {lo.rx.mean():.2f} vs {hi.rx.mean():.2f} (t {dr:+.1f}) | {J.pts.corr(J.ab):+.2f}  n {len(J)}')
