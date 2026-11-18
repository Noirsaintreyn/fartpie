"""Nov 18 2026 report 331. THE INPUTS THE BANK / HEDGE-FUND PAPERS ACTUALLY USE, on the small-cap sleeve.  Declared BEFORE running.
Source papers: Deutsche Bank positioning & flows (CTA / vol-control / systematic positioning percentiles, breadth, small-cap positioning, call/put), GS Flow Show & desk notes (flows, oil, copper, gold, dollar), Citi rotation note ('compare revisions, credit spreads, financing conditions, concentration and breadth'), GSAM Market Pulse ('earnings drivers, copper/gold, energy').  Proprietary positioning series are not available; each is replaced by a transparent price proxy (stated).
Each test has a FIXED direction, a transmission path (Harvest rule), and the same strict split.  Target: next-month return of the equal-weight 753-name universe (EW) and of IWM; signal at month-end t, return t+1.
 B1 CTA-flow proxy on IWM (12 trend models, window 21/63/126 x stop 3/4/5/6 ATR-equivalents, position mean of +1/-1/0; here the close-only variant): 1-month CHANGE in aggregate position > 0 (CTAs buying) -> up (flow persistence; Bouchaud: trend flows are slow-decaying);
     and EXTREME: position in its top decile of the trailing 5y -> LOWER forward return (DB: 'vulnerable to any vol shocks' when systematic positioning is in the top decile).
 B2 breadth washout: share of the 753 names above their own 200d average in the BOTTOM tercile (train cuts) -> HIGHER forward return (capitulation); breadth thrust (3m change in share > 0) -> up.   Path: forced-selling exhaustion vs participation.
 B3 broadening: RSP/SPY (equal- vs cap-weight S&P) 3m change > 0 -> small caps up.   Path: leadership broadening beyond megacap concentration (Citi: 'track concentration and breadth').
 B4 vol-control crowding: SPY 20d realized vol in the bottom tercile of its trailing 252d range (vol-control funds fully levered) -> LOWER forward return of EW (leverage can only be cut).
 B5 energy cost: oil (USO) 3m change < 0 -> small caps up (GSAM/Citi: falling energy helps users; small caps are mostly users).
 B6 copper/gold (growth proxy) 3m change > 0 -> small caps up (GSAM: copper support = growth).
Tests are time-series, ~100 months; six tests => one |t|>2 is expected by chance.  Admit only with the same sign in both halves AND |t|>=2 in one."""
import numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
C = pd.read_parquet(SP + 'sm_close.parquet').dropna(how='all', axis=1); MP = pd.read_parquet(SP + 'macro_px.parquet')
X = yf.download(['RSP', 'USO', 'HG=F', 'GC=F'], start='2015-01-01', end='2026-11-17', auto_adjust=True, progress=False)['Close']; MP = MP.join(X)
R = C.pct_change().replace([np.inf, -np.inf], np.nan).clip(-.9, 3); EW = (1 + R.mean(axis=1)).cumprod()
# --- B1 CTA proxy on IWM (close-only ATR proxy = 14d mean |close change|)
def cta(px):
    a = px.diff().abs().rolling(14).mean(); pos = []
    for w in (21, 63, 126):
        pk, tr_ = px.rolling(w).max(), px.rolling(w).min()
        for k in (3, 4, 5, 6):
            lo = px > pk - k * a; sh = px < tr_ + k * a; pos.append(np.where(lo & ~sh, 1.0, np.where(sh & ~lo, -1.0, 0.0)))
    return pd.Series(np.mean(pos, axis=0), index=px.index)
pos = cta(MP.IWM.dropna()); M = pd.DataFrame({'EW': EW, 'IWM': MP.IWM}).resample('ME').last(); fw = M.pct_change().shift(-1)
dpos = pos.resample('ME').last().diff(); ppos = pos.resample('ME').last(); ptop = ppos.rolling(60, min_periods=36).apply(lambda s: (s.iloc[-1] >= s.quantile(.9)), raw=False).astype(float)
above = (C > C.rolling(200).mean()).where(C.notna()).mean(axis=1).resample('ME').last(); MM = MP.resample('ME').last()
vol = np.log(MP.SPY.dropna()).diff().rolling(20).std().mul(np.sqrt(252)); vpct = vol.rolling(252).apply(lambda s: (s <= s.iloc[-1]).mean(), raw=False).resample('ME').last()
tr = lambda ix: (ix >= '2019-01-01') & (ix < '2022-01-01'); te = lambda ix: (ix >= '2018-01-01') & ~tr(ix)
q = above[tr(above.index)].quantile([1 / 3, 2 / 3]).values; vq = vpct[tr(vpct.index)].quantile(1 / 3)
SIG = {'B1a CTA buying (dpos>0)': dpos > 0, 'B1b CTA extreme long (top decile)': ptop > 0, 'B2a breadth washout (bottom tercile)': above <= q[0], 'B2b breadth thrust (3m chg>0)': (above - above.shift(3)) > 0,
       'B3 broadening (RSP/SPY 3m up)': ((MM.RSP / MM.SPY) / (MM.RSP / MM.SPY).shift(3) - 1) > 0, 'B4 vol-control crowded (low vol pct)': vpct <= vq, 'B5 oil falling (3m)': (MM.USO / MM.USO.shift(3) - 1) < 0,
       'B6 copper/gold rising (3m)': ((MM['HG=F'] / MM['GC=F']) / (MM['HG=F'] / MM['GC=F']).shift(3) - 1) > 0}
DIR = {k: (-1 if k.startswith(('B1b', 'B4')) else 1) for k in SIG}
def tt(d, se): return f'{d * 100:+5.2f}% (t{d / se:+4.1f})'
print('Next-month return when signal ON minus OFF; sign shown AS DECLARED (positive = the paper is right).  [train 2019-21 | test 2018+2022-26]')
for k, s in SIG.items():
    s = s.reindex(fw.index); out = []
    for nm in ('EW', 'IWM'):
        m = fw[nm]; r = []
        for lab, f in (('tr', tr), ('te', te)):
            ix = m.index[f(m.index) & m.notna() & s.notna()]; a, b = m[ix][s[ix].astype(bool)], m[ix][~s[ix].astype(bool)]
            r.append('n/a' if len(a) < 5 or len(b) < 5 else f'{tt(DIR[k] * (a.mean() - b.mean()), np.sqrt(a.var() / len(a) + b.var() / len(b)))} on{len(a)}/off{len(b)}')
        out.append(f'{nm}: {r[0]} | {r[1]}')
    print(f'  {k:38s} ' + '   '.join(out))
print('\nCTA proxy sanity: corr(IWM position, next-month IWM)', round(pd.concat([ppos, fw.IWM], axis=1).dropna().corr().iloc[0, 1], 3), '| share of months long', round((ppos > 0).mean(), 2))
