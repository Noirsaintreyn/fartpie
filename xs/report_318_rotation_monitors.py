"""Nov 18 2026 report 318. Desk-doc D1a (sector rotation) cross-asset monitors, built exactly as defined there, from data on disk (Fed-model yield gap untestable: no PE/yield series): (M1) copper/gold ratio 20d rate of change > +2% (reflation); (M2) DXY below its 60d MA, proxied by FXE (euro, ~57% of DXY) ABOVE its 60d MA (weak dollar); (M3) SOX vs QQQ: semiconductor basket (NVDA AVGO AMD MU LRCX QCOM, equal weight) outperforming QQQ on each of the last 5 sessions. All lagged one day. Score = count of the three.
THEORY (declared): institutional flows follow macro regime (doc Part I); reflation/weak-dollar/semi-leadership = risk-on cyclical regime, so (a) screener flip returns are higher and (b) the NQ algo trend-day engine earns more when the score is high. Why it would fail: our flips are reversal entries after bear legs (report 302 found low index VR better), the monitors are slow, and our universe is megacap. Train/test as elsewhere; terciles not needed (score 0-3)."""
import numpy as np, pandas as pd, glob, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; UP = '/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/'
def dclose(path):
    H = pd.read_parquet(path); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H['day'] = H.dt.dt.normalize(); return H.groupby('day').close.last()
cu = dclose(glob.glob(SP + 'lsefut/lse_ohlcv_cache/COPPER.F_15m_*')[0]); au = dclose(glob.glob(SP + 'lsefut/lse_ohlcv_cache/GOLD.F_15m_*')[0]); fx = dclose(glob.glob(UP + 'FXE_15m_*')[0]); qq = dclose(glob.glob(UP + 'QQQ_15m_*')[0])
sem = pd.concat([dclose(glob.glob(UP + f'{t}_15m_*')[0]).pct_change() for t in ('NVDA', 'AVGO', 'AMD', 'MU', 'LRCX', 'QCOM')], axis=1).mean(axis=1)
cg = (cu / au); M1 = (cg / cg.shift(20) - 1) > .02; M2 = fx > fx.rolling(60).mean(); rel = sem - qq.pct_change(); M3 = (rel > 0).rolling(5).sum() == 5
F = pd.concat([M1.rename('M1'), M2.rename('M2'), M3.rename('M3')], axis=1).dropna().astype(int).shift(1); F['score'] = F.sum(axis=1); F = F.dropna()
print('monitor on-share:', F[['M1', 'M2', 'M3']].mean().round(2).to_dict(), '| score dist', F.score.value_counts(normalize=True).sort_index().round(2).to_dict())
D = pd.read_parquet(SP + 'd312.parquet')[['ALL', 'M1', 'V15a', 'ON+', 'ON-', 'tr', 'rng', 'adr']].rename(columns={'M1': 'm1leg'}).join(F, how='inner')
print('\n(a) NQ algo pts/day by monitor state (train | test)')
for c in ('M1', 'M2', 'M3'):
    print(f'  {c} off/on ALL: {D[D.tr & (D[c] == 0)].ALL.mean():+6.1f}|{D[~D.tr & (D[c] == 0)].ALL.mean():+6.1f}  /  {D[D.tr & (D[c] == 1)].ALL.mean():+6.1f}|{D[~D.tr & (D[c] == 1)].ALL.mean():+6.1f}   (n on {int((D.tr & (D[c] == 1)).sum())}|{int((~D.tr & (D[c] == 1)).sum())})')
for s in range(4):
    z = D.score == s; print(f'  score {s}: ALL {D[D.tr & z].ALL.mean():+6.1f}|{D[~D.tr & z].ALL.mean():+6.1f}  n{int((D.tr & z).sum())}|{int((~D.tr & z).sum())}')
B = pd.read_parquet(SP + 'b302.parquet'); B['d'] = pd.to_datetime(B.entry).dt.tz_localize(None).dt.normalize() if getattr(pd.to_datetime(B.entry).dt, 'tz', None) is not None else pd.to_datetime(B.entry).dt.normalize()
B = B.join(F, on='d').dropna(subset=['score']); print('\n(b) screener flip ret % by monitor state (train | test): mean (t) n')
def cell(m, lab):
    o = []
    for nm, t in (('tr', B.tr), ('te', ~B.tr)):
        r = B[t & m].ret; o.append(f'{nm} {100 * r.mean():+.2f} (t{r.mean() / (r.std() / np.sqrt(len(r))) if len(r) > 2 else np.nan:+.1f}) n{len(r)}')
    print(f'  {lab:14s}', ' | '.join(o))
for c in ('M1', 'M2', 'M3'): cell(B[c] == 0, c + ' off'); cell(B[c] == 1, c + ' on')
for s in range(4): cell(B.score == s, f'score {s}')
