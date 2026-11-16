"""Report 198: does flow speed at the flip identify BIG large-cap flips? DECLARED before results.
Tick counts do not exist for the stocks (15m OHLCV only), so the tick-rate z-score is replaced by its volume-clock analogue: RVOL = volume of the signal bar (the 15m bar that closed just before entry) / median volume of the same 15m slot over the previous 20 sessions (shifted, no lookahead); z = log RVOL. Mechanism: a flip on unusually fast volume = participation by large players -> bigger follow-through.
Sample: xs/ext/wide_T.parquet flips with sp & ratio>=1.5 (the live universe), train entry<2022, test 2022+. H: IC(logRVOL, ret)>0 same sign both eras AND top tercile mean ret > bottom tercile both eras; BIG = ret>=5% (also top-decile ret) AUC>.5 both eras. PASS = same sign, |t|>=2, n>=30 both eras.
Report both tracks pooled and split 15m/1H. Live-book check: tie-break by RVOL (ratio first) not tested here."""
import glob, os, numpy as np, pandas as pd
from scipy import stats
W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); T = W[W.sp & (W.ratio >= 1.5)].copy().reset_index(drop=True)
rv = {}
for f in glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/*_15m_*.parquet'):
    tk = os.path.basename(f).split('_15m_')[0]
    if tk not in set(T.tk): continue
    B = pd.read_parquet(f, columns=['datetime', 'volume']); B['dt'] = pd.to_datetime(B.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = B.dt.dt.hour * 60 + B.dt.dt.minute; B = B[(m >= 570) & (m < 960)].drop_duplicates('dt').copy(); B['slot'] = (B.dt.dt.hour * 60 + B.dt.dt.minute); B = B.sort_values('dt')
    B['med'] = B.groupby('slot').volume.transform(lambda s: s.shift(1).rolling(20, min_periods=10).median()); B['rv'] = np.log((B.volume + 1) / (B.med + 1))
    rv[tk] = B.set_index('dt').rv
def feat(tk, e):
    s = rv.get(tk)
    if s is None: return np.nan
    k = s.index.searchsorted(e) - 1
    return s.iloc[k] if k >= 0 and (e - s.index[k]) <= np.timedelta64(30, 'm') else np.nan
T['rv'] = [feat(a, b) for a, b in zip(T.tk, T.entry)]; T = T.dropna(subset=['rv']); T['era'] = np.where(T.entry < '2022-01-01', 'train', 'test')
def tt(x): x = np.asarray(x, float); return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 5 and x.std() > 0 else np.nan
print('flips with RVOL:', len(T), T.groupby('era').size().to_dict())
for nm, S in (('ALL', T), ('15m track', T[~T.h1]), ('1H track', T[T.h1])):
    print(f'\n--- {nm}')
    for e in ('train', 'test'):
        s = S[S.era == e]; ic = s.rv.corr(s.ret, method='spearman'); q = s.rv.quantile([1 / 3, 2 / 3]).values; lo = s[s.rv <= q[0]].ret; hi = s[s.rv >= q[1]].ret
        big = (s.ret >= .05).astype(int); auc = stats.mannwhitneyu(s.rv[big == 1], s.rv[big == 0]).statistic / (big.sum() * (1 - big).sum()) if big.sum() > 5 else np.nan
        diff = hi.mean() - lo.mean(); se = np.sqrt(hi.var() / len(hi) + lo.var() / len(lo))
        print(f'  {e:5s} n{len(s):5d} IC {ic:+.3f} | low-RVOL mean {lo.mean()*100:5.2f}% hi-RVOL {hi.mean()*100:5.2f}% diff {diff*100:+5.2f}pp t{diff/se:+5.1f} | BIG(>=5%) rate {big.mean():.1%} AUC {auc:.3f} | win {np.mean(s.ret>0):.0%}')
T.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/flip_rvol198.pkl')
