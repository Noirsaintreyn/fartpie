"""Weekly small-cap scan from lowpx_daily (through last saved day): (1) fresh vector bull flips (last 3 sessions), (2) big movers (+15% on >=3x volume) in the last 5 sessions with vector state, (3) names within 3% of their bull trigger (about to flip). $1-10, 20d median dollar volume >= $500k. Flip candidates ranked by the frozen report_90 logistic (rank features)."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; DD = pickle.load(open(SC + 'lowpx_daily.pkl', 'rb')); sic = pd.read_csv(SC + 'ext/sic.csv').drop_duplicates('tk').set_index('tk')
rows = []; flipdates = {}
for tk, q in DD.items():
    q = q[['Open', 'High', 'Low', 'Close', 'Volume']].copy(); q.columns = ['open', 'high', 'low', 'close', 'volume']; q = q[~q.index.duplicated()].dropna(); q.index = pd.to_datetime(q.index).tz_localize(None).normalize()
    if len(q) < 200: continue
    o = q.open.values; h = q.high.values; l = q.low.values; c = q.close.values; v = q.volume.values.astype(float); n = len(q); reg, lev, atr, _ = run_state_machine(q.assign(datetime=q.index)); i = n - 1
    dv = pd.Series(c * v).rolling(20).median().values; vm = pd.Series(v).rolling(20).mean().shift(1).values
    if not (1 <= c[i] < 10 and dv[i] >= 5e5): continue
    tr = np.maximum(h - l, np.maximum(abs(h - np.r_[c[0], c[:-1]]), abs(l - np.r_[c[0], c[:-1]]))); a10 = pd.Series(tr).rolling(10).mean().values; a60 = pd.Series(tr).rolling(60).mean().values; lr = pd.Series(np.log(c)).diff(); lo60 = pd.Series(l).rolling(60).min().values
    last_flip = max([k for k in range(n - 1, 0, -1) if reg[k] == 1 and reg[k - 1] != 1][:1] or [-1]); age = i - last_flip if reg[i] == 1 and last_flip >= 0 else None
    r1 = c / np.r_[np.nan, c[:-1]] - 1; mover = [(q.index[k].date(), round(r1[k] * 100), round(v[k] / vm[k], 1)) for k in range(n - 5, n) if r1[k] >= .15 and v[k] >= 3 * vm[k]]
    T = lev[i] + .35 * atr[i] if reg[i] != 1 and not np.isnan(lev[i]) else np.nan
    rows.append(dict(tk=tk, px=c[i], date=q.index[i].date(), state='BULL' if reg[i] == 1 else 'BEAR', age=age, mover=mover, dist_to_trig=(T / c[i] - 1) if T == T else np.nan, level=lev[i], risk_to_level=(c[i] - lev[i]) / c[i] if reg[i] == 1 else np.nan, VOLX=v[i] / vm[i], VR=a10[i] / a60[i], EXT=c[i] / lo60[i] - 1, R60=c[i] / c[i - 60] - 1, BLOW=lr.rolling(20).std().values[i] / lr.rolling(120).std().values[i], LDV=np.log(dv[i]), LPX=np.log(c[i]), dv=dv[i], sic2=(sic.sic.get(tk, np.nan) // 100) if tk in sic.index else np.nan, flipday=q.index[last_flip].date() if last_flip >= 0 and reg[i] == 1 else None))
R = pd.DataFrame(rows); print('as of', R.date.max(), 'names', len(R), 'bull', (R.state == 'BULL').sum())
fl = R[(R.state == 'BULL') & (R.age.notna()) & (R.age <= 3)].copy(); recent = R[(R.state == 'BULL') & (R.age.notna()) & (R.age <= 10)]
fl['PEERS'] = [((recent.sic2 == s) & (recent.tk != t)).sum() if s == s else np.nan for s, t in zip(fl.sic2, fl.tk)]; R['PEERS'] = R.sic2.map(recent.groupby('sic2').size()).fillna(0)
co = dict(VOLX=.22, PEERS=-.22, VR=.21, EXT=.5, R60=.03, BLOW=.08, LDV=-.13, LPX=-.14)
def score(z, ref): return sum(w * z[f].rank(pct=True) for f, w in co.items() if f in z) 
Rr = R.copy(); Rr['PEERS'] = R['PEERS']; fl['score'] = score(fl.assign(PEERS=fl.PEERS.fillna(0)), None); fl = fl.sort_values('score', ascending=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
cols = ['tk', 'px', 'flipday', 'age', 'VOLX', 'EXT', 'VR', 'PEERS', 'dv', 'risk_to_level', 'score']; print('\n(1) FRESH BULL FLIPS (last 3 sessions), ranked by frozen score:'); print(fl[cols].round(2).to_string(index=False))
mv = R[R.mover.map(len) > 0].sort_values('px'); print('\n(2) MOVERS last 5 sessions (date, % day, volume x) with vector state:'); print(mv[['tk', 'px', 'state', 'age', 'mover', 'risk_to_level', 'dv']].round(2).to_string(index=False))
nr = R[(R.state == 'BEAR') & (R.dist_to_trig.between(0, .03))].sort_values('dist_to_trig'); print('\n(3) BEAR names within 3% of their bull trigger (could flip on a close above):'); print(nr[['tk', 'px', 'dist_to_trig', 'VOLX', 'EXT', 'dv']].round(3).to_string(index=False))
