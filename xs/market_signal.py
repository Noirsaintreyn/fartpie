"""Oct 27 2026: one market signal from every component - a vote, each +1 / -1 (0 = neutral), all known at the prior close; trade QQQ (and
SPY) close-to-close. Votes are fixed by convention up front (no fitting):
 vec_d   QQQ daily vector bull / bear                     vec_1h  QQQ 1h vector state at the last bar of the day
 volx    volatility cross: 10-day realized vol below / above 60-day (calming = +1)
 har     HAR next-day vol forecast in the bottom / top third of its past year (calm = +1)
 vixts   VIX / VIX3M < 0.95 contango +1, > 1.0 backwardation -1
 credit  HYG / IEF 50-day trend up +1 / down -1          breadth RSP / SPY 50-day trend
 sma200  SPY over / under its 200-day                    usd     dollar (UUP) 50-day down +1 / up -1
 cot     asset managers' E-mini S&P positioning (3-year percentile) < 20% +1, > 80% -1 (crowded)
Signals: each vote alone (long when +1, else cash); vote sum >= k (long) for k = 0..4; exposure scaled by the vote (sum / n, floor 0);
long / short versions. Cash earns 0. Compared with buy-and-hold."""
import glob, os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])
MD = '/tmp/claude-0/ml/macro'; P = pd.read_parquet(f'{MD}/px.parquet'); P.index = pd.to_datetime(P.index).tz_localize(None).normalize()
def etf(t):
    d = pd.read_parquet(f'/mnt/user-data/uploads/fartpie/etf_daily/{t}.parquet'); d = d.set_index(pd.to_datetime(d.datetime).dt.tz_localize(None).dt.normalize()); return d[~d.index.duplicated(keep='last')].sort_index()
OUT = {}
for IDX in ('QQQ', 'SPY'):
    D = etf(IDX); D = D[D.index >= '2015-01-01']; c = D.close
    V = pd.DataFrame(index=D.index)
    V['vec_d'] = vec(D[['open', 'high', 'low', 'close']].reset_index(drop=True))
    f15 = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{IDX}_15m_*.parquet')
    if f15:
        H = load(f15[0]); r1 = agg(H, '1h', '9h30min'); s1 = pd.Series(vec(r1), index=r1.dt); V['vec_1h'] = s1.groupby(s1.index.normalize()).last().reindex(V.index).ffill()
    lr = np.log(c).diff(); V['volx'] = np.where(lr.rolling(10).std() < lr.rolling(60).std(), 1, -1)
    x = np.log(np.log(D.high / D.low).clip(lower=1e-5)); sig = np.exp(-0.6186 + 0.2939 * x + 0.3661 * x.rolling(5).mean() + 0.1783 * x.rolling(22).mean())
    pr = sig.rolling(252, min_periods=120).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True); V['har'] = np.where(pr < 1 / 3, 1, np.where(pr > 2 / 3, -1, 0))
    M = P.reindex(V.index).ffill()
    ts = M['^VIX'] / M['^VIX3M']; V['vixts'] = np.where(ts < .95, 1, np.where(ts > 1.0, -1, 0))
    r50 = lambda s: s / s.shift(50) - 1
    V['credit'] = np.sign(r50(M.HYG / M.IEF)); V['breadth'] = np.sign(r50(M.RSP / M.SPY)); V['sma200'] = np.sign(M.SPY / M.SPY.rolling(200).mean() - 1); V['usd'] = -np.sign(r50(M.UUP))
    C = pd.read_parquet(f'{MD}/tff_all.parquet'); C = C[C.CFTC_Contract_Market_Code.astype(str) == '13874A']; C['rd'] = pd.to_datetime(C['Report_Date_as_YYYY-MM-DD'])
    x = C.groupby('rd').sum(numeric_only=True).sort_index(); net = (x.Asset_Mgr_Positions_Long_All - x.Asset_Mgr_Positions_Short_All) / x.Open_Interest_All
    ci = net.rolling(156, min_periods=52).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True); ci.index = ci.index + pd.Timedelta(days=6)
    ci = ci.reindex(V.index.union(ci.index)).ffill().reindex(V.index); V['cot'] = np.where(ci < .2, 1, np.where(ci > .8, -1, 0))
    V = V.fillna(0).shift(1)                                                # known at the prior close
    ret = c.pct_change(); V['ret'] = ret; OUT[IDX] = V.dropna()
PER = [('2016-06-01', '2019-01-01'), ('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01')]
def st(r):
    eq = (1 + r).cumprod(); y = len(r) / 252; cg = eq.iloc[-1] ** (1 / y) - 1; dd = (eq / eq.cummax() - 1).min(); sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    return cg, dd, sh
for IDX, V in OUT.items():
    votes = [c for c in V.columns if c != 'ret']; V['sum'] = V[votes].sum(axis=1); n = len(votes)
    print(f'\n===== {IDX}  ({n} votes)  CAGR / max DD / Sharpe')
    print(f'{"":<40}' + ''.join(f'{a[:4]}-{b[2:4]:>14}' for a, b in PER))
    def show(nm, pos):
        r = V.ret * pos; out = []
        for a, b in PER:
            k = (V.index >= a) & (V.index < b); cg, dd, sh = st(r[k]); out.append(f'{cg:>+7.1%}/{dd:>4.0%}/{sh:>4.1f}')
        print(f'  {nm:<38}' + ''.join(f'{o:>19}' for o in out) + f'   time in {pos.clip(lower=0).mean():.0%}', flush=True)
    show('buy and hold', pd.Series(1.0, index=V.index))
    for c in votes: show(f'{c} alone (long when +1)', (V[c] > 0).astype(float))
    for k in range(0, 6): show(f'vote sum >= {k} -> long', (V['sum'] >= k).astype(float))
    show('exposure = max(sum, 0) / n', (V['sum'].clip(lower=0) / n))
    show('exposure = (sum + n) / 2n (always some)', (V['sum'] + n) / (2 * n))
    show('long / short by sign of the sum', np.sign(V['sum']))
    if 'vec_1h' in V: show('vec_d + 1h agree long, else cash', ((V.vec_d > 0) & (V.vec_1h > 0)).astype(float))
    show('vec_d and SPY > 200d', ((V.vec_d > 0) & (V.sma200 > 0)).astype(float))
