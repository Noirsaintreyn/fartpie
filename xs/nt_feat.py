"""Nov 2 2026: shared features for the next research round (outside review, 7 proposals). HYPOTHESES DECLARED BEFORE ANY RESULT:
 H6 joint-jump tie-break  When more same-bar flips arrive than free slots, taking the ones with the LEAST co-jump history with the
                          positions already held (shared +2 sigma up days, trailing 250d) raises return/DD.
 H2 signed gap tails      A walk-forward forecast of P(next overnight gap > +2 sigma) - P(gap < -2 sigma) ranks trades: higher = better.
 H3 peer lead-lag         A flip whose sector peers have already moved up more than it has (since the prior close, at the entry bar;
                          and over the prior 1 / 5 days vs its sector ETF) is a lagging reaction: higher residual = better trade.
 H1 competing risks       In-trade: at each 10:00 checkpoint after entry, a walk-forward model of the VALUE OF HOLDING (live return
                          minus exit-now return) exits trades with a predicted negative value; this beats the live exits.
 H5 jump types            A multiclass model of the outcome type (none / gap-led jump / intraday jump / market-wide jump) gives a
                          better ranker than the options ratio and than a single binary jump model.
 PASS RULE (all): beats live return/DD in >= 70% of 40 matched runs in BOTH 2020-21 and 2022-26 (models walk-forward by year,
 so the first test year is 2020), AND a neighbouring setting beats live in >= 60%. Anything that passes goes to forward paper-tracking
 only (no untouched period is left). Outputs: /tmp/claude-0/ml/nt_X.parquet (trades + entry features), nt_CK.parquet (checkpoints)."""
import glob, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
LSE = '/mnt/user-data/uploads/fartpie/lse_clean'; ETF = '/mnt/user-data/uploads/fartpie/etf_daily'; C2 = 2e-4
X = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet').sort_values('entry').reset_index(drop=True)
SEC = json.load(open('/home/claude/fartpie/sp500_sectors.json'))
ETFMAP = {'Information Technology': 'XLK', 'Consumer Discretionary': 'XLY', 'Industrials': 'XLI', 'Financials': 'XLF', 'Communication Services': 'XLC',
          'Consumer Staples': 'XLP', 'Health Care': 'XLV', 'Energy': 'XLE', 'Materials': 'XLB', 'Utilities': 'XLU', 'Real Estate': 'XLRE'}
def bars(tk):
    f = glob.glob(f'{LSE}/{tk}_15m_*.parquet')
    if not f: return None
    H = pd.read_parquet(f[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    H['m'] = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(H.m >= 570) & (H.m < 960) & (H.volume > 0)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
    H['day'] = H.dt.dt.normalize(); return H
TKS = sorted(set(X.tk) | {'QQQ'}); B, D = {}, {}
for tk in TKS:
    H = bars(tk)
    if H is None: continue
    B[tk] = H; g = H.groupby('day'); d = pd.DataFrame({'o': g.open.first(), 'c': g.close.last(), 'hi': g.high.max()})
    d['pc'] = d.c.shift(1); d['r'] = d.c / d.pc - 1; d['sd'] = d.r.rolling(20).std().shift(1)
    d['on'] = np.log(d.o / d.pc); d['sdon'] = d.on.rolling(60).std().shift(1); z = d.on / d.sdon
    d['upg'] = (z > 2).astype(float); d['dng'] = (z < -2).astype(float); d['big'] = (d.r > 2 * d.sd).astype(float)
    for w in (60, 250): d[f'fup{w}'] = d.upg.rolling(w).mean().shift(1); d[f'fdn{w}'] = d.dng.rolling(w).mean().shift(1)
    d['skon'] = (z ** 3).clip(-50, 50).rolling(120).mean().shift(1); d['onm20'] = d.on.rolling(20).mean().shift(1) / d.sdon
    d['rv'] = d.sd; d['y_up'] = d.upg.shift(-1); d['y_dn'] = d.dng.shift(-1)          # target: the NEXT session's gap
    D[tk] = d
print('bars loaded', len(B), flush=True)
# --- H2: pooled walk-forward logistic for signed gap tails (features known at the prior close of day d; target = gap on d+1)
from sklearn.linear_model import LogisticRegression
P = pd.concat([D[t].assign(tk=t) for t in D if t != 'QQQ']).reset_index().rename(columns={'index': 'day'})
FX = ['fup60', 'fdn60', 'fup250', 'fdn250', 'skon', 'onm20', 'rv']
P = P.dropna(subset=FX); P['yr'] = P.day.dt.year
for tgt in ('y_up', 'y_dn'):
    P['p_' + tgt] = np.nan
    for Y in range(2020, 2027):
        tr = (P.yr < Y) & P[tgt].notna() & (P.day < pd.Timestamp(f'{Y}-01-01') - pd.Timedelta(days=3)); te = P.yr == Y
        if tr.sum() < 2000 or te.sum() == 0: continue
        mu, sdv = P.loc[tr, FX].mean(), P.loc[tr, FX].std()
        m = LogisticRegression(C=.5, max_iter=500).fit((P.loc[tr, FX] - mu) / sdv, P.loc[tr, tgt]); P.loc[te, 'p_' + tgt] = m.predict_proba((P.loc[te, FX] - mu) / sdv)[:, 1]
P['gskew'] = P.p_y_up - P.p_y_dn; P['gskew_raw'] = P.fup250 - P.fdn250
X = X.merge(P[['tk', 'day', 'gskew', 'gskew_raw', 'p_y_up', 'p_y_dn']].rename(columns={'day': 'd'}), on=['tk', 'd'], how='left')
# --- H3: peer lead-lag
OPN = pd.DataFrame({t: B[t].set_index('dt').open for t in B}).sort_index()
PCm = pd.DataFrame({t: D[t].pc for t in D}); SDm = pd.DataFrame({t: D[t].sd for t in D}); Rm = pd.DataFrame({t: D[t].r for t in D})
dayidx = OPN.index.normalize()
MOVE = (OPN / PCm.reindex(dayidx).values - 1) / SDm.reindex(dayidx).values              # move since prior close, in daily sigma, at each bar open
EX = {}
for e in set(ETFMAP.values()):
    f = glob.glob(f'{ETF}/{e}_yf.parquet')
    if f: q = pd.read_parquet(f[0]); EX[e] = q['Close'].pct_change()
res_now, res1, res5, pjump = [], [], [], []
for tk, e, d in zip(X.tk.values, X.entry.values, X.d.values):
    sec = SEC.get(tk); peers = [p for p in MOVE.columns if p != tk and p != 'QQQ' and SEC.get(p) == sec]
    try: row = MOVE.loc[e]
    except KeyError: row = None
    if row is not None and peers and pd.notna(row.get(tk, np.nan)):
        pm = row[peers].mean(); res_now.append(pm - row[tk] if pd.notna(pm) else np.nan)
    else: res_now.append(np.nan)
    et = ETFMAP.get(sec); r = Rm[tk] if tk in Rm else None
    if et in EX and r is not None:
        er = EX[et]; prv = r.index[r.index < d]
        if len(prv) >= 6:
            d1, d5 = prv[-1], prv[-5:]; s = SDm[tk].get(d, np.nan)
            res1.append((er.get(d1, np.nan) - r.get(d1, np.nan)) / s); res5.append((er.reindex(d5).sum() - r.reindex(d5).sum()) / (s * np.sqrt(5)))
        else: res1.append(np.nan); res5.append(np.nan)
    else: res1.append(np.nan); res5.append(np.nan)
    prv = Rm.index[Rm.index < d]
    if len(prv) and peers:
        d1 = prv[-1]; pj = any((Rm.at[d1, p] > 2 * SDm.at[d1, p]) for p in peers if pd.notna(Rm.at[d1, p]) and pd.notna(SDm.at[d1, p]))
        own = Rm[tk].get(d1, np.nan) > 2 * SDm[tk].get(d1, np.nan) if tk in Rm else False
        pjump.append(float(pj and not own))
    else: pjump.append(np.nan)
X['res_now'], X['res1'], X['res5'], X['pjump'] = res_now, res1, res5, pjump
print('entry features done', flush=True)
# --- H6 co-jump matrix (trailing 250d count of shared big-up days), monthly snapshots
BIG = pd.DataFrame({t: D[t].big for t in D if t != 'QQQ'}).fillna(0)
CJ = {}
for mth in pd.date_range('2019-01-01', X.entry.max() + pd.offsets.MonthBegin(1), freq='MS'):
    w = BIG[(BIG.index < mth) & (BIG.index >= mth - pd.Timedelta(days=365))]
    if len(w) > 100: CJ[mth] = (w.T @ w) / max(len(w), 1)
pd.to_pickle(CJ, '/tmp/claude-0/ml/nt_cj.pkl')
# --- H1 / H5: in-trade checkpoints and outcome decomposition
QD = D['QQQ']; CK = []; lab = []
for i, (tk, e, x, r) in enumerate(zip(X.tk.values, X.entry.values, X.exit.values, X.ret.values)):
    if tk not in B: lab.append((np.nan, np.nan, np.nan)); continue
    H = B[tk]; t = H.dt.values; a, b = np.searchsorted(t, e), np.searchsorted(t, x)
    if a >= len(t) or b <= a: lab.append((np.nan, np.nan, np.nan)); continue
    o, c, hi, m, dy = H.open.values, H.close.values, H.high.values, H.m.values, H.day.values; px = o[a]; d = D[tk]
    # decomposition of the trade's log move into overnight gaps vs intraday
    seg = H.iloc[a:b + 1]; on_sum = 0.0; prev = None
    for dd_, gg in seg.groupby('day'):
        if prev is not None: on_sum += np.log(gg.open.iloc[0] / prev)
        prev = gg.close.iloc[-1]
    tot = np.log(o[b] / px); qbig = 0.0
    days = pd.unique(dy[a:b + 1])
    for dd_ in days:
        if dd_ in QD.index and QD.at[dd_, 'r'] > 1.0 * (QD.at[dd_, 'sd'] if pd.notna(QD.at[dd_, 'sd']) else 1): qbig = 1.0
    lab.append((on_sum, tot - on_sum, qbig))
    peak = px; nd = 0; last = dy[a]
    for j in range(a + 1, b):
        peak = max(peak, hi[j - 1])
        if dy[j] != last: nd += 1; last = dy[j]
        if m[j] == 600 and nd >= 1:
            dd_ = dy[j]; pcv = d.pc.get(dd_, np.nan); sdv = d.sd.get(dd_, np.nan)
            rest = H.iloc[j:b]; futbig = 0.0
            for d2 in pd.unique(rest.day.values):
                if d2 > dd_ and d2 in d.index and d.at[d2, 'big'] == 1: futbig = 1.0; break
            CK.append(dict(i=i, ck=t[j], nd=nd, cum=c[j - 1] / px - 1, pk=peak / px - 1, today=(c[j - 1] / pcv - 1) / sdv if sdv else np.nan,
                           gap=d.on.get(dd_, np.nan) / d.sdon.get(dd_, np.nan) if d.sdon.get(dd_, 0) else np.nan, exnow=o[j] / px - 1 - C2,
                           hold=r - (o[j] / px - 1 - C2), futbig=futbig))
L = np.array(lab, dtype=float); X['lg_on'], X['lg_in'], X['q_up'] = L[:, 0], L[:, 1], L[:, 2]
X.to_parquet('/tmp/claude-0/ml/nt_X.parquet'); CK = pd.DataFrame(CK); CK.to_parquet('/tmp/claude-0/ml/nt_CK.parquet')
print('done', len(X), 'trades', len(CK), 'checkpoints', X[['gskew', 'res_now', 'res1', 'res5', 'pjump']].notna().mean().round(2).to_dict(), flush=True)
