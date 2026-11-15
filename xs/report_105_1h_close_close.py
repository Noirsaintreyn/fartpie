"""Nov 15 2026 VARIANT of report_103 (Rey: entry at the 1H CLOSE of the flip bar, exit at the 1H CLOSE of the bear-flip bar; everything else identical, same 100bp). Original: FROZEN SCORE ON 1H FLIPS (Rey: 'test the model with 1h'). Declared before results. DATA LIMIT: 60 days of 5m bars (Aug 20-Nov 12 2026) resampled to RTH 1H (09:30 anchored), ~2,000 low-priced names, ONE regime, survivorship-flattered -> every number is a LEAD, not a validated finding; no train/test split possible. The score is NOT refit: the same frozen weights as scan_week (VOLX .22, PEERS -.22, VR .21, EXT .5, R60 .03, BLOW .08, LDV -.13, LPX -.14) on percentile ranks, ranks taken across all 1H flips in the window. Features from DAILY bars through the day BEFORE the flip day (VR, EXT, R60, BLOW, LDV, LPX); VOLX = flip-bar 1H volume / mean of previous 100 1H bars; PEERS = same-2-digit-SIC 1H flips in prior 5 sessions. Universe: close $1-10, 20d median daily dollar volume >= $500k, SIC 67xx (funds) excluded. Entry next 1H open after the flip bar, exit next-bar open after opposite 1H flip (still-open marked at last close), 100bp round trip. Metrics: n, mean/median net, share >=+10%, share >=+30%, ex-top-3 mean, Spearman IC of score vs net return, first-half vs second-half (split at Oct 6), top-quintile vs rest. Then the LIVE 1H list as of the last bar: bull-state names ranked by score with flip time, flip price, now, and distance to ratchet."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from vector_pine import run_state_machine
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; M5 = pickle.load(open(SC + 'lowpx_5m.pkl', 'rb')); DD = pickle.load(open(SC + 'lowpx_daily.pkl', 'rb')); sic = pd.read_csv(SC + 'ext/sic.csv').drop_duplicates('tk').set_index('tk')
co = dict(VOLX=.22, PEERS=-.22, VR=.21, EXT=.5, R60=.03, BLOW=.08, LDV=-.13, LPX=-.14); COST = .01
def cl(x):
    x = x[['Open', 'High', 'Low', 'Close', 'Volume']].copy(); x.columns = ['open', 'high', 'low', 'close', 'volume']; x.index = x.index.tz_convert('America/New_York') if x.index.tz is not None else x.index; x = x[~x.index.duplicated()]; m = x.index.hour * 60 + x.index.minute; return x[(m >= 570) & (m < 960)]
rows = []; live = []
for tk, x in M5.items():
    if sic.sic.get(tk, 0) // 100 == 67: continue
    q = DD.get(tk)
    if q is None or len(q) < 200: continue
    q = q[['Open', 'High', 'Low', 'Close', 'Volume']].copy(); q.columns = ['open', 'high', 'low', 'close', 'volume']; q = q[~q.index.duplicated()].dropna(); q.index = pd.to_datetime(q.index).tz_localize(None).normalize()
    c = q.close.values; h = q.high.values; l = q.low.values; v = q.volume.values.astype(float); tr = np.maximum(h - l, np.maximum(abs(h - np.r_[c[0], c[:-1]]), abs(l - np.r_[c[0], c[:-1]])))
    a10 = pd.Series(tr).rolling(10).mean().values; a60 = pd.Series(tr).rolling(60).mean().values; lr = pd.Series(np.log(c)).diff(); b20 = lr.rolling(20).std().values; b120 = lr.rolling(120).std().values; lo60 = pd.Series(l).rolling(60).min().values; dv = pd.Series(c * v).rolling(20).median().values
    F = pd.DataFrame(dict(VR=a10 / a60, EXT=c / lo60 - 1, R60=c / pd.Series(c).shift(60).values - 1, BLOW=b20 / b120, LDV=np.log(dv), LPX=np.log(c), dvv=dv, cc=c), index=q.index).shift(1)
    x = cl(x); x = x[x.volume > 0]
    if len(x) < 300: continue
    B = x.resample('60min', offset='30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna(subset=['open']); B = B[B.index.hour * 60 + B.index.minute >= 570]
    if len(B) < 150: continue
    reg, lev, atr, _ = run_state_machine(B.assign(datetime=B.index)); o = B.open.values; cc = B.close.values; n = len(B); vol = B.volume.values.astype(float); vm = pd.Series(vol).rolling(100, min_periods=30).mean().shift(1).values; dts = pd.to_datetime(B.index.date); sic2 = sic.sic.get(tk, np.nan) // 100 if tk in sic.index else np.nan
    for i in range(1, n - 1):
        if not (reg[i] == 1 and reg[i - 1] != 1): continue
        d = dts[i]
        if d not in F.index: continue
        f = F.loc[d]
        if not (1 <= f.cc < 10 and f.dvv >= 5e5) or f[['VR', 'EXT', 'R60', 'BLOW', 'LDV', 'LPX']].isna().any(): continue
        j = i + 1
        while j < n and reg[j] != -1: j += 1
        mk = j >= n; ex = cc[-1] if mk else cc[j]; path = (B.high.values[i + 1:(j + 1 if not mk else n)].max() / cc[i] - 1) if i + 1 < n else np.nan
        rows.append(dict(tk=tk, t=B.index[i], d=d, entry=cc[i], ex=ex, mk=mk, bars=(j - i), peak=path, VOLX=vol[i] / vm[i] if vm[i] == vm[i] else np.nan, sic2=sic2, VR=f.VR, EXT=f.EXT, R60=f.R60, BLOW=f.BLOW, LDV=f.LDV, LPX=f.LPX))
        if mk: live.append(dict(tk=tk, flip_t=B.index[i], flip_px=cc[i], now=cc[-1], ratchet=lev[-1], bars_since=n - 1 - i))
A = pd.DataFrame(rows).dropna(subset=['VOLX']); print('1H bull flips', len(A), 'names', A.tk.nunique(), 'window', A.d.min().date(), A.d.max().date())
A = A.sort_values('t'); pe = []
for _, r in A.iterrows(): pe.append(((A.sic2 == r.sic2) & (A.tk != r.tk) & (A.t < r.t) & (A.t >= r.t - pd.Timedelta(days=7))).sum() if r.sic2 == r.sic2 else 0)
A['PEERS'] = pe; A['score'] = sum(w * A[f].rank(pct=True) for f, w in co.items()); A['r'] = A.ex / A.entry - 1 - COST
def st(z, lab):
    if len(z) < 5: print(f' {lab:28s} n {len(z)}'); return
    s = z.r.sort_values(ascending=False); print(f' {lab:28s} n {len(z):5d} mean {z.r.mean()*100:+7.2f}% med {z.r.median()*100:+6.2f}% >=10% {(z.r>=.1).mean()*100:5.1f}% >=30% {(z.r>=.3).mean()*100:4.1f}% ex-top3 {s.iloc[3:].mean()*100:+6.2f}% win {(z.r>0).mean()*100:4.1f}% avg hold bars {z.bars.mean():5.1f}')
print('\nALL / SCORE QUINTILES (net of 100bp, hold to 1H bear flip; still-open marked at last close)'); st(A, 'ALL 1H flips'); A['q'] = pd.qcut(A.score, 5, labels=False)
for k in range(5): st(A[A.q == k], f'score quintile {k+1}' + (' (top)' if k == 4 else ''))
ic = spearmanr(A.score, A.r); print(f' Spearman IC score vs net return {ic[0]:+.3f} (p {ic[1]:.3f}); vs win10 flag {spearmanr(A.score, A.r>=.1)[0]:+.3f}'); 
top = A[A.score >= A.score.quantile(.8)]; rest = A[A.score < A.score.quantile(.8)]; st(top, 'TOP 20% score'); st(rest, 'rest')
for lab, z in (('first half (<Oct 6)', A[A.d < '2026-10-06']), ('second half (>=Oct 6)', A[A.d >= '2026-10-06'])):
    zt = z[z.score >= z.score.quantile(.8)]; st(z, lab + ' ALL'); st(zt, lab + ' top20')
cd = top.assign(day=top.d.astype(str)).groupby('day').r.mean(); ca = A.assign(day=A.d.astype(str)).groupby('day').r.mean(); dd = (cd - ca.reindex(cd.index)).dropna(); print(f' day-clustered top-20 minus all-flip mean: {dd.mean()*100:+.2f}pp t {dd.mean()/(dd.std()/np.sqrt(len(dd))):+.2f} over {len(dd)} days')
print('\nPrice buckets, top20 vs all:'); 
for lo, hi in ((1, 2), (2, 5), (5, 10)):
    z = A[(A.entry >= lo) & (A.entry < hi)]; st(z, f'${lo}-{hi} all'); st(z[z.score >= A.score.quantile(.8)], f'${lo}-{hi} top20')
A.to_pickle(SC + 'ext/flips_1h_score_close.pkl')
Lv = pd.DataFrame(live); 
if len(Lv):
    last = A.t.max(); Lv = Lv.merge(A[['tk', 't', 'score', 'VOLX', 'PEERS']].rename(columns={'t': 'flip_t'}), on=['tk', 'flip_t'], how='left').dropna(subset=['score']); Lv['since_flip'] = Lv.now / Lv.flip_px - 1; Lv['to_ratchet'] = Lv.now / Lv.ratchet - 1; Lv = Lv.sort_values('score', ascending=False)
    pd.set_option('display.width', 250); print('\nLIVE 1H BULL STATE at last bar (flip time, flip-bar next-open, now, vs ratchet), top 25 by frozen score:'); print(Lv.head(25)[['tk', 'flip_t', 'bars_since', 'flip_px', 'now', 'since_flip', 'to_ratchet', 'VOLX', 'PEERS', 'score']].round(3).to_string(index=False))
