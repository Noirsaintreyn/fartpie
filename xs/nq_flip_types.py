"""Nov 4 2026: what KIND of vector flip works in what regime, for the NQ NY AM trade. DECLARED BEFORE RESULTS.
Trades: 15m QQQ vector flips, signal bars closing 09:30-11:30, any day, up to 2 a day, flat 12:00, 1 NQ 75/50, 1 pt cost (QQQ proxy -> NQ pts), 2018 - Sep 2026.
LABELS at the signal (only data before the entry bar):
  flip type: DIR long/short | ORDER first flip of the day / later | TIME 09:45-10:00 / 10:00-10:30 / 10:30-11:30 | GAP with / against the 09:30 gap | AMT aligned / not
             DISP flip-bar range in 15m ATR (small / mid / big) | VOL5 15m volume vs its 20-day same-time average (low / mid / high) | VWAP distance in ATR, in the trade's direction
             (behind VWAP / near / extended) | PRIOR how long the previous vector regime lasted (short chop < 8 bars / medium / long > 26 bars)
  regime: HAR tercile | HURST (R/S on the last 100 15m returns: < .45 mean-reverting / .45-.55 / > .55 trending) | WAVE wavelet low-frequency energy share of the last 64 5m returns
          (db4, 4 levels; terciles) | GMM 3 states on daily [return, |return|, log RV] fit on 2016-2021 (labelled calm / normal / stressed by mean RV) | HMM 2 states on daily returns
          fit on 2016-2021, FORWARD-filtered (no smoothing) -> bull / bear state | QZ QQQ 20d z-score sign | DVOL QQQ daily volume vs 20d (low / normal / high)
STEP 1 every label: n, win, avg pts in 2018-21 (in-sample) and 2022-26 (out-of-sample, split 22-23 / 24-26).
STEP 2 setup score chosen ON 2018-21 ONLY: each label level whose in-sample avg beats the in-sample overall by >= 4 pts with n >= 40 scores +1, below by >= 4 pts scores -1.
        great = top in-sample third of the score, bad = bottom third (cut-offs fixed in-sample) -> applied unchanged to 2022-26.
STEP 3 long / short: both | long only | long + shorts only in the HMM bear state | long + shorts only when QQQ z20 < 0.
STEP 4 sizing on 2022-26 with the in-sample score (realistic fresh-eval months + 2026): F flat 10 MNQ (= 1 NQ) 75/50 | M1 bad 7 / normal 10 / great 20 MNQ, all 75/50 |
        M2 bad 7 MNQ 75/50, normal 10 MNQ 75/50, great 20 MNQ at 37.5/25 (a win is still ~$1,500) | M3 like M2 but great 30 MNQ at 25/17 (live idea)."""
_s = open('xs/nq_size_tp.py').read(); exec(_s[:_s.index("print('\\nby context")])
import pywt
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM
C = C.reset_index(drop=True)
# ---------------------------------------------------------------- flip-type labels
end_i = pd.Series(np.arange(len(H15)), index=pd.DatetimeIndex(H15.end)); kk = end_i.reindex(pd.DatetimeIndex(t5[C.j.values])).values.astype(int)
hh, ll, cc_ = H15.high.values, H15.low.values, H15.close.values; pc_ = np.r_[cc_[0], cc_[:-1]]; tr15 = np.maximum(hh - ll, np.maximum(abs(hh - pc_), abs(ll - pc_)))
atr15 = pd.Series(tr15).ewm(alpha=1 / 14, adjust=False).mean().values
C['DIR'] = np.where(C.dir > 0, 'long', 'short'); C['ORDER'] = np.where(C.groupby('day').j.rank(method='first') == 1, 'first flip', 'later flip')
mm_ = m5[C.j.values]; C['TIME'] = np.where(mm_ <= 600, '09:45-10:00', np.where(mm_ <= 630, '10:00-10:30', '10:30-11:30'))
pcl = np.array([c5[np.searchsorted(t5, np.datetime64(d)) - 1] for d in C.day]); opn_ = np.array([o5[np.searchsorted(t5, np.datetime64(d + pd.Timedelta(hours=9, minutes=30)))] for d in C.day])
C['GAP'] = np.where(np.sign(opn_ - pcl) == C.dir, 'with gap', 'against gap'); C['AMT'] = np.where(C.aligned, 'AMT aligned', 'AMT not aligned')
disp = (hh[kk] - ll[kk]) / atr15[np.maximum(kk - 1, 0)]; C['disp'] = disp
v15 = pd.Series(H5.volume.values.astype(float)).rolling(3).sum().values; vj = v15[C.j.values - 1]
tod = pd.Series(v15).groupby(m5).transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean()).values; C['vol5'] = vj / tod[C.j.values - 1]
vw = []
for j, d_, a_ in zip(C.j.values, C.dir.values, atr15[np.maximum(kk - 1, 0)]):
    i0 = j
    while i0 > 0 and d5[i0 - 1] == d5[j] and m5[i0 - 1] >= 570: i0 -= 1
    w = vol5[i0:j]; v_ = (tp[i0:j] * w).sum() / w.sum() if j > i0 and w.sum() > 0 else np.nan; vw.append((o5[j] - v_) * d_ / a_)
C['vwapd'] = vw
rg = H15.reg.values; flips = np.where(np.r_[False, rg[1:] != rg[:-1]])[0]; prev_len = []
for k in kk:
    f_ = flips[flips < k]; prev_len.append(k - f_[-1] if len(f_) else np.nan)
C['prior'] = prev_len
def ter(x, a, b, labs): return np.where(x < a, labs[0], np.where(x > b, labs[2], labs[1]))
q_ = lambda col: C.loc[C.day < P0, col].quantile([1 / 3, 2 / 3]).values                     # tercile cut-offs from 2018-21 only
a, b = q_('disp'); C['DISP'] = ter(C.disp, a, b, ['small flip bar', 'mid flip bar', 'big flip bar'])
a, b = q_('vol5'); C['VOL5'] = ter(C.vol5, a, b, ['low volume', 'mid volume', 'high volume'])
a, b = q_('vwapd'); C['VWAP'] = ter(C.vwapd, a, b, ['behind VWAP', 'near VWAP', 'extended past VWAP'])
C['PRIOR'] = ter(C.prior, 8, 26, ['after chop (<8 bars)', 'after medium regime', 'after long regime (>26)'])
# ---------------------------------------------------------------- regime labels
C['HAR'] = ter(prior(HP, C.day), 1 / 3, 2 / 3, ['HAR low', 'HAR mid', 'HAR high'])
r15 = np.diff(np.log(cc_), prepend=np.nan)
def hurst(x):
    x = x[np.isfinite(x)]
    if len(x) < 64: return np.nan
    ns, rs = [8, 16, 32, 64], []
    for n in ns:
        m = len(x) // n; seg = x[:m * n].reshape(m, n); z = np.cumsum(seg - seg.mean(1, keepdims=True), 1); r = z.max(1) - z.min(1); s = seg.std(1)
        rs.append(np.mean(r[s > 0] / s[s > 0]))
    return np.polyfit(np.log(ns), np.log(rs), 1)[0]
C['hurst'] = [hurst(r15[max(k - 100, 0):k]) for k in kk]; C['HURST'] = ter(C.hurst, .45, .55, ['Hurst < .45 (mean-rev)', 'Hurst .45-.55', 'Hurst > .55 (trending)'])
r5 = np.diff(np.log(c5), prepend=np.nan); ws = []
for j in C.j.values:
    x = r5[j - 64:j]; x = x[np.isfinite(x)]
    if len(x) < 64: ws.append(np.nan); continue
    co = pywt.wavedec(x, 'db4', level=4, mode='periodization'); e = np.array([np.sum(c_ ** 2) for c_ in co]); ws.append((e[0] + e[1]) / e.sum())
C['wave'] = ws; a, b = q_('wave'); C['WAVE'] = ter(C.wave, a, b, ['wavelet noisy', 'wavelet mid', 'wavelet trendy'])
dq = Dq.copy(); dq['ret'] = np.log(dq['Adj Close']).diff(); dq['lrv'] = np.log(RV.reindex(dq.index)); dq = dq.dropna(subset=['ret', 'lrv'])
F_ = np.c_[dq.ret, dq.ret.abs(), dq.lrv]; tr_ = dq.index < '2022-01-01'
gm = GaussianMixture(3, random_state=0, n_init=5).fit(F_[tr_]); order = np.argsort(gm.means_[:, 2]); names = {order[0]: 'GMM calm', order[1]: 'GMM normal', order[2]: 'GMM stressed'}
gstate = pd.Series([names[s] for s in gm.predict(F_)], index=dq.index)
hm = GaussianHMM(2, covariance_type='diag', n_iter=200, random_state=0).fit(dq.ret.values[tr_].reshape(-1, 1))
bear = int(np.argmin(hm.means_[:, 0])); A_ = hm.transmat_; mu = hm.means_[:, 0]; sd = np.sqrt(hm.covars_[:, 0, 0]) if hm.covars_.ndim == 3 else np.sqrt(hm.covars_[:, 0])
alpha = hm.startprob_.copy(); pb = []
for x in dq.ret.values:                                                              # forward filter only: P(state_t | returns up to t)
    lik = np.exp(-.5 * ((x - mu) / sd) ** 2) / sd; alpha = (alpha @ A_) * lik; alpha /= alpha.sum(); pb.append(alpha[bear])
hmm_bear = pd.Series(pb, index=dq.index)
C['GMM'] = pd.Series(gstate).reindex(pd.DatetimeIndex(C.day) - pd.Timedelta(days=1), method='ffill').values
C['HMM'] = np.where(prior(hmm_bear, C.day) > .5, 'HMM bear', 'HMM bull'); C['QZ'] = np.where(prior(XQ.qz, C.day) < 0, 'QQQ z20 < 0', 'QQQ z20 >= 0')
qv = Dq['Volume']; C['DVOL'] = ter(prior(qv / qv.rolling(20).mean(), C.day), .85, 1.15, ['day volume low', 'day volume normal', 'day volume high'])
LABELS = ['DIR', 'ORDER', 'TIME', 'GAP', 'AMT', 'DISP', 'VOL5', 'VWAP', 'PRIOR', 'HAR', 'HURST', 'WAVE', 'GMM', 'HMM', 'QZ', 'DVOL']
# ---------------------------------------------------------------- trades with labels
def run_l(Cc, T=75, S=50, per_day=2, flat_m=720, cost=1.0, Tcol=None, Scol=None):
    out = []
    for day, g in Cc.sort_values('j').groupby('day'):
        busy = -1; n = 0
        for idx, c in g.iterrows():
            if n >= per_day or c.j <= busy: continue
            e = o5[c.j]; d = c.dir; kk_ = c.k; T_ = c[Tcol] if Tcol else T; S_ = c[Scol] if Scol else S; res = None
            for q in range(c.j, len(H5)):
                if d5[q] != d5[c.j] or m5[q] >= flat_m: res = ((c5[q - 1] - e) * d * kk_, q - 1); break
                lo_ = ((l5[q] if d > 0 else h5[q]) - e) * d * kk_; hi_ = ((h5[q] if d > 0 else l5[q]) - e) * d * kk_
                if lo_ <= -S_: res = (-S_, q); break
                if hi_ >= T_: res = (T_, q); break
            if res: out.append((idx, day, res[0] - cost)); busy = res[1]; n += 1
    return pd.DataFrame(out, columns=['idx', 'day', 'net']).set_index('idx')
R = run_l(C); Z = C.join(R[['net']], how='inner'); Z['per'] = np.where(Z.day < P0, 'IS 18-21', np.where(Z.day < '2024-01-01', 'OOS 22-23', 'OOS 24-26'))
print(f'trades {len(Z)} (in-sample 2018-21 {np.sum(Z.day < P0)}, 2022-26 {np.sum(Z.day >= P0)})', flush=True)
print('\nSTEP 1: avg pts per trade by label   [in-sample 2018-21 | out-of-sample 22-23 | 24-26]   (n, win)', flush=True)
base_is = Z[Z.day < P0].net.mean(); base_oos = Z[Z.day >= P0].net.mean(); print(f'  ALL  IS {base_is:+.1f} | OOS {base_oos:+.1f}', flush=True)
for col in LABELS:
    for v, g in Z.groupby(col):
        p = {k_: g[g.per == k_] for k_ in ('IS 18-21', 'OOS 22-23', 'OOS 24-26')}
        cells = ' | '.join(f'{k_[-5:]} {x.net.mean():+6.1f} (n{len(x):>3} w{x.net.gt(0).mean():.0%})' if len(x) else f'{k_[-5:]}   n/a' for k_, x in p.items())
        print(f'  {col:<6} {str(v):<27} {cells}', flush=True)
# ---------------------------------------------------------------- STEP 2: score chosen in-sample
IS = Z[Z.day < P0]; rules = []
for col in LABELS:
    for v, g in IS.groupby(col):
        if len(g) >= 40 and g.net.mean() - base_is >= 4: rules.append((col, v, +1))
        elif len(g) >= 40 and g.net.mean() - base_is <= -4: rules.append((col, v, -1))
print('\nSTEP 2: in-sample rules:', ', '.join(f'{c}={v} ({s:+d})' for c, v, s in rules), flush=True)
C['score'] = sum(np.where(C[c] == v, s, 0) for c, v, s in rules); Z['score'] = C.score.reindex(Z.index)
lo_c, hi_c = Z[Z.day < P0].score.quantile([1 / 3, 2 / 3]).values
if hi_c > lo_c: bad_m, great_m = C.score <= lo_c, C.score >= hi_c
else: bad_m, great_m = C.score < lo_c, C.score > hi_c
C['setup'] = np.where(bad_m, 'bad', np.where(great_m, 'great', 'normal')); Z['setup'] = C.setup.reindex(Z.index)
print(f'  score cut-offs (in-sample terciles): bad <= {lo_c:.0f}, great >= {hi_c:.0f}; 2022-26 mix: ' + str(Z[Z.day >= P0].setup.value_counts(normalize=True).round(2).to_dict()), flush=True)
for v, g in Z.groupby('setup'):
    a_, b_, c_ = g[g.per == 'IS 18-21'], g[g.per == 'OOS 22-23'], g[g.per == 'OOS 24-26']
    print(f'  {v:<7} IS {a_.net.mean():+6.1f} (n{len(a_)}) | OOS 22-23 {b_.net.mean():+6.1f} (n{len(b_)} w{b_.net.gt(0).mean():.0%}) | 24-26 {c_.net.mean():+6.1f} (n{len(c_)} w{c_.net.gt(0).mean():.0%})', flush=True)
# ---------------------------------------------------------------- STEP 3: direction
def months2(dv, a='2022-01-01', target=3000, mll=2000, dll=1000): return months(dv, a, target, mll, dll)
def mline(nm, dv):
    M_ = months2(dv); M6 = M_[M_.month >= '2026-01']
    print(f'  {nm:<44} months passed {M_.res.eq("passed").mean():.0%} lost {M_.res.eq("lost").mean():.0%} neither {M_.res.eq("neither").mean():.0%} | P&L mean {M_.pnl.mean():+,.0f} worst {M_.pnl.min():+,.0f} | 2026 passed {M6.res.eq("passed").sum()}/{len(M6)} lost {M6.res.eq("lost").sum()}', flush=True)
print('\nSTEP 3: long / short (2022-26, 1 NQ 75/50)', flush=True)
for nm, msk in (('both directions', np.ones(len(C), bool)), ('long only', C.dir > 0), ('long + shorts only in HMM bear', (C.dir > 0) | (C.HMM == 'HMM bear')),
                ('long + shorts only when QQQ z20 < 0', (C.dir > 0) | (C.QZ == 'QQQ z20 < 0'))):
    Rr = run_l(C[msk]); post = Rr[Rr.day >= P0]; mline(f'{nm} (n{len(post)}, avg {post.net.mean():+.1f})', (Rr.net * 20).groupby(Rr.day).sum())
# ---------------------------------------------------------------- STEP 4: sizing with the in-sample setup score
print('\nSTEP 4: sizing on 2022-26 with the in-sample setup score (realistic months)', flush=True)
Rf = run_l(C); Rf = Rf.join(C.setup); mline('F flat 10 MNQ (1 NQ) 75/50', (Rf.net * 20).groupby(Rf.day).sum())
mline('M1 bad 7 / normal 10 / great 20 MNQ, 75/50', (Rf.net * Rf.setup.map({'bad': 14, 'normal': 20, 'great': 40})).groupby(Rf.day).sum())
C['Tm'] = np.where(C.setup == 'great', 37.5, 75.0); C['Sm'] = np.where(C.setup == 'great', 25.0, 50.0); Rm = run_l(C, Tcol='Tm', Scol='Sm').join(C.setup)
mline('M2 great 20 MNQ at 37.5/25, others 7 / 10 MNQ 75/50', (Rm.net * Rm.setup.map({'bad': 14, 'normal': 20, 'great': 40})).groupby(Rm.day).sum())
C['Tm'] = np.where(C.setup == 'great', 25.0, 75.0); C['Sm'] = np.where(C.setup == 'great', 17.0, 50.0); Rm = run_l(C, Tcol='Tm', Scol='Sm').join(C.setup)
mline('M3 great 30 MNQ at 25/17, others 7 / 10 MNQ 75/50', (Rm.net * Rm.setup.map({'bad': 14, 'normal': 20, 'great': 60})).groupby(Rm.day).sum())
g37 = run_l(C[C.setup == 'great'], 37.5, 25); g75 = run_l(C[C.setup == 'great'], 75, 50)
print(f'  great setups 2022-26: 75/50 win {g75[g75.day >= P0].net.gt(0).mean():.0%} avg {g75[g75.day >= P0].net.mean():+.1f} pts | 37.5/25 win {g37[g37.day >= P0].net.gt(0).mean():.0%} avg {g37[g37.day >= P0].net.mean():+.1f} pts', flush=True)
C.drop(columns=['Tm', 'Sm']).to_parquet('/tmp/claude-0/ml/nq_flip_types_C.parquet')
