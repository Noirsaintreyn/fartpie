"""Nov 4 2026: the NQ flip system on ES (SPY 15m proxy -> ES pts via ES=F prior close). Dow / Russell: no multi-year intraday DIA / IWM data here. DECLARED BEFORE RESULTS.
Signals: SPY 15m vector flip (live settings, premarket + RTH state), signal bars closing 09:30-11:30, entry at the next 15m open, both ways, <= 2 a day, flat 12:00.
Exits 30 / 20 ES pts (= $1,500 / $1,000 per ES, like NQ 75 / 50), path checked on 15m bars (stop first), 0.5 ES pt cost.
Rule A on ES: skip when the 60-day SPY volume profile (15m RTH bars, volume spread over each bar's range) is thin on the path to target (bottom third of 2018-21), entry at a monthly LVN,
  or a hard break (flip bar range top third in 15m ATR + close in the outer 25% + volume above its 20-day same-time average).
Sizing: 1 ES (= 10 MES), 0.5 / 0.3 by Yang-Zhang 20d high (SPY daily) + entry stretched past session VWAP (top third), x0.7 while the last 20 trades are net negative; eval skips flips against the gap.
Report: per trade by period (every flip / Rule A) + coin-flip; Lucid 50K ($88) and Topstep sims for every flip at 1 ES and the full ES system; calendar years."""
_ad = open('xs/nq_adaptive_size.py').read(); exec(_ad[:_ad.rindex("if __name__ == '__main__':")])
S = load(glob.glob(f'{ROOT}/lse_ohlcv_cache/SPY_15m_*.parquet')[0]).reset_index(drop=True); S['reg'] = np.asarray(vec(S[['open', 'high', 'low', 'close']]))
S['end'] = S.dt + pd.Timedelta('15min'); S['me'] = S.end.dt.hour * 60 + S.end.dt.minute; S['d'] = S.dt.dt.normalize()
so, sh, sl, sc, sv, sm, sd = S.open.values, S.high.values, S.low.values, S.close.values, S.volume.values.astype(float), S.m.values, S.d.values
srt = (sm >= 570) & (sm < 960); stp = (sh + sl + sc) / 3
ESd = yf.download('ES=F', start='2017-12-01', interval='1d', progress=False, auto_adjust=False)['Close'].squeeze(); ESd.index = ESd.index.normalize(); ESd = ESd.sort_index()
SPd = yf.download('SPY', start='2016-01-01', interval='1d', progress=False, auto_adjust=False); SPd.columns = [c[0] if isinstance(c, tuple) else c for c in SPd.columns]; SPd.index = SPd.index.normalize()
n_ = 20; kz = .34 / (1.34 + (n_ + 1) / (n_ - 1)); o_, h_, l_, c_ = SPd.Open, SPd.High, SPd.Low, SPd.Close
yzS = np.log(o_ / c_.shift(1)).rolling(n_).var() + kz * np.log(c_ / o_).rolling(n_).var() + (1 - kz) * (np.log(h_ / c_) * np.log(h_ / o_) + np.log(l_ / c_) * np.log(l_ / o_)).rolling(n_).mean()
YZS = yzS.rolling(252, min_periods=120).rank(pct=True)
pc_ = np.r_[sc[0], sc[:-1]]; atr = pd.Series(np.maximum(sh - sl, np.maximum(abs(sh - pc_), abs(sl - pc_)))).ewm(alpha=1 / 14, adjust=False).mean().values
tod = pd.Series(sv).groupby(sm).transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean()).values
BWs = .0005; DPs = {}
for d, ix in pd.Series(np.arange(len(S))[srt]).groupby(sd[srt]):
    ii = ix.values; a = np.floor(np.log(sl[ii]) / BWs).astype(int); b = np.floor(np.log(sh[ii]) / BWs).astype(int); lo = a.min(); v = np.zeros(b.max() - lo + 1)
    for x, y, vv in zip(a - lo, b - lo, sv[ii]): v[x:y + 1] += vv / (y - x + 1)
    DPs[pd.Timestamp(d)] = (lo, v)
DLs = pd.DatetimeIndex(sorted(DPs)); pcache = {}
def profS(day, n):
    key = (day, n)
    if key in pcache: return pcache[key]
    i = DLs.searchsorted(day); sel = DLs[max(i - n, 0):i]
    if len(sel) < n: pcache[key] = None; return None
    lo = min(DPs[x][0] for x in sel); hi = max(DPs[x][0] + len(DPs[x][1]) for x in sel); v = np.zeros(hi - lo)
    for x in sel: a, w = DPs[x]; v[a - lo:a - lo + len(w)] += w
    pcache[key] = (lo, v); return pcache[key]
rows = []
for k in range(2, len(S) - 1):
    r = S.reg.values
    if r[k] == r[k - 1] or r[k - 1] == 0 or not (570 < S.me.values[k] <= 690): continue
    j = k + 1
    if sd[j] != sd[k]: continue
    day = pd.Timestamp(sd[k]); lv = ESd.reindex([day - pd.Timedelta(days=1)], method='ffill').values[0]
    if not np.isfinite(lv): continue
    d_ = int(r[k]); e = so[j]; kk = lv / e; eb = int(np.floor(np.log(e) / BWs)); tb = int(np.floor(np.log(e + d_ * 30 / kk) / BWs))
    P60, P21 = profS(day, 60), profS(day, 21)
    ah = np.nan
    if P60 is not None:
        lo, v = P60; bb = np.arange(min(eb + d_, tb), max(eb + d_, tb) + 1) - lo; ok = (bb >= 0) & (bb < len(v)); xv = np.zeros(len(bb)); xv[ok] = v[bb[ok]]; ah = xv.mean() / v.mean()
    lvn = False
    if P21 is not None:
        lo, v = P21; rel = eb - lo; lvn = 0 <= rel < len(v) and (v < v[rel]).mean() <= .25
    clv = (sc[k] - sl[k]) / (sh[k] - sl[k]) if sh[k] > sl[k] else .5; clv = clv if d_ > 0 else 1 - clv; disp = (sh[k] - sl[k]) / atr[k - 1]
    i0 = j
    while i0 > 0 and sd[i0 - 1] == sd[j] and sm[i0 - 1] >= 570: i0 -= 1
    w = sv[i0:j]; vw = (stp[i0:j] * w).sum() / w.sum() if j > i0 and w.sum() > 0 else e
    prevc = sc[np.where((sd < sd[j]) & srt)[0][-1]] if ((sd < sd[j]) & srt).any() else np.nan; o930 = so[np.where((sd == sd[j]) & (sm == 570))[0][0]] if ((sd == sd[j]) & (sm == 570)).any() else np.nan
    rows.append(dict(day=day, j=j, k_=k, dir=d_, kk=kk, ahead=ah, lvn=lvn, clv=clv, disp=disp, vol=sv[k] / tod[k] if tod[k] > 0 else np.nan, vwapd=(e - vw) * d_ / atr[k - 1],
                     gap=np.sign(o930 - prevc) if np.isfinite(o930) and np.isfinite(prevc) else 0))
E = pd.DataFrame(rows); isb = E.day < P0
E['THIN'] = E.ahead < E.loc[isb, 'ahead'].quantile(1 / 3); dq3 = E.loc[isb, 'disp'].quantile(2 / 3); E['HARD'] = (E.disp >= dq3) & (E.clv >= .75) & (E.vol > 1)
E['VWAPX'] = E.vwapd > E.loc[isb, 'vwapd'].quantile(2 / 3); E['YZH'] = prior(YZS, E.day) > 2 / 3; E['GAPAG'] = (E.gap != 0) & (E.gap != E.dir)
E['RULEA'] = ~(E.THIN | E.lvn | E.HARD)
def simE(j, d, kk, T=30.0, St=20.0):
    e = so[j]; path = []
    for q in range(j, len(S)):
        if sd[q] != sd[j] or sm[q] >= 720: path.append((sc[q - 1] - e) * d * kk); return path[-1], q - 1, path
        lo_ = ((sl[q] if d > 0 else sh[q]) - e) * d * kk; hi_ = ((sh[q] if d > 0 else sl[q]) - e) * d * kk
        if lo_ <= -St: path += [min(hi_, T), -St]; return -St, q, path
        if hi_ >= T: path += [lo_, T]; return T, q, path
        path += [hi_, lo_]
    return None
def seqE(mask, dirs=None):
    out = {}
    for day, g in E[mask].sort_values('j').groupby('day'):
        busy = -1; nn = 0
        for idx, r in g.iterrows():
            if nn >= 2 or r.j <= busy: continue
            res = simE(int(r.j), int(r.dir if dirs is None else dirs.loc[idx]), r.kk)
            if res is None: continue
            out[idx] = res; busy = res[1]; nn += 1
    return out
PERe = lambda d: np.select([d < P0, d < pd.Timestamp('2024-01-01'), d < pd.Timestamp('2026-01-01')], ['18-21', '22-23', '24-25'], '2026')
rng = np.random.default_rng(23); allm = pd.Series(True, index=E.index)
print(f'ES (SPY proxy) flips {len(E)}; Rule A keeps {E.RULEA.mean():.0%}', flush=True)
for nm, m in (('every flip', allm), ('Rule A', E.RULEA)):
    res = seqE(m); net = pd.Series({i: v[0] - .5 for i, v in res.items()}); per = PERe(E.loc[net.index].day)
    cf = np.median([pd.Series({i: v[0] - .5 for i, v in seqE(m, pd.Series(rng.choice([-1, 1], len(E)), index=E.index)).items()}).mean() for _ in range(10)])
    print(f'  {nm:<10} n{len(net)} win {net.gt(0).mean():.0%} avg {net.mean():+.2f} ES pts | ' + ' / '.join(f'{p} {net[per == p].mean():+5.2f} (n{(per == p).sum()})' for p in ('18-21', '22-23', '24-25', '2026')) + f' | coin-flip {cf:+.2f}', flush=True)
def toD(res, size_fn):
    D = [[] for _ in D_]
    for idx, (net, qe, p, *_) in res.items():
        day = E.at[idx, 'day']
        if day not in dpos: continue
        k = dpos[day]; s_ = size_fn(idx); p = (np.array(p) - .5) * 2.5; D[k].append(((s_,), (net - .5) * 2.5, p.min(), p))     # ES pts x 2.5 = NQ-equivalent $20 units
    return D
resA_full = seqE(E.RULEA); netA = pd.Series({i: v[0] - .5 for i, v in resA_full.items()}).loc[lambda s: s.index.isin(E.sort_values(['day', 'j']).index)]
ordE = E.loc[netA.index].sort_values(['day', 'j']).index; mlt = pd.Series(1.0, index=E.index); hist = []
for idx in ordE: mlt.loc[idx] = .7 if len(hist) >= 20 and sum(hist[-20:]) < 0 else 1.0; hist.append(netA.loc[idx])
szA = lambda idx: (1.0, .5, .3)[min(int(bool(E.at[idx, 'YZH'])) + int(bool(E.at[idx, 'VWAPX'])), 2)] * float(mlt.loc[idx])
DS['ES_all'] = toD(seqE(allm), lambda idx: 1.0); DS['ES_Ae'] = toD(seqE(E.RULEA & ~E.GAPAG), szA); DS['ES_Af'] = toD(resA_full, szA)
VAR = {'ES every flip, 1 ES': ('ES_all', 'ES_all', 'ES_all'), 'ES full system': ('ES_Ae', 'ES_Af', 'ES_Af')}
yrs_ = [(y, int(np.searchsorted(D_, pd.Timestamp(f'{y}-01-01'))), int(np.searchsorted(D_, pd.Timestamp(f'{y + 1}-01-01')))) for y in range(2022, 2027)]
def job14(a):
    r = job10(a); f = TOP if a[1] == 'TOPSTEP' else LUC; r['cal'] = {y: carS(f, s, e, VAR[a[0]])['net'] for y, s, e in yrs_}; return r
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in VAR for fn in ('LUCID 50K', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job14, jobs)
    print('\nES FIRM SIMS (SPY 15m ends Aug 2026)', flush=True)
    for r in RES:
        print(f'  {r["nm"]:<20} {r["fn"]:<9} | eval pass {r["passed"]:4.0%} | funded paid {r["paid_f"]:4.0%} | evals -> payout {r["paid_e"]:4.0%} | net/eval {r["net_eval"]:+6,.0f} | 12m median {r["k_med"]:+6,.0f} positive {r["k_pos"]:4.0%} | calendar ' + ' '.join(f'{y} {v:+,.0f}' for y, v in r['cal'].items()), flush=True)
