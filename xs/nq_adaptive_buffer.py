"""Nov 4 2026: ADAPTIVE flip buffer on the vector for the NQ NY AM trade. DECLARED BEFORE RESULTS.
The vector flips when a 15m close breaks the level by BREAK_ATR x ATR (live 0.35; already scales with volatility through ATR). Question: should the multiplier itself adapt?
Trade: QQQ 15m flips (premarket + RTH state), signal bars closing 09:30-11:30, both ways, <= 2 a day, 1 NQ 75/50, flat 12:00, 1 pt cost (QQQ proxy -> NQ pts).
VARIANTS (per 15m bar, known at that bar's close):
  STATIC   0.20 | 0.35 (live) | 0.50 | 0.70
  ER-CHOP  efficiency ratio ER20 = |close - close 20 bars ago| / sum of |bar-to-bar moves| (terciles from 2018-21 bars): choppy (low ER) 0.55 / mid 0.35 / trending (high ER) 0.20
  ER-INV   the reverse: choppy 0.20 / trending 0.55
  ER-CONT  0.35 x (median ER / ER), clipped 0.15 - 0.80
  VOL-EXP  ATR14 / ATR100 terciles: expanding 0.55 / mid 0.35 / contracting 0.20
Per variant: trades, avg pts in 2018-21 / 2022-23 / 2024-26, coin-flip direction avg (10 draws), realistic months 2022 - Oct 2026 (fresh 50K eval each month), 2026."""
_r = open('xs/nq_regime_switch.py').read(); exec(_r[:_r.index("# ---------------------------------------------------------------- RSI, ORB")])
src = open('xs/signal_lab.py').read(); vs = src[src.index('def vec(df, **kw):'):src.index('def donch(')]
vs = vs.replace('def vec(df, **kw):', 'def vec_ad(df, BAarr, **kw):').replace('level - a * BA', 'level - a * BAarr[i]').replace('level + a * BA', 'level + a * BAarr[i]')
assert vs.count('BAarr[i]') == 2; exec(vs)
X = H15[['open', 'high', 'low', 'close']].reset_index(drop=True); cl_ = X.close.values; nb = len(X)
d1 = np.abs(np.diff(cl_, prepend=cl_[0])); er = np.abs(cl_ - np.r_[np.full(20, np.nan), cl_[:-20]]) / pd.Series(d1).rolling(20).sum().values
atr14 = compute_atr_series(X.high.values, X.low.values, cl_, 14); atr100 = compute_atr_series(X.high.values, X.low.values, cl_, 100); vr = atr14 / atr100
isb = (H15.dt < P0).values
def terc(x): a, b = np.nanquantile(x[isb], [1 / 3, 2 / 3]); return a, b
ea, eb = terc(er); va, vb = terc(vr); emed = np.nanmedian(er[isb])
def pick(x, a, b, lo, mid, hi): return np.where(np.isnan(x), mid, np.where(x < a, lo, np.where(x > b, hi, mid)))
VAR = {'STATIC 0.20': np.full(nb, .20), 'STATIC 0.35 (live)': np.full(nb, .35), 'STATIC 0.50': np.full(nb, .50), 'STATIC 0.70': np.full(nb, .70),
       'ER-CHOP chop .55 / trend .20': pick(er, ea, eb, .55, .35, .20), 'ER-INV chop .20 / trend .55': pick(er, ea, eb, .20, .35, .55),
       'ER-CONT .35 x median/ER': np.clip(.35 * emed / np.where(np.isnan(er) | (er <= 0), emed, er), .15, .80), 'VOL-EXP expand .55 / contract .20': pick(vr, va, vb, .20, .35, .55)}
def cands_r(reg, lo=570, hi=690):
    me = H15.me.values; ends = H15.end.values; dts = H15.dt.values; out = []
    for k in range(2, nb):
        if reg[k] == reg[k - 1] or reg[k - 1] == 0 or not (lo < me[k] <= hi): continue
        day = pd.Timestamp(dts[k]).normalize(); j = np.searchsorted(t5, np.datetime64(ends[k]))
        if j < len(H5) and d5[j] == np.datetime64(day): out.append((day, int(j), int(reg[k])))
    return kcol(out)
def run_t(Cc):
    out = []
    for day, g in Cc.sort_values('j').groupby('day'):
        busy = -1; n = 0
        for j, d, kk in zip(g.j.values, g.dir.values, g.k.values):
            if n >= 2 or j <= busy or not np.isfinite(kk): continue
            e = o5[j]; res = None
            for q in range(j, len(H5)):
                if d5[q] != d5[j] or m5[q] >= 720: res = ((c5[q - 1] - e) * d * kk, q - 1); break
                lo_ = ((l5[q] if d > 0 else h5[q]) - e) * d * kk; hi_ = ((h5[q] if d > 0 else l5[q]) - e) * d * kk
                if lo_ <= -50: res = (-50, q); break
                if hi_ >= 75: res = (75, q); break
            if res: out.append((day, res[0] - 1.0)); busy = res[1]; n += 1
    return pd.DataFrame(out, columns=['day', 'net'])
rng = np.random.default_rng(5); chk = np.asarray(vec(X))
print(f'15m bars {nb}; live vector reproduced by vec_ad(0.35): {np.mean(np.asarray(vec_ad(X, np.full(nb, .35))) == chk):.1%} of bars; ER terciles {ea:.2f}/{eb:.2f}, ATR ratio terciles {va:.2f}/{vb:.2f}', flush=True)
for nm, BA in VAR.items():
    reg = np.asarray(vec_ad(X, BA)); Cc = cands_r(reg); R = run_t(Cc); R['per'] = np.where(R.day < P0, '18-21', np.where(R.day < '2024-01-01', '22-23', '24-26'))
    cf = [run_t(Cc.assign(dir=rng.choice([-1, 1], len(Cc)))).net.mean() for _ in range(10)]
    M_ = months((R.net * 20).groupby(R.day).sum()); M6 = M_[M_.month >= '2026-01']; pm = R.groupby('per').net.agg(['mean', 'size'])
    print(f'  {nm:<34} flips/day {len(Cc) / len(days):.2f} trades {len(R)} | avg ' + ' / '.join(f'{p} {pm.loc[p, "mean"]:+5.1f} (n{int(pm.loc[p, "size"])})' for p in ('18-21', '22-23', '24-26')) +
          f' | coin-flip {np.median(cf):+.1f} | months passed {M_.res.eq("passed").mean():4.0%} lost {M_.res.eq("lost").mean():4.0%} mean {M_.pnl.mean():+5,.0f} | 2026 passed {M6.res.eq("passed").sum()}/{len(M6)} lost {M6.res.eq("lost").sum()}', flush=True)
