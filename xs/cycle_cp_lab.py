"""Two trend-model ideas in the live engine (Oct 26 2026)
 1 ADAPTIVE CYCLE  Ehlers' Hilbert-transform (homodyne) dominant cycle on (h+l)/2, causal. The vector's pivot length follows it:
     cyc04  L = round(0.4 x cycle) clipped 4-16 (about 8 at a typical 20-bar cycle)      cyc05  L = round(0.5 x cycle) clipped 4-20
 2 CHANGE POINTS   Bayesian online change-point detection (Adams & MacKay 2007) on 15m returns scaled by their 260-bar volatility:
     Gaussian, unknown mean (prior N(0, 0.5^2)), hazard 1/100, run lengths up to 300. 'break' = P(run length <= 4) > 0.5 with the
     short-run mean beyond +/-0.5 sd.
     bocpd_exit   in a bull vector, a bearish break turns the 15m state bear until the vector itself resets
     bocpd_entry  in a bear vector, a bullish break turns it bull early; that holds until the vector flips bull (hand-off) or the
                  posterior mean of the current regime turns negative
python3 xs/cycle_cp_lab.py BARS_DIR OUT_DIR WIDE_T [TK ...]"""
import sys, os, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, 'structure_lab.py')).read().split("W = pd.read_parquet(WT)")[0])

def dominant_cycle(h, l):
    p = (h + l) / 2; n = len(p); sm = np.zeros(n); de = np.zeros(n); I1 = np.zeros(n); Q1 = np.zeros(n); I2 = np.zeros(n); Q2 = np.zeros(n)
    Re = np.zeros(n); Im = np.zeros(n); P = np.full(n, 20.0); SP = np.full(n, 20.0)
    def hb(x, i, f): return (.0962 * x[i] + .5769 * x[i - 2] - .5769 * x[i - 4] - .0962 * x[i - 6]) * f
    for i in range(6, n):
        sm[i] = (4 * p[i] + 3 * p[i - 1] + 2 * p[i - 2] + p[i - 3]) / 10; f = .075 * P[i - 1] + .54
        de[i] = hb(sm, i, f); Q1[i] = hb(de, i, f); I1[i] = de[i - 3]
        jI = hb(I1, i, f); jQ = hb(Q1, i, f)
        I2[i] = .2 * (I1[i] - jQ) + .8 * I2[i - 1]; Q2[i] = .2 * (Q1[i] + jI) + .8 * Q2[i - 1]
        Re[i] = .2 * (I2[i] * I2[i - 1] + Q2[i] * Q2[i - 1]) + .8 * Re[i - 1]; Im[i] = .2 * (I2[i] * Q2[i - 1] - Q2[i] * I2[i - 1]) + .8 * Im[i - 1]
        q = P[i - 1]
        if Im[i] != 0 and Re[i] != 0: q = 2 * np.pi / abs(np.arctan(Im[i] / Re[i]))
        q = min(max(q, .67 * P[i - 1]), 1.5 * P[i - 1]); q = min(max(q, 6), 50); P[i] = .2 * q + .8 * P[i - 1]; SP[i] = .33 * P[i] + .67 * SP[i - 1]
    return SP

def vec_adaptive(df, Ls):
    """the vector with a per-bar pivot length (pivot at i - L confirmed by L bars each side)"""
    P = dict(BASE); high, low, close = df['high'].values, df['low'].values, df['close'].values; n = len(df)
    atr = compute_atr_series(high, low, close, P['ATR_LEN']); ef, es = ema(close, P['FAST_LEN']), ema(close, P['SLOW_LEN'])
    pl = np.full(n, np.nan); ph = np.full(n, np.nan); patr = np.full(n, np.nan)
    for i in range(n):
        L = int(Ls[i])
        if i < 2 * L: continue
        c = low[i - L]
        if c < low[i - 2 * L:i - L].min() and c <= low[i - L + 1:i + 1].min(): pl[i] = c
        c = high[i - L]
        if c > high[i - 2 * L:i - L].max() and c >= high[i - L + 1:i + 1].max(): ph[i] = c
        patr[i] = atr[i - L]
    LA, BA, MM, MB, CB = P['LEVEL_ATR'], P['BREAK_ATR'], P['MIN_MOVE_ATR'], P['MIN_BARS_RATCH'], P['CONFIRM_BARS']
    regime, level, bc, ac, lr = 0, np.nan, 0, 0, None; regs = np.zeros(n, int)
    for i in range(n):
        a = atr[i]
        if np.isnan(a) or np.isnan(ef[i]) or np.isnan(es[i]) and regime == 0: regs[i] = regime; continue
        cs = pl[i] - patr[i] * LA if not np.isnan(pl[i]) and not np.isnan(patr[i]) else np.nan
        cr = ph[i] + patr[i] * LA if not np.isnan(ph[i]) and not np.isnan(patr[i]) else np.nan
        if regime == 0: regime = 1 if ef[i] >= es[i] else -1; level = low[i] - a * LA if regime == 1 else high[i] + a * LA; lr = i
        sb = regime == 1 and close[i] < level - a * BA; rb = regime == -1 and close[i] > level + a * BA
        if regime == 1: bc, ac = (bc + 1 if sb else 0), 0
        elif regime == -1: ac, bc = (ac + 1 if rb else 0), 0
        if regime == 1 and bc >= CB:
            regime, bc, ac = -1, 0, 0; nr = cr if (not np.isnan(cr) and cr > close[i]) else high[i] + a * LA; level = max(nr, close[i] + a * LA); lr = i
        elif regime == -1 and ac >= CB:
            regime, bc, ac = 1, 0, 0; ns = cs if (not np.isnan(cs) and cs < close[i]) else low[i] - a * LA; level = min(ns, close[i] - a * LA); lr = i
        elif regime == 1 and not np.isnan(cs) and cs < close[i] and cs > level + a * MM and (lr is None or i - lr >= MB): level, lr = cs, i
        elif regime == -1 and not np.isnan(cr) and cr > close[i] and cr < level - a * MM and (lr is None or i - lr >= MB): level, lr = cr, i
        regs[i] = regime
    return regs

def bocpd(x, hazard=1 / 100, tau2=.25, R=300, k=4):
    """-> P(run length <= k), short-run posterior mean, full posterior mean of the current regime (per bar, causal)"""
    n = len(x); pr = np.zeros(R + 1); pr[0] = 1.0; mu = np.zeros(R + 1); v = np.full(R + 1, tau2)
    cp = np.full(n, np.nan); ms = np.full(n, np.nan); mf = np.full(n, np.nan)
    for t in range(n):
        xt = x[t]
        if np.isnan(xt): continue
        s2 = 1 + v; pred = np.exp(-.5 * (xt - mu) ** 2 / s2) / np.sqrt(s2)
        g = pr * pred; new = np.empty_like(pr); new[0] = (g * hazard).sum(); new[1:] = g[:-1] * (1 - hazard); new[-1] += g[-1] * (1 - hazard)
        new /= new.sum()
        vn = 1 / (1 / v + 1); mn = vn * (mu / v + xt)
        mu2 = np.empty_like(mu); v2 = np.empty_like(v); mu2[0], v2[0] = 0.0, tau2; mu2[1:], v2[1:] = mn[:-1], vn[:-1]
        mu, v, pr = mu2, v2, new
        cp[t] = pr[:k + 1].sum(); w = pr[1:k + 1]; ms[t] = (w * mu[1:k + 1]).sum() / w.sum() if w.sum() > 0 else 0.0; mf[t] = (pr * mu).sum()
    return cp, ms, mf

def cp_states(reg, cp, ms, mf, mode, ct=.5, mt=.5):
    out = reg.copy(); s = None
    for i in range(len(reg)):
        brk_dn = cp[i] > ct and ms[i] < -mt; brk_up = cp[i] > ct and ms[i] > mt
        if mode == 'exit':
            if s == -1 and reg[i] != 1: s = None                                   # vector reset itself -> back to normal
            if s is None and reg[i] == 1 and brk_dn: s = -1
        else:
            if s == 1 and (reg[i] == 1 or mf[i] < 0): s = None if reg[i] == 1 else -1
            if s == -1 and reg[i] != -1: s = None
            if s is None and reg[i] == -1 and brk_up: s = 1
            if s == -1 and reg[i] == -1: out[i] = -1; continue
        if s is not None: out[i] = s
    return out

W = pd.read_parquet(WT); CORE = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
for tk in (sys.argv[4:] or CORE):
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    H = load(fs[tk]); rows = []; hb_ = agg(H, '1h', '9h30min')
    def add(var, a, b):
        for h1t in (False, True): rows.extend((tk, var, h1t, 'trade') + t for t in run(H, a, b, h1t)[0])
    D15 = dominant_cycle(H.high.values, H.low.values); D1 = dominant_cycle(hb_.high.values, hb_.low.values)
    for nm, k, hi_ in ([] if os.getenv('ROBUST') else  (('cyc04', .4, 16), ('cyc05', .5, 20))):
        L15 = np.clip(np.round(k * D15), 4, hi_); L1 = np.clip(np.round(k * D1), 4, hi_)
        add(nm, vec_adaptive(H, L15), known(H, hb_, vec_adaptive(hb_, L1), '1h'))
    reg, s1 = states(H, vec)
    r = np.log(H.close).diff(); x = (r / r.rolling(260, min_periods=60).std().shift(1)).values
    cp, ms, mf = bocpd(x)
    if os.getenv('ROBUST'):                                                       # robustness grid around the first pass
        for ct, mt in ((.4, .5), (.6, .5), (.5, .3), (.5, .8), (.7, .8)):
            add(f'cpE_{ct}_{mt}', cp_states(reg, cp, ms, mf, 'entry', ct, mt), s1); add(f'cpX_{ct}_{mt}', cp_states(reg, cp, ms, mf, 'exit', ct, mt), s1)
        both = cp_states(cp_states(reg, cp, ms, mf, 'entry'), cp, ms, mf, 'exit'); add('cp_both', both, s1)
    else:
        add('bocpd_exit', cp_states(reg, cp, ms, mf, 'exit'), s1)
        add('bocpd_entry', cp_states(reg, cp, ms, mf, 'entry'), s1)
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'kind', 'entry', 'exit', 'ret']).to_parquet(fo)
    print(tk, len(rows), f'median cycle 15m {np.median(D15[500:]):.1f} 1h {np.median(D1[200:]):.1f}', flush=True)
