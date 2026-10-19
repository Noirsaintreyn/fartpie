"""'Reach terrain' chart: THOD/TLOD + levels on NON-LINEAR axes.
  * TIME is a fisheye: the newest candles are wide, history is compressed (x = -((age+1)^0.62 - 1)).
  * The FUTURE is a probability field, not a forecast line: for every price and every minute until 12:00 it shows the volatility-aware odds (bot sigma + 30-min RV + time left) of a price at that distance
    being touched by 12:00 if price were still sitting where it is now. Iso-odds contours at 50 / 25 / 10 %; the 50 % and 25 % contours are the T1 / T2 target distances and they tighten as the clock runs down.
  * THOD/TLOD 1/2/3 sigma lines and the ML-filtered levels sit on the field with their own odds; the right-hand ruler reads in odds, not price.
Candles are coloured by the vector regime (blue +, red -).  No predicted HOD/LOD, no vector line."""
import json
import os
import pickle
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vchart as VC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, 'reach_model_v2.json') if os.path.exists(os.path.join(ROOT, 'reach_model_v2.json')) else os.path.join(ROOT, 'reach_model_vol.json')
TRADE_MODEL = os.path.join(ROOT, 'trade_model_v2.json'); HMM_PKL = os.path.join(ROOT, 'hmm_intraday.pkl')
NYTZ = ZoneInfo('America/New_York')
EQUITY_SHIFT = -0.41      # logit offset fit on SPY QQQ NVDA AAPL TSLA MSFT (60d of 5m): -0.40 first half / -0.42 second half
GAMMA = 0.62
HOD_C, LOD_C, LVL_C = '#2dd4a7', '#c4b5fd', '#8da2c0'
OPEN_C = '#ffa24a'
HOD2_C, LOD2_C = '#a7f3e0', '#ede9fe'      # second (longer) timeframe: lighter + dashed


CORR930_MODE = os.environ.get('CORR930', 'off')          # 'off' (default) | 'compare' (raw primary + corrected overlay) | 'on' (corrected only)
_CORR930 = None


def load_corr930():
    """monotone (isotonic) map raw v2 odds -> corrected odds for the 09:30 snapshot (corr930.json: x = raw, y = corrected, strictly increasing); None if the file is missing"""
    global _CORR930
    if _CORR930 is None:
        path = os.path.join(ROOT, 'corr930.json'); _CORR930 = json.load(open(path)) if os.path.exists(path) else {}
    return (np.array(_CORR930['x']), np.array(_CORR930['y'])) if _CORR930 and _CORR930.get('type') == 'isotonic' else None


_OPEN_MAP = None


def calib_open(p):
    """isotonic map raw v2 open-state touch odds -> calibrated (fit on 2025 + last month, 248 sessions); identity if the map file is missing"""
    global _OPEN_MAP
    if _OPEN_MAP is None:
        path = os.path.join(ROOT, 'open_calibration_map.json'); _OPEN_MAP = json.load(open(path)) if os.path.exists(path) else {}
    return float(np.interp(p, _OPEN_MAP['x'], _OPEN_MAP['y'])) if _OPEN_MAP else float(p)


def live_open_states(src, now=None):
    """src: dict tf -> backend data response for '5m', '15m', '1h'.  Open states known NOW from COMPLETED bars of each timeframe, merged by price (0.5pt)."""
    now = pd.Timestamp(now or datetime.now(NYTZ).replace(tzinfo=None)); mins = {'5m': 5, '15m': 15, '1h': 60}; fr = {}
    for tf, r in src.items():
        d = VC._frame(r['priceData']); fr[tf] = d[d['dt'] + pd.Timedelta(minutes=mins[tf]) <= now].reset_index(drop=True)
    chart = fr['5m']; price = float(chart['close'].iloc[-1]); merged = {}
    for tf, d in fr.items():
        for o in VC.open_states(d, chart, price, ks=(8, 21), per_side=6):
            e = merged.setdefault(round(o['price'] * 2) / 2, dict(price=o['price'], t=o['t'], tf=[], side=o['side'])); e['tf'].append(tf); e['t'] = min(e['t'], o['t'])
    order = ['5m', '15m', '1h']
    return [dict(price=e['price'], t=e['t'], tf='/'.join(sorted(set(e['tf']), key=order.index)), side=e['side']) for e in merged.values()]


class Reach:
    """vectorised copy of the saved logistic touch model.  Varying inputs: [log bps distance, minutes left, log bot sigma bps, log RV bps/min].
    v2 adds the symmetric path-shape block (lr, lrng30, wick, n_tot, t_any) - constants for the current moment, set with set_extra()."""
    def __init__(self, path=MODEL):
        m = json.load(open(path)); self.mu, self.sd, self.coef, self.b0 = np.array(m['mu']), np.array(m['sd']), np.array(m['coef']), m['intercept']
        self.features = m.get('features', ['lbps', 'tl', 'lsig', 'lrv']); self.cext = 0.0; self.shift = 0.0; self.cmap = None       # shift = logit recalibration for non-NQ assets

    def set_extra(self, vals):
        """vals: dict of the extra feature values at this moment (None -> neutral, i.e. the training mean)"""
        c = 0.0
        for k, name in enumerate(self.features[4:], start=4):
            if vals is not None and name in vals: c += self.coef[k] * (vals[name] - self.mu[k]) / self.sd[k]
        self.cext = float(c)

    def p(self, dist_bps, tl, sig_bps, rv_bps):
        f = np.stack(np.broadcast_arrays(np.log(np.maximum(dist_bps, 0.3)), np.asarray(tl, float), np.log(sig_bps), np.log(rv_bps)), -1)
        z = (f - self.mu[:4]) / self.sd[:4]; p = 1 / (1 + np.exp(-(self.b0 + self.cext + self.shift + z @ self.coef[:4])))
        if self.cmap is not None: p = np.interp(p, self.cmap[0], self.cmap[1])
        return p

    def dist_bps(self, prob, tl, sig_bps, rv_bps):
        if self.cmap is not None: prob = np.interp(prob, self.cmap[1], self.cmap[0])          # raw probability that the corrected curve maps to `prob`
        rest = (np.log(sig_bps) - self.mu[2]) / self.sd[2] * self.coef[2] + (np.log(rv_bps) - self.mu[3]) / self.sd[3] * self.coef[3] + (np.asarray(tl, float) - self.mu[1]) / self.sd[1] * self.coef[1]
        z0 = (np.log(prob / (1 - prob)) - self.b0 - self.cext - self.shift - rest) / self.coef[0]; return np.exp(z0 * self.sd[0] + self.mu[0])


class Trade:
    """TP-before-SL logistic: [log TP bps, log SL bps, minutes left, log sigma bps, log RV] x intraday-HMM high-vol-state interactions.  Weak discrimination (AUC ~0.56-0.59): context, not a signal."""
    def __init__(self, path=TRADE_MODEL):
        m = json.load(open(path)); self.F, self.mu, self.sd, self.coef, self.b0 = m['features'], np.array(m['mu']), np.array(m['sd']), np.array(m['coef']), m['intercept']

    def p(self, tp_bps, sl_bps, tl, sig_bps, rv_bps, hmm_hi):
        base = dict(lT=np.log(tp_bps), lS=np.log(sl_bps), tl=tl, lsig=np.log(sig_bps), lrv=np.log(rv_bps)); v = dict(base); v['hmm_hi'] = hmm_hi
        for k, x in base.items(): v[f'hmm_hi*{k}'] = hmm_hi * x
        z = (np.array([v[f] for f in self.F]) - self.mu) / self.sd; return float(1 / (1 + np.exp(-(self.b0 + z @ self.coef))))


def live_state(df_all, now=None, nb=12, cap=180):
    """symmetric path-shape block + intraday-HMM high-vol probability from the COMPLETED 5m bars (a still-forming last bar is left out, as in training).  None on failure."""
    try:
        now = now or datetime.now(NYTZ).replace(tzinfo=None); dt = pd.to_datetime(df_all['dt']).reset_index(drop=True); n = len(df_all)
        i = n - 1 if dt.iloc[-1] + pd.Timedelta(minutes=5) <= pd.Timestamp(now) else n - 2
        O, H, L, C = (df_all[k].values for k in ('open', 'high', 'low', 'close')); t = dt.iloc[i] + pd.Timedelta(minutes=5)
        start = int(dt.searchsorted(t.normalize() - pd.Timedelta(hours=6)))
        P = C[i]; h6, l6 = H[i - 5:i + 1].max(), L[i - 5:i + 1].min(); r30 = max(h6 - l6, 1e-6); r5 = max(H[i] - L[i], 1e-6)
        uw = (h6 - max(O[i - 5], P)) / r30; lw = (min(O[i - 5], P) - l6) / r30
        cm_h = np.maximum.accumulate(H[start:i + 1]); cm_l = np.minimum.accumulate(L[start:i + 1]); up = np.r_[False, H[start + 1:i + 1] > cm_h[:-1]]; dn = np.r_[False, L[start + 1:i + 1] < cm_l[:-1]]

        def since(a):
            w = np.where(a)[0]; return min((len(a) - 1 - w[-1]) * 5, cap) if len(w) else cap
        sym = dict(lr=float(np.log((r5 / P * 1e4 + .5) / (r30 / P * 1e4 + .5))), lrng30=float(np.log(r30 / P * 1e4 + .5)), wick=float(uw + lw), n_tot=float(int(up[-nb:].sum()) + int(dn[-nb:].sum())), t_any=float(min(since(up), since(dn))))
        hmm_hi = None
        if os.path.exists(HMM_PKL):
            hmm, hi = pickle.load(open(HMM_PKL, 'rb')); r = np.r_[0.0, np.diff(np.log(C[:i + 1]))] * 1e4; hmm_hi = float(hmm.predict_proba(r[-100:].reshape(-1, 1))[-1, hi])
        return sym, hmm_hi
    except Exception as e:
        print(f'live_state failed: {e}'); return None, None


def xpos(age):
    return -((np.asarray(age, float) + 1.0) ** GAMMA - 1.0)


TARGET_SETS = {'t1': ([0.50], ['#ffffff'], [1.6], ['-'], [(0.50, 'T1 50%')], ['50']),
               't1t2': ([0.25, 0.50], ['#bfe3f5', '#ffffff'], [1.0, 1.5], ['-', '-'], [(0.25, 'T2 25%'), (0.50, 'T1 50%')], ['50', '25']),
               'full': ([0.10, 0.25, 0.50, 0.75], ['#6ea8c9', '#bfe3f5', '#ffffff', '#ffffff'], [0.8, 1.0, 1.5, 0.8], ['-', '-', '-', ':'], [(0.10, '10%'), (0.25, 'T2 25%'), (0.50, 'T1 50%')], ['90', '75', '50', '25', '10', '5'])}


def render(ticker, tf, resp, hl, levels, now=None, nbars=150, minutes_to_noon=None, hl_lines=None, targets='t1t2', levels_tf=None, thod_tf=None, extra_lines=None, auto_window=True, log_to=None, after=None, replay_note=None, open_states=None, corr930=None, cones=True):
    """hl = the 1h hod_lod response (its sigma drives the odds model).  hl_lines = hod_lod response whose THOD/TLOD lines are drawn (None -> no lines).
    levels = list of levels to draw (None/[] -> none).  targets = 't1' | 't1t2' | 'full' (which odds contours/ticks to show).  cones=False drops the odds field and the T1/T2 contours/labels/ticks (the THOD/TLOD lines and their touch odds stay)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans'}); MONO = 'DejaVu Sans Mono'
    df_all = VC._frame(resp['priceData']); reg_all = VC.vector_regime(df_all)
    price = float(df_all['close'].iloc[-1]); closes = df_all['close'].values
    h1, l1 = hl['hod']['1std'], hl['lod']['1std']; anchor = (h1 + l1) / 2; sg = (h1 - l1) / 2; sig_bps = sg / price * 1e4
    if auto_window:       # start the chart BEFORE the earliest last-touch among the lines that will be drawn, so every line's history (and where price reacted to it) is visible
        tmp = df_all.iloc[-nbars:]; lo0 = min(tmp['low'].min(), price - 3.35 * sg); hi0 = max(tmp['high'].max(), price + 3.35 * sg); pad0 = (hi0 - lo0) * .04
        cand = [hh[side][key] for hh in ([hl_lines] if hl_lines else []) + [h_ for _, h_ in (extra_lines or [])] for key in ('1std', '2std', '3std') for side in ('hod', 'lod')]
        cand += [l['price'] for l in sorted([l for l in (levels or []) if abs(l['price'] - price) > 1.5], key=lambda l: abs(l['price'] - price)) if lo0 - pad0 < l['price'] < hi0 + pad0][:7]
        L_, H_ = df_all['low'].values, df_all['high'].values; firsts = []
        for v in cand:
            if lo0 - pad0 < v < hi0 + pad0:
                hit = np.where((L_ <= v) & (H_ >= v))[0]
                if len(hit): firsts.append(hit[-1])                                            # last touch of this line
        if firsts: nbars = int(np.clip(len(df_all) - min(firsts) + 20, nbars, min(len(df_all), 700)))
    df = df_all.iloc[-nbars:].reset_index(drop=True); reg = reg_all[-nbars:]; n = len(df)
    global GAMMA
    GAMMA = 0.62 if n <= 150 else max(0.62 - 0.12 * min((n - 150) / 550, 1), 0.5)                 # compress history more when the window is long
    rv_bps = float(np.std(np.diff(np.log(closes[-7:]))) / np.sqrt(5) * 1e4 + 1e-6)
    last_dt = pd.Timestamp(df['dt'].iloc[-1]) + pd.Timedelta(minutes=5)
    if minutes_to_noon is None: minutes_to_noon = 720 - (last_dt.hour * 60 + last_dt.minute)
    H = minutes_to_noon if minutes_to_noon > 0 else 145; tl_now = float(np.clip(H, 1, 145)); M = Reach()
    equity = not str(ticker).endswith('=F'); M.shift = EQUITY_SHIFT if equity else 0.0                         # NQ-fit odds overstate equity reach by ~7pp: logit offset from the 6-ticker test
    sym_live, hmm_hi = live_state(df_all); M.set_extra(sym_live)                         # v2 path-shape block (neutral if it could not be computed)
    mode = corr930 if corr930 is not None else CORR930_MODE; cp = load_corr930(); Mc = None
    corr_active = mode in ('on', 'compare') and cp is not None and not equity and 135 <= minutes_to_noon <= 160          # the correction is fit for the 09:30 snapshot only (tl = 145)
    if corr_active:
        Mc = Reach(); Mc.shift = M.shift; Mc.set_extra(sym_live); Mc.cmap = cp
        if mode == 'on': M, Mc = Mc, None
    d1 = float(M.dist_bps(0.50, tl_now, sig_bps, rv_bps)) / 1e4 * price; d2 = float(M.dist_bps(0.25, tl_now, sig_bps, rv_bps)) / 1e4 * price
    trade = None
    if hmm_hi is not None and os.path.exists(TRADE_MODEL) and not equity:      # the HMM TP-before-SL model does not transfer to equities
        TM = Trade(); trade = {}
        for nm_, dd in (('t1', d1), ('t2', d2)):
            for fr, key in ((0.5, 'half'), (1.0, 'full')): trade[f'{nm_}_{key}'] = TM.p(dd / price * 1e4, max(dd * fr / price * 1e4, 1.0), tl_now, sig_bps, rv_bps, hmm_hi)
    dc1 = float(Mc.dist_bps(0.50, tl_now, sig_bps, rv_bps)) / 1e4 * price if Mc is not None else None; dc2 = float(Mc.dist_bps(0.25, tl_now, sig_bps, rv_bps)) / 1e4 * price if Mc is not None else None
    render.last = dict(corr930=(mode if corr_active else f'inactive ({mode})'), t1_corr=dc1, t2_corr=dc2, trade=trade, hmm_hi=hmm_hi, sym_ok=sym_live is not None, price=price, sigma=sg, rv_pt=rv_bps * price / 1e4, rv_bps=rv_bps, t1=d1, t2=d2, vector=int(reg[-1]), minutes_to_noon=minutes_to_noon, now=last_dt,
                       hod1=float(M.p(sg / price * 1e4, tl_now, sig_bps, rv_bps)), anchor=anchor)
    W = 13.0; xf0 = float(1 - 0.5 ** GAMMA); xw = lambda m: xf0 + W * np.asarray(m, float) / H          # minutes from now -> x
    after_df, reg_after, maxm = None, None, 0.0
    if after:                                                       # REPLAY: candles that actually printed after the as-of moment, drawn on the same minutes-to-noon scale (and beyond it)
        after_df = VC._frame(after); reg_after = VC.vector_regime(pd.concat([df_all, after_df], ignore_index=True))[len(df_all):]
        maxm = float(((pd.to_datetime(after_df['dt']) + pd.Timedelta(minutes=5)) - last_dt).dt.total_seconds().max() / 60)
    xend = xf0 + W * max(1.0, maxm / H)
    span = df['high'].max() - df['low'].min(); lo = min(df['low'].min(), price - 3.35 * sg); hi = max(df['high'].max(), price + 3.35 * sg)
    if after_df is not None: lo, hi = min(lo, after_df['low'].min()), max(hi, after_df['high'].max())          # replay: keep every later candle on the chart
    pad = (hi - lo) * .04; ylo, yhi = lo - pad, hi + pad
    fig = plt.figure(figsize=(13.4, 8.7), dpi=125); fig.patch.set_facecolor('#05070d')
    bgax = fig.add_axes([0, 0, 1, 1], zorder=-5); bgax.axis('off'); g = np.linspace(0, 1, 256).reshape(-1, 1)
    bgax.imshow(np.dstack([0.02 + 0.045 * g, 0.03 + 0.06 * g, 0.05 + 0.13 * g]), aspect='auto', extent=[0, 1, 0, 1], origin='lower')
    ax = fig.add_axes([0.05, 0.245, 0.70, 0.60]); ax.set_facecolor('none')
    xmin = float(xpos(n - 1)) - 0.6; ax.set_xlim(xmin, xend + 0.25); ax.set_ylim(ylo, yhi)
    # ---------------- probability terrain (future)
    nx, ny = 360, 520; xs = np.linspace(xf0, xf0 + W, nx); ys = np.linspace(ylo, yhi, ny); X, Y = np.meshgrid(xs, ys)
    mins_elapsed = (X - xf0) / W * H; tl = np.clip(H - mins_elapsed, 1, 145); P = M.p(np.abs(Y - price) / price * 1e4, tl, sig_bps, rv_bps)
    up = (Y >= price)[..., None]; col_up, col_dn = np.array([0.18, 0.83, 0.65]), np.array([0.77, 0.71, 0.99])
    rgb = np.where(up, col_up, col_dn); alpha = np.clip(P, 0, 1) ** 1.15 * 0.62; img = np.dstack([np.broadcast_to(rgb, (ny, nx, 3)), alpha])
    if cones: ax.imshow(img, extent=[xf0, xf0 + W, ylo, yhi], origin='lower', aspect='auto', zorder=1, interpolation='bilinear')
    tl_, tc_, tw_, ts_, tlabs_, ticks_ = TARGET_SETS[targets]
    cs = ax.contour(X, Y, P, levels=tl_, colors=tc_, linewidths=tw_, linestyles=ts_, alpha=0.9, zorder=2) if cones else None
    if Mc is not None and cones:                                                                    # compare: corrected contours as amber dashes over the raw ones
        Pc = Mc.p(np.abs(Y - price) / price * 1e4, tl, sig_bps, rv_bps); ax.contour(X, Y, Pc, levels=tl_, colors='#ffb454', linewidths=1.3, linestyles='--', alpha=0.95, zorder=2.2)
        for lvl_, txt in tlabs_:
            for s_ in (1, -1):
                yy_ = price + s_ * float(Mc.dist_bps(lvl_, tl_now, sig_bps, rv_bps)) / 1e4 * price
                if ylo < yy_ < yhi: ax.text(xf0 + 4.2, yy_ + s_ * (yhi - ylo) * 0.008, txt.split()[0] + 'c', color='#ffb454', fontsize=7.2, va='bottom' if s_ == 1 else 'top', family=MONO, zorder=6)
    for lvl_, txt in (tlabs_ if cones else []):
        for s in (1, -1):
            d = float(M.dist_bps(lvl_, tl_now, sig_bps, rv_bps)) / 1e4 * price; yy = price + s * d
            if ylo < yy < yhi: ax.text(xf0 + 0.25, yy + s * (yhi - ylo) * 0.008, txt, color='#e8eaed', fontsize=7.2, va='bottom' if s == 1 else 'top', family=MONO, alpha=0.9, zorder=6)
    # ---------------- fisheye candles
    for i, r in df.iterrows():
        age = n - 1 - i; xc = float(xpos(age)); w = float(xpos(age - 0.5) - xpos(age + 0.5)); bw = max(w * 0.74, 0.03)
        c = VC.POS if reg[i] == 1 else (VC.NEG if reg[i] == -1 else VC.NEUT); top, bot = max(r['open'], r['close']), min(r['open'], r['close'])
        ax.plot([xc, xc], [r['low'], r['high']], color=c, lw=3.0, alpha=0.10, zorder=3.5, solid_capstyle='round'); ax.plot([xc, xc], [r['low'], r['high']], color=c, lw=0.8, zorder=4)
        ax.add_patch(plt.Rectangle((xc - bw / 2, bot), bw, max(top - bot, 1e-9), facecolor=c, edgecolor=c, lw=0.4, alpha=0.95, zorder=4.5))
    # now marker with ripples
    for s_, a_ in ((520, .05), (260, .09), (110, .18)): ax.scatter([xf0], [price], s=s_, color='#e8eaed', alpha=a_, zorder=5.5)
    ax.scatter([xf0], [price], s=22, color='#ffffff', zorder=6); ax.axhline(price, xmin=0, xmax=1, color='#e8eaed', lw=0.5, alpha=0.25, zorder=2)
    ax.axvline(xf0, color='#e8eaed', lw=0.6, alpha=0.30, ls=(0, (1, 3)), zorder=2)
    if after:                                                       # replay / 4pm post: mark the 09:30 open so it is clear where the levels started
        x930 = xf0 + W * (((last_dt.normalize() + pd.Timedelta(hours=9, minutes=30)) - last_dt).total_seconds() / 60) / H
        ax.axvline(x930, color='#ffd166', lw=1.1, ls=(0, (5, 3)), alpha=0.8, zorder=2.3); ax.text(x930 + 0.12, ylo + (yhi - ylo) * 0.965, '09:30', color='#ffd166', fontsize=8, va='top', family=MONO, zorder=6)
    # ---------------- THOD/TLOD lines + levels as nodes
    items = []                                                      # (price, label, color, odds, kind)
    oc_map = {}                                                     # compare mode: corrected odds by line price
    ocf = lambda p_: (float(Mc.p(abs(p_ - price) / price * 1e4, tl_now, sig_bps, rv_bps)) if Mc is not None else None)
    lows, highs = df['low'].values, df['high'].values

    def touch_x(v):                                                 # x of the LAST candle whose range contains the line (left edge if never touched in view)
        hit = np.where((lows <= v) & (highs >= v))[0]
        return float(xpos(n - 1 - hit[-1])) if len(hit) else xmin

    def touch_history(v, c):                                        # every earlier touch (first candle of each run): hollow dot + a faint segment from the first touch to the last
        hit = np.where((lows <= v) & (highs >= v))[0]
        if len(hit) < 2: return
        starts = [hit[0]] + [h for p, h in zip(hit[:-1], hit[1:]) if h - p > 1]
        xs_ = [float(xpos(n - 1 - i)) for i in starts][-14:]
        ax.plot([xs_[0], float(xpos(n - 1 - hit[-1]))], [v, v], color=c, lw=0.7, alpha=0.28, ls=(0, (2, 3)), zorder=2.4)
        if VC.TOUCH_MARKS: ax.scatter(xs_[:-1], [v] * (len(xs_) - 1), s=11, facecolor='none', edgecolor=c, linewidths=0.8, alpha=0.85, zorder=5.0)

    def after_touch(v, c):                                          # replay: every touch of the line by the candles that printed after the as-of moment
        if after_df is None: return
        lo_, hi_ = after_df['low'].values, after_df['high'].values; hit = np.where((lo_ <= v) & (hi_ >= v))[0]
        if not len(hit): return
        starts = [hit[0]] + [h for p, h in zip(hit[:-1], hit[1:]) if h - p > 1]
        xs_ = [xf0 + W * (((pd.Timestamp(after_df['dt'].iloc[i]) + pd.Timedelta(minutes=2.5)) - last_dt).total_seconds() / 60) / H for i in starts]
        if VC.TOUCH_MARKS: ax.scatter(xs_, [v] * len(xs_), s=34, facecolor=c, edgecolor='white', linewidths=0.9, zorder=5.6)

    if after_df is not None:
        bw_ = W * 5 / H * 0.74
        for j in range(len(after_df)):
            r = after_df.iloc[j]; xc = xf0 + W * (((pd.Timestamp(r['dt']) + pd.Timedelta(minutes=2.5)) - last_dt).total_seconds() / 60) / H
            c = VC.POS if reg_after[j] == 1 else (VC.NEG if reg_after[j] == -1 else VC.NEUT); top, bot = max(r['open'], r['close']), min(r['open'], r['close'])
            ax.plot([xc, xc], [r['low'], r['high']], color=c, lw=0.8, zorder=4.0); ax.add_patch(plt.Rectangle((xc - bw_ / 2, bot), bw_, max(top - bot, 1e-9), facecolor=c, edgecolor=c, lw=0.3, alpha=0.95, zorder=4.5))
        if xend > xf0 + W: ax.axvline(xf0 + W, color='#e8eaed', lw=0.8, alpha=0.45, ls=(0, (4, 3)), zorder=2.5); ax.text(xf0 + W + 0.1, ylo + (yhi - ylo) * 0.015, '12:00 · odds horizon', color='#8a94a6', fontsize=7.5, zorder=6)
    ltag = f'{levels_tf} ' if (levels_tf and levels_tf != tf) else ''
    sets = ([(thod_tf, hl_lines, HOD_C, LOD_C, ':')] if hl_lines else []) + [(lab_, h_, HOD2_C, LOD2_C, '--') for lab_, h_ in (extra_lines or [])]
    bmarks = []                                                     # every THOD/TLOD line for the price strip, even if it is off the chart
    for tfl, hh, ch_, cl_, lsty in sets:
        ttag = f'{tfl} ' if (tfl and (tfl != tf or extra_lines)) else ''
        for kk, key in ((1, '1std'), (2, '2std'), (3, '3std')):
            for nm, v, c in (('THOD', hh['hod'][key], ch_), ('TLOD', hh['lod'][key], cl_)):
                o = float(M.p(abs(v - price) / price * 1e4, tl_now, sig_bps, rv_bps)); on = ylo < v < yhi
                bmarks.append((v, f'{ttag}{nm} {kk}σ' + ('' if on else ' (off)'), c, o)); oc_map[round(float(v), 2)] = ocf(v)
                if on:
                    items.append((v, f'{ttag}{nm} {kk}σ', c, o, 'band'))
                    x0 = touch_x(v); VC._glow(ax, [x0, xend], [v, v], c, lw=0.9, ls=lsty, zorder=2.6, alpha=0.9)
                    if VC.TOUCH_MARKS and x0 > xmin: ax.scatter([x0], [v], s=16, color=c, zorder=5.2)
                    touch_history(v, c); after_touch(v, c)
    lv = sorted([l for l in (levels or []) if ylo < l['price'] < yhi and abs(l['price'] - price) > 1.5], key=lambda l: abs(l['price'] - price))[:7]
    for l in lv:
        o = float(M.p(abs(l['price'] - price) / price * 1e4, tl_now, sig_bps, rv_bps)); items.append((l['price'], f"{ltag}{l['name'].split('+')[0][:11]}", LVL_C, o, 'level')); oc_map[round(float(l['price']), 2)] = ocf(l['price'])
        yy = l['price']; st = min(l['strength'] / 100, 1)
        x0 = touch_x(yy); VC._glow(ax, [x0, xend], [yy, yy], LVL_C, lw=0.7 + 1.0 * st, ls='-', zorder=2.8, alpha=0.8)
        if VC.TOUCH_MARKS and x0 > xmin: ax.scatter([x0], [yy], s=16, color=LVL_C, zorder=5.2)
        touch_history(yy, LVL_C); after_touch(yy, LVL_C)
        for s_, a_ in ((260 * st + 60, .07), (110 * st + 30, .15)): ax.scatter([xend], [yy], s=s_, color=LVL_C, alpha=a_, zorder=5)
        ax.scatter([xend], [yy], s=22, facecolor='#05070d', edgecolor=LVL_C, linewidths=1.3, zorder=5.5)
    render.last['bands'] = [(lab, float(p), float(o)) for p, lab, c, o in bmarks]
    for o in (open_states or []):                                   # untouched wave pivots (open states): line from the pivot candle, diamond at the pivot, filled dot at every later touch
        p_ = float(o['price'])
        if not (ylo < p_ < yhi): continue
        ix_ = int(np.searchsorted(df['dt'].values, np.datetime64(pd.Timestamp(o['t'])))); older = ix_ <= 0 or ix_ >= n
        x0 = xmin if older else float(xpos(n - 1 - ix_)); raw_o = float(M.p(abs(p_ - price) / price * 1e4, tl_now, sig_bps, rv_bps)); odds_ = raw_o if (corr_active and Mc is None) else calib_open(raw_o); oc_map[round(p_, 2)] = ocf(p_)       # 'on': M is already the corrected curve
        VC._glow(ax, [x0, xend], [p_, p_], OPEN_C, lw=0.9, ls='-', zorder=2.7, alpha=0.9)
        ax.scatter([x0], [p_], marker='<' if older else 'D', s=30, facecolor=OPEN_C, edgecolor='white', linewidths=0.6, zorder=5.3)
        items.append((p_, f"open {o.get('tf', '')}", OPEN_C, odds_, 'open')); after_touch(p_, OPEN_C)
    # ---------------- axes decor
    pt = sorted(({0, 12, 36, 72} if n <= 220 else {0, 72}) | {a for a in (150, 300, 500) if a < n - 20 and (n <= 220 or a > 100)} | {n - 1}); ax.set_xticks([float(xpos(a)) for a in pt] + [float(xw(m)) for m in (60, 120, 180) if H - m >= 40] + [xf0 + W] + [float(xw(H + 60 * k)) for k in range(1, 8) if H + 60 * k <= maxm + 30])
    ax.set_xticklabels([pd.Timestamp(df['dt'].iloc[n - 1 - a]).strftime('%a %H:%M') for a in pt] + [(last_dt + pd.Timedelta(minutes=m)).strftime('%H:%M') for m in (60, 120, 180) if H - m >= 40] + ['12:00'] + [(last_dt + pd.Timedelta(minutes=H + 60 * k)).strftime('%H:%M') for k in range(1, 8) if H + 60 * k <= maxm + 30], color='#8a94a6', fontsize=8)
    ax.tick_params(axis='y', colors='#8a94a6', labelsize=8)
    for s in ax.spines.values(): s.set_color('#1d2735')
    ax.grid(color='#1d2735', lw=0.4, alpha=0.5, zorder=0)
    # ---------------- right ladder: odds ruler
    rx = fig.add_axes([0.755, 0.245, 0.235, 0.60]); rx.set_facecolor('none'); rx.set_ylim(ylo, yhi); rx.set_xlim(0, 1); rx.axis('off')
    for pr, lab in [(int(k) / 100, k) for k in (ticks_ if cones else [])]:
        for s in (1, -1):
            yy = price + s * float(M.dist_bps(pr, tl_now, sig_bps, rv_bps)) / 1e4 * price
            if ylo < yy < yhi: rx.plot([0.0, 0.035], [yy, yy], color='#e8eaed', lw=0.9, alpha=0.55); rx.text(0.045, yy, f'{lab}%', color='#6b7686', fontsize=6.8, va='center', family=MONO)
    rx.plot([0.0, 0.0], [ylo, yhi], color='#2b3447', lw=0.8)
    items.append((price, 'NOW', '#ffffff', 1.0, 'now')); ys_ = VC.spread([i[0] for i in items], (yhi - ylo) * 0.034)
    for (p, lab, c, o, kind), y in zip(items, ys_):
        if kind == 'now': rx.text(0.32, y, f'NOW {price:,.1f}', color='#ffffff', fontsize=9.5, fontweight='bold', va='center', family=MONO); continue
        rx.plot([0.2, 0.3], [p, y], color=c, lw=0.6, alpha=0.6); rx.text(0.32, y, f'{lab} {p:,.0f}', color=c, fontsize=8.2, va='center', family=MONO)
        rx.add_patch(plt.Rectangle((0.78, y - (yhi - ylo) * 0.007), 0.14 * min(o / 0.9, 1), (yhi - ylo) * 0.014, color=c, alpha=0.8)); rx.text(0.945 if oc_map.get(round(float(p), 2)) is None else 0.80, y, (f'{o*100:.0f}%' if oc_map.get(round(float(p), 2)) is None else f'{o*100:.0f}→{oc_map[round(float(p), 2)]*100:.0f}%'), color='#e8eaed', fontsize=8, va='center', family=MONO)
    # ---------------- bottom strip: exact prices to mark (nearest tick), high -> low
    marks = [(p, lab, c, o) for (p, lab, c, o, kind) in items if kind in ('level', 'open')] + bmarks
    for nm_, pr_ in ((('T1 50%', 0.50), ('T2 25%', 0.25)) if cones else ()):
        if nm_.startswith('T2') and targets == 't1': continue
        d_ = float(M.dist_bps(pr_, tl_now, sig_bps, rv_bps)) / 1e4 * price
        marks += [(price + d_, f'{nm_} target ▲', '#ffffff', pr_), (price - d_, f'{nm_} target ▼', '#ffffff', pr_)]
        if Mc is not None:
            dcc = float(Mc.dist_bps(pr_, tl_now, sig_bps, rv_bps)) / 1e4 * price; marks += [(price + dcc, f'{nm_[:2]}c corrected ▲', '#ffb454', pr_), (price - dcc, f'{nm_[:2]}c corrected ▼', '#ffb454', pr_)]
    marks.append((price, 'NOW', '#e8eaed', 1.0)); marks.sort(key=lambda m: -m[0])
    fig.text(0.05, 0.200, 'PRICES TO MARK  (high → low · nearest tick · % = odds of being touched by 12:00)', color='#8a94a6', fontsize=8.5, fontweight='bold')
    ncol = 4 if len(marks) <= 20 else 5; nrow = int(np.ceil(len(marks) / ncol)); colw = 0.238 if ncol == 4 else 0.19; mfs = 8.6 if ncol == 4 else 7.9
    for k, (p, lab, c, o) in enumerate(marks):
        col_, row_ = divmod(k, nrow); fig.text(0.05 + col_ * colw, 0.168 - row_ * 0.029, f'{round(p * 4) / 4:>10,.2f}  {lab[:(17 if ncol == 4 else 15)]:<{17 if ncol == 4 else 15}}{o * 100:>3.0f}%', color=c, fontsize=mfs, family=MONO)
    # ---------------- titles
    fig.text(0.05, 0.955, f'{ticker}  {tf}  ·  ' + ('REACH TERRAIN' if cones else 'THOD / TLOD + TOUCH ODDS'), color='#e8eaed', fontsize=16, fontweight='bold')
    fig.text(0.05, 0.922, f'{last_dt:%a %b %d %H:%M} ET · σ {sg:.0f}pt · recent vol {rv_bps * price / 1e4:.1f}pt/min · {minutes_to_noon} min to 12:00 · fisheye time (recent bars wide)' + (' · field = odds of touching by 12:00' if cones else ' · % on each line = odds of being touched by 12:00'), color='#8a94a6', fontsize=8.5)
    if not cones: fig.text(0.05, 0.893, 'blue = positive vector, red = negative · not a price forecast', color='#8a94a6', fontsize=8.5)
    if cones: fig.text(0.05, 0.893, 'brighter = more likely to be touched · white contours: ' + {'t1': 'T1 (50%)', 't1t2': 'T1 (50%) / T2 (25%)', 'full': 'T1 / T2 / 10%'}[targets] + ' target distance if price stays here · blue = positive vector, red = negative · not a price forecast' + (' · open % calibrated' if open_states else '') + (' · amber = corrected (odds raw→corr)' if Mc is not None else (' · 09:30-corrected odds' if corr_active else '')), color='#8a94a6', fontsize=8.5)
    if trade and cones:
        fig.text(0.05, 0.866, f"TP before stop (model, weak signal) · T1 ±{d1:.0f}pt: stop ½ {trade['t1_half']*100:.0f}% / stop 1× {trade['t1_full']*100:.0f}%  ·  T2 ±{d2:.0f}pt: stop ½ {trade['t2_half']*100:.0f}% / stop 1× {trade['t2_full']*100:.0f}%  ·  intraday state: {'HIGH-vol' if hmm_hi > 0.5 else 'low-vol'} ({hmm_hi*100:.0f}%)", color='#c9d3e3', fontsize=8.5)
    if replay_note: fig.text(0.05, 0.034, replay_note, color='#9ad1ff', fontsize=8.5)
    if equity: fig.text(0.05, 0.034, 'Equity: odds were fit on NQ and recalibrated on 6 US stocks/ETFs (60 days, logit offset -0.41); TP-before-stop odds are not shown (that model does not transfer).', color='#ffb454', fontsize=8.5)
    if minutes_to_noon > 145 or minutes_to_noon <= 0: fig.text(0.05, 0.012, 'Odds were fit on 09:40–11:30 decisions, so the field is flat until ~09:35 and indicative before then.', color='#ffb454', fontsize=8.5)
    if log_to:                                                       # keep each post's lines/levels so later charts can show earlier sessions' levels and how price reacted
        try:
            cm_ = load_corr930(); cdist = None
            if cm_ is not None and not equity and 135 <= minutes_to_noon <= 160:
                Mr = Reach(); Mr.shift = M.shift; Mr.set_extra(sym_live); Mq = Reach(); Mq.shift = M.shift; Mq.set_extra(sym_live); Mq.cmap = cm_; cdist = dict(t1_raw=float(Mr.dist_bps(0.50, tl_now, sig_bps, rv_bps)) / 1e4 * price, t2_raw=float(Mr.dist_bps(0.25, tl_now, sig_bps, rv_bps)) / 1e4 * price, t1_corr=float(Mq.dist_bps(0.50, tl_now, sig_bps, rv_bps)) / 1e4 * price, t2_corr=float(Mq.dist_bps(0.25, tl_now, sig_bps, rv_bps)) / 1e4 * price)
            rec = dict(ts=str(last_dt), ticker=str(ticker), price=price, contours=cdist, bands=[(lab, float(p)) for p, lab, c, o in bmarks], levels=[(it[1], float(it[0])) for it in items if it[4] == 'level'])
            old = json.load(open(log_to)) if os.path.exists(log_to) else []; old.append(rec); json.dump(old[-60:], open(log_to, 'w'))
        except Exception as e: print(f'levels log failed: {e}')
    buf = __import__('io').BytesIO(); fig.savefig(buf, format='png', facecolor=fig.get_facecolor()); plt.close(fig); return buf.getvalue()


if __name__ == '__main__':
    import backend_client as B
    import analysis as A
    bk = B.Backend(); root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = bk.data('NQ=F', '5m'); h1 = bk.hod_lod('NQ=F', '1h'); r1h = bk.data('NQ=F', '1h'); h4 = bk.hod_lod('NQ=F', '4h')
    png = render('NQ', '5m', r, h1, A.all_levels(r1h), hl_lines=h4, targets='t1', levels_tf='1h', thod_tf='4h'); open(os.path.join(root, 'reach_terrain_t1_4h_1hlevels.png'), 'wb').write(png)
    png = render('NQ', '5m', r, h1, None, hl_lines=None, targets='t1'); open(os.path.join(root, 'reach_terrain_t1_only.png'), 'wb').write(png); print('ok')
