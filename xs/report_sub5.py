"""SUB-$5 mid/small book: 2026 stats, VWAP-skip variant returns, last 5 signals (charts). Entry price < $5 at the entry bar (point-in-time), 40 bp round trip, 8 slots, higher flip-volume first, 1h vector bull flip entry next open, exit next open after the bear flip. Filters: A all flips, B vol>=1.5x, C vol>=1.5x + low-vol day, C+V = C and flip-bar close <= 1 ATR above session VWAP (skip extended). Universe = cached screener lists (survivorship-biased: only names alive today)."""
import sys, os, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import midsmall_screener as M
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb'))
def atr14(h, l, c): pc = np.r_[c[0], c[:-1]]; tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))); return pd.Series(tr).rolling(14).mean().values
R = []
for tk, Y in data.items():
    try: Yc = M.closed_only(Y); F = M.features(Yc)
    except Exception: continue
    n = len(F); r = F.reg.values; o, h, l, c, v = (Yc[k].values.astype(float) for k in ('open', 'high', 'low', 'close', 'volume')); at = atr14(h, l, c); day = Yc.index.normalize()
    vw = (pd.Series((h + l + c) / 3 * v).groupby(day).cumsum() / (pd.Series(v).groupby(day).cumsum() + 1e-9)).values; op = None
    for i in range(1, n):
        if r[i] == r[i - 1] or r[i] == 0: continue
        if r[i] == 1 and i + 1 < n and op is None and o[i + 1] < 5 and at[i] == at[i]: op = dict(tk=tk, fi=i, ei=i + 1, epx=o[i + 1], et=Yc.index[i + 1], vr=float(F.vrel.iat[i]) if F.vrel.iat[i] == F.vrel.iat[i] else np.nan, lv=bool(F.lv.iat[i]), vwx=(c[i] - vw[i]) / at[i])
        elif r[i] == -1 and op is not None and i + 1 < n: op['xpx'] = o[i + 1]; op['xt'] = Yc.index[i + 1]; op['xi'] = i + 1; op['open'] = False; R.append(op); op = None
        elif r[i] == -1: op = None
    if op is not None: op['xpx'] = c[-1]; op['xt'] = Yc.index[-1]; op['xi'] = n - 1; op['open'] = True; R.append(op)
T = pd.DataFrame(R); T['ret'] = T.xpx / T.epx - 1; T = T[T.et >= pd.Timestamp('2023-12-01')].reset_index(drop=True); print('sub-5 trades', len(T), 'names', T.tk.nunique(), flush=True)
T.to_pickle(SC + 'sub5_T.pkl'); COST = .004; K = 8
FIL = {'A all flips': np.ones(len(T), bool), 'B vol>=1.5x': (T.vr >= 1.5).values, 'C vol>=1.5x + low-vol day': ((T.vr >= 1.5) & T.lv).values, 'C+V (C, not >1 ATR above VWAP)': ((T.vr >= 1.5) & T.lv & (T.vwx <= 1)).values, 'B+V (B, not >1 ATR above VWAP)': ((T.vr >= 1.5) & (T.vwx <= 1)).values}
def sim(d, t0, t1):
    d = d[(d.et >= t0) & (d.et < t1)]; rows = sorted(d.to_dict('records'), key=lambda r: (r['et'], -(r['vr'] if r['vr'] == r['vr'] else 0))); eq = 1.0; op = []; curve = [1.0]; n = w = 0; hold = []; taken = []
    def close(p):
        nonlocal eq, n, w
        eq += p['sz'] * (p['ret'] - COST); curve.append(eq); n += 1; w += p['ret'] - COST > 0; hold.append((p['xt'] - p['et']) / np.timedelta64(1, 'D'))
    for r in rows:
        still = []
        for p in op:
            if p['xt'] <= r['et']: close(p)
            else: still.append(p)
        op = still
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r); taken.append(r)
    for p in op: close(p)
    cv = np.array(curve); return eq - 1, (cv / np.maximum.accumulate(cv) - 1).min(), n, (w / n if n else np.nan), (np.mean(hold) if hold else np.nan), taken
PER = {'2024 (Dec23-Dec24)': (pd.Timestamp('2023-12-01'), pd.Timestamp('2025-01-01'), 13 / 12), '2025': (pd.Timestamp('2025-01-01'), pd.Timestamp('2026-01-01'), 1.0), '2026 YTD (to Nov 10)': (pd.Timestamp('2026-01-01'), pd.Timestamp('2027-01-01'), 314 / 365), 'FULL': (pd.Timestamp('2023-12-01'), pd.Timestamp('2027-01-01'), 2.94)}
for fn, fm in FIL.items():
    print(f'\n## {fn}  (all sub-$5 flips passing: {fm.sum()})')
    for pn, (a, b, yrs) in PER.items():
        tr_, dd, n, w, hd, _ = sim(T[fm], a, b); ann = ((1 + tr_) ** (1 / yrs) - 1) * 100 if tr_ > -1 else -100
        d_ = T[fm & (T.et >= a).values & (T.et < b).values]; pt = (d_.ret - COST).mean() * 100 if len(d_) else float('nan')
        print(f'   {pn:22s} return {tr_ * 100:+7.1f}% (annualised {ann:+6.0f}%)  maxDD {dd * 100:5.1f}%  trades {n:4d}  win {w * 100:4.1f}%  avg hold {hd:4.1f}d  mean/trade {pt:+5.2f}%')
# --- last 5 signals of variant C (screener rules) among the 8-slot taken trades
fm = FIL['C vol>=1.5x + low-vol day']; _, _, _, _, _, taken = sim(T[fm], pd.Timestamp('2023-12-01'), pd.Timestamp('2027-01-01')); last5 = sorted(taken, key=lambda r: r['et'])[-5:]
BULL, BEAR, BG, FG = '#3b6cff', '#ff5d8f', '#0b0b0c', '#8a8d93'; files = []; rows = []
for k, t in enumerate(last5):
    Yc = M.closed_only(data[t['tk']]); F = M.features(Yc); fi, ei, xi = t['fi'], t['ei'], t['xi']; a = max(0, fi - 45); b = min(len(Yc), xi + 14); B = Yc.iloc[a:b]; rg = F.reg.values[a:b]; n = len(B)
    day = Yc.index.normalize(); vwa = (pd.Series((Yc.high.values + Yc.low.values + Yc.close.values) / 3 * Yc.volume.values).groupby(day).cumsum() / (pd.Series(Yc.volume.values).groupby(day).cumsum() + 1e-9)).values[a:b]
    fig = plt.figure(figsize=(11, 6), facecolor=BG); gs = fig.add_gridspec(2, 1, height_ratios=[5, 1.3], hspace=.06, left=.06, right=.93, top=.86, bottom=.07); ax = fig.add_subplot(gs[0]); vx = fig.add_subplot(gs[1], sharex=ax); rngv = B.high.max() - B.low.min()
    for q in range(n):
        c = BULL if rg[q] > 0 else BEAR; ax.plot([q, q], [B.low.iat[q], B.high.iat[q]], color=c, lw=.9); lo, hi = sorted([B.open.iat[q], B.close.iat[q]]); ax.add_patch(plt.Rectangle((q - .34, lo), .68, max(hi - lo, rngv * .002), color=c)); vx.bar(q, B.volume.iat[q], width=.7, color=c)
    ax.plot(range(n), vwa, color='#d0a94a', lw=1.0, ls='--', label='session VWAP'); ax.legend(loc='upper left', fontsize=7, facecolor=BG, edgecolor='#26282c', labelcolor=FG)
    ax.scatter([ei - a], [t['epx']], marker='^', s=110, color='#ffffff', edgecolor=BULL, zorder=6, lw=1.4)
    if not t['open']: ax.scatter([xi - a], [t['xpx']], marker='v', s=110, color='#ffffff', edgecolor=BEAR, zorder=6, lw=1.4)
    for a_ in (ax, vx): a_.set_facecolor(BG); [s_.set_color('#26282c') for s_ in a_.spines.values()]
    ax.yaxis.tick_right(); vx.yaxis.tick_right(); ax.tick_params(colors=FG, labelsize=8); vx.tick_params(colors=FG, labelsize=7); plt.setp(ax.get_xticklabels(), visible=False)
    st = max(1, n // 8); vx.set_xticks(list(range(0, n, st))); vx.set_xticklabels([f'{B.index[q]:%b %d}' for q in range(0, n, st)], color=FG, fontsize=8); ax.set_xlim(-1, n + 1)
    net = t['ret'] - COST; stt = 'OPEN (marked at last close)' if t['open'] else f"closed {t['xt']:%b %d %H:%M}"
    fig.text(.06, .945, f"{t['tk']}  ·  BUY {t['et']:%a %b %d %H:%M}  ·  {t['ret']:+.1%}", color='#f1f2f4', fontsize=18, fontweight='bold')
    fig.text(.06, .905, f"entry {t['epx']:.2f} ({t['vr']:.1f}x volume, low-vol day, {t['vwx']:+.1f} ATR vs VWAP)  ·  exit {t['xpx']:.2f}  ·  {stt}", color=FG, fontsize=9)
    p = f"xs/sub5_recent_{k + 1}_{t['tk']}.png"; fig.savefig(p, dpi=125, facecolor=BG); plt.close(fig); files.append(p)
    rows.append(f"{k + 1} {t['tk']:5s} BUY {t['et']:%b %d %H:%M} @ {t['epx']:.2f} -> {t['xpx']:.2f}  {t['ret']:+.1%} (net {net:+.1%})  {'OPEN' if t['open'] else 'closed ' + format(t['xt'], '%b %d')}  vol {t['vr']:.1f}x  vwap {t['vwx']:+.1f}ATR")
print('\nLAST 5 sub-$5 signals (filter C):'); print('\n'.join(rows)); print('FILES', files)
