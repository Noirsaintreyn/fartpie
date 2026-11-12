"""Last trades taken by the sub-$5 C + E + V book (vol>=1.5x, low-vol day, not <=3 sessions before earnings, not >1 ATR above session VWAP); 8 slots, 40bp. Charts: 1h candles, session VWAP, entry/exit, earnings reaction marker."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import midsmall_screener as M
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb'))
E = pd.read_pickle(SC + 'earnmac_T2.pkl'); V = pd.read_pickle(SC + 'sub5_T.pkl')[['tk', 'et', 'vwx']]; T = E[E.sub5].merge(V, on=['tk', 'et'], how='left'); COST = .004; K = 8
T = T[(T.vr >= 1.5) & T.lv & ~(T.dte <= 3) & ~(T.vwx > 1)]
rows = sorted(T.to_dict('records'), key=lambda r: (r['et'], -r['vr'])); eq = 1.0; op = []; taken = []
def close(p):
    global eq; eq += p['sz'] * (p['ret'] - COST)
for r in rows:
    still = []
    for p in op:
        if p['xt'] <= r['et']: close(p)
        else: still.append(p)
    op = still
    if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r); taken.append(r)
last = taken[-5:]; print('taken', len(taken)); BULL, BEAR, BG, FG = '#3b6cff', '#ff5d8f', '#0b0b0c', '#8a8d93'; files = []; out = []
for k, t in enumerate(last):
    Yc = M.closed_only(data[t['tk']]); F = M.features(Yc); fi = int(t['fi']); ei = fi + 1; xi = int(np.searchsorted(Yc.index.values, np.datetime64(t['xt']))); a = max(0, fi - 45); b = min(len(Yc), xi + 14); B = Yc.iloc[a:b]; rg = F.reg.values[a:b]; n = len(B)
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
    stt = 'OPEN (marked at last close)' if t['open'] else f"closed {t['xt']:%b %d %H:%M}"; net = t['ret'] - COST
    fig.text(.06, .945, f"{t['tk']}  ·  BUY {t['et']:%a %b %d %H:%M}  ·  {t['ret']:+.1%}", color='#f1f2f4', fontsize=18, fontweight='bold')
    fig.text(.06, .905, f"entry {t['epx']:.2f} ({t['vr']:.1f}x vol, low-vol day, {t['vwx']:+.1f} ATR vs VWAP, earnings in {int(t['dte'])} sessions)  ·  exit {t['xpx']:.2f}  ·  {stt}", color=FG, fontsize=9)
    p = f"xs/sub5_ev_{k + 1}_{t['tk']}.png"; fig.savefig(p, dpi=125, facecolor=BG); plt.close(fig); files.append(p)
    out.append(f"{k + 1} {t['tk']:5s} {t['et']:%b %d %H:%M} @ {t['epx']:.2f} -> {t['xpx']:.2f} {t['ret']:+.1%} net {net:+.1%} {'OPEN' if t['open'] else 'closed ' + format(t['xt'], '%b %d')} vol {t['vr']:.1f}x vwap {t['vwx']:+.1f}ATR dte {int(t['dte'])}")
print('\n'.join(out)); print(files)
