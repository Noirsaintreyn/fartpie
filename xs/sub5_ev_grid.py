"""Contact sheets for the sub-$5 C+E+V book: last 10 trades and last 10 winning (closed, net>0) trades. 1h candles, session VWAP, entry (^) / exit (v)."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import midsmall_screener as M
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb'))
E = pd.read_pickle(SC + 'earnmac_T2.pkl'); V = pd.read_pickle(SC + 'sub5_T.pkl')[['tk', 'et', 'vwx']]; T = E[E.sub5].merge(V, on=['tk', 'et'], how='left'); COST = .004; K = 8
T = T[(T.vr >= 1.5) & T.lv & ~(T.dte <= 3) & ~(T.vwx > 1)]; rows = sorted(T.to_dict('records'), key=lambda r: (r['et'], -r['vr'])); eq = 1.0; op = []; taken = []
def close(p):
    global eq; eq += p['sz'] * (p['ret'] - COST)
for r in rows:
    still = []
    for p in op:
        if p['xt'] <= r['et']: close(p)
        else: still.append(p)
    op = still
    if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r); taken.append(r)
BULL, BEAR, BG, FG = '#3b6cff', '#ff5d8f', '#0b0b0c', '#8a8d93'
def sheet(trs, title, path):
    fig, axs = plt.subplots(2, 5, figsize=(22, 9.5), facecolor=BG); fig.subplots_adjust(left=.02, right=.985, top=.88, bottom=.03, wspace=.12, hspace=.22)
    for ax, t in zip(axs.flat, trs):
        Yc = M.closed_only(data[t['tk']]); F = M.features(Yc); fi = int(t['fi']); ei = fi + 1; xi = int(np.searchsorted(Yc.index.values, np.datetime64(t['xt']))); a = max(0, fi - 24); b = min(len(Yc), xi + 8); B = Yc.iloc[a:b]; rg = F.reg.values[a:b]; n = len(B); rngv = B.high.max() - B.low.min()
        day = Yc.index.normalize(); vwa = (pd.Series((Yc.high.values + Yc.low.values + Yc.close.values) / 3 * Yc.volume.values).groupby(day).cumsum() / (pd.Series(Yc.volume.values).groupby(day).cumsum() + 1e-9)).values[a:b]
        for q in range(n):
            c = BULL if rg[q] > 0 else BEAR; ax.plot([q, q], [B.low.iat[q], B.high.iat[q]], color=c, lw=.8); lo, hi = sorted([B.open.iat[q], B.close.iat[q]]); ax.add_patch(plt.Rectangle((q - .35, lo), .7, max(hi - lo, rngv * .003), color=c))
        ax.plot(range(n), vwa, color='#d0a94a', lw=.9, ls='--'); ax.scatter([ei - a], [t['epx']], marker='^', s=80, color='#fff', edgecolor=BULL, zorder=6, lw=1.3)
        if not t['open']: ax.scatter([xi - a], [t['xpx']], marker='v', s=80, color='#fff', edgecolor=BEAR, zorder=6, lw=1.3)
        net = t['ret'] - COST; ax.set_facecolor(BG); [s.set_color('#26282c') for s in ax.spines.values()]; ax.set_xticks([]); ax.yaxis.tick_right(); ax.tick_params(colors=FG, labelsize=7); ax.set_xlim(-1, n + 1)
        ax.set_title(f"{t['tk']}  {net:+.1%}{' (open)' if t['open'] else ''}", color='#4ade80' if net > 0 else '#ff6b81', fontsize=13, fontweight='bold', loc='left')
        ax.text(0, 1.0, '', transform=ax.transAxes); ax.text(.0, -.02, f"{t['et']:%b %d} @ {t['epx']:.2f} -> {t['xpx']:.2f}", color=FG, fontsize=8, transform=ax.transAxes, va='top')
    for ax in list(axs.flat)[len(trs):]: ax.axis('off')
    fig.text(.02, .94, title, color='#f1f2f4', fontsize=20, fontweight='bold'); fig.text(.02, .91, 'sub-$5 book · vol>=1.5x, low-vol day, not <=3 sessions before earnings, not >1 ATR above VWAP · net of 40 bp · yellow dashed = session VWAP', color=FG, fontsize=10)
    fig.savefig(path, dpi=100, facecolor=BG); plt.close(fig)
last10 = taken[-10:]; wins = [t for t in taken if not t['open'] and t['ret'] - COST > 0][-10:]
sheet(last10, 'Last 10 trades', 'xs/sub5_ev_last10.png'); sheet(wins, 'Last 10 winning trades (closed)', 'xs/sub5_ev_wins10.png')
f = lambda t: f"{t['tk']:5s} {t['et']:%b %d} {t['epx']:.2f}->{t['xpx']:.2f} net {t['ret'] - COST:+.1%} {'OPEN' if t['open'] else ''}"
print('LAST 10'); print('\n'.join(f(t) for t in last10)); print('LAST 10 WINNERS'); print('\n'.join(f(t) for t in wins))
print('last10 net mean', np.mean([t['ret'] - COST for t in last10]).round(4), 'wins in last10', sum(t['ret'] - COST > 0 for t in last10))
