"""Contact sheets for the live $1-2 book (vol>=1.5x, low-vol day, not <=3 sessions before earnings, not >1 ATR above VWAP) tagged A-GRADE = daily vector bull. 1h candles, 50 / 200 EMA (1h), quarterly anchored VWAP, entry ^ / exit v."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.'); sys.path.insert(0, 'xs')
import midsmall_screener as M
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb')); COST = .004; K = 8
T = pd.read_pickle(SC + 'mtf_T.pkl'); T['xpx'] = T.epx * (1 + T.ret); V = pd.read_pickle(SC + 'sub5_T.pkl')[['tk', 'et', 'vwx']]; E = pd.read_pickle(SC + 'earnmac_T2.pkl')[['tk', 'et', 'dte']]
T = T.merge(V, on=['tk', 'et'], how='left').merge(E, on=['tk', 'et'], how='left')
T = T[(T.sig >= 1) & (T.sig < 2) & (T.vr >= 1.5) & T.lv & ~(T.dte <= 3) & ~(T.vwx > 1)]; print('live-rule $1-2 flips', len(T), ' A-GRADE (daily bull)', int(T.ad.sum()), ' mean net all / A / not-A', f'{(T.ret-COST).mean():+.2%} {(T[T.ad].ret-COST).mean():+.2%} {(T[~T.ad].ret-COST).mean():+.2%}')
A = T[T.ad]; rows = sorted(A.to_dict('records'), key=lambda r: (r['et'], -r['vr'])); eq = 1.0; op = []; taken = []
for r in rows:
    still = []
    for p in op:
        if p['xt'] <= r['et']: eq += p['sz'] * (p['ret'] - COST)
        else: still.append(p)
    op = still
    if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r); taken.append(r)
print('A-GRADE trades taken in the 8-slot book', len(taken), ' mean net', f"{np.mean([t['ret'] - COST for t in taken]):+.2%}", ' win', f"{np.mean([t['ret'] - COST > 0 for t in taken]):.0%}")
BULL, BEAR, BG, FG = '#3b6cff', '#ff5d8f', '#0b0b0c', '#8a8d93'
def sheet(trs, title, path):
    fig, axs = plt.subplots(2, 5, figsize=(22, 9.8), facecolor=BG); fig.subplots_adjust(left=.02, right=.985, top=.87, bottom=.03, wspace=.12, hspace=.24)
    for ax, t in zip(axs.flat, trs):
        Yc = M.closed_only(data[t['tk']].dropna(subset=['open', 'close'])); F = M.features(Yc); rg = F.reg.values; ei = int(np.searchsorted(Yc.index.values, np.datetime64(t['et']))); xi = int(np.searchsorted(Yc.index.values, np.datetime64(t['xt'])))
        c = Yc.close; e50 = c.ewm(span=50, adjust=False).mean().values; e200 = c.ewm(span=200, adjust=False).mean().values
        q = Yc.index.to_period('Q'); tp = (Yc.high + Yc.low + Yc.close) / 3; qv = ((tp * Yc.volume).groupby(q).cumsum() / (Yc.volume.groupby(q).cumsum() + 1e-9)).values
        a = max(0, ei - 40); b = min(len(Yc), xi + 8, a + 170); B = Yc.iloc[a:b]; n = len(B)
        for k in range(n):
            col = BULL if rg[a + k] > 0 else BEAR; ax.plot([k, k], [B.low.iat[k], B.high.iat[k]], color=col, lw=.8); lo, hi = sorted([B.open.iat[k], B.close.iat[k]]); ax.add_patch(plt.Rectangle((k - .35, lo), .7, max(hi - lo, (B.high.max() - B.low.min()) * .003), color=col))
        ax.plot(range(n), e50[a:b], color='#5ad1ff', lw=1.1); ax.plot(range(n), e200[a:b], color='#b58cff', lw=1.1); ax.plot(range(n), qv[a:b], color='#d0a94a', lw=1.0, ls='--')
        ax.scatter([ei - a], [t['epx']], marker='^', s=80, color='#fff', edgecolor=BULL, zorder=6, lw=1.3)
        if xi < b: ax.scatter([xi - a], [t['xpx']], marker='v', s=80, color='#fff', edgecolor=BEAR, zorder=6, lw=1.3)
        lo_, hi_ = B.low.min(), B.high.max(); pad = (hi_ - lo_) * .08; ax.set_ylim(lo_ - pad, hi_ + pad); net = t['ret'] - COST
        ax.set_facecolor(BG); [s.set_color('#26282c') for s in ax.spines.values()]; ax.set_xticks([]); ax.yaxis.tick_right(); ax.tick_params(colors=FG, labelsize=7); ax.set_xlim(-1, n + 1)
        ax.set_title(f"{t['tk']}  {net:+.1%}", color='#4ade80' if net > 0 else '#ff6b81', fontsize=13, fontweight='bold', loc='left'); ax.text(.0, -.02, f"{t['et']:%b %d '%y} @ {t['epx']:.2f} -> {t['xpx']:.2f} · vol {t['vr']:.1f}x", color=FG, fontsize=8, transform=ax.transAxes, va='top')
    for ax in list(axs.flat)[len(trs):]: ax.axis('off')
    fig.text(.02, .935, title, color='#f1f2f4', fontsize=20, fontweight='bold'); fig.text(.02, .905, '$1-2 book, A-GRADE (daily vector bull) · vol>=1.5x, low-vol day, no earnings <=3 sessions, <=1 ATR above VWAP · net of 40 bp · blue = 50 EMA, purple = 200 EMA (1h), yellow dashed = quarterly VWAP', color=FG, fontsize=10)
    fig.savefig(path, dpi=100, facecolor=BG); plt.close(fig)
last10 = taken[-10:]; wins = [t for t in taken if t['ret'] - COST > 0][-10:]
sheet(last10, 'Last 10 trades', 'xs/sub2_ev_last10.png'); sheet(wins, 'Last 10 winning trades', 'xs/sub2_ev_wins10.png')
f = lambda t: f"{t['tk']:5s} {t['et']:%b %d %y} {t['epx']:.2f}->{t['xpx']:.2f} net {t['ret'] - COST:+.1%}"
print('LAST 10'); print('\n'.join(f(t) for t in last10)); print('LAST 10 WINNERS'); print('\n'.join(f(t) for t in wins))
