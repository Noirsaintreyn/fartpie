"""Nov 22 2026: charts of 10 stocks that bull-flipped on BOTH the 15m and 1h vector in the last month (Oct 22 - Nov 20 2026): price, vector level (green = bull regime, red = bear), EMA 50 and EMA 200 on the same timeframe, flip markers.  One PNG per timeframe (5 x 2 panels).  Random 10 of the qualifying names, seed 11."""
import glob, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; T0 = pd.Timestamp('2026-10-22'); T1 = pd.Timestamp('2026-11-21')
def agg1h(H): g = H.resample('60min', origin='start_day', offset='30min'); return pd.DataFrame({'open': g.open.first(), 'high': g.high.max(), 'low': g.low.min(), 'close': g.close.last(), 'volume': g.volume.sum()}).dropna()
D = {}
for f in sorted(glob.glob(SP + 'lastweek/*.parquet')):
    tk = f.split('/')[-1][:-8]; H = pd.read_parquet(f); res = {}
    for tf, X in (('15m', H), ('1h', agg1h(H))):
        regs, lv, atr, fl = run_state_machine(X); bull = [i for i, d in fl if d == 1 and T0 <= X.index[i] < T1]; res[tf] = (X, regs, lv, fl, bull)
    if res['15m'][4] and res['1h'][4]: D[tk] = res
names = sorted(D); rng = np.random.default_rng(11); pick = sorted(rng.choice(names, size=min(10, len(names)), replace=False)); print('qualifying names %d; chosen %s' % (len(names), ' '.join(pick)))
INK, BLUE, ORG, GRN, RED, MUT = '#101826', '#2A62D6', '#A44F00', '#1B7F4B', '#B3261E', '#6B7280'
for tf in ('15m', '1h'):
    fig, axs = plt.subplots(5, 2, figsize=(22, 20), dpi=90); fig.patch.set_facecolor('#F5F3EE')
    for ax, tk in zip(axs.ravel(), pick):
        X, regs, lv, fl, bull = D[tk][tf]; c = X.close; e50 = c.ewm(span=50, adjust=False).mean(); e200 = c.ewm(span=200, adjust=False).mean(); m = (X.index >= T0) & (X.index < T1); idx = np.where(m)[0]; x = np.arange(len(idx)); a, z = idx[0], idx[-1]
        ax.set_facecolor('#F5F3EE'); ax.plot(x, c.values[a:z + 1], color=INK, lw=1.0, label='Price', zorder=3)
        lvl = lv[a:z + 1].copy(); rg = regs[a:z + 1]
        for sign, col in ((1, GRN), (-1, RED)):
            y = np.where(rg == sign, lvl, np.nan); ax.plot(x, y, color=col, lw=1.6, ls='-' if sign == 1 else '--', label='Vector level (%s)' % ('bull' if sign == 1 else 'bear'), zorder=2)
        ax.plot(x, e50.values[a:z + 1], color=BLUE, lw=1.3, label='EMA 50'); ax.plot(x, e200.values[a:z + 1], color=ORG, lw=1.3, ls=':', label='EMA 200')
        for i, d in fl:
            if a <= i <= z:
                xi = i - a
                if d == 1: ax.scatter(xi, X.low.values[i], marker='^', s=70, color=GRN, edgecolor=INK, lw=.6, zorder=5, label='Bull flip')
                else: ax.scatter(xi, X.high.values[i], marker='v', s=70, color=RED, edgecolor=INK, lw=.6, zorder=5, label='Bear flip')
        days = X.index[a:z + 1].normalize(); first = [k for k in range(len(days)) if k == 0 or days[k] != days[k - 1]]; step = max(1, len(first) // 8); ax.set_xticks([first[k] for k in range(0, len(first), step)]); ax.set_xticklabels([days[first[k]].strftime('%b %d') for k in range(0, len(first), step)], fontsize=9, color=MUT)
        for k in first: ax.axvline(k, color='#D9D5CB', lw=.5, zorder=0)
        lo = min(np.nanmin(c.values[a:z + 1]), np.nanmin(lvl)); hi = np.nanmax(c.values[a:z + 1]); pad = (hi - lo) * .05; ax.set_ylim(lo - pad, hi + pad); ax.tick_params(axis='y', labelsize=9, colors=MUT)
        nb = len(D[tk]['15m'][4]) if tf == '15m' else len(D[tk]['1h'][4]); ax.set_title('%s  ·  %d bull flips on this timeframe in the window' % (tk, nb), loc='left', fontsize=12, color=INK, fontweight='bold')
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
    h, l = axs[0, 0].get_legend_handles_labels(); seen = {}; [seen.setdefault(b_, a_) for a_, b_ in zip(h, l)]
    fig.legend(seen.values(), seen.keys(), loc='upper center', ncol=7, fontsize=11, frameon=False, bbox_to_anchor=(.5, .975)); fig.suptitle('%s vector flips, Oct 22 - Nov 20 2026 (price, vector level, EMA 50 / 200 on the same timeframe)' % tf, y=.995, fontsize=15, color=INK, x=.5)
    fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig(SP + 'chart_flips_%s.png' % tf, dpi=90, facecolor=fig.get_facecolor()); print('saved', tf)
