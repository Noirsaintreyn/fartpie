"""Nov 24 2026 report 439 = equity curve + max drawdown of the intraday negative-gamma flip model (report 428 candidate) for the #neg-gamma-flips channel.
Model: 15m vector bull flip, first gated flip per negative-gamma episode, options-ratio gate G20 (20d median >= 2), exit X3 (first of positive-gamma open or vector flip) and X1 (vector flip) for comparison, 4bp.
Portfolio: 6 slots, each trade sized 1/6 of equity at entry, first come first served (trades while all 6 are full are skipped), P&L booked at the exit (REALIZED basis: open-trade drawdowns are not marked, so the max drawdown shown is a floor).  Names = those with gex + greeks files (11), 2018-07 to 2025-12 (the gex files end 2025-12-30).  Lead, not an admitted rule."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
T = pd.read_parquet(SP + 'g15m428.parquet'); T = T[(T.gate == 'G20') & (T.arm == 'FLIP')].copy(); T['exit_t'] = T.entry + pd.to_timedelta(T.hold, unit='D')
def book(D, slots=6):
    D = D.sort_values('entry'); open_, out = [], []
    for r in D.itertuples():
        open_ = [x for x in open_ if x > r.entry]
        if len(open_) >= slots: continue
        open_.append(r.exit_t); out.append(r)
    X = pd.DataFrame(out); X['pnl'] = X.ret / slots; return X.sort_values('exit_t')
def curve(X):
    s = X.groupby(X.exit_t.dt.normalize()).pnl.apply(lambda p: (1 + p).prod() - 1); idx = pd.bdate_range(X.entry.min().normalize(), X.exit_t.max().normalize()); r = s.reindex(idx).fillna(0); eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
    yrs = (idx[-1] - idx[0]).days / 365.25; return eq, dd, dict(cagr=eq.iloc[-1] ** (1 / yrs) - 1, mdd=dd.min(), sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan, n=len(X), win=(X.ret > 0).mean(), per_yr=len(X) / yrs, tot=eq.iloc[-1] - 1)
res = {}
for ex in ('X3', 'X1'):
    X = book(T[T.exit == ex]); eq, dd, st = curve(X); res[ex] = (eq, dd, st); print(ex, {k: round(float(v), 3) for k, v in st.items()})
    for e in ('18-19', '20-21', '22', '23-26'):
        Xe = X[X.era == e]
        if len(Xe): print(f"   {e}: n {len(Xe)} win {(Xe.ret>0).mean()*100:.0f}% mean/trade {Xe.ret.mean()*100:.2f}%")
BG, PANEL, T1, MUT, GOOD, CRIT, GOLD = '#05080f', '#0a1322', '#dfe7f1', '#7d8ca0', '#4da3ff', '#ff4d5e', '#fbbf24'
fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6.4), facecolor=BG, gridspec_kw=dict(height_ratios=[3, 1.2], hspace=.08), sharex=True)
for ax in (a, b):
    ax.set_facecolor(PANEL); [s.set_visible(False) for s in ax.spines.values()]; ax.tick_params(colors=MUT, labelsize=8); ax.grid(color='#1a2638', lw=.5, alpha=.7)
e3, d3, s3 = res['X3']; e1, d1, s1 = res['X1']
a.plot(e3.index, (e3 - 1) * 100, color=GOOD, lw=1.8, label='exit: positive gamma or vector flip'); a.plot(e1.index, (e1 - 1) * 100, color=MUT, lw=1.1, ls='--', label='exit: vector flip only')
a.set_ylabel('return %', color=MUT, fontsize=8); a.legend(loc='upper left', frameon=False, labelcolor=T1, fontsize=8)
b.fill_between(d3.index, d3 * 100, 0, color=CRIT, alpha=.55); b.set_ylabel('drawdown %', color=MUT, fontsize=8)
fig.text(.075, .955, 'Negative-gamma 15m vector flips + options-ratio gate', color=T1, fontsize=14, fontweight='bold')
fig.text(.075, .915, f"2018-07 to 2025-12 · 11 names · 6 slots · {s3['n']:.0f} trades · total {s3['tot']*100:+.0f}% · CAGR {s3['cagr']*100:.1f}% · max DD {s3['mdd']*100:.1f}% · win {s3['win']*100:.0f}%", color=MUT, fontsize=9)
fig.text(.075, .025, 'Realized-P&L basis (open-trade drawdown not marked: real max DD is deeper). 4bp costs. Lead under test, not a validated rule.', color=MUT, fontsize=7.5)
fig.subplots_adjust(left=.075, right=.97, top=.88, bottom=.09); fig.savefig('/tmp/neggamma_backtest.png', facecolor=BG, dpi=150); print('saved')
