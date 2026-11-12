"""Equity curve of the sub-$5 book (8 slots, 40 bp round trip, realised at exit time) vs SPY buy-and-hold, Dec 2023 - Nov 2026."""
import sys, numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; T = pd.read_pickle(SC + 'sub5_T.pkl'); COST = .004; K = 8
def curve(d):
    rows = sorted(d.to_dict('records'), key=lambda r: (r['et'], -(r['vr'] if r['vr'] == r['vr'] else 0))); eq = 1.0; op = []; pts = [(pd.Timestamp('2023-12-01'), 1.0)]
    def close(p):
        nonlocal eq
        eq += p['sz'] * (p['ret'] - COST); pts.append((p['xt'], eq))
    for r in rows:
        still = []
        for p in op:
            if p['xt'] <= r['et']: close(p)
            else: still.append(p)
        op = still
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r)
    for p in op: close(p)
    s = pd.Series({t: v for t, v in sorted(pts)}); return s[~s.index.duplicated(keep='last')]
A = curve(T); CV = curve(T[((T.vr >= 1.5) & T.lv & (T.vwx <= 1)).values]); BV = curve(T[((T.vr >= 1.5) & (T.vwx <= 1)).values])
spy = yf.download('SPY', start='2023-12-01', progress=False, auto_adjust=True)['Close'].squeeze(); spy = spy / spy.iloc[0]; spy.index = spy.index.tz_localize(None) if spy.index.tz else spy.index
BG, FG = '#0b0b0c', '#8a8d93'; fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG); ax.set_facecolor(BG)
for s, lab, col, lw in ((A, 'Sub-$5 book, all flips (+91%/yr)', '#3b6cff', 2.2), (BV, 'Sub-$5, volume >= 1.5x, not extended from VWAP (+130%/yr)', '#7ad39b', 1.3), (CV, 'Sub-$5, low-vol day + volume + not extended (+71%/yr)', '#d0a94a', 1.3), (spy, 'SPY buy & hold', '#ff5d8f', 2.0)):
    ax.plot(s.index, s.values, color=col, lw=lw, label=lab)
ax.set_yscale('log'); ax.set_yticks([1, 2, 4, 8]); ax.set_yticklabels(['1x', '2x', '4x', '8x']); ax.tick_params(colors=FG); [s_.set_color('#26282c') for s_ in ax.spines.values()]; ax.grid(color='#1a1c20', lw=.6)
ax.legend(loc='upper left', facecolor=BG, edgecolor='#26282c', labelcolor='#cfd1d6', fontsize=9)
fig.text(.07, .945, 'Sub-$5 mid/small book vs SPY', color='#f1f2f4', fontsize=18, fontweight='bold'); fig.text(.07, .905, f'Growth of $1, log scale · 8 slots · 40 bp round trip · BACKTEST on survivors (names alive today), realised at exits · SPY +{(spy.iloc[-1] - 1) * 100:.0f}%, book +{(A.iloc[-1] - 1) * 100:.0f}%', color=FG, fontsize=9)
fig.subplots_adjust(top=.86, left=.07, right=.97, bottom=.08); fig.savefig('xs/sub5_vs_spy.png', dpi=130, facecolor=BG); print('spy', spy.iloc[-1], 'book', A.iloc[-1], BV.iloc[-1], CV.iloc[-1])
