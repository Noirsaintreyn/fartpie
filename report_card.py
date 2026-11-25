"""Nov 25 2026: one-page investor-style report image for a sub-$5 DAILY flip alert (daily_small.py). Header, KPI tiles, daily chart with the vector level, 5m chart, group-score ranks, track record (cumulative reach of +10%), catalyst / news / insider check, sign-off.
make_card(a, rep_text, d_df, five_m_png, path, ref) -> path. Never needs the network (everything is passed in)."""
import os, sys, textwrap, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xs'))
from vector_pine import run_state_machine
BG, PANEL, GRID, T1, T2, MUT = '#080c16', '#0f1626', '#1b2438', '#f1f5f9', '#94a3b8', '#56627a'
UP, DN, LVL, GOLD, GOOD, CRIT = '#3b82f6', '#fb7185', '#a3e635', '#fcd34d', '#34d399', '#f87171'
LAB = {'ATRP': 'Volatility (ATR % of price)', 'F11': 'Room above vector level', 'LTURN': 'Turnover (volume / shares)', 'LMC': 'Small market cap (lower = higher rank)'}
def _panel(fig, x, y, w, h, title=None):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.006', transform=fig.transFigure, fc=PANEL, ec=GRID, lw=.8, zorder=-8))
    if title: fig.text(x + .014, y + h - .022, title.upper(), fontsize=8.5, color=MUT, fontweight='bold', va='top')
def make_card(a, rep_text, d_df, five_m_png, path, ref, name=None):
    tk = a['tk']; F = a['F']; bt = ref['bt']; cv = ref.get('curve', {}); risk = F['F11']; size = float(a.get('size', 0.125))
    fig = plt.figure(figsize=(9.6, 13.2), facecolor=BG); fig.patches.append(plt.Rectangle((0, .985), 1, .015, transform=fig.transFigure, fc=GOOD, ec='none', zorder=-5))
    fig.text(.04, .958, 'DAILY BUY', fontsize=10, color=GOOD, fontweight='bold'); fig.text(.04, .925, tk, fontsize=30, color=T1, fontweight='bold')
    fig.text(.04, .903, (name or '')[:60], fontsize=11, color=T2)
    fig.text(.04, .884, f"Sub-$5 high-volatility group  ·  daily vector bull flip  ·  {a['day']:%a %b %d %Y} close {a['px']:.2f}", fontsize=10, color=T2)
    items = [('ENTRY', 'next open', f"~{a['px']:.2f} (flip close)"), ('VECTOR LEVEL', f"{a['lev']:.2f}", f"{risk:.0%} below price · broke {a['prev']:.2f}"), ('SIZE', f"{size:.1%}", 'of book · scaled by distance to level'), ('EXIT', 'vector bear flip', 'daily vector, no other stop')]
    for k, (h, v, s) in enumerate(items):
        x0 = .04 + k * .2375; fig.patches.append(FancyBboxPatch((x0, .815), .2225, .06, boxstyle='round,pad=0,rounding_size=0.006', transform=fig.transFigure, fc=PANEL, ec='none', zorder=-8))
        fig.text(x0 + .012, .861, h, fontsize=7.5, color=MUT, fontweight='bold'); fig.text(x0 + .012, .836, v, fontsize=14 if len(v) < 10 else 11.5, color=CRIT if h == 'VECTOR LEVEL' else T1, fontweight='bold'); fig.text(x0 + .012, .821, s, fontsize=6.6, color=T2)
    # daily chart
    _panel(fig, .04, .585, .92, .215, 'Daily chart · vector level'); ax = fig.add_axes([.065, .60, .875, .16]); ax.set_facecolor(PANEL)
    q = d_df.iloc[-90:].copy(); reg, lev, _, _ = run_state_machine(d_df.assign(datetime=d_df.index)); lev = pd.Series(lev, index=d_df.index).iloc[-90:]; rg = pd.Series(reg, index=d_df.index).iloc[-90:]
    for i, (dt_, r) in enumerate(q.iterrows()):
        c_ = UP if rg.iloc[i] == 1 else DN; ax.plot([i, i], [r.low, r.high], color=c_, lw=.8); ax.add_patch(plt.Rectangle((i - .34, min(r.open, r.close)), .68, max(abs(r.close - r.open), 1e-4), facecolor=c_ if r.close >= r.open else PANEL, edgecolor=c_, lw=.6))
    ax.step(range(len(q)), lev.values, where='mid', color=LVL, lw=1.3, alpha=.9, label='vector level'); fi = len(q) - 1; ax.scatter([fi], [q.close.iloc[-1] * 0.97], marker='^', color=GOLD, s=60, zorder=6); ax.annotate('FLIP', (fi, q.close.iloc[-1] * .97), xytext=(fi - 9, q.close.iloc[-1] * .9), color=GOLD, fontsize=8, fontweight='bold')
    ax.set_xlim(-1, len(q) + 1); ax.set_xticks([0, len(q) // 2, len(q) - 1]); ax.set_xticklabels([f'{q.index[i]:%b %d}' for i in (0, len(q) // 2, len(q) - 1)], color=MUT, fontsize=8)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=MUT, labelsize=8); ax.grid(color=GRID, lw=.5); ax.legend(loc='upper left', frameon=False, labelcolor=T2, fontsize=8); ax.text(.995, .02, 'blue = vector bull · rose = bear · hollow = down bar', transform=ax.transAxes, ha='right', color=MUT, fontsize=7)
    # 5m chart
    _panel(fig, .04, .325, .92, .25, '5-minute chart · flip day and after')
    if five_m_png and os.path.exists(five_m_png):
        im = plt.imread(five_m_png); h_ = im.shape[0]; im = im[int(h_ * .085):int(h_ * .765)]; a5 = fig.add_axes([.05, .33, .90, .218]); a5.imshow(im, aspect='auto'); a5.axis('off')
    # ranks
    _panel(fig, .04, .17, .445, .145, 'Why it qualifies')
    ar = fig.add_axes([.19, .18, .28, .095]); ar.set_facecolor(PANEL); Q = np.linspace(0, 1, 101); rk = []
    for f, sg in ref['signs'].items():
        x = F.get(f); p = float(np.interp(x, ref['ref'][f], Q)) if x == x and x is not None else np.nan; rk.append((f, p if sg > 0 else 1 - p))
    for i, (f, p) in enumerate(rk[::-1]): ar.barh(i, 1, color=GRID, height=.5); ar.barh(i, p if p == p else 0, color=UP, height=.5)
    ar.set_yticks(range(len(rk))); ar.set_yticklabels([LAB[f] for f, _ in rk[::-1]], color=T2, fontsize=7.2); ar.set_xlim(0, 1); ar.set_xticks([]); [s.set_visible(False) for s in ar.spines.values()]; ar.tick_params(length=0)
    fig.text(.47, .296, f"score {a['score']:.2f}  (cut {ref['cut']:.2f})", fontsize=9, color=GOOD, fontweight='bold', ha='right', va='top'); fig.text(.47, .178, 'bar = rank vs 2019-21 train flips', fontsize=6.4, color=MUT, ha='right')
    # track record
    _panel(fig, .515, .17, .445, .145, 'Track record of this setup'); at = fig.add_axes([.555, .20, .39, .07]); at.set_facecolor(PANEL)
    ks = [1, 2, 3, 5, 10, 20, 40]; ys = [cv.get(str(k), np.nan) * 100 for k in ks]; at.plot(ks, ys, color=GOOD, lw=2, marker='o', ms=3.5); at.set_ylim(0, 100); at.set_xscale('log'); at.set_xticks(ks); at.set_xticklabels([str(k) for k in ks], color=MUT, fontsize=7); at.tick_params(colors=MUT, labelsize=7); at.grid(color=GRID, lw=.5); [s.set_visible(False) for s in at.spines.values()]
    fig.text(.945, .298, f"{bt['reach']:.0%} reach +10%", fontsize=12, color=T1, fontweight='bold', ha='right', va='top'); fig.text(.945, .282, f"median {cv.get('median_days', 3):.0f} trading days to get there", fontsize=7, color=T2, ha='right', va='top'); fig.text(.53, .175, f"cumulative % of flips, 2022-Mar 2026, n {cv.get('n', bt['n'])}; the rest lose", fontsize=6.4, color=MUT)
    # catalyst
    _panel(fig, .04, .04, .92, .115, None); fig.text(.054, .148, 'CATALYST · NEWS · INSIDERS (last 10-30 days)', fontsize=8.5, color=MUT, fontweight='bold', va='top')
    L0 = [l.replace('⚠️ ', '').replace('• ', '').strip() for l in rep_text.split('\n')[1:-1] if l.strip()]; flag = L0[-1]; B_ = L0[:-1]; ins = [x for x in B_ if x.startswith('insiders')]; nw = [x for x in B_ if x.startswith('news:')]; oth = [x for x in B_ if x not in ins and x not in nw]; L = oth[:3] + ins + nw[:2] + [flag]
    bad = 'BAD CATALYST' in flag
    fig.text(.054, .132, '\n'.join(textwrap.shorten(x, 125, placeholder='...') for x in L[:7]), fontsize=7.4, color=T1, va='top', linespacing=1.4)
    if bad: fig.text(.945, .148, 'WARNING', fontsize=8, color=CRIT, fontweight='bold', ha='right', va='top')
    fig.text(.96, .012, '~Valiant', fontsize=13, color=T2, style='italic', family='DejaVu Serif', ha='right'); fig.text(.04, .012, 'Backtest in-sample and survivorship-biased; most trades lose 10-15%. Not advice.', fontsize=6.4, color=MUT)
    fig.savefig(path, dpi=130, facecolor=BG); plt.close(fig); return path
