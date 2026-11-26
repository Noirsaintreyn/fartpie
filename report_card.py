"""Nov 25 2026: one-page investor-style report image for a sub-$5 DAILY flip alert (daily_small.py). Header with the overall GRADE, KPI tiles, SETUP SCORECARD (every factor: today's reading, a tone and the stat of how that state did in the backtest), daily chart with the vector level, 5m chart, track record, catalyst / news / insider check, sign-off.
make_card(a, rep_text, d_df, five_m_png, path, ref, name=None) -> path; a['sc'] is the catalysts.scorecard() dict. No network needed."""
import os, sys, textwrap, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xs'))
from vector_pine import run_state_machine
BG, PANEL, GRID, T1, T2, MUT = '#080c16', '#0f1626', '#1b2438', '#f1f5f9', '#94a3b8', '#56627a'
UP, DN, LVL, GOLD, GOOD, CRIT = '#3b82f6', '#fb7185', '#a3e635', '#fcd34d', '#34d399', '#f87171'
GC = {'A': GOOD, 'B': '#60a5fa', 'C': T2, 'D': CRIT}; TONE = {'+': ('▲', GOOD), '0': ('●', MUT), '-': ('▼', CRIT)}
H = 15.4
def _panel(fig, x, y, w, h, title=None):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.005', transform=fig.transFigure, fc=PANEL, ec=GRID, lw=.8, zorder=-8))
    if title: fig.text(x + .014, y + h - .008, title.upper(), fontsize=8.5, color=MUT, fontweight='bold', va='top')
def make_card(a, rep_text, d_df, five_m_png, path, ref, name=None):
    tk = a['tk']; F = a['F']; bt = ref['bt']; cv = ref.get('curve', {}); risk = F['F11']; size = float(a.get('size', 0.125)); sc = a.get('sc') or {}; gr = sc.get('grade')
    fig = plt.figure(figsize=(9.6, H), facecolor=BG); fig.patches.append(plt.Rectangle((0, .988), 1, .012, transform=fig.transFigure, fc=GC.get(gr, GOOD), ec='none', zorder=-5))
    fig.text(.04, .971, 'DAILY BUY', fontsize=10, color=GOOD, fontweight='bold'); fig.text(.04, .938, tk, fontsize=30, color=T1, fontweight='bold'); fig.text(.04, .921, (name or '')[:60], fontsize=11, color=T2)
    fig.text(.04, .905, f"Sub-$5 high-volatility group  ·  daily vector bull flip  ·  {a['day']:%a %b %d %Y} close {a['px']:.2f}", fontsize=10, color=T2)
    if gr:
        fig.patches.append(FancyBboxPatch((.80, .9, ), .16, .075, boxstyle='round,pad=0,rounding_size=0.006', transform=fig.transFigure, fc=PANEL, ec=GC[gr], lw=1.6, zorder=-8)); fig.text(.88, .958, 'GRADE', fontsize=8, color=MUT, ha='center', fontweight='bold'); fig.text(.88, .911, gr, fontsize=34, color=GC[gr], ha='center', fontweight='bold')
    items = [('ENTRY', 'next open', f"~{a['px']:.2f} (flip close)"), ('VECTOR LEVEL', f"{a['lev']:.2f}", f"{risk:.0%} below price · broke {a['prev']:.2f}"), ('SIZE', f"{size:.1%}", 'of book · scaled by distance to level'), ('EXIT', 'vector bear flip', 'daily vector, no other stop')]
    for k, (h, v, s) in enumerate(items):
        x0 = .04 + k * .2375; fig.patches.append(FancyBboxPatch((x0, .835), .2225, .052, boxstyle='round,pad=0,rounding_size=0.005', transform=fig.transFigure, fc=PANEL, ec='none', zorder=-8))
        fig.text(x0 + .012, .875, h, fontsize=7.5, color=MUT, fontweight='bold'); fig.text(x0 + .012, .853, v, fontsize=14 if len(v) < 10 else 11.5, color=CRIT if h == 'VECTOR LEVEL' else T1, fontweight='bold'); fig.text(x0 + .012, .840, s, fontsize=6.6, color=T2)
    # scorecard
    rows = sc.get('rows', []); top = .82; rh = .0205; ph = .045 + rh * (len(rows) + 1.2); _panel(fig, .04, top - ph, .92, ph, 'Setup scorecard · every factor with how it did before')
    y = top - .034
    if gr:
        co = sc.get('cohort') or {}; ct = sc.get('cohort_test') or {}
        fig.text(.054, y, f'GRADE {gr}', fontsize=11, color=GC[gr], fontweight='bold', va='top'); fig.text(.17, y - .001, f"setups scoring like this: {co.get('hits', 0)}/{co.get('n', 0)} reached +10% ({co.get('hits', 0) / max(co.get('n', 1), 1):.0%}) since 2019 · test 2022+ only {ct.get('hits', 0)}/{ct.get('n', 0)} ({ct.get('hits', 0) / max(ct.get('n', 1), 1):.0%}) · median {((co.get('median') or 0)):+.0%}", fontsize=8, color=T1, va='top')
        y -= rh * 1.25
    for nm, today, tone, stat in rows:
        g_, c_ = TONE[tone]; fig.text(.054, y, nm, fontsize=7.6, color=T2, va='top'); fig.text(.215, y, textwrap.shorten(today, 40, placeholder='...'), fontsize=7.6, color=T1, va='top'); fig.text(.485, y, g_, fontsize=8, color=c_, va='top'); fig.text(.51, y, textwrap.shorten(stat, 92, placeholder='...'), fontsize=6.8, color=T2, va='top'); y -= rh
    # daily chart
    y1 = top - ph - .012; hd = .16; _panel(fig, .04, y1 - hd, .92, hd, 'Daily chart · vector level'); ax = fig.add_axes([.065, y1 - hd + .012, .875, hd - .035]); ax.set_facecolor(PANEL)
    q = d_df.iloc[-90:].copy(); reg, lev, _, _ = run_state_machine(d_df.assign(datetime=d_df.index)); lev = pd.Series(lev, index=d_df.index).iloc[-90:]; rg = pd.Series(reg, index=d_df.index).iloc[-90:]
    for i, (dt_, r) in enumerate(q.iterrows()):
        c_ = UP if rg.iloc[i] == 1 else DN; ax.plot([i, i], [r.low, r.high], color=c_, lw=.8); ax.add_patch(plt.Rectangle((i - .34, min(r.open, r.close)), .68, max(abs(r.close - r.open), 1e-4), facecolor=c_ if r.close >= r.open else PANEL, edgecolor=c_, lw=.6))
    ax.step(range(len(q)), lev.values, where='mid', color=LVL, lw=1.3, alpha=.9, label='vector level'); fi = len(q) - 1; ax.scatter([fi], [q.close.iloc[-1] * 0.97], marker='^', color=GOLD, s=60, zorder=6); ax.annotate('FLIP', (fi, q.close.iloc[-1] * .97), xytext=(fi - 9, q.close.iloc[-1] * .9), color=GOLD, fontsize=8, fontweight='bold')
    ax.set_xlim(-1, len(q) + 1); ax.set_xticks([0, len(q) // 2, len(q) - 1]); ax.set_xticklabels([f'{q.index[i]:%b %d}' for i in (0, len(q) // 2, len(q) - 1)], color=MUT, fontsize=8)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=MUT, labelsize=8); ax.grid(color=GRID, lw=.5); ax.legend(loc='lower left', frameon=False, labelcolor=T2, fontsize=8); ax.text(.995, .02, 'blue = vector bull · rose = bear · hollow = down bar', transform=ax.transAxes, ha='right', color=MUT, fontsize=7)
    # 5m chart
    y2 = y1 - hd - .012; h5 = .235; _panel(fig, .04, y2 - h5, .92, h5, '5-minute chart · flip day and after')
    if five_m_png and os.path.exists(five_m_png):
        im = plt.imread(five_m_png); h_ = im.shape[0]; im = im[int(h_ * .105):int(h_ * .765)]; a5 = fig.add_axes([.05, y2 - h5 + .004, .90, h5 - .026]); a5.imshow(im, aspect='auto'); a5.axis('off')
    # track record + catalysts
    y3 = y2 - h5 - .012; h3 = .148; _panel(fig, .04, y3 - h3, .34, h3, 'Track record'); at = fig.add_axes([.075, y3 - h3 + .034, .28, h3 - .07]); at.set_facecolor(PANEL)
    ks = [1, 2, 3, 5, 10, 20, 40]; ys = [cv.get(str(k), np.nan) * 100 for k in ks]; at.plot(ks, ys, color=GOOD, lw=2, marker='o', ms=3.5); at.set_ylim(0, 100); at.set_xscale('log'); at.set_xticks(ks); at.set_xticklabels([str(k) for k in ks], color=MUT, fontsize=7); at.tick_params(colors=MUT, labelsize=7); at.grid(color=GRID, lw=.5); [s.set_visible(False) for s in at.spines.values()]
    fig.text(.365, y3 - .008, f"{bt['reach']:.0%} reach +10%", fontsize=11, color=T1, fontweight='bold', ha='right', va='top'); fig.text(.365, y3 - .024, f"median {cv.get('median_days', 3):.0f} trading days", fontsize=7, color=T2, ha='right', va='top'); fig.text(.055, y3 - h3 + .006, f"cumulative % of flips by trading day, 2022-Mar 2026, n {cv.get('n', bt['n'])}", fontsize=6.2, color=MUT)
    _panel(fig, .39, y3 - h3, .57, h3, 'Verdict · more on the next pages')
    L0 = [l.replace('⚠️ ', '').replace('• ', '').replace('★ ', '').strip() for l in rep_text.split('\n')[1:-1] if l.strip()]; flag = L0[-1] if L0 else ''; B_ = L0[:-1]; ins = [x for x in B_ if x.startswith('insiders') or x.startswith('GRADE A') or x.startswith('insider size')]; nw = [x for x in B_ if x.startswith('news:')]; oth = [x for x in B_ if x not in ins and x not in nw]; L = ([flag] if flag else []) + ['', 'Page 2: overhead supply, breadth and past flips in this spot', 'Page 3: news, filings, insiders and what carried edge']
    fig.text(.404, y3 - .026, '\n'.join(textwrap.shorten(x, 82, placeholder='...') for x in L[:7]), fontsize=7, color=T1, va='top', linespacing=1.38)
    if 'BAD CATALYST' in flag: fig.text(.945, y3 - .008, 'WARNING', fontsize=8, color=CRIT, fontweight='bold', ha='right', va='top')
    fig.text(.96, .012, '~Valiant', fontsize=13, color=T2, style='italic', family='DejaVu Serif', ha='right'); fig.text(.04, .012, 'Backtest in-sample and survivorship-biased; most trades lose 10-15%. Grade ranks the setup, it is not a forecast. Not advice.', fontsize=6.4, color=MUT)
    fig.savefig(path, dpi=125, facecolor=BG); plt.close(fig); return path
