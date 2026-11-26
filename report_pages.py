"""Nov 26 2026 (Rey: 'the signal and what is behind it'): the small-cap daily flip report as three pictures.
 page 1  report_card.make_card      the signal: header, grade, KPI tiles, scorecard, charts, track record
 page 2  context_page               the single-stock page: overhead supply, breadth, volatility, and what happened the other times flips were in this exact spot (and this stock's own earlier flips)
 page 3  catalyst_page              news, filings, insiders and the factors that carried measured edge (short interest...)
History = flip_history.csv (xs/build_flip_history.py): every sub-$5 daily flip since 2019 with ATRP, overhead supply (OVH), breadth (BRD) and the outcome. Tier cuts frozen on train (flip_history_cuts.json). Small samples are labelled as such: n < 15 reads as an anecdote."""
import os, json, textwrap, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE = os.path.dirname(os.path.abspath(__file__))
BG, PANEL, GRID, T1, T2, MUT = '#080c16', '#0f1626', '#1b2438', '#f1f5f9', '#94a3b8', '#56627a'
BLUE, RED, GOOD, GOLD, AMB = '#3b82f6', '#fb7185', '#34d399', '#fcd34d', '#f59e0b'
_H = {}
def _hist():
    if not _H:
        _H['h'] = pd.read_csv(os.path.join(HERE, 'flip_history.csv'), parse_dates=['d']); _H['c'] = json.load(open(os.path.join(HERE, 'flip_history_cuts.json')))
    return _H['h'], _H['c']
def _tier(x, cut): return 0 if x <= cut[0] else (2 if x >= cut[1] else 1)
def _panel(fig, x, y, w, h, title=None):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.006', transform=fig.transFigure, fc=PANEL, ec=GRID, lw=.8, zorder=-8))
    if title: fig.text(x + .016, y + h - .012, title.upper(), fontsize=8, color=MUT, fontweight='bold', va='top')
def _head(fig, tag, tk, name, sub, accent=GOOD):
    fig.patches.append(plt.Rectangle((0, .988), 1, .012, transform=fig.transFigure, fc=accent, ec='none', zorder=-5))
    fig.text(.04, .968, tag, fontsize=9, color=accent, fontweight='bold', va='top'); fig.text(.04, .945, tk, fontsize=28, color=T1, fontweight='bold', va='top')
    fig.text(.04, .902, f"{name[:58]}  ·  {sub}" if name else sub, fontsize=9.5, color=T2, va='top')
def _foot(fig, txt): fig.text(.04, .012, txt, fontsize=7.4, color=MUT, style='italic'); fig.text(.96, .012, '~Valiant', fontsize=13, color=T2, style='italic', family='DejaVu Serif', ha='right')
def _pct(x): return f'{x:.0%}'
def spot_stats(h, tA, tO, tB, ex_d=None):
    """flips in the same ATRP / overhead / breadth tiers; relax breadth, then overhead, until n >= 20. -> (subset, label)"""
    for lab, m in (('same volatility, overhead and breadth', (h.tA == tA) & (h.tO == tO) & (h.tB == tB)), ('same volatility and overhead', (h.tA == tA) & (h.tO == tO)), ('same volatility', h.tA == tA)):
        s = h[m]
        if len(s) >= 20: return s, lab
    return h, 'all sub-$5 flips'
def context_page(a, d_df, path, name=None):
    h, cu = _hist(); h = h.copy(); tk = a['tk']; day = pd.Timestamp(a['day']); F = a['F']
    h['tA'] = [_tier(x, cu['atrp']) for x in h.ATRP]; h['tO'] = [_tier(x, cu['ovh']) if x == x else 1 for x in h.OVH]; h['tB'] = [_tier(x, cu['brd']) for x in h.BRD]
    b = d_df.tail(253); c = float(b.close.iloc[-1]); tp = ((b.high + b.low + b.close) / 3).values; vo = b.volume.values.astype(float); ov = float(vo[(tp > c) & (tp <= 1.5 * c)].sum() / max(vo.sum(), 1))
    brd = a.get('brd'); atr = float(F['ATRP']); tA = _tier(atr, cu['atrp']); tO = _tier(ov, cu['ovh']); tB = _tier(brd, cu['brd']) if brd is not None else 1
    base = float(h.hit.mean()); prior = h[(h.tk == tk) & (h.d < day)].sort_values('d')
    fig = plt.figure(figsize=(9.6, 12.4), facecolor=BG); _head(fig, 'SETUP CONTEXT  ·  PAGE 2 OF 3', tk, name or '', f"what sits behind the {day:%b %d} flip")
    # tiles
    tiles = [('VOLATILITY (ATR %)', f'{atr:.1%}', ['LOW', 'MID', 'HIGH'][tA] + ' vs the group', 'tA', tA), ('OVERHEAD SUPPLY', f'{ov:.0%}', ['CLEAR AIR', 'MODERATE', 'HEAVY'][tO], 'tO', tO), ('BREADTH TODAY', f'{brd}' if brd is not None else 'n/a', ['QUIET TAPE', 'NORMAL TAPE', 'BUSY TAPE'][tB] if brd is not None else '', 'tB', tB)]
    for k, (t_, v_, w_, col, ti) in enumerate(tiles):
        x0 = .04 + k * .3133; _panel(fig, x0, .745, .2933, .125, t_)
        fig.text(x0 + .016, .843, v_, fontsize=26, color=T1, fontweight='bold', va='top'); fig.text(x0 + .016, .800, w_, fontsize=9.5, color=GOOD if (col == 'tO' and ti == 0) or (col == 'tB' and ti == 2) or (col == 'tA' and ti == 2) else (RED if (col == 'tO' and ti == 2) or (col == 'tB' and ti == 0) else T2), fontweight='bold', va='top')
        s = h[h[col] == ti]; fig.text(x0 + .016, .776, f"reached +10%: {s.hit.mean():.0%} of {len(s)} flips", fontsize=7.6, color=MUT, va='top')
    # overhead chart
    _panel(fig, .04, .475, .5, .255, 'Volume at price · last 252 days'); ax = fig.add_axes([.075, .505, .43, .19]); ax.set_facecolor(PANEL); [s_.set_visible(False) for s_ in ax.spines.values()]; ax.tick_params(colors=MUT, labelsize=8, length=0)
    lo, hi = float(b.low.min()), float(b.high.max()); bins = np.linspace(lo, hi, 34); h_, e_ = np.histogram(tp, bins=bins, weights=vo); m = (e_[:-1] + e_[1:]) / 2; w = e_[1] - e_[0]
    ax.barh(m, h_ / h_.max() * 100, height=w * .86, color=[BLUE if x <= c else (RED if x <= 1.5 * c else '#475569') for x in m], alpha=.92, zorder=3); ax.axhline(c, color=T1, lw=1.5, zorder=5); ax.text(2, c, f' breakout {c:.2f}', color=T1, fontsize=8.5, fontweight='bold', va='bottom', zorder=6)
    ax.axhspan(c, min(1.5 * c, hi), color=RED, alpha=.07); ax.set_xticks([]); ax.set_xlim(0, 108); ax.set_ylim(lo, hi * 1.02); ax.yaxis.tick_right(); ax.tick_params(axis='y', labelcolor=T2)
    # exact-spot stats
    s, lab = spot_stats(h, tA, tO, tB); n = len(s); rch = float(s.hit.mean()); _panel(fig, .56, .475, .4, .255, 'Flips in this exact spot')
    fig.text(.576, .68, _pct(rch), fontsize=34, color=GOOD if rch > base + .04 else (RED if rch < base - .04 else T1), fontweight='bold', va='top'); fig.text(.576, .625, f'reached +10% ({int(s.hit.sum())} of {n})', fontsize=10, color=T1, fontweight='bold', va='top')
    fig.text(.576, .60, f"{lab}\nall sub-$5 flips: {_pct(base)}\nmedian peak {s.PEAK.median():+.0%}  ·  median exit {s.r.median():+.0%}\nmean exit {s.r.clip(s.r.quantile(.01), s.r.quantile(.99)).mean():+.0%} (a few big winners)", fontsize=8.6, color=T2, va='top', linespacing=1.55)
    if n < 15: fig.text(.576, .50, 'few cases: read as anecdote', fontsize=8, color=AMB, va='top')
    # this stock's history
    _panel(fig, .04, .06, .92, .395, f'{tk} · its own earlier flips'); np_ = len(prior)
    if np_ == 0: fig.text(.06, .43, 'First daily flip on record for this stock: no own history to compare.', fontsize=10, color=T2, va='top')
    else:
        k = int(prior.hit.sum()); fig.text(.06, .435, f'Earlier flips on {tk}: {k} of {np_} reached +10%', fontsize=12, color=T1, fontweight='bold', va='top')
        pt = prior.assign(tA=lambda x: [_tier(v, cu['atrp']) for v in x.ATRP], tO=lambda x: [_tier(v, cu['ovh']) if v == v else 1 for v in x.OVH]); sm = pt[(pt.tA == tA) & (pt.tO == tO)]
        if len(sm): fig.text(.06, .405, f"When it was in this spot before (same volatility and overhead): {int(sm.hit.sum())} of {len(sm)} reached +10%" + (f"  ·  when: {', '.join(f'{d:%b %Y}' for d in sm.d.tail(4))}" if len(sm) else ''), fontsize=9.2, color=GOLD, va='top')
        else: fig.text(.06, .405, 'It has not been in this exact spot (same volatility and overhead) before.', fontsize=9.2, color=T2, va='top')
        if np_ < 6: fig.text(.06, .382, f'only {np_} earlier flip(s): an anecdote, not a rate', fontsize=8, color=AMB, va='top')
        yy = .352; hdr = ['DATE', 'VOLATILITY', 'OVERHEAD', 'BREADTH', 'PEAK', 'EXIT', '']; xs = [.06, .19, .34, .49, .62, .74, .86]
        for x_, t_ in zip(xs, hdr): fig.text(x_, yy, t_, fontsize=7.5, color=MUT, fontweight='bold', va='top')
        yy -= .022
        for r in prior.tail(9).iloc[::-1].itertuples():
            ra = _tier(r.ATRP, cu['atrp']); ro = _tier(r.OVH, cu['ovh']) if r.OVH == r.OVH else 1; rb = _tier(r.BRD, cu['brd']); same = (ra == tA and ro == tO)
            vals = [f'{r.d:%b %d %Y}', f"{r.ATRP:.1%} {['low', 'mid', 'high'][ra]}", f"{r.OVH:.0%} {['clear', 'mod', 'heavy'][ro]}" if r.OVH == r.OVH else 'n/a', f"{int(r.BRD)} {['quiet', 'norm', 'busy'][rb]}", f'{r.PEAK:+.0%}', f'{r.r:+.0%}']
            for x_, v_ in zip(xs, vals): fig.text(x_, yy, v_, fontsize=8.4, color=GOLD if same else T1, va='top')
            fig.text(xs[6], yy, '✓ +10%' if r.hit else '✗ missed', fontsize=8.4, color=GOOD if r.hit else RED, fontweight='bold', va='top'); yy -= .0265
        fig.text(.06, .092, 'gold rows = the same volatility and overhead as today. PEAK = best price above entry before the vector exit; EXIT = return at the vector exit.', fontsize=7.4, color=MUT, va='top')
    _foot(fig, 'Info page, no grade points: these readings overlap volatility inside the high-ATR group (report 502). Small counts are anecdotes.'); fig.savefig(path, dpi=130, facecolor=BG); plt.close(fig); return path
def catalyst_page(a, rep_text, path, name=None):
    """catalysts report restyled as page 3: the factors that carried measured edge first, then news / filings / insiders."""
    sc = a.get('sc') or {}; tk = a['tk']; day = pd.Timestamp(a['day'])
    L = [l.replace('• ', '', 1).strip() for l in rep_text.split('\n')[1:] if l.strip() and not l.strip().startswith('*~')]
    edge, news, filings, ins, other = [], [], [], [], []
    for r in sc.get('rows', []):
        nm, today, tone, stat = r
        if nm in ('Short interest', 'Insider buying', 'Earnings timing', 'Bad 8-K (7 days)') and today != 'unavailable' and (tone != '0' or (nm == 'Short interest' and stat.strip())): edge.append((nm, today, tone, stat))
    for s in L:
        if s.startswith('news:'): news.append(s[5:].strip())
        elif s.startswith('8-K') or 'filing' in s.lower()[:14] or s.startswith('earnings'): filings.append(s)
        elif s.startswith('insider') or s.startswith('★'): ins.append(s.replace('insiders 30d (open market): ', '').replace('insiders 30d: ', '').lstrip('★ ').replace('GRADE A: ', 'Grade A: ').replace('insider size: ', ''))
        elif s.startswith('no clear') or s.startswith('⚠') or s.startswith('catalyst:'): other.append(s)
        else: filings.append(s)
    verdict = (other[-1][9:] + ' catalyst' if other[-1].startswith('catalyst:') else other[-1]) if other else 'no clear catalyst'
    if any(x.startswith('insider size') or 'usual state' in x for x in ins): ins = [x for x in ins if not x.startswith('no open-market')]
    sections = [('WHAT CARRIED EDGE', [(f'{nm}  ·  {today}', stat, {'+': GOOD, '0': MUT, '-': RED}[tone]) for nm, today, tone, stat in edge] or [('Nothing beyond volatility carried measured edge on this name', '', MUT)]),
                ('INSIDERS', [(x, '', GOOD if ('Grade A' in x or x.startswith('first buying')) else (AMB if 'usual 30d' in x or 'sale(s)' in x else MUT)) for x in ins] or [('no open-market insider trades in 30 days (the usual state)', '', MUT)]),
                ('FILINGS AND EVENTS', [(x, '', AMB if ('offering' in x or '5.03' in x) else BLUE) for x in filings] or [('nothing notable filed', '', MUT)]),
                ('NEWS', [(x, '', BLUE) for x in news] or [('no recent headlines', '', MUT)])]
    wrap = lambda t, w: textwrap.wrap(t.replace('$', r'\$'), w) or ['']
    hu = [.62 + sum(.3 * len(wrap(t, 84)) + (.25 * len(wrap(s, 96)) if s else 0) + .14 for t, s, _ in rows) for _, rows in sections]; Ht = 2.05 + sum(hu) + .14 * len(hu) + .75; fig = plt.figure(figsize=(9.6, Ht), facecolor=BG); u = 1 / Ht
    fig.patches.append(plt.Rectangle((0, 1 - .1 * u), 1, .1 * u, transform=fig.transFigure, fc=GOOD, ec='none', zorder=-5))
    fig.text(.04, 1 - .4 * u, 'CATALYSTS  ·  PAGE 3 OF 3', fontsize=9, color=GOOD, fontweight='bold', va='top'); fig.text(.04, 1 - .68 * u, tk, fontsize=28, color=T1, fontweight='bold', va='top'); fig.text(.04, 1 - 1.28 * u, f"{(name or '')[:56]}  ·  news, filings, insiders and what carried edge  ·  {day:%b %d}", fontsize=9.5, color=T2, va='top')
    y = 1 - 1.75 * u
    for (ttl, rows), hh in zip(sections, hu):
        ph = hh * u; fig.patches.append(FancyBboxPatch((.04, y - ph), .92, ph, boxstyle='round,pad=0,rounding_size=0.006', transform=fig.transFigure, fc=PANEL, ec=GRID, lw=.8, zorder=-8)); fig.text(.056, y - .13 * u, ttl, fontsize=8, color=MUT, fontweight='bold', va='top'); yy = y - .5 * u
        for t, s, col in rows:
            ws = wrap(t, 84); fig.patches.append(plt.Rectangle((.056, yy - (.3 * len(ws)) * u + .02 * u), .004, (.3 * len(ws) - .02) * u, transform=fig.transFigure, fc=col, ec='none'))
            fig.text(.07, yy, '\n'.join(ws), fontsize=10, color=T1, va='top', linespacing=1.35); yy -= .3 * len(ws) * u
            if s: ws2 = wrap(s, 96); fig.text(.07, yy + .02 * u, '\n'.join(ws2), fontsize=8, color=T2, va='top'); yy -= .25 * len(ws2) * u
            yy -= .14 * u
        y -= ph + .14 * u
    bad = verdict.startswith('⚠'); fig.text(.04, y - .12 * u, ('WARNING  ·  ' if bad else 'VERDICT  ·  ') + verdict.replace('⚠️ ', '').replace('⚠ ', ''), fontsize=10.5, color=RED if bad else T2, fontweight='bold', va='top')
    fig.text(.96, .012, '~Valiant', fontsize=13, color=T2, style='italic', family='DejaVu Serif', ha='right'); fig.text(.04, .012, 'Only factors with measured edge are scored; everything else is context.', fontsize=7.4, color=MUT, style='italic'); fig.savefig(path, dpi=130, facecolor=BG); plt.close(fig); return path
