"""Nov 14 2026 OPEN-ENTRY TEST (Rey: 'if we open above vector, buy at open instead of close; we enter so late'). Declared before results. Wide daily long flips (report_64 universe/filters, 2019-2026Q1, survivorship-biased). Vector flips bull when close > level + 0.35*ATR (CONFIRM_BARS=1). At day i's OPEN the prior-bar level and ATR are known, so trigger T = lev[i-1] + 0.35*ATR[i-1] (ATR[i-1] approximates ATR[i]). GAP day = regime bear at prior close AND open[i] > T. Causal rule: buy at open[i]; if the close confirms the flip, hold with the usual exit (open after the next opposite flip); if not, sell at close[i] (false entry). Costs 30bp per round trip, so false entries pay the cost too. Compared: STANDARD (enter open i+1 after confirmed flip), GAP-RULE (every gap day), HYBRID (open entry only when the flip day gapped, else standard) and the per-flip gain from entering at open[i] instead of open[i+1]. Note flip-bar volume BV is NOT known at the open, so no volume gate here. Eras: entry 2019-21 / 2022-26Q1. HUGE >= +100%."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(SC + 'wide_daily_ohlcv.pkl'); rows = []; COST = 0.003
for tk, B in D.items():
    B = B[~B.index.duplicated()]; n = len(B)
    if n < 400: continue
    o = B.open.values; h = B.high.values; l = B.low.values; c = B.close.values; v = B.volume.values.astype(float); idx = B.index; reg, lev, atr, _ = run_state_machine(B.assign(datetime=B.index))
    dv = pd.Series(c * v).rolling(20).median().shift(1).values; ch = np.array([i for i in range(1, n) if reg[i] != reg[i - 1] and reg[i] != 0])
    for i in range(131, n - 1):
        if reg[i - 1] != -1 or np.isnan(lev[i - 1]) or atr[i - 1] <= 0 or np.isnan(dv[i]) or dv[i] < 5e5 or c[i - 1] < 1: continue
        T = lev[i - 1] + 0.35 * atr[i - 1]
        if not o[i] > T: continue
        flip = reg[i] == 1 and reg[i - 1] == -1
        nx = ch[ch > i]
        if flip:
            if len(nx) == 0 or nx[0] + 1 >= n: continue
            ex = o[nx[0] + 1]; r_open = ex / o[i] - 1; r_std = ex / o[i + 1] - 1
        else: r_open = c[i] / o[i] - 1; r_std = np.nan
        rows.append(dict(tk=tk, d=idx[i], flip=flip, gap=o[i] / c[i - 1] - 1, over=(o[i] - T) / atr[i - 1], r_open=r_open, r_std=r_std, ro_next=(o[i + 1] / o[i] - 1)))
G = pd.DataFrame(rows); G = G[(G.d >= '2019-01-01') & (G.d < '2026-04-01')]
# all standard flips (gap or not) for the baseline
rows = []
for tk, B in D.items():
    B = B[~B.index.duplicated()]; n = len(B)
    if n < 400: continue
    o = B.open.values; c = B.close.values; v = B.volume.values.astype(float); idx = B.index; reg, lev, atr, _ = run_state_machine(B.assign(datetime=B.index)); dv = pd.Series(c * v).rolling(20).median().shift(1).values
    ch = [i for i in range(1, n) if reg[i] != reg[i - 1] and reg[i] != 0]
    for a in range(1, len(ch) - 1):
        i, j = ch[a], ch[a + 1]
        if reg[i] != 1 or j + 1 >= n or i + 1 >= n or i < 130 or c[i] < 1 or np.isnan(dv[i]) or dv[i] < 5e5 or np.isnan(lev[i]) or atr[i] <= 0: continue
        T = lev[i - 1] + 0.35 * atr[i - 1] if reg[i - 1] == -1 else np.nan
        rows.append(dict(tk=tk, d=idx[i], gapday=bool(o[i] > T) if T == T else False, r_std=o[j + 1] / o[i + 1] - 1, r_open=o[j + 1] / o[i] - 1))
S = pd.DataFrame(rows); S = S[(S.d >= '2019-01-01') & (S.d < '2026-04-01')]
def era(x): return np.where(x.d < '2022-01-01', '2019-21', '2022+')
S['era'] = era(S); G['era'] = era(G)
def line(lab, r, n_extra=''):
    r = pd.Series(r).dropna(); return f'  {lab:46s} n {len(r):6d} | mean net {(r.mean()-COST)*100:+6.2f}% | median {(r.median()-COST)*100:+6.2f}% | win {((r-COST)>0).mean()*100:4.1f}% | HUGE {(r>=1).sum():3d} ({(r>=1).mean()*100:4.2f}%) {n_extra}'
for e in ('2019-21', '2022+', 'ALL'):
    s = S if e == 'ALL' else S[S.era == e]; g = G if e == 'ALL' else G[G.era == e]; gf = g[g.flip]; gn = g[~g.flip]; sg = s[s.gapday]
    print(f'\n=== {e}'); print(line('STANDARD: every confirmed flip, enter next open', s.r_std))
    print(f'  flips whose flip day OPENED above trigger: {len(sg)} of {len(s)} ({len(sg)/len(s)*100:.1f}%); HUGE among them {int((sg.r_std>=1).sum())} of {int((s.r_std>=1).sum())} HUGE')
    print(line('  same flips, STANDARD entry', sg.r_std)); print(line('  same flips, OPEN-of-flip-day entry', sg.r_open))
    print(line('  flips with NO gap, standard', s[~s.gapday].r_std))
    hyb = np.where(s.gapday, s.r_open, s.r_std); print(line('HYBRID: open entry if gap else standard', hyb))
    print(line('GAP-RULE flip days only (hindsight selected)', gf.r_open))
    print(line('GAP-RULE false entries (opened above, no flip)', gn.r_open, f'| avg overshoot {gn.over.mean():.2f} ATR'))
    allg = pd.concat([gf.r_open, gn.r_open]); print(line('GAP-RULE tradable: ALL gap days (flip+false)', allg, f'| false share {len(gn)/len(g)*100:.0f}%'))
    print(f'  trades per year-ish: standard {len(s)}  gap-rule {len(g)}  (x{len(g)/len(s):.1f})')
    comb = np.r_[gf.r_open.values, gn.r_open.values, s[~s.gapday].r_std.values] - COST; print(f'  COMBINED TRADABLE (open entry on gap days incl. false ones + standard on no-gap flips): trades {len(comb)} | mean net {comb.mean()*100:+.2f}% | sum of trade returns {comb.sum()*100:,.0f}pp  vs STANDARD trades {len(s)} | mean {(s.r_std.mean()-COST)*100:+.2f}% | sum {(s.r_std-COST).sum()*100:,.0f}pp')
    print(f'  where the sum comes from: real flips gain {((gf.r_open.values-COST).sum()-(sg.r_std.values-COST).sum())*100:+,.0f}pp from entering at the open, false entries cost {(gn.r_open.values-COST).sum()*100:+,.0f}pp')
    for q in (0.5, 1.0):
        gg = g[g.over >= q]; print(line(f'  gap-rule only if open >= {q} ATR above trigger', pd.concat([gg[gg.flip].r_open, gg[~gg.flip].r_open]), f'| false {int((~gg.flip).sum()/max(len(gg),1)*100)}%'))
        ff = gg[gg.flip]; print(f'      same cohort: open-rule sum {(pd.concat([ff.r_open, gg[~gg.flip].r_open])-COST).sum()*100:+,.0f}pp vs standard entry on its real flips {(ff.r_std-COST).sum()*100:+,.0f}pp')
