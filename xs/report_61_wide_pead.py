"""Nov 14 2026 STRICT-SPLIT CHECK OF THE SMALL-CAP LEADS ON THE WIDE DAILY UNIVERSE (declared before results). The PEAD gate and the financing-8-K skip passed only on 19 months of 1h data; here they get 2019-21 vs 2022-26Q1 on ~1,650 names, daily vector long flips (same flips/universe as report_60, survivorship bias applies equally to every variant). Earnings reaction from 8-K item 2.02 + daily closes (reaction day = filing day if accepted before 16:00 ET else next day; z = reaction / std of prior 20 daily returns). Rules: PEAD GATE = skip flips AGAINST a big reaction (|z|>=2, reaction negative for a long flip) within 1-20 trading days after it; PEAD-ALIGNED = flips WITH a big reaction in that window; FIN SKIP = skip flips with an 8-K item 1.01/2.03/3.02 filed in the prior 5 trading days. Base filters: BV>=1.5 variants shown beside all-flips. 30bp costs. Output n, mean net %, median, win %, big(>=15%) %, clustered t of kept vs removed (by entry month)."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; SC0 = SC.replace('ext/', '')
src = open('/home/claude/fartpie/xs/report_56_pead_flips.py').read(); exec(src.split("# ---- large caps")[0].split('def show(')[0].replace("SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'", "SC0='x'"))
D = pd.read_pickle(SC + 'wide_daily_ohlcv.pkl'); dc = {tk: x.close for tk, x in D.items()}; P = pd.read_pickle(SC + 'wide_flips_ins.pkl'); P['d'] = 1
K = pd.read_csv(SC + '8k_wide.csv'); K['items'] = K['items'].astype(str); print('8-K rows', len(K), 'tickers', K.ticker.nunique(), flush=True)
ev = events(K, dc); print('earnings events', sum(len(v) for v in ev.values()), 'tickers', len(ev), flush=True); P = tag(P, ev, dc); P['ret_n'] = P.ret - 0.003
fin = K[K['items'].map(lambda s: bool({'1.01', '2.03', '3.02'} & set(s.split(','))))].copy(); fin['acc'] = pd.to_datetime(fin.accept, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); per = {tk: x.acc.sort_values().values for tk, x in fin.groupby('ticker')}; flag = np.zeros(len(P), bool); bd = pd.offsets.BDay(5)
for tk, ix in P.groupby('tk').groups.items():
    a = per.get(tk)
    if a is None: continue
    t = P.loc[ix, 't'].values; pos = np.searchsorted(a, t, side='right') - 1; ok = pos >= 0; last = a[np.clip(pos, 0, None)]; lim = (pd.to_datetime(P.loc[ix, 't']) - bd).values; flag[P.index.get_indexer(ix)] = ok & (last >= lim)
P['FIN'] = flag; big = P.z.abs() >= 2; win = P.lag.between(1, 20); against = big & win & (P.ALIGN < 0); aligned = big & win & (P.ALIGN > 0)
print('flips', len(P), '| in 20d window after a big reaction:', int((big & win).sum()), '| against', int(against.sum()), 'aligned', int(aligned.sum()), '| FIN flagged', int(P.FIN.sum()), flush=True)
def mc(S, m):
    s = S[m]; r = S[~m]; a = s.groupby(s.t.dt.to_period('M')).ret_n.mean(); b = r.groupby(r.t.dt.to_period('M')).ret_n.mean(); dm = (a - b).dropna(); return dm.mean() / (dm.std() / np.sqrt(len(dm))) if len(dm) > 5 else np.nan
def row(name, S, keep, base):
    keep = keep & base; s = S[keep]; r = S[base & ~keep]
    return f'   {name:34s} n {len(s):6d} | mean {s.ret_n.mean()*100:+6.2f}% | med {s.ret_n.median()*100:+5.2f} | win {(s.ret_n>0).mean()*100:4.1f} | big {(s.ret_n>=0.15).mean()*100:4.1f} | removed n {len(r):5d} mean {r.ret_n.mean()*100:+6.2f}% | t kept-vs-removed {mc(S[base], keep[base]) if (base & ~keep).sum()>5 else float("nan"):+.1f}'
for en, a, b in (('2019-21', '2019', '2022'), ('2022-26Q1', '2022', '2026-04'), ('ALL', '2019', '2026-04')):
    S = P[(P.t >= a) & (P.t < b)].reset_index(drop=True); big_ = S.z.abs() >= 2; win_ = S.lag.between(1, 20); ag = (big_ & win_ & (S.ALIGN < 0)).values; al = (big_ & win_ & (S.ALIGN > 0)).values; fn = S.FIN.values; bv = (S.BV >= 1.5).values; allm = np.ones(len(S), bool); print(f'\n=== {en}: {len(S)} flips')
    for tag_, base in (('all flips', allm), ('BV>=1.5', bv)):
        print(f'  -- base: {tag_} (n {base.sum()}, mean {S[base].ret_n.mean()*100:+.2f}%)')
        print(row('PEAD gate (skip against)', S, ~ag, base)); print(row('FIN skip', S, ~fn, base)); print(row('PEAD gate + FIN skip', S, ~ag & ~fn, base)); print(row('only PEAD-aligned flips', S, al, base))
        if tag_ == 'BV>=1.5': print(row('BV + PEAD gate + FIN skip + INS>=1', S, ~ag & ~fn & (S.NI >= 1).values, base))
