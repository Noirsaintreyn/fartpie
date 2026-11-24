"""Nov 24 2026 report 414. Rey (institutional trader example): 'looking for big moves - catalyst/news + macro + financials - then enter those big moves with vector'.
DECLARED: news proxies in the 5 trading days up to the prior close: VOL = max daily volume / prior-20d mean volume >= 2.0; GAP = max |open/prev close - 1| >= 3%; EARN = a 10-Q/10-K filed date (sec_q.parquet) inside the window.  CAT = VOL or GAP.  Entry = gated 15m flip at flip-day close (entry410 rows), 21d hold, -8% close stop, 4bp.
Arms: ALL | CAT | NOCAT | CAT&UPGAP (largest gap >0) | EARN.  PASS: CAT beats NOCAT and ALL in win rate and median in EVERY era (18-19, 20-21, 22, 23-26), month-clustered |t| >= 2 pooled, and top-5 name share of profit not worse than ALL.  Eras shown, nothing fitted per era."""
import numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
E = pd.read_parquet(SP + 'entry410.parquet'); E['de'] = pd.to_datetime(E.de).astype('datetime64[ns]')
D = pd.concat([pd.read_parquet(SP + 'lse_daily_54.parquet'), pd.read_parquet(SP + 'lse_daily_pool.parquet')]).drop_duplicates(['symbol', 'ts']); D['day'] = pd.to_datetime(D.ts).dt.normalize(); print(D.columns.tolist())
Q = pd.read_parquet(SP + 'sec_q.parquet'); Q['filed'] = pd.to_datetime(Q.filed); FD = {t: np.sort(g.filed.dropna().unique()) for t, g in Q.groupby('tk')}
F = {}
for tk, g in D.groupby('symbol'):
    g = g.sort_values('day').drop_duplicates('day').set_index('day'); vr = g.volume / g.volume.rolling(20).mean().shift(1); gp = g.open / g.close.shift(1) - 1
    F[tk] = pd.DataFrame({'vol': vr.rolling(5).max(), 'gmax': gp.abs().rolling(5).max(), 'gup': gp.rolling(5).max()}).shift(1)
rows = []
for tk, d in zip(E.tk, E.de):
    f = F.get(tk); 
    if f is None or d not in f.index: rows.append((np.nan,) * 4); continue
    r = f.loc[d]; fd = FD.get(tk, []); w = np.datetime64(d - pd.Timedelta(days=8)); earn = bool(((fd >= w) & (fd <= np.datetime64(d))).any()) if len(fd) else False
    rows.append((r.vol, r.gmax, r.gup, earn))
E[['vol', 'gmax', 'gup', 'earn']] = pd.DataFrame(rows, index=E.index); E = E[E.vol.notna()].copy()
E['cat'] = (E.vol >= 2) | (E.gmax >= .03); E['upg'] = E.cat & (E.gup >= .03); E['earn'] = E.earn.astype(bool)
e_ = pd.to_datetime(E.entry); E['era'] = pd.cut(e_.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26']); E['mo'] = e_.dt.to_period('M'); print('flips', len(E), '| CAT share', E.cat.mean().round(3))
ARM = {'ALL': lambda d: d.cat | ~d.cat, 'CAT': lambda d: d.cat, 'NOCAT': lambda d: ~d.cat, 'CAT&UPGAP': lambda d: d.upg, 'EARN': lambda d: d.earn}
for H in (21, 5, 63):
    c = f'r{H}'; B = E[E[c].notna()]; allm = B.groupby('mo')[c].mean(); print(f'\n=== HOLD {H}d === arm n | win% med% mean% | t vs ALL | top5 share | era win/med')
    for nm, f in ARM.items():
        d = B[f(B)]; 
        if len(d) < 50: print(nm, len(d)); continue
        m = (d.groupby('mo')[c].mean() - allm).dropna(); t = m.mean() / (m.std(ddof=1) / np.sqrt(len(m))) if len(m) > 3 and m.std() > 0 else np.nan
        pn = d.groupby('tk')[c].sum().sort_values(ascending=False); sh = pn.head(5).sum() / pn[pn > 0].sum()
        s = f"{nm:10s}{len(d):5d} | {(d[c]>0).mean()*100:5.1f} {d[c].median()*100:5.2f} {d[c].mean()*100:5.2f} | {t:+.1f} | {sh:.0%} |"
        for era in ('18-19', '20-21', '22', '23-26'):
            x = d[d.era == era]; s += f" {(x[c]>0).mean()*100:4.1f}/{x[c].median()*100:5.2f}(n{len(x)}) |" if len(x) >= 15 else ' n<15 |'
        print(s)
