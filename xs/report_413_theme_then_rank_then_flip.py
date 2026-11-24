"""Nov 23 2026 report 413. Rey: 'real analysis first; if money is moving into tech and we have 5 tech names, analyze each fundamentally, then use the vector to enter the best' + 'analysis in 2020 is different than post-2023 trend years'.
DECLARED (before any result): THEME = sector ETF (XL*) whose 20d return minus SPY is in the top 3 of the sectors AND 20d avg dollar volume / 60d avg dollar volume > 1 (relative strength + rising participation), lagged 1 day.  RANK = within the sector, among the 54 research names with fundamentals (report 406 F percentile Fp, as of prior day), TOP = within-sector Fp percentile >= 0.6 (needs >= 3 names).  ENTRY = the live gated 15m flip at flip-day close (entry410 rows), fixed hold 21d, -8% close stop, 4bp.
ARMS: ALL | HOT | HOT&TOP | HOT&BOT (control) | TOP-any (rank only, no theme) | COLD&TOP.
PASS (H): HOT&TOP beats ALL AND beats HOT in win rate and median in EVERY era (2018-19, 2020-21, 2022, 2023-26 - no per-era fitting) with month-clustered |t| >= 2 pooled; report name concentration (top-5 name share of arm profit).  Same 21d hold also shown at 5d/63d for robustness."""
import numpy as np, pandas as pd, json
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
E = pd.read_parquet(SP + 'entry410.parquet'); E['de'] = pd.to_datetime(E.de).astype('datetime64[ns]'); E['d0'] = E.de - pd.Timedelta(days=1)
sec = json.load(open('sp500_sectors.json')); E['sec'] = E.tk.map(sec); print('flips', len(E), 'no sector', E.sec.isna().sum())
ETF = {'Information Technology': 'XLK', 'Health Care': 'XLV', 'Financials': 'XLF', 'Consumer Discretionary': 'XLY', 'Consumer Staples': 'XLP', 'Energy': 'XLE', 'Industrials': 'XLI', 'Materials': 'XLB', 'Utilities': 'XLU', 'Real Estate': 'XLRE', 'Communication Services': 'XLC'}
px = {}; dv = {}
for s, e in list(ETF.items()) + [('SPY', 'SPY')]:
    try: d = pd.read_parquet(f'data/yahoo/daily_v/{e}.parquet')
    except Exception: print('missing', e); continue
    px[e] = d.close; dv[e] = d.close * d.volume
P = pd.DataFrame(px).sort_index(); V = pd.DataFrame(dv).sort_index(); P.index = pd.to_datetime(P.index); V.index = P.index
rs = (P.pct_change(20).sub(P.pct_change(20)['SPY'], axis=0)).drop(columns='SPY'); vr = V.rolling(20).mean() / V.rolling(60).mean(); rk = rs.rank(axis=1, ascending=False)
HOT = ((rk <= 3) & (vr.drop(columns='SPY') > 1)).shift(1)
T6 = pd.read_parquet(SP + 'stock406.parquet')[['tk', 't', 'Fp']].dropna(); T6['t'] = T6.t.astype('datetime64[ns]')
FP = T6.pivot_table(index='t', columns='tk', values='Fp').sort_index(); FP = FP.reindex(P.index.union(FP.index)).ffill(limit=45).reindex(P.index)
names = sorted(E.tk.unique()); SEC = pd.Series({t: sec.get(t) for t in names})
def within_pct(day):
    row = FP.loc[:day].iloc[-1] if len(FP.loc[:day]) else None; out = {}
    if row is None: return out
    for s, g in SEC.groupby(SEC):
        v = row.reindex(g.index).dropna()
        if len(v) >= 3: out.update(v.rank(pct=True).to_dict())
    return out
cache = {}; hot = []; wp = []
for tk, s, d in zip(E.tk, E.sec, E.d0):
    h = HOT[ETF[s]].reindex([d], method='ffill').iloc[0] if s in ETF else np.nan; hot.append(h)
    if d not in cache: cache[d] = within_pct(d)
    wp.append(cache[d].get(tk, np.nan))
E['hot'] = pd.Series(hot).astype(float).values == 1; E['wp'] = wp; E['top'] = E.wp >= .6; E['bot'] = E.wp <= .4; E = E[E.wp.notna()].copy()
e_ = pd.to_datetime(E.entry); E['era'] = pd.cut(e_.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26']); E['mo'] = e_.dt.to_period('M'); print('flips with within-sector rank', len(E), '| hot share', E.hot.mean().round(3))
ARM = {'ALL': lambda d: d.hot | ~d.hot, 'HOT': lambda d: d.hot, 'HOT&TOP': lambda d: d.hot & d.top, 'HOT&BOT': lambda d: d.hot & d.bot, 'TOP-any': lambda d: d.top, 'COLD&TOP': lambda d: ~d.hot & d.top}
for H in (21, 5, 63):
    c = f'r{H}'; B = E[E[c].notna()]; print(f'\n=== HOLD {H}d ===  arm n | win% med% mean% | t vs ALL | top5 name share | by era win/med: ' + ' | '.join(['18-19', '20-21', '22', '23-26']))
    allm = B.groupby('mo')[c].mean()
    for nm, f in ARM.items():
        d = B[f(B)]
        if len(d) < 50: print(nm, len(d)); continue
        m = (d.groupby('mo')[c].mean() - allm).dropna(); t = m.mean() / (m.std(ddof=1) / np.sqrt(len(m))) if len(m) > 3 else np.nan
        pn = d.groupby('tk')[c].sum().sort_values(ascending=False); sh = pn.head(5).sum() / pn[pn > 0].sum() if (pn > 0).any() else np.nan
        s = f"{nm:9s}{len(d):5d} | {(d[c]>0).mean()*100:5.1f} {d[c].median()*100:5.2f} {d[c].mean()*100:5.2f} | {t:+.1f} | {sh:.0%} |"
        for era in ('18-19', '20-21', '22', '23-26'):
            x = d[d.era == era]; s += f" {(x[c]>0).mean()*100:4.1f}/{x[c].median()*100:5.2f} (n{len(x)}) |" if len(x) >= 15 else ' n<15 |'
        print(s)
E.to_parquet(SP + 'theme413.parquet')
