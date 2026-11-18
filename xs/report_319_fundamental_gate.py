"""Nov 18 2026 report 319. THESIS-THEN-ENTRY test: does a point-in-time fundamental gate pick flips (vector = entry) that do better? Data: SEC companyfacts JSON for the 54 screener names (annual facts, usable from the 'filed' date). Fields (declared, one definition each, set a priori from desk doc D1a / Sloan 1996 / Novy-Marx): ACCR = (NetIncomeLoss - NetCashProvidedByOperatingActivities)/Assets (high = low earnings quality); GPA = GrossProfit/Assets (profitability); ROA = NetIncomeLoss/Assets; GROW = Revenue yoy growth. Latest 10-K FY row with filed date <= flip entry. Terciles cut on train. Hypotheses (directions fixed): high GPA/ROA/GROW better, high ACCR worse. Why it would fail: 54 megacaps, big-cap quality premium small, flips already filter on price turn. Pass = same sign both halves and |t(top-bottom)|>=2 in one."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
def series(f, tags, unit='USD'):
    for t in tags:
        g = f.get('us-gaap', {}).get(t)
        if g and unit in g.get('units', {}):
            r = pd.DataFrame(g['units'][unit]); r = r[(r.form == '10-K') & (r.fp == 'FY')] if 'fp' in r else r
            if len(r): r['end'] = pd.to_datetime(r.end); r['filed'] = pd.to_datetime(r.filed); r['dur'] = (r.end - pd.to_datetime(r.start)).dt.days if 'start' in r else 365; r = r[(r.dur > 300) | r.dur.isna()] if 'start' in r else r; return r.sort_values('filed').drop_duplicates('end', keep='last')[['end', 'filed', 'val']].set_index('end')
    return None
rows = []
for tk in sorted(set(pd.read_parquet(SP + 'b302.parquet').tk)):
    try: f = json.load(open(f'/home/claude/fartpie/data/fund/sec/{tk}.json'))['facts']
    except Exception: continue
    ni = series(f, ['NetIncomeLoss', 'ProfitLoss']); cfo = series(f, ['NetCashProvidedByUsedInOperatingActivities']); gp = series(f, ['GrossProfit']); rev = series(f, ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet'])
    at = series(f, ['Assets'])
    if at is None or ni is None: continue
    A = at.rename(columns={'val': 'assets', 'filed': 'f_at'}); X = A.join(ni.val.rename('ni')); 
    if cfo is not None: X = X.join(cfo.val.rename('cfo'))
    if gp is not None: X = X.join(gp.val.rename('gp'))
    if rev is not None: X = X.join(rev.val.rename('rev'))
    X['tk'] = tk; X['ROA'] = X.ni / X.assets; X['ACCR'] = (X.ni - X.get('cfo', np.nan)) / X.assets; X['GPA'] = X.get('gp', np.nan) / X.assets; X['GROW'] = X.get('rev', pd.Series(dtype=float)).pct_change() if 'rev' in X else np.nan
    X = X.reset_index().rename(columns={'f_at': 'filed'}); rows.append(X[['tk', 'filed', 'ROA', 'ACCR', 'GPA', 'GROW']])
F = pd.concat(rows).dropna(subset=['filed']); F['filed'] = F.filed.astype('datetime64[ns]'); F = F.sort_values('filed'); print('names with fundamentals', F.tk.nunique(), 'rows', len(F))
B = pd.read_parquet(SP + 'b302.parquet'); e = pd.to_datetime(B.entry); B['ed'] = (e.dt.tz_localize(None) if e.dt.tz is not None else e).dt.normalize(); B['ed'] = B.ed.astype('datetime64[ns]'); B = B.reset_index().rename(columns={'index': 'i'}).sort_values('ed')
M = pd.merge_asof(B, F.sort_values('filed'), left_on='ed', right_on='filed', by='tk', direction='backward', tolerance=pd.Timedelta(days=540)); M = M.dropna(subset=['filed']); print('flips with a point-in-time fundamental row', len(M), 'of', len(B))
def t(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))
for f, hyp in (('GPA', 'high better'), ('ROA', 'high better'), ('GROW', 'high better'), ('ACCR', 'high worse')):
    q = M[M.tr][f].quantile([1 / 3, 2 / 3]).values; out = []
    for nm, m in (('train', M.tr), ('test', ~M.tr)):
        z = M[m].dropna(subset=[f]); lo = z[z[f] <= q[0]].ret; hi = z[z[f] >= q[1]].ret; out.append(f'{nm} low {100 * lo.mean():+.2f}% high {100 * hi.mean():+.2f}% diff {100 * (hi.mean() - lo.mean()):+.2f}pp t{t(hi, lo):+.1f} sd {100 * lo.std():.1f}/{100 * hi.std():.1f}')
    print(f'{f:5s} ({hyp}): ', ' | '.join(out))
print('\ndisqualifier rule (doc): ACCR > 0.08 excluded:'); z = M.dropna(subset=['ACCR'])
for nm, m in (('train', z.tr), ('test', ~z.tr)): a = z[m & (z.ACCR > .08)].ret; b = z[m & (z.ACCR <= .08)].ret; print(f'  {nm} ACCR>.08 {100 * a.mean():+.2f}% (n{len(a)}) vs rest {100 * b.mean():+.2f}% (n{len(b)})')
