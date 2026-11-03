"""Nov 3 2026: hold through the report, then exit right after it depending on how bad it is. Live-engine trades, live book (6 slots, ratio tie-break, 8% breaker).
Checkpoint = 30 min after the report time (extended-hours bar close), exit fill there minus 15 bp. Triggers (DECLARED BEFORE RESULTS):
 R1/R2/R3  reaction (checkpoint vs last regular close) <= -1/-2/-3%        [price-based; close to the Oct 25 news_exit test]
 S5/S10    EPS surprise <= -5% / -10% (known at the print, before price settles)   [new]
 S0R1      EPS surprise < 0 AND reaction <= -1%                                     [new]
 GR        graded: reaction <= -2% OR surprise <= -10%                              [new]
Plus a diagnostic: after the checkpoint, does the stock keep falling to the next open (remaining return), by reaction bucket and surprise bucket?"""
import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
src = open('/home/claude/fartpie/xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
L = prep(glob.glob('tm/*.parquet')); L = L[L['var'] == 'vec'].reset_index(drop=True)
FULL = {}
def full(tk):
    if tk not in FULL:
        f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
        if not f: FULL[tk] = None
        else:
            B_ = pd.read_parquet(f[0], columns=['datetime', 'close']); B_['dt'] = pd.to_datetime(B_.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); FULL[tk] = B_.drop_duplicates('dt').sort_values('dt').set_index('dt').close
    return FULL[tk]
def px_at(S_, t):
    k = S_.index.searchsorted(t - pd.Timedelta('15min'), 'right') - 1
    return (S_.iloc[k], S_.index[k] + pd.Timedelta('15min')) if k >= 0 and t - S_.index[k] < pd.Timedelta('60min') else (None, None)
REP = {}
for f in glob.glob('earn/*.csv'):
    tk = os.path.basename(f)[:-4]
    try: e = pd.read_csv(f); t = pd.to_datetime(e.iloc[:, 0], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); s = pd.to_numeric(e['Surprise(%)'], errors='coerce')
    except Exception: continue
    REP[tk] = [(tt, ss) for tt, ss in zip(t, s) if ss == ss]
VN = ['R1', 'R2', 'R3', 'S5', 'S10', 'S0R1', 'GR']
def fire(v, rea, sur):
    return {'R1': rea <= -.01, 'R2': rea <= -.02, 'R3': rea <= -.03, 'S5': sur <= -5, 'S10': sur <= -10, 'S0R1': sur < 0 and rea <= -.01, 'GR': rea <= -.02 or sur <= -10}[v]
mods = {v: [] for v in VN}; ev = []
for x in L.itertuples():
    rec = x._asdict(); hit = None; S_ = full(x.tk); b = bars(x.tk)
    if S_ is not None and b is not None and x.tk in REP:
        op, dc = b; e, xt = pd.Timestamp(x.entry), pd.Timestamp(x.exit); px0 = op.get(e); D = dc.index
        for tt, sur in REP[x.tk]:
            d = tt.normalize(); ac = tt.hour >= 12; g = d if not ac else (D[D > d][0] if (D > d).any() else None)
            if g is None or g not in D: continue
            gopen = g + pd.Timedelta('9h30min'); pdy = D[D < g]
            if not (len(pdy) and e < gopen <= xt and px0 == px0 and px0): continue
            c0 = dc[pdy[-1]]; ct = tt + pd.Timedelta('30min'); p, tc = px_at(S_, ct)
            if p is None: continue
            o1 = op.get(gopen, np.nan); hit = dict(rea=p / c0 - 1, sur=sur, tc=tc, p=p, px0=px0, rem=o1 / p - 1, gap=o1 / c0 - 1, up=c0 / px0 - 1); break
    if hit: ev.append(dict(tk=x.tk, entry=x.entry, **{k: hit[k] for k in ('rea', 'sur', 'rem', 'gap', 'up')}))
    for v in VN:
        r = dict(rec)
        if hit and fire(v, hit['rea'], hit['sur']): r['exit'], r['ret'] = np.datetime64(hit['tc']), hit['p'] * (1 - .0015) / hit['px0'] - 1
        mods[v].append(r)
E = pd.DataFrame(ev); TK = taken(L); key = set(zip(TK.tk, TK.entry.astype('datetime64[ns]'))); E['taken'] = [(a, np.datetime64(b, 'ns')) in key for a, b in zip(E.tk, E.entry)]; T = E[E.taken]
print(f'book trades held through a report with a 30-min checkpoint: {len(T)} (all engine trades {len(E)})')
print('\nDIAGNOSTIC: after the 30-min checkpoint, the move to the next 09:30 open (remaining) by bucket   n | mean remaining | share worse than -3% | mean total gap')
for nm, col, bins in (('reaction', 'rea', [-1, -.05, -.03, -.02, -.01, 0, .02, 9]), ('EPS surprise %', 'sur', [-1e9, -10, -5, 0, 5, 10, 1e9])):
    print(' ' + nm); g_ = T.groupby(pd.cut(T[col], bins)); 
    for k, g in g_:
        if len(g): print(f'   {str(k):<18} {len(g):4d} | {g.rem.mean():+.2%} | {np.mean(g.rem < -.03):5.1%} | {g.gap.mean():+.2%}')
P = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01')), (np.datetime64('2026-01-01'), np.datetime64('2027-01-01'))]
print('\nbook  2019-21 / 2022-26 / 2026 YTD   (return, max DD)')
def show(nm, Z):
    Z = pd.DataFrame(Z) if not isinstance(Z, pd.DataFrame) else Z; print(f'  {nm:<8}' + ''.join(f'{c:>+8.1%} /{d:>4.0%}   ' for c, d in (sim(Z, a, b_) for a, b_ in P)))
show('hold', L)
for v in VN: show(v, mods[v])
