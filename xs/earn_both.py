"""Nov 3 2026: pre-report guard vs post-report cut vs both, same earnings events. Live-engine trades, live book (6 slots, ratio tie-break, 8% breaker).
HOLD  no earnings rule | GUARD  sell at the close before a report if up >= 3% but < typical move (mean |gap| of last 8 reports) [live since 10/26]
POST1/POST2  hold through, then sell 30 min after the report (extended-hours fill - 15bp) if the reaction <= -1% / -2%
G+P1 / G+P2  guard first; if it did not fire, the post-report cut.   Also: book-% saved per rule = sum over held-through trades of (new return - hold return) / 6."""
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
    try: e = pd.read_csv(f); t = pd.to_datetime(e.iloc[:, 0], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    except Exception: continue
    REP[tk] = sorted(set(t))
MODES = ['GUARD', 'POST1', 'POST2', 'G+P1', 'G+P2']; mods = {m: [] for m in MODES}; ev = []
for x in L.itertuples():
    rec = x._asdict(); info = None; S_ = full(x.tk); b = bars(x.tk)
    if S_ is not None and b is not None and x.tk in REP:
        op, dc = b; e, xt = pd.Timestamp(x.entry), pd.Timestamp(x.exit); px0 = op.get(e); D = dc.index
        for tt in REP[x.tk]:
            d = tt.normalize(); ac = tt.hour >= 12; g = d if not ac else (D[D > d][0] if (D > d).any() else None)
            if g is None or g not in D: continue
            gopen = g + pd.Timedelta('9h30min'); pdy = D[D < g]
            if not (len(pdy) and e < gopen <= xt and px0 == px0 and px0): continue
            pv = pdy[-1]; c0 = dc[pv]; mv = []
            for tt2 in [z for z in REP[x.tk] if z < tt][-8:]:
                d2 = tt2.normalize(); g2 = d2 if tt2.hour < 12 else (D[D > d2][0] if (D > d2).any() else None)
                if g2 is None or g2 not in D: continue
                p2 = D[D < g2]
                if len(p2) and (g2 + pd.Timedelta('9h30min')) in op.index: mv.append(abs(op[g2 + pd.Timedelta('9h30min')] / dc[p2[-1]] - 1))
            up = c0 / px0 - 1; typ = np.mean(mv) if len(mv) >= 4 else np.nan
            p, tc = px_at(S_, tt + pd.Timedelta('30min'))
            if p is None: continue
            info = dict(up=up, typ=typ, cut=(pv + pd.Timedelta('15h59min'), up), rea=p / c0 - 1, tc=tc, pr=p * (1 - .0015) / px0 - 1); break
    if info: ev.append(dict(tk=x.tk, entry=x.entry, hold=x.ret, exit=x.exit, tc=info['tc'], pr=info['pr'], **{k: info[k] for k in ('up', 'typ', 'rea')}))
    for m in MODES:
        r = dict(rec)
        if info:
            gfire = info['up'] >= .03 and info['typ'] == info['typ'] and info['up'] < info['typ']
            th = {'POST1': .01, 'POST2': .02, 'G+P1': .01, 'G+P2': .02}.get(m)
            if m in ('GUARD', 'G+P1', 'G+P2') and gfire: r['exit'], r['ret'] = np.datetime64(info['cut'][0]), info['cut'][1]
            elif th is not None and info['rea'] <= -th: r['exit'], r['ret'] = np.datetime64(info['tc']), info['pr']
        mods[m].append(r)
E = pd.DataFrame(ev); TK = taken(L); key = set(zip(TK.tk, TK.entry.astype('datetime64[ns]'))); E['taken'] = [(a, np.datetime64(b, 'ns')) in key for a, b in zip(E.tk, E.entry)]; T = E[E.taken]
print(f'book trades held through a report (30-min checkpoint available): {len(T)}')
P = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01')), (np.datetime64('2026-01-01'), np.datetime64('2027-01-01'))]
print('\nbook  2019-21 / 2022-26 / 2026 YTD   (return, max DD)')
def show(nm, Z):
    Z = pd.DataFrame(Z) if not isinstance(Z, pd.DataFrame) else Z; print(f'  {nm:<7}' + ''.join(f'{c:>+8.1%} /{d:>4.0%}   ' for c, d in (sim(Z, a, b_) for a, b_ in P)))
show('HOLD', L)
for m in MODES: show(m, mods[m])
# per-event accounting on the book's own trades
M = {m: pd.DataFrame(mods[m]) for m in MODES}; Lk = L.set_index(['tk', 'entry']).ret
print('\nbook-% saved per rule on held-through trades (sum of (new return - hold return) / 6), and how many it changed')
for m in MODES:
    d = M[m].set_index(['tk', 'entry']).ret - Lk; d = d[d.index.isin(list(zip(T.tk, T.entry)))]; ch = d[d.abs() > 1e-9]
    print(f'  {m:<6} changed {len(ch):3d} trades   total {ch.sum() / 6:+.2%} of book over 2019-26   saved on losers {ch[ch > 0].sum() / 6:+.2%}   given up on winners {ch[ch < 0].sum() / 6:+.2%}')
E.to_parquet('/tmp/claude-0/ml/earn_both_E.parquet')

T = E[E.taken].copy(); T['d'] = T.pr - T.hold; T['hd'] = (pd.to_datetime(T.exit) - pd.to_datetime(T.tc)).dt.total_seconds() / 86400
print('\nPER-TRADE: cutting 30 min after the report vs holding to the live exit, by how bad the reaction was (book trades held through a report)')
print('  reaction bucket        n   mean (cut - hold)   sum/6 (book %)   share where cutting was better   median days the hold lasted after the report')
for lo, hi in ((-1, -.08), (-.08, -.05), (-.05, -.03), (-.03, -.02), (-.02, -.01), (-.01, 0), (0, 1)):
    g = T[(T.rea > lo) & (T.rea <= hi)]
    if len(g): print(f'  ({lo:+.0%},{hi:+.0%}]  {len(g):6d}   {g.d.mean():+8.2%}   {g.d.sum() / 6:+9.2%}   {np.mean(g.d > 0):10.0%}   {g.hd.median():6.1f}')
for th in (.01, .02, .03, .05, .08):
    g = T[T.rea <= -th]; print(f'  rule "cut if reaction <= -{th:.0%}": fires {len(g):3d}, net {g.d.sum() / 6:+.2%} of book over 2019-26, mean per fired trade {g.d.mean():+.2%}')
for yrs, nm in (((2019, 2021), '2019-21'), ((2022, 2026), '2022-26')):
    g = T[(T.rea <= -.03) & (pd.to_datetime(T.entry).dt.year.between(*yrs))]; print(f'    reaction <= -3%, {nm}: n {len(g)}, net {g.d.sum() / 6:+.2%} of book')
