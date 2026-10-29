"""Oct 29 2026: sell at the 09:30 opening print when a held name OPENS below its stop, instead of waiting for a bar to close below it.
Active stop approximated: 15m-track trades -> the 15m vector stop, 1h-track -> the 1h vector stop, as known at the last bar before 09:30
(vector on 04:00-16:00 bars like live). If the open is below it and the trade's actual exit is later, the exit moves to that 09:30 open."""
import os, glob, numpy as np, pandas as pd
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
src = open('/home/claude/fartpie/xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
import sys; sys.path.insert(0, '/home/claude/fartpie'); cwd = os.getcwd(); os.chdir('/home/claude/fartpie'); import live_screener as LS; os.chdir(cwd)
L = prep(glob.glob('tm/*.parquet')); L = L[L['var'] == 'vec'].sort_values('entry').reset_index(drop=True)
def et(df): df = df.copy(); df['dt'] = pd.to_datetime(df.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); return df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
NX, NR, HIT, REC = [], [], [], []
for tk, g in L.groupby('tk'):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: NX += list(g.exit); NR += list(g.ret); HIT += [False] * len(g); continue
    X = et(pd.read_parquet(f[0])); m = X.dt.dt.hour * 60 + X.dt.dt.minute; X = X[(m >= 240) & (m < 960)].reset_index(drop=True)
    V = LS.vec(X[['dt', 'open', 'high', 'low', 'close', 'volume']]); st15 = pd.Series(np.where(V.reg == 1, V['stop'], np.nan), index=V.dt + pd.Timedelta('15min'))
    H1 = LS.resample_1h(X); V1 = LS.vec(H1); st1 = pd.Series(np.where(V1.reg == 1, V1['stop'], np.nan), index=V1.dt + pd.Timedelta('1h'))
    op = X.set_index('dt').open; o930 = op[op.index.time == pd.Timestamp('09:30').time()]; o945 = op[op.index.time == pd.Timestamp('09:45').time()]
    for e, x, r, h1 in zip(g.entry, g.exit, g.ret, g.h1):
        px = op.get(e, np.nan); new = None
        if px == px:
            for d, o in o930[(o930.index > e) & (o930.index < x)].items():
                s = (st1 if h1 else st15); s = s[s.index <= d]
                if not len(s) or not s.iloc[-1] == s.iloc[-1]: continue
                if o < s.iloc[-1]: new = (d, o); break
        if new:
            NX.append(new[0]); NR.append(new[1] / px - 1); HIT.append(True)
            REC.append(dict(tk=tk, d=new[0], o930=new[1], stop=s.iloc[-1], old_exit=x, old_ret=r, new_ret=new[1] / px - 1, o945=o945.get(new[0] + pd.Timedelta('15min'), np.nan)))
        else: NX.append(x); NR.append(r); HIT.append(False)
    print(tk, end=' ', flush=True)
order = L.sort_values('tk', kind='stable').index; G = L.loc[order].copy(); G['exit2'], G['ret2'], G['hit'] = NX, NR, HIT; G = G.sort_values('entry')
R = pd.DataFrame(REC); R['same_day'] = R.old_exit.dt.normalize() == R.d.dt.normalize(); R['h'] = np.where(R.d.dt.year <= 2021, '2019-21', '2022-26')
print(f'\n\ngap-below-stop opens while held: {len(R)} of {len(G)} trades ({len(R) / len(G):.1%}); system exited the same day {R.same_day.mean():.0%} of them')
print('trade return change (new - old), mean %:'); print((R.assign(chg=R.new_ret - R.old_ret).groupby(['h', 'same_day']).chg.agg(['size', 'mean', 'median']) * [1, 100, 100]).round(2).to_string())
P = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def sh(nm, Z): print(f'  {nm:<38}' + ''.join(f'{c:>+8.1%} /{d:>4.0%}   ' for c, d in (sim(Z, a, b) for a, b in P)), flush=True)
print('\nbook, yearly return / max DD')
sh('live', G)
Z = G.copy(); Z['exit'], Z['ret'] = Z.exit2, Z.ret2; sh('sell at the 09:30 open if it opens below the stop', Z)
Z = G.copy(); k = Z.hit & (Z.exit2.dt.normalize() == Z.exit.dt.normalize()); Z.loc[k, 'exit'], Z.loc[k, 'ret'] = Z.exit2[k], Z.ret2[k]
sh('  ... only when the system exits that day anyway', Z)
