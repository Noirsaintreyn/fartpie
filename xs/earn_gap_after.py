"""Oct 28 2026: after a bad earnings gap, does getting out early (premarket) beat the system's exit (open of the 9:45 bar), or does waiting help?
All reports for the 54 names 2019-26 with a gap <= -5% (prior close -> 9:30 open); prices from 15m bars incl. premarket (04:00+).
Returns are measured vs the prior close: sell at 08:00 premarket, 09:30 open, 09:45 (system), 10:30, the gap day's close, +1 day, +5 days."""
import os, glob, numpy as np, pandas as pd
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
rows = []
for f in glob.glob('earn/*.csv'):
    tk = os.path.basename(f)[:-4]; b = bars(tk)
    try: e = pd.read_csv(f)
    except Exception: continue
    if b is None or not len(e): continue
    op, dc = b; D = dc.index
    for tt in pd.to_datetime(e.iloc[:, 0], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None):
        d = tt.normalize(); g = d if tt.hour < 12 else (D[D > d][0] if (D > d).any() else None)
        if g is None or g not in D or g < pd.Timestamp('2019-01-01'): continue
        k = D.get_loc(g)
        if k < 1 or k + 5 >= len(D): continue
        c0 = dc.iloc[k - 1]; at = lambda h: op.get(g + pd.Timedelta(h), np.nan)
        o = at('9h30min')
        if not o == o or o / c0 - 1 > -.05: continue
        rows.append(dict(tk=tk, g=g, gap=o / c0 - 1, pm8=at('8h') / c0 - 1, o930=o / c0 - 1, o945=at('9h45min') / c0 - 1, o1030=at('10h30min') / c0 - 1,
                         close=dc.iloc[k] / c0 - 1, d1=dc.iloc[k + 1] / c0 - 1, d5=dc.iloc[k + 5] / c0 - 1))
R = pd.DataFrame(rows); R['h'] = np.where(R.g.dt.year <= 2021, '2019-21', '2022-26')
cols = ['pm8', 'o930', 'o945', 'o1030', 'close', 'd1', 'd5']
print(f'{len(R)} gaps <= -5%   (returns vs the prior close; higher = better exit)')
print((R.groupby('h')[cols].mean() * 100).round(2).to_string())
print('\nvs the system exit (09:45): share of gaps where each exit was better')
print(pd.DataFrame({c: (R[c] > R.o945).groupby(R.h).mean() for c in cols if c != 'o945'}).round(2).to_string())
print('\nmedian'); print((R.groupby('h')[cols].median() * 100).round(2).to_string())
print('\ndeep gaps <= -8%'); Z = R[R.gap <= -.08]; print(len(Z)); print((Z[cols].mean() * 100).round(2).to_string())
