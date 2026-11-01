"""Nov 1 2026: macro event days, found from QQQ's own volume (no calendar file): 08:30 data days (CPI, jobs, PCE...) = the 08:30 15m bar's volume
>= 4x its 60-day median; FOMC days = Wednesday 14:00 bar volume >= 2.5x its median (top ~8/yr). Tests: avg trade by entry day type,
skip entries on those days, skip FOMC-day entries before 14:15, and skip the day before a data day."""
import os, glob, numpy as np, pandas as pd
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
src = open('/home/claude/fartpie/xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
Q = pd.read_parquet(glob.glob('/mnt/user-data/uploads/fartpie/lse_clean/QQQ_15m_*.parquet')[0]); Q['dt'] = pd.to_datetime(Q.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
Q['d'] = Q.dt.dt.normalize(); Q['t'] = Q.dt.dt.strftime('%H:%M')
def spike(t, k):
    v = Q[Q.t == t].set_index('d').volume; return v[v / v.rolling(60, min_periods=20).median().shift(1) >= k].index
DATA = spike('08:30', 4.0); F = [d for d in spike('14:00', 2.5) if d.dayofweek == 2]
print(f'data days {len(DATA)} (~{len(DATA) / 7.8:.0f}/yr), FOMC-like days {len(F)} (~{len(F) / 7.8:.0f}/yr); sample FOMC: {[f"{d:%Y-%m-%d}" for d in F[-8:]]}')
print(f'sample data days: {[f"{d:%Y-%m-%d}" for d in DATA[-10:]]}')
L = pd.read_parquet('/tmp/claude-0/ml/mtf_states.parquet').sort_values('entry').reset_index(drop=True); L['d'] = L.entry.dt.normalize()
DS, FS = set(DATA), set(F); TD = pd.DatetimeIndex(sorted(Q.d.unique())); nxt = {TD[i]: TD[i + 1] for i in range(len(TD) - 1)}
L['data'] = L.d.isin(DS); L['fomc'] = L.d.isin(FS); L['pre_data'] = L.d.map(nxt).isin(DS)
TK = taken(L).merge(L[['tk', 'entry', 'data', 'fomc', 'pre_data']], on=['tk', 'entry']); TK['h'] = np.where(TK.entry.dt.year <= 2021, '2019-21', '2022-26')
for c in ('data', 'fomc', 'pre_data'):
    t = TK.groupby([c, 'h']).ret.agg(['size', 'mean']).unstack('h'); print(f'\n{c}: avg trade % (n)'); print((t * [1, 1, 100, 100]).round(2).to_string())
P = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def sh(nm, Z): print(f'  {nm:<44}' + ''.join(f'{c:>+8.1%} /{d:>4.0%}   ' for c, d in (sim(Z, a, b) for a, b in P)), flush=True)
print('\nbook (yearly return / max DD)'); sh('live', L)
sh('skip entries on data days', L[~L.data]); sh('skip entries on FOMC days', L[~L.fomc])
sh('skip FOMC-day entries before 14:15', L[~(L.fomc & (L.entry.dt.hour * 60 + L.entry.dt.minute < 855))])
sh('skip entries the day before a data day', L[~L.pre_data]); sh('skip data + FOMC days', L[~(L.data | L.fomc)])
print('\njobs report = first Friday of the month (approx; data-day volume detection is too sparse before 2025)')
L['nfp'] = (L.d.dt.dayofweek == 4) & (L.d.dt.day <= 7)
TK = taken(L).merge(L[['tk', 'entry', 'nfp']], on=['tk', 'entry']); TK['h'] = np.where(TK.entry.dt.year <= 2021, '2019-21', '2022-26')
print((TK.groupby(['nfp', 'h']).ret.agg(['size', 'mean']).unstack('h') * [1, 1, 100, 100]).round(2).to_string())
sh('skip entries on jobs-report Fridays', L[~L.nfp])
