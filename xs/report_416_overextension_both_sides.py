"""Nov 24 2026 report 416. Rey: overextended + hype names (the trader's short side), signal exits only.
DECLARED: EXT at the prior daily close = 20d return >= 2 x (60d daily sigma x sqrt20) AND 5d mean volume >= 1.5 x 60d mean (extended on hype).  Vector state machine on RTH 15m bars of the 54 research names.  LONG trade = regime flips to +1, enter next bar open, exit next bar open after regime flips to -1.  SHORT trade = mirror (enter on flip to -1, exit on flip to +1; borrow ignored, noted).  Cost 4bp round trip.  No time holds, no stops other than the flip.
H_S: SHORT on EXT beats SHORT non-EXT and zero in win rate, median, mean in every era (18-19, 20-21, 22, 23-26), month-clustered |t| >= 2 pooled.  H_L: LONG on EXT does worse than LONG non-EXT (avoid-filter) in every era.
Caveat: the 54 names are selected by the options-ratio gate (winners), biasing shorts down; comparisons are EXT vs non-EXT within the same direction."""
import glob, numpy as np, pandas as pd
from xs.vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
D1 = pd.concat([pd.read_parquet(SP + 'lse_daily_54.parquet'), pd.read_parquet(SP + 'lse_daily_pool.parquet')]).drop_duplicates(['symbol', 'ts']); D1['day'] = pd.to_datetime(D1.ts).dt.normalize()
tks = sorted(pd.read_parquet(SP + 'b302.parquet').tk.unique()) if glob.glob(SP + 'b302.parquet') else None
if tks is None: tks = sorted({f.split('/')[-1].split('_15m')[0] for f in glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/*_15m_2018-01-01_*.parquet')})
rows = []
for tk in tks:
    fs = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'); g = D1[D1.symbol == tk].sort_values('day').drop_duplicates('day').set_index('day')
    if not fs or len(g) < 100: continue
    H = pd.read_parquet(fs[0]); tc = [c for c in H.columns if c in ('datetime', 'ts', 'timestamp')][0]; H['dt'] = pd.to_datetime(H[tc], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].sort_values('dt').reset_index(drop=True)
    regs, lv, atr, fl = run_state_machine(H[['high', 'low', 'close']].reset_index(drop=True)); r = pd.Series(regs); o = H.open.values; d = H.dt.values
    r20 = g.close.pct_change(20); sg = g.close.pct_change().rolling(60).std() * np.sqrt(20); vr = g.volume.rolling(5).mean() / g.volume.rolling(60).mean(); EXT = ((r20 >= 2 * sg) & (vr >= 1.5)).shift(1)
    pos = 0; e = None
    for i in range(2, len(H)):
        k = regs[i - 1]; kp = regs[i - 2]
        if pos == 0 and k != kp and k != 0: pos = k; e = i; ed = pd.Timestamp(d[i]).normalize(); ext = EXT.reindex([ed], method='ffill').iloc[0]; ep = o[i]
        elif pos != 0 and k == -pos and i > e:
            ret = pos * (o[i] / ep - 1) - 4e-4; rows.append(dict(tk=tk, side=pos, entry=pd.Timestamp(d[e]), exit=pd.Timestamp(d[i]), ret=ret, ext=bool(ext) if ext == ext else np.nan)); pos = 0
            if k != 0 and k != kp: pass
T = pd.DataFrame(rows).dropna(subset=['ext']); T['ext'] = T.ext.astype(bool); T = T[T.entry >= '2018-07-01']; e_ = T.entry; T['era'] = pd.cut(e_.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26']); T['mo'] = e_.dt.to_period('M'); T['hold'] = (T.exit - T.entry).dt.total_seconds() / 86400
print('trades', len(T), '| EXT share', T.ext.mean().round(3)); print('arm n | win% med% mean% hold(d) | EXT-nonEXT mean pp (t) | era win/med: 18-19 | 20-21 | 22 | 23-26')
for side, nm in ((1, 'LONG'), (-1, 'SHORT')):
    S = T[T.side == side]; non = S[~S.ext].groupby('mo').ret.mean()
    for lab, d in ((nm + ' all', S), (nm + ' EXT', S[S.ext]), (nm + ' nonEXT', S[~S.ext])):
        if len(d) < 30: print(lab, len(d)); continue
        mm = (S[S.ext].groupby('mo').ret.mean() - non).dropna(); t = mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm))) if len(mm) > 3 else np.nan
        s = f"{lab:14s}{len(d):6d} | {(d.ret>0).mean()*100:5.1f} {d.ret.median()*100:6.2f} {d.ret.mean()*100:6.2f} {d.hold.mean():5.2f} | {mm.mean()*100:+.2f} ({t:+.1f}) |"
        for era in ('18-19', '20-21', '22', '23-26'): x = d[d.era == era]; s += f" {(x.ret>0).mean()*100:4.1f}/{x.ret.median()*100:5.2f}(n{len(x)}) |" if len(x) >= 15 else ' n<15 |'
        print(s)
T.to_parquet(SP + 'ext416.parquet')
