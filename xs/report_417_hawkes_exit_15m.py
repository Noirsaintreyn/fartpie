"""Nov 24 2026 report 417. Rey: 'try hawkes on the lower time frame'; exits only from signals.
DECLARED: live book entries (12,360 gated 15m flips).  15m RTH bars per name: event bar if bar return >= 1 x rolling-96-bar sigma (shifted) AND bar volume >= rolling-96 mean (up-excitation with volume).  lambda_i = sum_s event_s exp(-(i-s)/tau), tau = 8 bars (2h).  HAWKES-DEATH: from the bar after the entry bar, exit (next bar open) at the first bar where lambda <= theta x peak lambda since entry (peak >= 1), theta = .25.  Wrong-exit = the live vector exit; exit = whichever first (MIN).
Neighbours: (tau,theta) = (4,.25) (16,.25) (8,.15) (8,.40).
PASS: MIN raises win rate AND median vs LIVE at trade level with every era not worse, and in the 6-slot true-MTM book improves Calmar and MDD with CAGR >= 90% of live in BOTH halves; >= 3 of 4 neighbours same direction."""
import glob, numpy as np, pandas as pd
_t = open('xs/report_402_five_ideas.py').read(); exec(_t[:_t.index("rows = []; TK0, s0")])
TKS = sorted(B0.tk.unique()); BAR = {}
for tk in TKS:
    fs = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')
    if not fs: continue
    H = pd.read_parquet(fs[0]); tc = [c for c in H.columns if c in ('datetime', 'ts', 'timestamp')][0]; H['dt'] = pd.to_datetime(H[tc], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
    r = H.close.pct_change(); sg = r.rolling(96).std().shift(1); vm = H.volume.rolling(96).mean().shift(1); BAR[tk] = (H.dt.values, H.open.values, ((r >= sg) & (H.volume >= vm)).astype(float).values)
def lam(ev, tau):
    o = np.zeros(len(ev)); a = np.exp(-1 / tau); s = 0.
    for i in range(len(ev)): s = s * a + ev[i]; o[i] = s
    return o
def modify(Bv, tau, th):
    L = {tk: lam(v[2], tau) for tk, v in BAR.items()}; xs_ = []; rs = []; ne = 0
    for tk, e, x, r0 in zip(Bv.tk, Bv.entry, Bv.exit, Bv.ret):
        xo, ro = x, r0
        if tk in BAR:
            dts, op, _ = BAR[tk]; l = L[tk]; a = np.searchsorted(dts, np.datetime64(e)); b = np.searchsorted(dts, np.datetime64(x))
            if b - a > 2:
                pk = l[a]
                for j in range(a + 1, b - 1):
                    pk = max(pk, l[j])
                    if pk >= 1 and l[j] <= th * pk:
                        xo = pd.Timestamp(dts[j + 1]); ro = op[j + 1] / op[a] - 1 - 2e-4; ne += 1; break
        xs_.append(xo); rs.append(ro)
    out = Bv.copy(); out['exit'] = xs_; out['ret'] = rs; return out, ne
def tl(B, lab):
    n = B.ret - 4e-4; e = pd.to_datetime(B.entry); era = pd.cut(e.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26']); hold = (pd.to_datetime(B.exit) - e).dt.total_seconds() / 86400
    s = f"{lab:30s} n{len(B):6d} win {(n>0).mean()*100:5.1f} med {n.median()*100:6.2f} mean {n.mean()*100:5.2f} hold {hold.mean():5.2f}d |"
    for k in ('18-19', '20-21', '22', '23-26'): m = (era == k).values; s += f" {(n[m]>0).mean()*100:4.1f}/{n[m].median()*100:5.2f} |"
    print(s)
print('names with bars', len(BAR), '| trade level: win med mean hold | era win/med 18-19 | 20-21 | 22 | 23-26'); tl(B0, 'LIVE vector exit')
TK0, s0 = live_fn(B0); b0 = score('LIVE', TK0, s0); print(line(b0) + f" | CAGR halves {b0['ca'][0]:.1%} | {b0['cb'][0]:.1%}", flush=True)
for tau, th in ((8, .25), (4, .25), (16, .25), (8, .15), (8, .40)):
    Bm, ne = modify(B0, tau, th); tl(Bm, f'MIN tau{tau} th{th} ({ne} H-exits)'); TKm, sm = live_fn(Bm); d = score(f'book MIN tau{tau} th{th}', TKm, sm); print(line(d) + f" | CAGR halves {d['ca'][0]:.1%} | {d['cb'][0]:.1%}", flush=True)
