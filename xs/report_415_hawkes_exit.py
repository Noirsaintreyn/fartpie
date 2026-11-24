"""Nov 24 2026 report 415. Rey: no time-based holds; exit when the thesis dies (vector flip or Hawkes intensity), 'as soon as we know we're wrong'.
DECLARED: live book entries (12,360 gated 15m flips).  Daily Hawkes-style intensity per name: event on day t if close-to-close ret >= 1 x 20d sigma AND volume >= 1 x prior-20d mean (up-excitation with volume); lambda_t = sum_s event_s * exp(-(t-s)/tau), tau = 5d.  HAWKES-DEATH exit: from the first close after the entry day, exit at the first close where lambda <= theta x (peak lambda since entry day close) and that peak >= 1 (move had energy and it has died); theta = 0.25.  Wrong-exit = the live vector exit.  Exit = whichever comes first (MIN).
Neighbours: (tau,theta) = (3,.25) (8,.25) (5,.15) (5,.40).
PASS: MIN arm raises win rate AND median vs LIVE (trade level, every era 18-19/20-21/22/23-26 not worse, pooled gain) and in the 6-slot true-MTM book improves Calmar and MDD with CAGR >= 90% of live in both halves; >= 3 of 4 neighbours same direction."""
import numpy as np, pandas as pd
_t = open('xs/report_402_five_ideas.py').read(); exec(_t[:_t.index("rows = []; TK0, s0")])
D1 = pd.concat([pd.read_parquet(SP + 'lse_daily_54.parquet'), pd.read_parquet(SP + 'lse_daily_pool.parquet')]).drop_duplicates(['symbol', 'ts']); D1['day'] = pd.to_datetime(D1.ts).dt.normalize()
G = {}
for tk, g in D1.groupby('symbol'):
    g = g.sort_values('day').drop_duplicates('day').set_index('day'); r = g.close.pct_change(); sd = r.rolling(20).std().shift(1); vr = g.volume / g.volume.rolling(20).mean().shift(1)
    G[tk] = ((r >= sd) & (vr >= 1)).astype(float)
def lam(ev, tau): return ev.ewm(alpha=1 - np.exp(-1 / tau), adjust=False).mean() / (1 - np.exp(-1 / tau)) if False else ev.copy().pipe(lambda e: pd.Series(_ex(e.values, tau), index=e.index))
def _ex(v, tau):
    o = np.zeros(len(v)); a = np.exp(-1 / tau); s = 0.
    for i, x in enumerate(v): s = s * a + x; o[i] = s
    return o
def modify(Bv, tau, th):
    L = {tk: lam(ev, tau) for tk, ev in G.items()}; out = Bv.copy(); ne = 0; xs_ = []; rs = []
    for i, (tk, e, x) in enumerate(zip(Bv.tk, Bv.entry, Bv.exit)):
        xo = x; ro = Bv.ret.iloc[i]
        if tk in L:
            l = L[tk]; d0 = pd.Timestamp(e).normalize(); d1 = pd.Timestamp(x).normalize(); w = l[(l.index >= d0) & (l.index <= d1)]
            if len(w) > 1:
                pk = w.iloc[0]
                for dd, v in w.iloc[1:].items():
                    pk = max(pk, v)
                    if pk >= 1 and v <= th * pk and dd < d1:
                        xo = dd + pd.Timedelta(hours=16); ro = px_at(tk, xo) / px_at(tk, e) - 1; ne += 1; break
        xs_.append(xo); rs.append(ro)
    out['exit'] = xs_; out['ret'] = rs; return out, ne
def tl(B, lab):
    n = B.ret - 4e-4; e = pd.to_datetime(B.entry); era = pd.cut(e.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26']); hold = (pd.to_datetime(B.exit) - e).dt.total_seconds() / 86400
    s = f"{lab:22s} n{len(B):6d} win {(n>0).mean()*100:5.1f} med {n.median()*100:6.2f} mean {n.mean()*100:5.2f} hold {hold.mean():5.2f}d |"
    for k in ('18-19', '20-21', '22', '23-26'): m = (era == k).values; s += f" {(n[m]>0).mean()*100:4.1f}/{n[m].median()*100:5.2f} |"
    print(s)
print('trade level: win med mean hold | era win/med 18-19 | 20-21 | 22 | 23-26'); tl(B0, 'LIVE vector exit')
TK0, s0 = live_fn(B0); b0 = score('LIVE', TK0, s0); print(line(b0) + f" | CAGR halves {b0['ca'][0]:.1%} | {b0['cb'][0]:.1%}", flush=True)
for tau, th in ((5, .25), (3, .25), (8, .25), (5, .15), (5, .40)):
    Bm, ne = modify(B0, tau, th); tl(Bm, f'MIN tau{tau} th{th} ({ne} H-exits)'); TKm, sm = live_fn(Bm); d = score(f'book MIN tau{tau} th{th}', TKm, sm); print(line(d) + f" | CAGR halves {d['ca'][0]:.1%} | {d['cb'][0]:.1%}", flush=True)
