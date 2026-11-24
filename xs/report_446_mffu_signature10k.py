"""Nov 24 2026 report 446. Rey: is the futures prop (MFFU Signature 10K) better, and run the model on it. DECLARED BEFORE RESULTS.
RULES (official plan page read 2026-11-24, per Rey's paste): $10,000 account, eval target +$1,000 (balance >= $11,000), STATIC max loss $500 (floor $9,500, never trails/moves, treated as INCLUDING open P&L), no daily loss limit, no consistency, min 1 trading day, 10 micros max (no minis), no time limit; sim-funded same $500 static floor, payout any time when flat, min $250, 80% split, no buffer, withdraw down to $10,000 (floor stays 9,500 so each withdrawal restores a $500 cushion), flat by 16:10 ET (this model is flat at 12:00), eval $116.50 promo, $0 activation, reset price unknown -> rebuy at $116.50 (assumption).
MODEL: the base NQ flip trade of nq_firms.py / report 24x (QQQ 15m vector flip, bars closing 09:30-11:30, both ways, <= 2 a day, TP 75 / SL 50 NQ points, flat 12:00, 1 pt cost, QQQ proxy -> NQ points) with its 5m path kept; open P&L enforced in real time (bar order against us). Sessions 2022-2026 (the framework's out-of-train window). MNQ = $2/pt per micro.
SIZING arms: fixed n micros in {1,2,3,5}; CUSHION f: n = floor(f * cushion / (50 pts * $2)) micros, f in {.25,.4,.6}, cushion = equity above the 9,500 floor at the trade's entry, skip if n < 1, cap 10.
PAYOUT policies (mine, declared): withdraw (equity-above-$10,000 minus buffer B) when it reaches T: (T=250,B=0) (500,0) (1000,0) (750,B=500).
OUTPUT (Rey's format): share PASSED by 30/60/90 calendar days from the start session; share with FIRST PAYOUT by 60/90/120d; eval fail %; funded blown before 1st payout; net $ per eval bought (80% payouts - $116.50); 12-month careers with rebuys (start monthly 2022-01..2025-10): paid at least once, median / 10th pct net.
BAR (Rey's): >= 1 payout a month on average and <= ~$250 spent in the longest drought; compare with the NQ/Lucid and perps results."""
_s = open('xs/nq_firms.py').read(); exec(_s[:_s.index("F = {\n")])
import numpy as np, pandas as pd
FEE, MLL, TGT, PM, SPL = 116.5, 500.0, 1000.0, 2.0, 0.8
def run_day(k, bal, size_fn):
    """one session at balance bal (realized). returns (new_bal, breached, ntr). equity = bal + realized-so-far + open path."""
    cum = 0.0; ntr = 0
    for th, net, mae, p in DAY[k]:
        cush = bal + cum - (10000 - MLL); n = size_fn(cush)
        if n < 1: continue
        m = PM * n; ntr += 1
        for x in p:
            if bal + cum + x * m <= 10000 - MLL: return bal + cum + x * m, True, ntr
        cum += net * m
    return bal + cum, False, ntr
def mk_size(kind, v):
    if kind == 'fixed': return lambda c: v
    return lambda c: int(min(10, np.floor(v * c / (50 * PM))))
def eval_phase(s, end, sz):
    bal = 10000.0
    for k in range(s, end):
        bal, br, nt = run_day(k, bal, sz)
        if br: return 'fail', k
        if bal >= 10000 + TGT: return 'pass', k
    return 'open', end
def funded(s, end, sz, T, B):
    bal = 10000.0; pays = []
    for k in range(s, end):
        bal, br, nt = run_day(k, bal, sz)
        if br: return 'lost', k, pays
        prof = bal - 10000 - B
        if prof >= T: pays.append((k, SPL * prof)); bal -= prof
    return 'open', end, pays
def career(s, end, sz, T, B):
    k = s; fees = got = 0.0; first = None; evals = 0
    while k < end:
        evals += 1; fees += FEE; r, k2 = eval_phase(k, end, sz)
        if r == 'fail': k = k2 + 1; continue
        if r == 'open': break
        fr, k3, pays = funded(k2 + 1, end, sz, T, B); got += sum(a for _, a in pays)
        if pays and first is None: first = (evals, k)
        if fr == 'lost': k = k3 + 1; continue
        break
    return dict(net=got - fees, paid=first is not None, evals=evals, fees=fees)
cal = pd.DatetimeIndex(D_); last = int(np.searchsorted(D_, pd.Timestamp('2025-09-01')))
mstarts = [int(np.searchsorted(D_, pd.Timestamp(f'{y}-{m:02d}-01'))) for y in range(2022, 2026) for m in range(1, 13) if (y, m) <= (2025, 10)]
tr = pd.DataFrame([(net, mae) for dd in DAY for th, net, mae, p in dd], columns=['net', 'mae']); print(f'base trades {len(tr)}: avg {tr.net.mean():+.1f} pts win {(tr.net>0).mean()*100:.0f}% avg MAE {tr.mae.mean():.1f} pts', flush=True)
ARMS = [('fixed', n) for n in (1, 2, 3, 5)] + [('cush', f) for f in (.25, .4, .6)]; POL = [(250, 0), (500, 0), (1000, 0), (750, 500)]
print('\narm | policy | passed<=30/60/90d | 1st payout<=60/90/120d | eval fail% open% | blown before 1st payout (of passed) | net$/eval | 12m careers: paid%, med net, p10, evals bought')
for kind, v in ARMS:
    sz = mk_size(kind, v)
    for T, B in POL:
        A = []
        for s in range(last):
            r, k = eval_phase(s, NS, sz); rec = dict(r=r, tp=(cal[k] - cal[s]).days if r == 'pass' else np.nan, tf=np.nan, np_=0, got=0.0, lost=False)
            if r == 'pass':
                fr, k2, pays = funded(k + 1, NS, sz, T, B); rec.update(np_=len(pays), got=sum(a for _, a in pays), lost=fr == 'lost' and not pays, tf=(cal[pays[0][0]] - cal[s]).days if pays else np.nan)
            A.append(rec)
        A = pd.DataFrame(A); Kc = pd.DataFrame([career(s, min(s + 252, NS), sz, T, B) for s in mstarts]); a = lambda c, d: (A[c] <= d).mean() * 100
        print(f"{kind}{v} | T{T} B{B} | {a('tp',30):3.0f} {a('tp',60):3.0f} {a('tp',90):3.0f} | {a('tf',60):3.0f} {a('tf',90):3.0f} {a('tf',120):3.0f} | {(A.r=='fail').mean()*100:3.0f} {(A.r=='open').mean()*100:3.0f} | {A.lost.sum()/max((A.r=='pass').sum(),1)*100:3.0f}% | {(A.got-FEE).mean():+6.0f} | {Kc.paid.mean()*100:3.0f}% {Kc.net.median():+6.0f} {Kc.net.quantile(.1):+6.0f} {Kc.evals.mean():4.1f}", flush=True)
