"""Nov 14 2026 NEW IDEAS, hypothesis-first. (A) GAP TAX: how much of a flip's return is lost between the confirming close and the next open, and is entering at the close (last minutes) better? Daily flips 2019-Mar 2026 as before (close>=$1, dv>=$500k), entry next open (system) vs entry at the flip-bar close (feasible only if the flip is knowable ~5 min before the close; check below with 5m data), exit open after opposite flip, 30bp both. H_A: close entry beats next-open entry by >0 in both eras, paired month-clustered t>=2. (A2) FEASIBILITY, 60d of 5m: among bear/neutral days, does price at 15:55 (close of the 15:50 bar) > T identify the flip days? Precision/recall; mean 15:55 price vs next open vs close on true flips. (B) TRUE vs FALSE CROSS classifier on intraday crosses of T (60d of 5m, $0.5-10, dv>=$300k): features at cross time: VOLPACE = cumulative RTH volume to the cross bar / median full-RTH-day volume of prior days (>=5), TMIN = minute of day of cross, GAP = day open / prior close - 1, EXT = fill/T - 1, PRIOR5 = 5-day return. Predicted: VOLPACE high -> true; TMIN early -> true; GAP large and EXT large -> false. Split the 60 days into first/second half by date; PASS = predicted direction with AUC beyond 0.55/0.45 in BOTH halves. (C) FLIP CONTAGION: number of other vector flips in the same 2-digit-SIC industry in the 5 calendar days before (not including) the flip; terciles within month; H_C: more peers flipping first = better (theme wave) ; PASS both eras, pooled clustered t>=2. Caveats: survivorship-flattered, 60d one regime for B/A2."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; X = SC + 'ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl')
rows = []
for tk, B in D.items():
    B = B[~B.index.duplicated()]; B = B[B.index >= '2018-06-01']
    if len(B) < 300: continue
    o = B.open.values; c = B.close.values; v = B.volume.values.astype(float); idx = B.index; n = len(B); reg, _, _, _ = run_state_machine(B.assign(datetime=B.index)); dv = pd.Series(c * v).rolling(20).median().values; i = 130
    while i < n - 2:
        if reg[i] == 1 and reg[i - 1] != 1:
            j = i + 1
            while j < n and reg[j] != -1: j += 1
            if j + 1 < n and '2019-01-01' <= str(idx[i])[:10] < '2026-03-01' and c[i] >= 1 and dv[i] >= 5e5: rows.append((tk, idx[i], o[j + 1] / o[i + 1] - 1 - .003, o[j + 1] / c[i] - 1 - .003, o[i + 1] / c[i] - 1, c[i]))
            i = j
        i += 1
F = pd.DataFrame(rows, columns=['tk', 'd', 'rA', 'rC', 'gap', 'px']); F['era'] = np.where(F.d < '2022-01-01', '2019-21', '2022-26Q1'); F['mo'] = F.d.dt.to_period('M')
def ct(a, b, key='mo', z=None):
    d = (z.groupby(key)[a].mean() - z.groupby(key)[b].mean()).dropna(); return d.mean() / (d.std() / np.sqrt(len(d)))
print('(A) GAP TAX, all flips: next-open entry (system) vs flip-bar-close entry')
for lab, sub in (('ALL', F), ('$1-10', F[F.px < 10]), ('$10+', F[F.px >= 10])):
    for en in ('2019-21', '2022-26Q1'):
        z = sub[sub.era == en]; print(f'  {lab:6s} {en:9s} n {len(z):5d} gap (next open vs flip close) mean {z.gap.mean()*100:+5.2f}% med {z.gap.median()*100:+5.2f}% | system mean {z.rA.mean()*100:+5.2f}% med {z.rA.median()*100:+5.2f}% | close-entry mean {z.rC.mean()*100:+5.2f}% med {z.rC.median()*100:+5.2f}% | paired +{(z.rC-z.rA).mean()*100:.2f}pp t {ct("rC","rA",z=z):+.1f} | huge% sys {(z.rA>=1).mean()*100:.2f} close {(z.rC>=1).mean()*100:.2f}')
# (A2) + (B) from 5m
M5 = pickle.load(open(SC + 'lowpx_5m.pkl', 'rb')); DD = pickle.load(open(SC + 'lowpx_daily.pkl', 'rb'))
def clean(x):
    x = x[['Open', 'High', 'Low', 'Close', 'Volume']].copy(); x.columns = ['open', 'high', 'low', 'close', 'volume']; x.index = x.index.tz_convert('America/New_York') if x.index.tz is not None else x.index; x = x[~x.index.duplicated()]; m = x.index.hour * 60 + x.index.minute; return x[(m >= 570) & (m < 960)]
ev = []; day = []
for tk, x in M5.items():
    x = clean(x); x = x[x.volume > 0]
    if len(x) < 400 or (x.close * x.volume).groupby(x.index.date).sum().median() < 3e5 or not 0.5 <= x.close.median() < 10: continue
    q = DD.get(tk)
    if q is None or len(q) < 120: continue
    q = q[['Open', 'High', 'Low', 'Close', 'Volume']].copy(); q.columns = ['open', 'high', 'low', 'close', 'volume']; q = q[~q.index.duplicated()]; q.index = pd.to_datetime(q.index).tz_localize(None).normalize(); reg, lev, atr, _ = run_state_machine(q.assign(datetime=q.index)); o = q.open.values; c = q.close.values; n = len(q)
    x = x.assign(date=pd.to_datetime(x.index.date), m=x.index.hour * 60 + x.index.minute); g = dict(list(x.groupby('date'))); dvol = x.groupby('date').volume.sum()
    for i in range(6, n - 1):
        dts = q.index[i]
        if dts not in g or not 0.5 <= c[i] < 10 or reg[i - 1] == 1 or np.isnan(lev[i - 1]) or np.isnan(atr[i - 1]): continue
        T = lev[i - 1] + .35 * atr[i - 1]; b = g[dts]; flip = reg[i] == 1; late = b[b.m <= 950]; p1555 = late.close.iloc[-1] if len(late) else np.nan
        day.append((tk, dts, flip, p1555 > T if not np.isnan(p1555) else False, p1555, o[i + 1], c[i], T))
        hit = b[b.high >= T]
        if hit.empty: continue
        prior = dvol[dvol.index < dts].tail(20)
        if len(prior) < 5: continue
        k = hit.index[0]; cum = b.volume[b.index <= k].sum(); fill = max(T, hit.open.iloc[0])
        ev.append((tk, dts, flip, cum / prior.median(), hit.m.iloc[0], b.open.iloc[0] / c[i - 1] - 1, fill / T - 1, c[i - 1] / c[i - 6] - 1, fill, c[i], o[i + 1]))
Dy = pd.DataFrame(day, columns=['tk', 'd', 'flip', 'sig', 'p1555', 'nopen', 'close', 'T']); print('\n(A2) FEASIBILITY, price at 15:55 > T as a flip detector (bear/neutral days, 60d 5m):')
tp = ((Dy.sig) & Dy.flip).sum(); fp = ((Dy.sig) & ~Dy.flip).sum(); fn = ((~Dy.sig) & Dy.flip).sum(); print(f'  flip days {int(Dy.flip.sum())}: caught {tp}, missed {fn}; false alarms {fp}  -> precision {tp/max(tp+fp,1)*100:.0f}% recall {tp/max(tp+fn,1)*100:.0f}%')
s = Dy[Dy.sig & Dy.flip]; print(f'  on caught flips: 15:55 price vs next open {np.mean(s.nopen/s.p1555-1)*100:+.2f}% (gap you still eat), vs flip-day close {np.mean(s.close/s.p1555-1)*100:+.2f}%')
f_ = Dy[Dy.sig & ~Dy.flip]; print(f'  false alarms: 15:55 price -> final close {np.mean(f_.close/f_.p1555-1)*100:+.2f}%, -> next open {np.mean(f_.nopen/f_.p1555-1)*100:+.2f}% (cost of acting on them)')
E_ = pd.DataFrame(ev, columns=['tk', 'd', 'flip', 'VOLPACE', 'TMIN', 'GAP', 'EXT', 'PRIOR5', 'fill', 'cl', 'nopen']); E_ = E_.replace([np.inf, -np.inf], np.nan).dropna(); print('\n(B) TRUE vs FALSE cross classifier: n crosses', len(E_), 'true share', round(E_.flip.mean(), 3))
def auc(s, y): r = s.rank(); n1 = y.sum(); n0 = len(y) - n1; return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
E_ = E_.sort_values('d'); mid = E_.d.iloc[len(E_) // 2]; E_['half'] = np.where(E_.d < mid, 'H1', 'H2'); pred = {'VOLPACE': 1, 'TMIN': -1, 'GAP': -1, 'EXT': -1, 'PRIOR5': 0}
for f_, sg in pred.items():
    a = [auc(E_[E_.half == h][f_], E_[E_.half == h].flip) for h in ('H1', 'H2')]; ok = sg != 0 and all((x - .5) * sg >= .05 for x in a); print(f'  {f_:8s} AUC(true flip) H1 {a[0]:.3f} H2 {a[1]:.3f}  predicted {"high" if sg>0 else "low" if sg<0 else "none"} -> {"PASS" if ok else "no pass"}')
E_['B'] = np.where(E_.flip, E_.nopen / E_.fill - 1, E_.cl / E_.fill - 1); w = lambda s: s.clip(-.5, 1)
E_['vt'] = pd.qcut(E_.VOLPACE.rank(method='first'), 3, labels=False); print('  every-cross return by VOLPACE tercile (true -> next open [proxy for hold], false -> same-day close; winsorized):')
for k in (0, 1, 2): z = E_[E_.vt == k]; print(f'    T{k} n {len(z)} true share {z.flip.mean()*100:.0f}% mean {w(z.B).mean()*100:+.2f}% med {z.B.median()*100:+.2f}%')
# (C) contagion
Fc = pd.read_pickle(X + 'flip_context.pkl'); sic = pd.read_csv(X + 'sic.csv').drop_duplicates('tk').set_index('tk').sic; Fc['g'] = (Fc.tk.map(sic) // 100); Fc = Fc.dropna(subset=['g']); Fc['dn'] = Fc.d.dt.normalize(); cnt = []
by = {g: np.sort(z.dn.values.astype('datetime64[D]').astype(int)) for g, z in Fc.groupby('g')}
for g, d in zip(Fc.g, Fc.dn): a = by[g]; di = np.datetime64(d, 'D').astype(int); cnt.append(((a >= di - 5) & (a < di)).sum())
Fc['peers'] = cnt; Fc['pt'] = Fc.groupby('mo').peers.transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=False) if len(s) >= 9 else np.nan); print('\n(C) FLIP CONTAGION: peers = other flips in same 2-digit SIC in prior 5 calendar days')
for en in ('2019-21', '2022-26Q1', 'POOLED'):
    z = Fc if en == 'POOLED' else Fc[Fc.era == en]; a = z[z.pt == 2].groupby('mo').r.mean(); b = z[z.pt == 0].groupby('mo').r.mean(); d = (a - b).dropna(); print(f'  {en:9s} T0 n {int((z.pt==0).sum())} mean {z[z.pt==0].r.mean()*100:+5.1f}% huge {(z[z.pt==0].r>=1).mean()*100:.1f}% || T2 n {int((z.pt==2).sum())} mean {z[z.pt==2].r.mean()*100:+5.1f}% huge {(z[z.pt==2].r>=1).mean()*100:.1f}% || top-bottom {(z[z.pt==2].r.mean()-z[z.pt==0].r.mean())*100:+.1f}pp t {d.mean()/(d.std()/np.sqrt(len(d))):+.1f}')
