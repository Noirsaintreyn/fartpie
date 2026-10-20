"""Stream dolt_option_chain_slice.csv -> per (date, symbol) IV surface features. stdlib only.
near expiry: DTE closest to 30 (14..60); far expiry: DTE closest to 91 (61..150).
iv at target delta = IV of contract with delta closest to target (within 0.1)."""
import csv, sys, os, datetime as dt
SRC = os.path.join(os.path.dirname(__file__), 'raw', 'dolt_option_chain_slice.csv')
OUT = os.path.join(os.path.dirname(__file__), 'ext', 'opt_features.csv')
D = dt.date.fromisoformat

def pick(rows, cp, target):
    best = None
    for r in rows:
        if r[4] != cp: continue
        try: d = float(r[8]); v = float(r[7])
        except: continue
        if v <= 0: continue
        e = abs(d - target)
        if best is None or e < best[0]: best = (e, v)
    return best[1] if best and best[0] <= 0.1 else ''

def feats(date, sym, rows):
    d0 = D(date)
    byexp = {}
    for r in rows: byexp.setdefault(r[2], []).append(r)
    exps = [(abs((D(e) - d0).days - t), e) for e in byexp for t in ()]
    def choose(lo, hi, tgt):
        c = [((abs((D(e) - d0).days - tgt)), e) for e in byexp if lo <= (D(e) - d0).days <= hi]
        return byexp[min(c)[1]] if c else None
    near, far = choose(14, 60, 30), choose(61, 150, 91)
    out = [date, sym, '', '', '', '', '', '', '']
    if near:
        c50, p50, p25, c25 = pick(near, 'Call', .5), pick(near, 'Put', -.5), pick(near, 'Put', -.25), pick(near, 'Call', .25)
        # call-put IV spread at matched strikes, |delta| 0.2..0.8 on the call
        calls = {r[3]: r for r in near if r[4] == 'Call'}
        diffs = []
        for r in near:
            if r[4] != 'Put' or r[3] not in calls: continue
            c = calls[r[3]]
            try:
                if 0.2 <= float(c[8]) <= 0.8 and float(c[7]) > 0 and float(r[7]) > 0:
                    diffs.append(float(c[7]) - float(r[7]))
            except: pass
        cps = sum(diffs) / len(diffs) if diffs else ''
        out[2:7] = [c50, p50, p25, c25, cps]
    if far:
        out[7] = pick(far, 'Call', .5); out[8] = pick(far, 'Put', -.5)
    return out

with open(SRC, newline='') as f, open(OUT, 'w', newline='') as g:
    rd = csv.reader(f); next(rd); w = csv.writer(g)
    w.writerow(['date', 'sym', 'iv_c50', 'iv_p50', 'iv_p25', 'iv_c25', 'cp_spread', 'iv_c50_far', 'iv_p50_far'])
    key, buf, n = None, [], 0
    for r in rd:
        if len(r) < 13: continue
        k = (r[0], r[1])
        if k != key:
            if buf: w.writerow(feats(key[0], key[1], buf)); n += 1
            key, buf = k, []
            if n % 20000 == 0: print(n, key, flush=True)
        buf.append(r)
    if buf: w.writerow(feats(key[0], key[1], buf))
print('done', n)
