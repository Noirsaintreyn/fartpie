"""Do Rey's level detectors (HDBSCAN, OPTICS, GMM, TDA, KDE, MeanShift; QQQ daily + 4h) beat PLACEBO levels (level +/-0.5 ATR), and does pairing with HVN distance / GEX sign / vol (YZ, HAR, VXN) help?
Events = first touch per level per day; R-multiples: R1 stop 0.10 ATR 1:1, R2 0.10 1.5:1, R3 0.20 1.5:1 (2:3 RR = R2/R3). Edge = real - placebo mean R. Eras A <2022, B >=2022. Nov 11 2026."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
E = pd.concat([pd.read_pickle(f) for f in glob.glob(SC + 'lvl_events_*.pkl')], ignore_index=True); E['era'] = np.where(E.day < '2022-01-01', 'A', 'B')
print(f'events {len(E)}  methods {sorted(E.method.unique())}')
def t2(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if len(a) > 20 and len(b) > 20 else np.nan
def row(lab, S, cfg):
    r, p = S[S.plc == 0], S[S.plc == 1]; out = f' {lab:30} n {len(r):5d} real {r[cfg].mean():+.3f}R placebo {p[cfg].mean():+.3f}R edge {r[cfg].mean()-p[cfg].mean():+.3f} (t {t2(r[cfg], p[cfg]):+.1f})'
    eds = []
    for e in 'AB':
        re, pe = r[r.era == e][cfg], p[p.era == e][cfg]; eds.append((re.mean() - pe.mean(), t2(re, pe))); out += f' | {e} edge {eds[-1][0]:+.3f} (t {eds[-1][1]:+.1f})'
    ok = all(np.isfinite(x[1]) and abs(x[1]) >= 2 for x in eds) and np.sign(eds[0][0]) == np.sign(eds[1][0]) and eds[0][0] > 0
    print(out + (' PASS' if ok else '')); return ok
passes = []
for cfg in ('R1', 'R2', 'R3'):
    print(f'\n=== {cfg} by method x timeframe ===')
    for (m, tf), S in E.groupby(['method', 'tf']):
        if m == 'Random' or True:
            if row(f'{m} {tf}', S, cfg): passes.append((cfg, m, tf, 'alone'))
print('\n=== PAIRING (R2): real vs placebo within bucket ===')
for (m, tf), S in E.groupby(['method', 'tf']):
    if m == 'Random': continue
    for lab, mask in (('HVN within 0.15 ATR', S.hvd < .15), ('HVN far (>0.5 ATR)', S.hvd > .5), ('GEX positive', S.gex > 0), ('GEX negative', S.gex < 0),
                      ('YZ vol high third', S.yz >= S.yz.quantile(2/3)), ('YZ vol low third', S.yz <= S.yz.quantile(1/3)), ('HAR high third', S.har >= S.har.quantile(2/3)), ('VXN high third', S.vxn >= S.vxn.quantile(2/3)),
                      ('HVN near & GEX +', (S.hvd < .15) & (S.gex > 0)), ('HVN near & YZ low', (S.hvd < .15) & (S.yz <= S.yz.quantile(1/3)))):
        if mask.sum() > 200 and row(f'{m} {tf} | {lab}', S[mask], 'R2'): passes.append(('R2', m, tf, lab))
print('\nPASS (edge>0 vs placebo, |t|>=2 both eras):', passes if passes else 'none')
