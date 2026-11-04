"""Nov 4 2026: capacity check - slow trend models at N slots (1/N each) vs live at the same N. DECLARED BEFORE RESULTS: N in {10, 15}; R3 daily EMA10/20, R4 Donchian20+trail, R5 ensemble, R1 1h EMA cross."""
import re
_s = open('xs/ema_replace.py').read(); exec(_s[:_s.index("B0 = ev(XL)")])
sim_src = open('xs/stop_1h_ev.py').read(); body = sim_src[sim_src.index('def sim('):sim_src.index('def ev(')]
for N in (10, 15):
    ns = {}; exec(body.replace('>= 6', f'>= {N}').replace('/ 6', f'/ {N}').replace('def sim(', 'def simN(') , globals(), ns); globals()['simN'] = ns['simN']
    def evN(Y):
        HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(Y.tk, Y.entry.astype('int64'))], dtype=np.uint64); K = Y.ratio.values.astype(float); res = []
        for i in range(1, 41):
            dl = ((HSH * 2654435761 + i * 40503) % 1000) < 50; m = ~dl; res.append(stat2(simN(Y.entry.values[m], Y.exit.values[m], Y.ret.values[m], K[m], Y.tk.values[m])))
        return np.array(res)
    B0 = evN(XL); print(f'\n== {N} slots @1/{N}: 40 matched runs, 2019-21 / 2022-26'); line(f'L live', B0)
    for kind, nm in (('R3', 'daily EMA10/20'), ('R4', 'Donchian20 + 3ATR trail'), ('R5', 'Donchian ensemble'), ('R1', '1h EMA20/50')): line(f'{kind} {nm}', evN(mk(kind)), B0)
