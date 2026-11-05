"""Nov 4 2026, POST-HOC after nq_adaptive_buffer.log: STATIC 0.20 looked better than the live 0.35 in 2018-21 and 2022-23 (not 2024-26). Neighbour check: 0.10 / 0.15 / 0.25 / 0.30."""
_nbx = open('xs/nq_adaptive_buffer.py').read(); _nbx = _nbx.replace("VAR = {", "VAR0 = {"); exec(_nbx[:_nbx.index("rng = np.random.default_rng(5)")])
VAR = {f'STATIC {b:.2f}': np.full(nb, b) for b in (.10, .15, .20, .25, .30, .35)}
exec(_nbx[_nbx.index("rng = np.random.default_rng(5)"):].replace("print(f'15m bars", "0 and print(f'15m bars"))
