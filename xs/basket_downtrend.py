"""Oct 27 2026: more positions (basket) vs bigger positions when SPY is in a downtrend (daily vector bear + under its 200-day)."""
import numpy as np, pandas as pd
src = open('xs/market_gate.py').read(); exec(src[:src.index("print(f'{\"\":<60}")])
ss2 = ss.replace('def sim_cap(', 'def sim_cw(').replace("for e, x, r_, ww, cp in", "for e, x, r_, ww, cp, wn in").replace("Z.w.values, Z.cap.values)", "Z.w.values, Z.cap.values, Z.wn.values)").replace("pend.append((x, r_, ww / n))", "pend.append((x, r_, ww / wn))")
exec(ss2)
def show2(nm, Z):
    R_ = [sim_cw(Z, a, b) for a, b in PER]
    print(f'  {nm:<52}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in R_) + '  ret/DD ' + ' '.join(f'{c / -d:.1f}' for c, d in R_[:2]), flush=True)
bad = sig['SPY vector bear AND under 200-day']; z0 = L.copy(); z0['cap'] = 6; z0['wn'] = 6
print(f'{"":<54}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}'); show2('live (6 x 1/6)', z0)
for cap, wn, lab in ((7, 6, '7 x 1/6 (117%)'), (8, 6, '8 x 1/6 (133%)'), (9, 6, '9 x 1/6 (150%)'), (10, 6, '10 x 1/6 (167%)'), (8, 8, '8 x 1/8 (100%)'), (10, 10, '10 x 1/10 (100%)')):
    z = z0.copy(); z['cap'] = np.where(bad, cap, 6); z['wn'] = np.where(bad, wn, 6); show2(f'downtrend: {lab}', z)
z = z0.copy(); z['w'] = np.where(bad, 1.5, 1.0); show2('downtrend: 6 x 1.5/6 (bigger size, same basket)', z)
z = z0.copy(); z['cap'] = np.where(bad, 8, 6); z['w'] = np.where(bad, 1.5, 1.0); show2('downtrend: both (8 slots, 1.5x)', z)
b = L[L.inbook]; print('extra-slot candidates: downtrend signals NOT taken by the 6-slot book, avg', f"{L[(~L.inbook) & bad].ret.mean():+.2%}", 'vs taken downtrend', f"{b[bad[L.inbook.values]].ret.mean():+.2%}")
