"""report 145: IFVG (the one ICT entry that passed both eras, report 144) aligned with prior-day volume-profile LVN / TPO single prints / shelves / gamma walls. DECLARED BEFORE RESULTS.
Rey: 'fvg aligned with tpo single print or vp lvn will prob work'. Mechanism: single prints / LVNs are fast-travel zones (little prior acceptance) -> an inverted gap sitting at one acts as a fast reaction level;
shelves (high-volume node edges, VAL/POC/VAH) = acceptance, price sticks -> weaker. Alignment (same rule as xs/ict.py): IFVG zone [lo,hi] within TOL ADR (default 0.03) of a prior-day level (levels from the prior session's profile, so known at the open).
Slices: LVN, SP, LVN|SP ('fast travel'), SHELF, WALL, none. PASS = same sign, |t|>=2 both eras, n>=30; also aligned vs non-aligned difference. Same trades as report 144 (2R, 0.5 pt cost)."""
import numpy as np, pandas as pd, warnings, sys, os; TOL=float(os.getenv('TOL','0.03')); warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/ict.py').read(); exec(_s[:_s.index('def zones_and_trades')])
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
T=pd.read_pickle(SC+'ext/uni_ifvg.pkl'); T=T[T.model.isin(['IFVG','UNI'])].copy()
adr={}
for k,d in enumerate(days):
    if k>=14: adr[d]=np.mean([np.nanmax(h[gi[days[q]]])-np.nanmin(l[gi[days[q]]]) for q in range(k-14,k)])
for nm in ('LVN','SP','SHELF','WALL'):
    T[nm]=[ (d in LEV and d in adr and any((lv>=z0-TOL*adr[d]) and (lv<=z1+TOL*adr[d]) for lv in np.atleast_1d(LEV[d][nm]).astype(float))) for d,z0,z1 in zip(T.d,T.zlo,T.zhi)]
T['FAST']=T.LVN|T.SP; T['NONE']=~(T.LVN|T.SP|T.SHELF|T.WALL); T.to_pickle(SC+'ext/uni_ifvg_vp.pkl')
def t_(a): return a.mean()/(a.std()/np.sqrt(len(a))) if len(a)>8 else np.nan
def line(lab,m):
    a,b_=T[m&T.A].pts,T[m&~T.A].pts; ps=len(a)>=30 and len(b_)>=30 and np.sign(a.mean())==np.sign(b_.mean()) and abs(t_(a))>=2 and abs(t_(b_))>=2
    return f'  {lab:28s} A n{len(a):4d} {a.mean():+6.2f} win{100*(a>0).mean():3.0f}% t{t_(a):+5.1f} | B n{len(b_):4d} {b_.mean():+6.2f} win{100*(b_>0).mean():3.0f}% t{t_(b_):+5.1f}{"  <== PASS" if ps else ""}'
for mdl in ('IFVG','UNI'):
    M=T.model==mdl; print(f'\n== {mdl} ({int(M.sum())} trades)'); print(line('all',M))
    for c in ('LVN','SP','FAST','SHELF','WALL','NONE'): print(line(c+' aligned',M&T[c])); print(line(c+' NOT',M&~T[c]))
