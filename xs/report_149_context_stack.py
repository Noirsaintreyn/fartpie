"""report 149: CONTEXT STACK for ICT entries on NQ (Rey: 'ICT traders pair the signal with context; any trade we take, we say there's an FVG AND we're trending up / vector bullish / overnight inventory aligns, else skip').
DECLARED BEFORE RESULTS. Signals: IFVG, UNICORN (report 144 trades, 2R, 0.5 pt cost) and plain FVG (report 148 trades, all non-overlapping). Day context from idle_state (known by the 09:30 open): 
 ALIGNED flags with trade direction dr (+1 long): drive (f_drive_with = overnight drive in trade direction; f_drive_against), swept (swept lows & long / swept highs & short), gap (gap up & long / gap down & short), open vs prior range (open above/below), prior close (top/bot of range), nq5 trend (f_nq5_up/dn = 5d trend), tnx/dxy/vix direction is NOT used (not about NQ direction).
 Direction convention: for each directional pair (up/with, dn/against) align = ONE of: with-flow (long with up-type, short with dn-type) -> test WITH and AGAINST as separate slices (ICT says with; our earlier chain result says the sign flips by regime), plus the non-directional: f_on_wide (wide overnight), vol state (vxn_pct terciles cut on A), trend_eff10 terciles cut on A.
PASS = same sign, |t|>=2 both eras, n>=30 each. STACK: pre-declared ICT stack = (drive WITH or swept-in-direction) AND (nq5 trend WITH); also its complement. Selection-free: no thresholds fitted on B. 12 contexts x 3 signals: expect ~1 chance PASS; compare stacked vs unstacked on the same trades."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
ctx=pd.read_pickle(SC+'idle_state.pkl'); ctx.index=pd.to_datetime(ctx.index)
U=pd.read_pickle(SC+'ext/uni_ifvg.pkl'); F=pd.read_pickle(SC+'ext/fvg_all_vp.pkl'); F['model']='FVG'
sig={'IFVG':U[U.model=='IFVG'][['d','dr','pts','A']],'UNICORN':U[U.model=='UNI'][['d','dr','pts','A']],'FVG':F[['d','dr','pts','A']]}
def t_(a): return a.mean()/(a.std()/np.sqrt(len(a))) if len(a)>8 else np.nan
for nm,T in sig.items():
    T=T.join(ctx,on='d'); T=T[T.trade.notna()].copy(); L=T.dr>0
    f=lambda c:T[c]==1
    T['drive_with']=f('f_drive_with'); T['drive_ag']=f('f_drive_against')
    T['swept_dir']=(L&f('f_swept_lo'))|(~L&f('f_swept_hi')); T['swept_opp']=(L&f('f_swept_hi'))|(~L&f('f_swept_lo'))
    T['gap_with']=(L&f('f_gap_up'))|(~L&f('f_gap_dn')); T['gap_ag']=(L&f('f_gap_dn'))|(~L&f('f_gap_up'))
    T['open_with']=(L&f('f_open_above'))|(~L&f('f_open_below')); T['open_ag']=(L&f('f_open_below'))|(~L&f('f_open_above'))
    T['pclose_with']=(L&f('f_pclose_top'))|(~L&f('f_pclose_bot')); T['pclose_ag']=(L&f('f_pclose_bot'))|(~L&f('f_pclose_top'))
    T['trend_with']=(L&f('f_nq5_up'))|(~L&f('f_nq5_dn')); T['trend_ag']=(L&f('f_nq5_dn'))|(~L&f('f_nq5_up')); T['on_wide']=f('f_on_wide')
    q=T[T.A].vxn_pct.quantile([1/3,2/3]).values; T['vol_lo']=T.vxn_pct<=q[0]; T['vol_hi']=T.vxn_pct>q[1]
    q=T[T.A].trend_eff10.quantile([1/3,2/3]).values; T['eff_hi']=T.trend_eff10>q[1]; T['eff_lo']=T.trend_eff10<=q[0]
    T['STACK']=(T.drive_with|T.swept_dir)&T.trend_with; T['NOT_STACK']=~T.STACK
    T['STACK3']=T.drive_with&T.trend_with&T.open_with
    def line(lab,m):
        a,b=T[m&T.A].pts,T[m&~T.A].pts; ps=len(a)>=30 and len(b)>=30 and np.sign(a.mean())==np.sign(b.mean()) and abs(t_(a))>=2 and abs(t_(b))>=2
        return f'  {lab:14s} A n{len(a):4d} {a.mean():+6.2f} win{100*(a>0).mean():3.0f}% t{t_(a):+5.1f} | B n{len(b):4d} {b.mean():+6.2f} win{100*(b>0).mean():3.0f}% t{t_(b):+5.1f}{"  <== PASS" if ps else ""}'
    print(f'\n===== {nm}: {len(T)} trades'); print(line('ALL',T.pts.notna()))
    for c in ('drive_with','drive_ag','swept_dir','swept_opp','gap_with','gap_ag','open_with','open_ag','pclose_with','pclose_ag','trend_with','trend_ag','on_wide','vol_lo','vol_hi','eff_hi','eff_lo','STACK','NOT_STACK','STACK3'): print(line(c,T[c]))
