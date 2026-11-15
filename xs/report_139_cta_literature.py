"""report 139: CTA / bank-research features on all three engines. DECLARED BEFORE RESULTS.
Hypotheses (mechanism first): (H1) time-series momentum agreement across 1/3/6/12m (MAC, TS12, TTR) = trend funds are already positioned/buying, so flips inside an established trend run further (continuation), flips against it are bounces (the take-profit-vs-move distinction Rey described);
(H2) Daniel-Moskowitz: after market drawdown with high vol (PANIC) momentum-type longs crash/rebound violently -> worse flips; (H3) multi-scale energy ENR: persistent slow-scale energy = big moves; (H4) negative skew/lottery (SKEW, KURT) = already-extended tails.
Same harness as report 138 (strict split, PASS = same sign both halves |t|>=2; AUC add-on to OS)."""
import sys, numpy as np, pandas as pd; sys.path.insert(0,'/home/claude/fartpie/xs')
import feats139 as F9, report_138_structure_models as R
SP=R.SP; W=pd.read_pickle(R.X+'wide_daily_ohlcv.pkl')
C=pd.DataFrame({k:v.close[~v.index.duplicated()] for k,v in W.items()}); F9.MKT=np.log(C).diff().clip(-.5,.5).mean(axis=1)
R.dfeat=F9.dfeat; R.FE=F9.FE
R.part(sys.argv[1])
