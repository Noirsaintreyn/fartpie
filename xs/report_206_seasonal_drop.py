"""Report 206: does dropping a calendar month (December) or the last 3 months help? Rey heard it; the message was cut off, so tested both readings. DECLARED: H_DEC: December entries are worse than the rest (sign - in train 2019-21 AND test 2022+); H_Q4: Oct-Dec entries worse; H_EARN: earnings-season months (Jan, Apr, Jul, Oct) beat the other months (supply and flip returns). Evidence is by-calendar-month (year-paired where possible), not a pick of the best month. (1) sub-$5 flips (report 203 table, entry month of the signal day): mean, ex-best-3, median, BIG rate by month, train | test. (2) large-cap book (report 196 sim: 6 slots, 8% breaker, 2bp) CAGR / maxDD with Dec, Q4, or each month removed from NEW entries. (3) 2019-2026 doubler supply by calendar month (names doubling within 60d)."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
Z=pd.read_pickle(X+'selectors203.pkl'); Z['mo']=Z.d.dt.month; Z=Z[Z.era!='late26']
print('(1) sub-$5 flips by entry month (train 2019-21 | test 2022-Feb26): n, mean%, median%, BIG%')
for m in range(1,13):
    c=[]
    for e in ('train','test'):
        s=Z[(Z.era==e)&(Z.mo==m)]; c.append(f'n{len(s):4d} {s.r.mean()*100:6.1f} {s.r.median()*100:6.1f} {s.big.mean()*100:4.0f}%' if len(s)>=5 else 'n<5')
    print(f'  {m:2d}  '+'   |   '.join(c))
for nm,ms in (('DEC',[12]),('Q4 Oct-Dec',[10,11,12]),('earnings months Jan/Apr/Jul/Oct',[1,4,7,10])):
    c=[]
    for e in ('train','test'):
        s=Z[Z.era==e]; a=s[s.mo.isin(ms)]; b=s[~s.mo.isin(ms)]; r=a.r.sort_values(); c.append(f'{nm}: n{len(a)} mean {a.r.mean()*100:5.1f}% x3 {r.iloc[:-3].mean()*100:5.1f}% med {a.r.median()*100:5.1f}% BIG {a.big.mean()*100:4.1f}% vs rest mean {b.r.mean()*100:5.1f}% med {b.r.median()*100:5.1f}% BIG {b.big.mean()*100:4.1f}%')
    print('  '+'\n  '.join(c))
print('\n(2) large-cap book (hm_fill clone, one-per-name off), exclusions on NEW entries: CAGR / maxDD  2019-21 | 2022-26 | 2019-26 | 2026 YTD')
sys.argv=['x','/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet']; src=open('report_196_one_slot_per_name.py').read(); src=src[:src.index("P = [(np.datetime64")]; ns={}; exec(src,ns)
T0=ns['T'].copy(); P=[(np.datetime64(a),np.datetime64(b)) for a,b in (('2019-01-01','2022-01-01'),('2022-01-01','2027-01-01'),('2019-01-01','2027-01-01'),('2026-01-01','2027-01-01'))]
def run(nm,drop):
    ns['T']=T0[~T0.entry.dt.month.isin(drop)].reset_index(drop=True); res=[ns['sim'](a,b) for a,b in P]; print(f'  {nm:22s}'+''.join(f'{c:>+9.1%} /{d:>5.0%}' for c,d,_ in res),flush=True)
run('baseline (all months)',[]); run('drop December',[12]); run('drop Oct-Dec',[10,11,12]); run('drop Dec + Jan',[12,1])
for m in range(1,13): run(f'drop month {m}',[m])
