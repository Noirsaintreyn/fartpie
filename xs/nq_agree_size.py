"""Nov 5 2026: LEVERAGE UP when the overnight (Globex) vector agrees? DECLARED BEFORE RESULTS. Only Nov 2024 - Oct 2026 has overnight bars (yfinance NQ=F 1h Globex vector),
so this is a 2-year prop test: evals started every session Nov 18 2024 - Oct 31 2025, followed to Oct 2026; 12-month runs from each month Dec 2024 - Oct 2025. Lucid 50K.
  BASE | UP15 x1.5 when the 1h Globex vector agrees with the flip | UP125 x1.25 | SPLIT x1.5 agree / x0.7 disagree (about the same average size as BASE)"""
_cg = open('xs/nq_context_guards.py').read(); exec(_cg[:_cg.rindex("if __name__ == '__main__':")])
N1 = pd.read_parquet('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/yf1h/NQ_F.parquet'); N1 = N1[N1.index.dayofweek < 5].reset_index(); N1.columns = ['dt'] + list(N1.columns[1:])
N1['reg'] = np.asarray(vec(N1[['open', 'high', 'low', 'close']])); N1['end'] = N1.dt + pd.Timedelta('1h')
te = np.asarray(pd.DatetimeIndex(t5[XA.j.values]), dtype='datetime64[ns]'); ii = np.searchsorted(N1.end.values, te, side='right') - 1
covered = te > np.datetime64(N1.dt.min() + pd.Timedelta(days=5)); st1 = np.where(ii >= 0, N1.reg.values[np.clip(ii, 0, None)], 0)
AGR = pd.Series(np.where(covered, np.where(st1 == XA.dir.values, 'agree', 'disagree'), 'n/a'), index=XA.index)
print('Rule A trades by 1h Globex agreement:', AGR.loc[nets.index].value_counts().to_dict(), flush=True)
VAR = {}
for nm, (ua, ud) in {'BASE': (1.0, 1.0), 'UP15': (1.5, 1.0), 'UP125': (1.25, 1.0), 'SPLIT': (1.5, .7)}.items():
    fm = pd.Series(np.select([AGR == 'agree', AGR == 'disagree'], [ua, ud], 1.0), index=XA.index)
    DS[f'e_A{nm}'] = to_D(ev_res, eqm(.3) * fm); DS[f'f_A{nm}'] = to_D(full, eqm(.7) * fm); VAR[nm] = (f'e_A{nm}', f'f_A{nm}', f'f_A{nm}')
def job(nm):
    v = VAR[nm]; f = LUC; s0 = int(np.searchsorted(D_, pd.Timestamp('2024-11-18'))); s1 = int(np.searchsorted(D_, pd.Timestamp('2025-11-01'))); A = []
    for s in range(s0, s1):
        res, k, fees = evS(f, s, NS, DS[v[0]]); r = dict(res=res, fees=fees + (f['act'] if res == 'pass' else 0), npay=0, got=0.0, fres=None)
        if res == 'pass': fr, k2, pays = fuS(f, k + 1, NS, DS[v[1]], DS[v[2]]); r.update(fres=fr, npay=len(pays), got=sum(a for _, a in pays))
        A.append(r)
    A = pd.DataFrame(A); Fd = A[A.res == 'pass']
    K = pd.DataFrame([carS(f, int(np.searchsorted(D_, pd.Timestamp(f'{y}-{m:02d}-01'))), min(int(np.searchsorted(D_, pd.Timestamp(f'{y}-{m:02d}-01'))) + 252, NS), v)
                      for y, m in [(2024, 12)] + [(2025, m) for m in range(1, 11)]])
    return dict(nm=nm, passed=(A.res == 'pass').mean(), paid_f=(Fd.npay > 0).mean(), lost_f=(Fd.fres == 'lost').mean(), paid_e=(A.npay > 0).mean(), net=(A.got - A.fees).mean(),
                k_med=K.net.median(), k_min=K.net.min(), k_pos=(K.net > 0).mean())
if __name__ == '__main__':
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job, list(VAR))
    print('\nLUCID 50K, evals started Nov 2024 - Oct 2025 (followed to Oct 2026)', flush=True)
    for r in RES:
        print(f'  {r["nm"]:<6} | eval pass {r["passed"]:4.0%} | funded paid {r["paid_f"]:4.0%} blown {r["lost_f"]:4.0%} | evals -> payout {r["paid_e"]:4.0%} | net/eval {r["net"]:+6,.0f} | 12m runs median {r["k_med"]:+6,.0f} worst {r["k_min"]:+6,.0f} positive {r["k_pos"]:4.0%}', flush=True)
