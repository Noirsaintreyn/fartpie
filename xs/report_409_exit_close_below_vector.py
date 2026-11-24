"""Nov 23 2026 report 409. Rey: 'don't hold either: a volatility exit (NHP + volume) or just a close below the vector'.
THIS RUN = EXIT B ONLY (the unambiguous one): exit at the OPEN of the first 15m bar after a CLOSE below the vector level (the ratcheting stop line, no break-ATR buffer, no confirm bars, no 1H-sticky, no giveback rule).
Same 12,360 live-book candidate flips and entries; compare with the LIVE exit on identical entries: trade win rate, median, mean, mean hold (days), and the 6-slot live book (true daily MTM, report 402 score()).
Declared pass: Calmar and MDD better in both halves with CAGR not lower by > 3 pts, or win rate and median trade higher in both halves."""
import sys, glob, numpy as np, multiprocessing as mp
_t = open('xs/report_402_five_ideas.py').read(); exec(_t[:_t.index("rows = []; TK0, s0")])
import report_295_flip_quality as q
def work(tk):
    f = sorted(glob.glob(UP + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))[-1]; H = q.build(f); d = H.dt.values; o = H.open.values; c = H.close.values; lv = H.lvl.values; out = []
    for e_t in B0[B0.tk == tk].entry.values:
        e = int(np.searchsorted(d, np.datetime64(e_t)))
        if e >= len(d) - 2: continue
        j = None
        for k in range(e, len(d) - 1):
            if k > e - 1 and c[k] < lv[k] and k >= e: j = k + 1; break
        if j is None: continue
        out.append((tk, pd.Timestamp(e_t), pd.Timestamp(d[j]), o[j] / o[e] - 1 - 2e-4))
    return out
if __name__ == '__main__':
    with mp.Pool(2) as P: R = P.map(work, sorted(B0.tk.unique()))
    A = pd.DataFrame([x for r in R for x in r], columns=['tk', 'entry', 'exitA', 'retA']); A.to_parquet(SP + 'exitA409.parquet')
    M = B0.merge(A, on=['tk', 'entry'], how='inner'); print('flips matched', len(M), 'of', len(B0), flush=True)
    M['hold_live'] = (pd.to_datetime(M.exit) - M.entry).dt.total_seconds() / 86400; M['hold_A'] = (M.exitA - M.entry).dt.total_seconds() / 86400
    for lab, col, hc in (('LIVE exit', 'ret', 'hold_live'), ('EXIT B close<vector', 'retA', 'hold_A')):
        n = M[col] - 2e-4; tr = M.entry < '2022-01-01'; ok = M.entry >= '2019-01-01'
        print(f"{lab:22s} n {len(M)} win {(n>0).mean():5.1%} med {n.median()*100:5.2f}% mean {n.mean()*100:5.2f}% hold mean {M[hc].mean():5.2f}d med {M[hc].median():5.2f}d | train win {(n[tr&ok]>0).mean():5.1%} med {n[tr&ok].median()*100:5.2f} | test win {(n[~tr]>0).mean():5.1%} med {n[~tr].median()*100:5.2f}")
    BA = B0.merge(A, on=['tk', 'entry'], how='inner'); BA['exit'] = BA.exitA; BA['ret'] = BA.retA
    TK0, s0 = live_fn(B0); b = score('LIVE', TK0, s0); print('\n' + line(b) + f" | CAGR halves {b['ca'][0]:.1%} | {b['cb'][0]:.1%}")
    TKs, s = live_fn(BA); d = score('book: EXIT B close<vector', TKs, s); print(line(d) + f" | CAGR halves {d['ca'][0]:.1%} | {d['cb'][0]:.1%}")
