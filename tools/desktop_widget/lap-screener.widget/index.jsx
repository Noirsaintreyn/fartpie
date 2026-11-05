// L'Argent Parle - LIVE VECTOR SCREENER widget for Übersicht (https://tracesof.net/uebersicht/). Data: the site's snapshot (the same one the website shows), every minute.
// Shows: book since the live start vs QQQ / SPY with a sparkline, open positions (P&L, peak, distance to stop, entry day), today's BUY / SELL signals,
// the last closed trades, and the eligible names ranked by options ratio (blue border = held).
import { css } from "uebersicht";

export const command = "bash lap-screener.widget/fetch.sh";
export const refreshFrequency = 60 * 1000;

const C = { bg: "rgba(5,8,15,0.88)", panel: "#0a1322", t1: "#dfe7f1", t2: "#b8c4d4", mut: "#7d8ca0", grid: "#1a2638", blue: "#3d93ff", good: "#4da3ff", crit: "#ff4d5e", warn: "#f5b041" };
export const className = `
  left: 24px; bottom: 24px; width: 380px; font-family: -apple-system, "SF Pro Text", Helvetica, sans-serif; color: ${C.t1};
  background: ${C.bg}; border: 1px solid ${C.grid}; border-radius: 14px; padding: 14px 16px; font-size: 12px; -webkit-font-smoothing: antialiased;
`;
const row = css`display: flex; justify-content: space-between; align-items: baseline; padding: 4px 0; border-top: 1px solid ${C.grid};`;
const head = css`display: flex; justify-content: space-between; align-items: center;`;
const sec = css`display: flex; justify-content: space-between; color: ${C.mut}; font-size: 10px; letter-spacing: .08em; text-transform: uppercase; margin: 12px 0 4px;`;
const num = css`font-variant-numeric: tabular-nums;`;
const pct = (x, d = 2) => (x == null || isNaN(x) ? "–" : `${x > 0 ? "+" : ""}${Number(x).toFixed(d)}%`);
const col = (x) => (x == null || isNaN(x) ? C.mut : x >= 0 ? C.good : C.crit);
const mins = (iso) => (iso ? Math.round((Date.now() - new Date(iso).getTime()) / 60000) : null);
const ago = (m) => (m == null ? "" : m < 60 ? `${m}m ago` : m < 1440 ? `${Math.round(m / 60)}h ago` : `${Math.round(m / 1440)}d ago`);

const Spark = ({ curve }) => {
  if (!curve || !curve.t || curve.t.length < 2) return null;
  const W = 380, H = 44, xs = curve.t, n = xs.length;
  const vals = [...(curve.book || []), ...(curve.qqq || [])].filter((v) => v != null);
  if (!vals.length) return null;
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals), span = hi - lo || 1;
  const X = (i) => (i / (n - 1)) * W, Y = (v) => H - ((v - lo) / span) * H;
  const path = (arr) => arr.map((v, i) => (v == null ? null : `${X(i).toFixed(1)},${Y(v).toFixed(1)}`)).filter(Boolean).join(" ");
  return (
    <svg width={W} height={H} style={{ display: "block", margin: "6px 0 2px" }}>
      <line x1="0" x2={W} y1={Y(0)} y2={Y(0)} stroke={C.grid} strokeWidth="1" />
      <polyline points={path(curve.qqq || [])} fill="none" stroke={C.mut} strokeWidth="1" strokeDasharray="3 3" />
      <polyline points={path(curve.book || [])} fill="none" stroke={C.blue} strokeWidth="1.6" />
    </svg>
  );
};

export const render = ({ output, error }) => {
  let d = null; try { d = JSON.parse(output); } catch (e) {}
  if (error || !d || d.error || !d.snapshot)
    return <div><b style={{ fontSize: 13 }}>Vector screener</b><div style={{ color: C.mut, marginTop: 6 }}>{(d && d.error) || "waiting for the site…"}</div></div>;
  const s = d.snapshot, h = s.history || {}, pos = s.positions || [], sig = s.signals || [], watch = (s.watch || []).slice(0, 8);
  const trades = h.trades || [], since = {}; trades.filter((t) => t.open).forEach((t) => { since[t.name] = (t.entry || "").split("  ")[0]; });
  const closed = trades.filter((t) => !t.open).slice(0, 3), age = mins(s.asof), stale = s.market_open && age != null && age > 30;
  return (
    <div>
      <div className={head}>
        <b style={{ fontSize: 13 }}>Vector screener</b>
        <span style={{ color: stale ? C.warn : C.mut }}>
          <span style={{ color: s.market_open ? C.good : C.mut }}>●</span> {s.market_open ? "open" : "closed"} · {ago(age)}{stale ? " · stale" : ""}
        </span>
      </div>
      <div className={head} style={{ marginTop: 8 }}>
        <span>
          <span className={num} style={{ fontSize: 20, fontWeight: 600, color: col(h.book) }}>{pct(h.book)}</span>
          <span style={{ color: C.mut }}> since {h.since || "start"}</span>
        </span>
        <span className={num} style={{ color: C.mut, textAlign: "right" }}>QQQ {pct(h.qqq)} · SPY {pct(h.spy)}</span>
      </div>
      <Spark curve={h.curve} />
      <div className={num} style={{ color: C.mut, fontSize: 11 }}>realized {pct(h.realized)} · open {pct(h.open)} · {pos.length}/{s.slots || "–"} slots · ratio ≥ {s.cutoff}</div>

      <div className={sec}><span>Positions</span><span>P&L · peak · to stop</span></div>
      {pos.length === 0 && <div style={{ color: C.mut }}>flat</div>}
      {pos.map((p) => {
        const last = Number(p.last), stop = Number(p.stop), room = last && stop ? ((last - stop) / last) * 100 : null;
        return (
          <div className={row} key={p.name}>
            <span style={{ width: 120 }}>
              <b style={{ color: p.selling ? C.crit : C.t1 }}>{p.name}{p.selling ? " ↓ SELL" : ""}</b>
              <span style={{ color: C.mut, fontSize: 10 }}> {since[p.name] || ""}</span>
            </span>
            <span className={num} style={{ width: 62, textAlign: "right", color: col(p.pnl), fontWeight: 600 }}>{pct(p.pnl)}</span>
            <span className={num} style={{ width: 62, textAlign: "right", color: C.mut }}>{pct(p.peak)}</span>
            <span className={num} style={{ width: 62, textAlign: "right", color: room != null && room < 1 ? C.warn : C.mut }}>{room == null ? "–" : `-${room.toFixed(1)}%`}</span>
          </div>
        );
      })}

      {sig.length > 0 && <div className={sec}><span>Today's signals</span></div>}
      {sig.slice(0, 6).map((x, i) => (
        <div className={row} key={i}>
          <span className={num} style={{ width: 44, color: C.mut }}>{x.time}</span>
          <span style={{ width: 48, fontWeight: 600, color: x.action === "SELL" ? C.crit : C.good }}>{x.action}</span>
          <span style={{ flex: 1, fontWeight: 600 }}>{x.ticker}{x.grade ? <span style={{ color: C.mut, fontWeight: 400 }}> · {x.grade}</span> : null}</span>
          <span className={num} style={{ color: C.t2 }}>{x.price}{x.pnl != null ? <span style={{ color: col(x.pnl) }}> {pct(x.pnl)}</span> : null}</span>
        </div>
      ))}

      {closed.length > 0 && <div className={sec}><span>Last closed</span></div>}
      {closed.map((t, i) => (
        <div className={row} key={i}>
          <span style={{ width: 70, fontWeight: 600 }}>{t.name}{t.lock ? <span style={{ color: C.mut, fontWeight: 400 }}> lock</span> : null}</span>
          <span style={{ flex: 1, color: C.mut, fontSize: 11 }}>{(t.exit || "").split("  ")[0]}</span>
          <span className={num} style={{ color: col(t.ret), fontWeight: 600 }}>{pct(t.ret)}</span>
        </div>
      ))}

      <div className={sec}><span>Eligible, ranked by options ratio</span></div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {watch.map((w) => (
          <span key={w.tk} className={num} style={{ background: C.panel, border: `1px solid ${w.held ? C.blue : C.grid}`, borderRadius: 6, padding: "2px 6px", color: w.held ? C.t1 : C.t2 }}>
            {w.tk} {Number(w.ratio).toFixed(1)}
          </span>
        ))}
      </div>
    </div>
  );
};
