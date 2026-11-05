// L'Argent Parle desktop widget for Übersicht (https://tracesof.net/uebersicht/). Shows the live screener book from the site:
// open positions (P&L, peak, distance to stop), new signals, and the top-ranked names by options ratio. Refreshes every minute.
import { css } from "uebersicht";

export const command = "bash lap.widget/fetch.sh";
export const refreshFrequency = 60 * 1000;

const C = { bg: "rgba(5,8,15,0.88)", panel: "#0a1322", t1: "#dfe7f1", t2: "#b8c4d4", mut: "#7d8ca0", grid: "#1a2638", blue: "#3d93ff", good: "#4da3ff", crit: "#ff4d5e" };
export const className = `
  left: 24px; bottom: 24px; width: 340px; font-family: -apple-system, "SF Pro Text", Helvetica, sans-serif; color: ${C.t1};
  background: ${C.bg}; border: 1px solid ${C.grid}; border-radius: 14px; padding: 14px 16px; font-size: 12px; -webkit-font-smoothing: antialiased;
`;
const row = css`display: flex; justify-content: space-between; align-items: baseline; padding: 4px 0; border-top: 1px solid ${C.grid};`;
const head = css`display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;`;
const sec = css`color: ${C.mut}; font-size: 10px; letter-spacing: .08em; text-transform: uppercase; margin: 10px 0 4px;`;
const num = css`font-variant-numeric: tabular-nums;`;
const pct = (x) => (x == null || isNaN(x) ? "–" : `${x > 0 ? "+" : ""}${Number(x).toFixed(2)}%`);
const ago = (iso) => { if (!iso) return ""; const m = Math.round((Date.now() - new Date(iso).getTime()) / 60000); return m < 60 ? `${m}m ago` : m < 1440 ? `${Math.round(m / 60)}h ago` : `${Math.round(m / 1440)}d ago`; };

export const render = ({ output, error }) => {
  let d = null; try { d = JSON.parse(output); } catch (e) {}
  if (error || !d || d.error || !d.snapshot) return <div><b>L'Argent Parle</b><div style={{ color: C.mut, marginTop: 6 }}>{(d && d.error) || "waiting for the site…"}</div></div>;
  const s = d.snapshot, pos = s.positions || [], sig = s.signals || [], watch = (s.watch || []).slice(0, 6);
  const tot = pos.reduce((a, p) => a + (Number(p.pnl) || 0), 0) / Math.max(s.slots || pos.length || 1, 1);
  return (
    <div>
      <div className={head}>
        <b style={{ fontSize: 13 }}>L'Argent Parle</b>
        <span style={{ color: C.mut }}>
          <span style={{ color: s.market_open ? C.good : C.mut }}>●</span> {s.market_open ? "open" : "closed"} · {ago(s.asof)}
        </span>
      </div>
      <div className={head} style={{ marginBottom: 2 }}>
        <span style={{ color: C.t2 }}>{pos.length}/{s.slots || "–"} slots</span>
        <span className={num} style={{ color: tot >= 0 ? C.good : C.crit, fontWeight: 600 }}>book {pct(tot)}</span>
      </div>
      <div className={sec}>Positions</div>
      {pos.length === 0 && <div style={{ color: C.mut }}>flat</div>}
      {pos.map((p) => {
        const last = Number(p.last), stop = Number(p.stop), room = last && stop ? ((last - stop) / last) * 100 : null;
        return (
          <div className={row} key={p.name}>
            <span style={{ width: 70, fontWeight: 600, color: p.selling ? C.crit : C.t1 }}>{p.name}{p.selling ? " ↓" : ""}</span>
            <span className={num} style={{ width: 70, textAlign: "right", color: p.pnl >= 0 ? C.good : C.crit }}>{pct(p.pnl)}</span>
            <span className={num} style={{ width: 80, textAlign: "right", color: C.mut }}>peak {pct(p.peak)}</span>
            <span className={num} style={{ width: 80, textAlign: "right", color: C.mut }}>stop {room == null ? "–" : `-${room.toFixed(1)}%`}</span>
          </div>
        );
      })}
      {sig.length > 0 && <div className={sec}>New signals</div>}
      {sig.map((x, i) => (
        <div className={row} key={i}><span style={{ fontWeight: 600 }}>{x.name || x.tk}</span><span style={{ color: (x.side || x.kind || "").toString().toUpperCase().startsWith("S") ? C.crit : C.good }}>{x.side || x.kind || ""} {x.price || x.entry || ""}</span></div>
      ))}
      <div className={sec}>Top ranked (options ratio)</div>
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
