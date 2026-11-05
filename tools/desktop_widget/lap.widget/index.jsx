// L'Argent Parle desktop widget for Übersicht (https://tracesof.net/uebersicht/). Refreshes every minute.
// Top: the live screener book from the site (positions with P&L / peak / distance to stop, new signals, top-ranked names by options ratio).
// Bottom: ADD ACCOUNT light - the account-start signals known at the prior close (tools/account_signals.py; backtest xs/nq_accounts3.py).
import { css } from "uebersicht";

export const command = "bash lap.widget/fetch.sh";
export const refreshFrequency = 60 * 1000;

const C = { bg: "rgba(5,8,15,0.88)", panel: "#0a1322", t1: "#dfe7f1", t2: "#b8c4d4", mut: "#7d8ca0", grid: "#1a2638", blue: "#3d93ff", good: "#4da3ff", crit: "#ff4d5e", go: "#34d399" };
export const className = `
  left: 24px; bottom: 24px; width: 360px; font-family: -apple-system, "SF Pro Text", Helvetica, sans-serif; color: ${C.t1};
  background: ${C.bg}; border: 1px solid ${C.grid}; border-radius: 14px; padding: 14px 16px; font-size: 12px; -webkit-font-smoothing: antialiased;
`;
const row = css`display: flex; justify-content: space-between; align-items: baseline; padding: 4px 0; border-top: 1px solid ${C.grid};`;
const head = css`display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;`;
const sec = css`display: flex; justify-content: space-between; color: ${C.mut}; font-size: 10px; letter-spacing: .08em; text-transform: uppercase; margin: 12px 0 4px;`;
const num = css`font-variant-numeric: tabular-nums;`;
const pct = (x) => (x == null || isNaN(x) ? "–" : `${x > 0 ? "+" : ""}${Number(x).toFixed(2)}%`);
const ago = (iso) => { if (!iso) return ""; const m = Math.round((Date.now() - new Date(iso).getTime()) / 60000); return m < 60 ? `${m}m ago` : m < 1440 ? `${Math.round(m / 60)}h ago` : `${Math.round(m / 1440)}d ago`; };
const fmtDay = (d) => { if (!d) return ""; const t = new Date(`${d}T12:00:00`); return t.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" }); };

const Book = ({ site }) => {
  if (!site || site.error || !site.snapshot) return <div style={{ color: C.mut }}>{(site && site.error) || "waiting for the site…"}</div>;
  const s = site.snapshot, pos = s.positions || [], sig = s.signals || [], watch = (s.watch || []).slice(0, 6);
  const tot = pos.reduce((a, p) => a + (Number(p.pnl) || 0), 0) / Math.max(s.slots || pos.length || 1, 1);
  return (
    <div>
      <div className={head} style={{ marginBottom: 2 }}>
        <span style={{ color: C.t2 }}>
          <span style={{ color: s.market_open ? C.good : C.mut }}>●</span> {s.market_open ? "open" : "closed"} · {ago(s.asof)} · {pos.length}/{s.slots || "–"} slots
        </span>
        <span className={num} style={{ color: tot >= 0 ? C.good : C.crit, fontWeight: 600 }}>book {pct(tot)}</span>
      </div>
      <div className={sec}><span>Positions</span></div>
      {pos.length === 0 && <div style={{ color: C.mut }}>flat</div>}
      {pos.map((p) => {
        const last = Number(p.last), stop = Number(p.stop), room = last && stop ? ((last - stop) / last) * 100 : null;
        return (
          <div className={row} key={p.name}>
            <span style={{ width: 70, fontWeight: 600, color: p.selling ? C.crit : C.t1 }}>{p.name}{p.selling ? " ↓" : ""}</span>
            <span className={num} style={{ width: 70, textAlign: "right", color: p.pnl >= 0 ? C.good : C.crit }}>{pct(p.pnl)}</span>
            <span className={num} style={{ width: 84, textAlign: "right", color: C.mut }}>peak {pct(p.peak)}</span>
            <span className={num} style={{ width: 84, textAlign: "right", color: C.mut }}>stop {room == null ? "–" : `-${room.toFixed(1)}%`}</span>
          </div>
        );
      })}
      {sig.length > 0 && <div className={sec}><span>New signals</span></div>}
      {sig.map((x, i) => (
        <div className={row} key={i}><span style={{ fontWeight: 600 }}>{x.name || x.tk}</span><span style={{ color: (x.side || x.kind || "").toString().toUpperCase().startsWith("S") ? C.crit : C.good }}>{x.side || x.kind || ""} {x.price || x.entry || ""}</span></div>
      ))}
      <div className={sec}><span>Top ranked (options ratio)</span></div>
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

const Accounts = ({ acct }) => {
  if (!acct || !acct.signals) return <div><div className={sec}><span>Add account?</span></div><div style={{ color: C.mut }}>signals unavailable (check ~/fartpie/.venv311 and yfinance)</div></div>;
  return (
    <div>
      <div className={sec}><span>Add account?</span><span>for {fmtDay(acct.for_session)}</span></div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "2px 0 4px" }}>
        <span style={{ width: 10, height: 10, borderRadius: 5, background: acct.green ? C.go : C.mut, boxShadow: acct.green ? `0 0 8px ${C.go}` : "none" }} />
        <b style={{ color: acct.green ? C.go : C.t2 }}>{acct.green ? `GREEN · ${acct.on.length} of 4 on` : "no signal · evals still +EV"}</b>
      </div>
      {acct.signals.map((s) => (
        <div className={row} key={s.key}>
          <span style={{ width: 62, fontWeight: 600, color: s.on ? C.go : s.stale ? C.crit : C.mut }}>{s.on ? "●" : "○"} {s.key}</span>
          <span className={num} style={{ flex: 1, color: C.t2, fontSize: 11 }}>{s.value}</span>
          <span className={num} style={{ width: 70, textAlign: "right", color: C.mut, fontSize: 11 }}>{Math.round(s.payout_on * 100)}% / {Math.round(s.payout_off * 100)}%</span>
        </div>
      ))}
      <div style={{ color: C.mut, fontSize: 10, marginTop: 4 }}>right column: evals that paid out when the signal was on / off (2022-25)</div>
    </div>
  );
};

export const render = ({ output, error }) => {
  let d = null; try { d = JSON.parse(output); } catch (e) {}
  const site = d ? (d.site !== undefined ? d.site : d) : null, acct = d ? d.acct : null;
  return (
    <div>
      <div className={head}><b style={{ fontSize: 13 }}>L'Argent Parle</b>{error && <span style={{ color: C.crit }}>error</span>}</div>
      <Book site={site} />
      <Accounts acct={acct} />
    </div>
  );
};
