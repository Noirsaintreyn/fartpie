// L'Argent Parle - PROP ACCOUNTS widget for Übersicht. The ADD ACCOUNT light: account-start signals known at the prior close
// (tools/account_signals.py; backtest xs/nq_accounts3.py / xs/nq_full_system.py). Refreshes every 5 minutes (the script caches for 30).
import { css } from "uebersicht";

export const command = "bash lap-prop.widget/fetch.sh";
export const refreshFrequency = 5 * 60 * 1000;

const C = { bg: "rgba(5,8,15,0.88)", t1: "#dfe7f1", t2: "#b8c4d4", mut: "#7d8ca0", grid: "#1a2638", crit: "#ff4d5e", go: "#34d399" };
export const className = `
  left: 456px; bottom: 24px; width: 330px; font-family: -apple-system, "SF Pro Text", Helvetica, sans-serif; color: ${C.t1};
  background: ${C.bg}; border: 1px solid ${C.grid}; border-radius: 14px; padding: 14px 16px; font-size: 12px; -webkit-font-smoothing: antialiased;
`;
const row = css`display: flex; justify-content: space-between; align-items: baseline; padding: 4px 0; border-top: 1px solid ${C.grid};`;
const num = css`font-variant-numeric: tabular-nums;`;
const fmtDay = (d) => (d ? new Date(`${d}T12:00:00`).toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" }) : "");

export const render = ({ output }) => {
  let a = null; try { a = JSON.parse(output); } catch (e) {}
  if (!a || a.error || !a.signals) return <div><b style={{ fontSize: 13 }}>Prop accounts</b><div style={{ color: C.mut, marginTop: 6 }}>{(a && a.error) || "loading signals…"}</div></div>;
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <b style={{ fontSize: 13 }}>Add an account?</b><span style={{ color: C.mut }}>for {fmtDay(a.for_session)}</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "8px 0 6px" }}>
        <span style={{ width: 12, height: 12, borderRadius: 6, background: a.green ? C.go : C.mut, boxShadow: a.green ? `0 0 10px ${C.go}` : "none" }} />
        <b style={{ fontSize: 15, color: a.green ? C.go : C.t2 }}>{a.green ? `GREEN · ${a.on.length} of 4 on` : "no signal"}</b>
      </div>
      {a.signals.map((s) => (
        <div className={row} key={s.key}>
          <span style={{ width: 64, fontWeight: 600, color: s.on ? C.go : s.stale ? C.crit : C.mut }}>{s.on ? "●" : "○"} {s.key}</span>
          <span className={num} style={{ flex: 1, color: C.t2, fontSize: 11 }}>{s.value}</span>
          <span className={num} style={{ width: 64, textAlign: "right", color: C.mut, fontSize: 11 }}>{Math.round(s.payout_on * 100)}% / {Math.round(s.payout_off * 100)}%</span>
        </div>
      ))}
      <div style={{ color: C.mut, fontSize: 10, marginTop: 6, lineHeight: 1.4 }}>
        right: evals that paid out when on / off (2022-25). Prior close {a.asof}. Off still averaged +$2,647 per eval - green is timing, not permission.
      </div>
    </div>
  );
};
