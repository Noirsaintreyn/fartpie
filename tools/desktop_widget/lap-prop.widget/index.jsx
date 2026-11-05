// L'Argent Parle - prop accounts: is today a good day to open an evaluation? Account-start signals known at the prior close
// (tools/account_signals.py; backtest xs/nq_accounts3.py / xs/nq_full_system.py). Übersicht widget, refreshes every 5 minutes (the script caches for 30).
// MOVE IT: drag it anywhere (the spot is remembered). Double-click puts it back at the default spot.
import { css } from "uebersicht";

export const command = "bash lap-prop.widget/fetch.sh";
export const refreshFrequency = 5 * 60 * 1000;
export const className = `left: 0; top: 0;`;            // the panel positions itself (draggable), see Panel below

const C = { ink: "rgba(13,19,34,0.94)", vellum: "#ece2c9", vellum2: "#c9bfa6", pewter: "#8b8f99", gold: "#c6a465", rule: "rgba(198,164,101,0.28)", faint: "rgba(236,226,201,0.08)",
            gain: "#9cc3a0", loss: "#d4776f" };
const SERIF = `"Times New Roman", Times, serif`, SCRIPT = `"Snell Roundhand", "Apple Chancery", cursive`;
const box = css`
  position: absolute; width: 336px; color: ${C.vellum}; font-family: ${SERIF}; font-size: 13px; line-height: 1.35; cursor: grab; user-select: none;
  background: ${C.ink}; border: 1px solid ${C.rule}; border-radius: 6px; padding: 0 20px 16px; overflow: hidden;
  box-shadow: 0 18px 50px rgba(0,0,0,0.45); -webkit-font-smoothing: antialiased;
  &:active { cursor: grabbing; }
`;
const row = css`display: flex; align-items: flex-start; gap: 10px; padding: 7px 0; border-top: 1px solid ${C.faint};`;
const it = css`font-style: italic;`;
const PLAIN = { "VOL-HI": "Volatility in its top third", VXN: "VXN above its yearly median", COLD: "System in a cold streak", DIP: "QQQ stretched below average" };
const day = (d, opt) => (d ? new Date(`${d}T12:00:00`).toLocaleDateString("en-GB", opt) : "");

// ---- dragging: the panel's spot lives in localStorage, so it survives refreshes and restarts
const KEY = "lap-prop-pos", DEF = { x: 640, y: 64 };
const getPos = () => { try { const p = JSON.parse(window.localStorage.getItem(KEY)); return p && isFinite(p.x) && isFinite(p.y) ? p : DEF; } catch (e) { return DEF; } };
const startDrag = (e) => {
  if (e.button !== 0) return;
  const el = e.currentTarget, p = getPos(), dx = e.clientX - p.x, dy = e.clientY - p.y;
  const at = (ev) => ({ x: Math.max(0, ev.clientX - dx), y: Math.max(0, ev.clientY - dy) });
  const move = (ev) => { const q = at(ev); el.style.left = `${q.x}px`; el.style.top = `${q.y}px`; };
  const up = (ev) => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); try { window.localStorage.setItem(KEY, JSON.stringify(at(ev))); } catch (err) {} };
  window.addEventListener("mousemove", move); window.addEventListener("mouseup", up); e.preventDefault();
};
const resetPos = (e) => { try { window.localStorage.removeItem(KEY); } catch (err) {} e.currentTarget.style.left = `${DEF.x}px`; e.currentTarget.style.top = `${DEF.y}px`; };
const Panel = ({ children }) => { const p = getPos(); return <div className={box} style={{ left: p.x, top: p.y }} onMouseDown={startDrag} onDoubleClick={resetPos}>{children}</div>; };

const Guilloche = ({ w, h }) => {
  const lines = [];
  for (let k = 0; k < 16; k++) {
    const ph = (k / 16) * Math.PI * 2, pts = [];
    for (let x = 0; x <= w; x += 3) { const t = (x / w) * Math.PI * 4; pts.push(`${x},${(h / 2 + Math.sin(t + ph) * h * 0.3 + Math.sin(t * 2.6 - ph * 1.4) * h * 0.13).toFixed(1)}`); }
    lines.push(<polyline key={k} points={pts.join(" ")} fill="none" stroke={C.gold} strokeOpacity="0.13" strokeWidth="0.6" />);
  }
  return <svg width={w} height={h} style={{ position: "absolute", left: 0, top: 0 }}>{lines}</svg>;
};

export const render = ({ output }) => {
  let a = null; try { a = JSON.parse(output); } catch (e) {}
  const ok = a && !a.error && a.signals;
  return (
    <Panel>
      <div style={{ position: "relative", height: 62, margin: "0 -20px", padding: "0 20px", borderBottom: `1px solid ${C.rule}` }}>
        <Guilloche w={376} h={62} />
        <div style={{ position: "relative", display: "flex", justifyContent: "space-between", alignItems: "flex-end", height: 54 }}>
          <span style={{ fontFamily: SCRIPT, fontSize: 30, color: C.gold, lineHeight: 1 }}>Les comptes</span>
          {ok && <span className={it} style={{ color: C.vellum2, fontSize: 12, textAlign: "right" }}>for {day(a.for_session, { weekday: "long", day: "numeric", month: "long" })}</span>}
        </div>
      </div>
      {!ok && <div className={it} style={{ color: C.vellum2, marginTop: 14 }}>{(a && a.error) || "Reading yesterday's close…"}</div>}
      {ok && (
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, margin: "14px 0 2px" }}>
            <span style={{ width: 13, height: 13, borderRadius: 7, flex: "none", background: a.green ? C.gain : "transparent", border: `1px solid ${a.green ? C.gold : C.pewter}`,
                           boxShadow: a.green ? `0 0 0 3px rgba(156,195,160,0.15), 0 0 12px rgba(156,195,160,0.55)` : "none" }} />
            <span className={it} style={{ fontSize: 21, lineHeight: 1.15, color: a.green ? C.vellum : C.vellum2 }}>
              {a.green ? "A good day to open an evaluation." : "No signal today."}
            </span>
          </div>
          <div className={it} style={{ color: C.pewter, fontSize: 12, margin: "2px 0 10px 23px" }}>
            {a.green ? `${a.on.length} of 4 signals on at ${day(a.asof, { weekday: "long" })}'s close.` : `Evaluations still pay on average. This only times them.`}
          </div>
          {a.signals.map((s) => (
            <div className={row} key={s.key}>
              <span style={{ width: 9, height: 9, borderRadius: 5, marginTop: 4, flex: "none", background: s.on ? C.gain : "transparent", border: `1px solid ${s.on ? C.gain : s.stale ? C.loss : C.pewter}` }} />
              <span style={{ flex: 1 }}>
                <div style={{ color: s.on ? C.vellum : C.vellum2 }}>{PLAIN[s.key] || s.key}</div>
                <div className={it} style={{ color: s.stale ? C.loss : C.pewter, fontSize: 11.5 }}>{s.value}</div>
              </span>
              <span style={{ textAlign: "right", color: C.vellum2, whiteSpace: "nowrap" }}>
                {Math.round(s.payout_on * 100)}%<span className={it} style={{ color: C.pewter }}> vs </span>{Math.round(s.payout_off * 100)}%
              </span>
            </div>
          ))}
          <div className={it} style={{ color: C.pewter, fontSize: 11, marginTop: 8, lineHeight: 1.4 }}>
            Share of 2022–25 evaluations that reached a payout when each signal was on, then off.
          </div>
        </div>
      )}
    </Panel>
  );
};
