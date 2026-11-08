// L'Argent Parle - LA MATINÉE: the 09:00 ET morning report as a desktop page. Übersicht widget, reads ~/fartpie/live/analysis.json (the Mac screener's
// report: market read, indices and sectors with daily / weekly trend, the day's calendar, headlines with a one-line read). Refreshes every 5 minutes.
// Same engraved look as the screener widget: Times New Roman + Snell Roundhand. Drag it anywhere (remembered); double-click resets.
import { css } from "uebersicht";

export const command = "bash lap-report.widget/fetch.sh";
export const refreshFrequency = 5 * 60 * 1000;
export const className = `left: 0; top: 0;`;

const C = { ink: "rgba(13,19,34,0.94)", vellum: "#ece2c9", vellum2: "#c9bfa6", pewter: "#8b8f99", gold: "#c6a465", rule: "rgba(198,164,101,0.28)", faint: "rgba(236,226,201,0.08)",
            gain: "#9cc3a0", loss: "#d4776f" };
const SERIF = `"Times New Roman", Times, serif`, SCRIPT = `"Snell Roundhand", "Apple Chancery", cursive`;
const box = css`
  position: absolute; width: 400px; color: ${C.vellum}; font-family: ${SERIF}; font-size: 13px; line-height: 1.35; cursor: grab; user-select: none;
  background: ${C.ink}; border: 1px solid ${C.rule}; border-radius: 6px; padding: 0 20px 16px; overflow: hidden;
  box-shadow: 0 18px 50px rgba(0,0,0,0.45); -webkit-font-smoothing: antialiased;
  &:active { cursor: grabbing; }
`;
const row = css`display: flex; align-items: baseline; padding: 3px 0; border-top: 1px solid ${C.faint};`;
const title = css`display: flex; justify-content: space-between; align-items: baseline; margin: 14px 0 4px; font-style: italic; color: ${C.gold}; font-size: 15px;`;
const it = css`font-style: italic;`;

const KEY = "lap-report-pos", DEF = { x: 24, y: 64 };
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
    for (let x = 0; x <= w; x += 3) { const t = (x / w) * Math.PI * 4.5; pts.push(`${x},${(h / 2 + Math.sin(t + ph) * h * 0.3 + Math.sin(t * 2.6 - ph * 1.4) * h * 0.13).toFixed(1)}`); }
    lines.push(<polyline key={k} points={pts.join(" ")} fill="none" stroke={C.gold} strokeOpacity="0.13" strokeWidth="0.6" />);
  }
  return <svg width={w} height={h} style={{ position: "absolute", left: 0, top: 0 }}>{lines}</svg>;
};
const Masthead = ({ right }) => (
  <div style={{ position: "relative", height: 62, margin: "0 -20px", padding: "0 20px", borderBottom: `1px solid ${C.rule}` }}>
    <Guilloche w={440} h={62} />
    <div style={{ position: "relative", display: "flex", justifyContent: "space-between", alignItems: "flex-end", height: 54 }}>
      <span style={{ fontFamily: SCRIPT, fontSize: 30, color: C.gold, lineHeight: 1 }}>La matinée</span>
      <span className={it} style={{ color: C.vellum2, fontSize: 12, textAlign: "right" }}>{right}</span>
    </div>
  </div>
);
const sgn = (x, d = 1) => (x == null || isNaN(x) ? "–" : `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(Number(x)).toFixed(d)}%`);
const hue = (x) => (x == null || isNaN(x) ? C.pewter : x >= 0 ? C.gain : C.loss);
const Trend = ({ v }) => <span style={{ color: v === "up" ? C.gain : v === "down" ? C.loss : C.pewter }}>{v === "up" ? "▲" : v === "down" ? "▼" : "–"}</span>;

export const render = ({ output, error }) => {
  let d = null; try { d = JSON.parse(output); } catch (e) {}
  if (error || !d || d.error || !d.report) return <Panel><Masthead right="" /><div className={it} style={{ color: C.vellum2, marginTop: 14 }}>{(d && d.error) || "Reading the morning report…"}</div></Panel>;
  const a = d.report, when = a.asof ? new Date(a.asof) : null;
  const today = when && when.toDateString() === new Date().toDateString();
  const head = when ? `${when.toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long" })}, ${a.time || ""}` : a.date || "";
  const idx = (a.sectors || []).filter((s) => ["SPY", "QQQ"].includes(s.etf)), all = (a.sectors || []).filter((s) => !["SPY", "QQQ"].includes(s.etf)).sort((x, y) => y.r5 - x.r5);
  const others = all.filter((s) => !s.mine), pick = new Set([...all.filter((s) => s.mine), ...others.slice(0, 2), ...others.slice(-2)].map((s) => s.etf));
  const sec = all.filter((s) => pick.has(s.etf));                                   // held sectors + the two leaders and two laggards over 5 days
  const cal = [...(a.calendar || [])].sort((x, y) => (y.impact === "High") - (x.impact === "High") || String(x.t).localeCompare(String(y.t))).slice(0, 4), news = (a.headlines || []).slice(0, 2);
  return (
    <Panel>
      <Masthead right={<span>{head}{!today && when ? <span><br /><span style={{ color: C.pewter }}>latest report</span></span> : null}</span>} />
      <div className={it} style={{ color: C.vellum, fontSize: 13, lineHeight: 1.42, marginTop: 12 }}>{a.market}</div>

      <div className={title}><span>Indices and sectors</span><span style={{ color: C.pewter, fontSize: 11 }}>trend day and week, then 1, 5 and 21 days</span></div>
      {[...idx, ...sec].map((s, i) => (
        <div className={row} key={s.etf} style={i === idx.length ? { borderTop: `1px solid ${C.rule}` } : {}}>
          <span style={{ flex: 1 }}>
            <span style={{ color: idx.includes(s) ? C.vellum : C.vellum2, borderBottom: s.mine ? `1px solid ${C.gold}` : "none" }}>{s.name}</span>
            <span className={it} style={{ color: C.pewter, fontSize: 11 }}> {s.etf}</span>
          </span>
          <span style={{ width: 22, textAlign: "center" }}><Trend v={s.d} /></span>
          <span style={{ width: 22, textAlign: "center" }}><Trend v={s.w} /></span>
          <span style={{ width: 54, textAlign: "right", color: hue(s.r1) }}>{sgn(s.r1)}</span>
          <span style={{ width: 54, textAlign: "right", color: hue(s.r5) }}>{sgn(s.r5)}</span>
          <span style={{ width: 58, textAlign: "right", color: hue(s.r21) }}>{sgn(s.r21)}</span>
        </div>
      ))}
      <div className={it} style={{ color: C.pewter, fontSize: 11, marginTop: 3 }}>Underlined: sectors you hold. The others are the week’s two leaders and two laggards.</div>

      {cal.length > 0 && <div className={title}><span>{today ? "Today's calendar" : "Calendar"}</span><span style={{ color: C.pewter, fontSize: 11 }}>ET</span></div>}
      {cal.map((c, i) => (
        <div className={row} key={i}>
          <span style={{ width: 44, color: C.pewter }}>{c.t}</span>
          <span style={{ flex: 1, color: c.impact === "High" ? C.vellum : C.vellum2, fontWeight: c.impact === "High" ? 600 : 400 }}>{c.title}</span>
          {c.prev ? <span className={it} style={{ color: C.pewter, fontSize: 11.5 }}>prev {c.prev}</span> : null}
        </div>
      ))}

      {news.length > 0 && <div className={title}><span>Headlines</span></div>}
      {news.map((n, i) => (
        <div key={i} style={{ padding: "5px 0", borderTop: `1px solid ${C.faint}` }}>
          <div style={{ color: C.vellum }}>{n.title}</div>
          <div className={it} style={{ color: C.pewter, fontSize: 12 }}>{n.read}</div>
        </div>
      ))}
    </Panel>
  );
};
