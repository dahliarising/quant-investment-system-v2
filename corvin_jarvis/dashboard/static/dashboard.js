const $ = (id) => document.getElementById(id);
const fmtKRW = (v) => v == null ? "—" : "₩" + Math.round(v).toLocaleString();
const pct = (v) => v == null ? "—" : (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
const cls = (v) => v == null ? "dim" : v >= 0 ? "up" : "down";
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));

function renderHero(t) {
  const pnl = t.equity_pnl_krw, p = t.equity_pnl_pct;
  $("p-hero").innerHTML = `
    <div><div class="big ${cls(pnl)}">${pnl != null && pnl >= 0 ? '+' : ''}${fmtKRW(pnl)}</div>
      <div class="sub ${cls(p)}">P&L ${pct(p)}</div></div>
    <div style="margin-left:auto;text-align:right">
      <div class="dim" style="font-size:11px">TOTAL ASSETS</div>
      <div class="tot">${fmtKRW(t.total_assets_krw)}</div>
      <div class="dim">CASH ${fmtKRW(t.deployable_krw)}</div></div>`;
}

// 실현 손익률(%) 추이 — 백엔드 equity_curve(스냅샷별 평단 대비 %)를 날짜순 좌→우로 그린다.
// 0% 기준선 위(초록)/아래(빨강), 마지막 점이 라이브 현재 손익률.
function renderReturnCurve(series) {
  const el = $("p-curve");
  const pts = (series || []).filter(p => p && p.pnl_pct != null);
  const last = pts.length ? pts[pts.length - 1].pnl_pct : null;
  const nowLbl = last == null ? "—" : (last >= 0 ? "+" : "") + last.toFixed(2) + "%";
  const nowCls = last == null ? "dim" : last >= 0 ? "up" : "down";
  if (pts.length < 2) {
    el.innerHTML = `<div class="curve-meta"><span class="dim">P&L% trend</span>
      <span class="now ${nowCls}">${nowLbl}</span></div>
      <div class="dim" style="flex:1;display:flex;align-items:center">need ≥2 snapshots…</div>`;
    return;
  }
  const vals = pts.map(p => p.pnl_pct);
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals), span = (hi - lo) || 1;
  const W = 600, H = 110, padL = 4, padR = 4;
  const x = (i) => padL + (i / (pts.length - 1)) * (W - padL - padR);
  const y = (v) => H - ((v - lo) / span) * (H - 12) - 6;
  const line = pts.map((p, i) => `${x(i).toFixed(0)},${y(p.pnl_pct).toFixed(1)}`).join(" ");
  const y0 = y(0).toFixed(1);
  const lastX = x(pts.length - 1).toFixed(0), lastY = y(last).toFixed(1);
  const tip = last >= 0 ? "#7fe0a0" : "#ff8a8a";
  el.innerHTML = `
    <div class="curve-meta"><span class="dim">P&L% trend · ${pts.length}pt</span>
      <span class="now ${nowCls}">${nowLbl}</span></div>
    <svg class="curve-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <line x1="0" y1="${y0}" x2="${W}" y2="${y0}" stroke="#5a4520" stroke-width="1" stroke-dasharray="4 4"/>
      <polyline fill="none" stroke="#ffae42" stroke-width="2" points="${line}"/>
      <circle class="curve-tip" cx="${lastX}" cy="${lastY}" r="3.5" fill="${tip}"/>
    </svg>
    <div class="curve-axis"><span>${esc(pts[0].date)}</span>
      <span>${hi >= 0 ? '+' : ''}${hi.toFixed(1)}% peak</span><span>${esc(pts[pts.length - 1].date)}</span></div>`;
}

function renderPositions(rows) {
  const price = (r) => r.price == null ? "—" : (r.ccy === "USD" ? "$" + r.price.toFixed(2) : "₩" + Math.round(r.price).toLocaleString());
  $("p-positions").innerHTML = `<table>
    <tr><th>TICKER</th><th>PRICE</th><th>DAY</th><th>PNL%</th><th>VALUE(₩)</th></tr>
    ${rows.map(r => `<tr><td><span class="sym">${esc(r.sym)}</span></td>
      <td>${price(r)}</td><td class="${cls(r.day_pct)}">${pct(r.day_pct)}</td>
      <td class="${cls(r.pnl_pct)}">${pct(r.pnl_pct)}</td>
      <td>${r.value_krw?Math.round(r.value_krw/1000).toLocaleString()+'k':'—'}</td></tr>`).join("")}
  </table>`;
}

function renderIndices(rows) {
  $("p-indices").innerHTML = rows.map(r =>
    `<div class="idx"><span>${esc(r.label)}</span><span>${r.price?.toLocaleString()} <span class="${cls(r.pct)}">${pct(r.pct)}</span></span></div>`).join("");
}

function renderAlloc(rows) {
  $("p-alloc").innerHTML = rows.map(r =>
    `<div class="bar"><div class="lab"><span>${esc(r.label)}</span><span>${r.pct}%</span></div>
     <div class="track"><div class="fill" style="width:${Math.min(r.pct,100)}%"></div></div></div>`).join("");
}

function renderSignals(rows) {
  const dot = (c) => c === "green" ? "g" : c === "red" ? "r" : "a";
  $("p-signals").innerHTML = rows.map(r =>
    `<div><span><span class="dot ${dot(r.color)}"></span>${esc(r.sym)}</span><span class="dim">${esc(r.zone)}</span></div>`).join("")
    || '<div class="dim">— no signals</div>';
}

function renderLog(rows) {
  $("p-log").innerHTML = rows.map(r => `<div><b>${esc(r.ts)}</b> ${esc(r.text)}</div>`).join("");
}

const POLY_HIST = {};  // question -> [prob,...] (확률 변화 스파크라인용 누적)
let POLY_ACTIVE = null;
function polySpark(hist) {
  if (!hist || hist.length < 2) return "";
  const w = 50, h = 12, mn = Math.min(...hist), mx = Math.max(...hist), sp = (mx - mn) || 1;
  const pts = hist.map((v, i) => `${(i / (hist.length - 1) * w).toFixed(0)},${(h - ((v - mn) / sp) * (h - 2) - 1).toFixed(1)}`).join(" ");
  return `<svg class="poly-spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline fill="none" stroke="#ffae42" stroke-width="1" points="${pts}"/></svg>`;
}
function renderPoly(rows) {
  const tabsEl = $("p-poly-tabs");
  if (!rows || !rows.length) { tabsEl.innerHTML = ""; $("p-poly").innerHTML = '<div class="dim">— no data</div>'; return; }
  // 확률 누적 + 급변 감지
  rows.forEach(r => {
    const h = POLY_HIST[r.question] || (POLY_HIST[r.question] = []);
    const prev = h.length ? h[h.length - 1] : null;
    r._moved = prev != null && Math.abs(r.prob - prev) >= 0.01;
    h.push(r.prob); if (h.length > 30) h.shift();
  });
  const cats = {};
  rows.forEach(r => { const c = r.category || "Other"; (cats[c] = cats[c] || []).push(r); });
  const names = Object.keys(cats);
  if (POLY_ACTIVE === null || !cats[POLY_ACTIVE]) POLY_ACTIVE = names[0];
  const vol = (v) => v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : Math.round(v / 1e3) + "k";
  const body = () => {
    $("p-poly").innerHTML = (cats[POLY_ACTIVE] || []).map(r => {
      const yes = r.prob >= 0.5, lead = Math.round((yes ? r.prob : 1 - r.prob) * 100);
      const hot = r.volume_usd >= 1e6 ? "⚡" : "";
      return `<div class="poly-row${r._moved ? ' poly-pulse' : ''}">
        <div class="q">${esc(r.question)}</div>
        <div class="poly-bot">
          <span class="${yes ? 'up' : 'down'}">${yes ? 'YES ▲' : 'NO ▼'}${lead}%</span>
          ${polySpark(POLY_HIST[r.question])}
          <span class="dim" style="font-size:10px">${hot}$${vol(r.volume_usd)}</span>
        </div>
        <div class="track"><div class="fill" style="width:${r.prob * 100}%"></div></div></div>`;
    }).join("");
  };
  tabsEl.innerHTML = names.map((n, i) =>
    `<span class="poly-tab ${n === POLY_ACTIVE ? 'active' : ''}" data-idx="${i}">${esc(n)}</span>`).join("");
  tabsEl.querySelectorAll(".poly-tab").forEach((t, i) => {
    t.onclick = () => { POLY_ACTIVE = names[i]; body();
      tabsEl.querySelectorAll(".poly-tab").forEach((x, j) => x.classList.toggle("active", j === i)); };
  });
  body();
}

function renderTicker(rows) {
  $("p-ticker").innerHTML = rows.map(r =>
    `<span><b>${esc(r.label)}</b> ${r.price} <span class="${cls(r.pct)}">${pct(r.pct)}</span></span>`).join("");
}

// 프로세싱 표현 — 여러 가닥 곡선 위를 점들이 흐르는 ambient 애니메이션 (SVG animateMotion).
let PROC_DONE = false;
function startProcessing() {
  if (PROC_DONE) return;
  PROC_DONE = true;
  const el = $("p-matrix");
  const W = 200, H = 80;
  const strands = [
    { y: 10, dur: 3.4 }, { y: 22, dur: 4.5 }, { y: 34, dur: 2.9 }, { y: 46, dur: 3.9 },
    { y: 58, dur: 5.1 }, { y: 70, dur: 3.2 }, { y: 16, dur: 4.1 },
  ];
  const d = (s) => `M0,${s.y} Q${W * 0.25},${s.y - 14} ${W * 0.5},${s.y} T${W},${s.y}`;
  const paths = strands.map(s => `<path d="${d(s)}" fill="none" stroke="#4a3818" stroke-width="1.4"/>`).join("");
  const dots = strands.map((s, i) => {
    const dot = (r, fill, op, mul, begin) =>
      `<circle r="${r}" fill="${fill}" opacity="${op}"><animateMotion dur="${(s.dur * mul).toFixed(1)}s" repeatCount="indefinite" begin="${begin}s" path="${d(s)}"/></circle>`;
    return dot(3.2, "#ffd27f", 0.95, 1, (i * 0.4).toFixed(1)) + dot(2.2, "#ffae42", 0.55, 1.5, (i * 0.6 + 1).toFixed(1));
  }).join("");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="width:100%;height:100%">${paths}${dots}</svg>
    <div class="proc-label">processing…</div>`;
}

async function refresh() {
  try {
    const snap = await (await fetch("/api/snapshot")).json();
    const state = snap.market_state || "open";
    renderHero(snap.totals || {});
    renderReturnCurve(snap.equity_curve || []);
    renderPositions(snap.positions || []);
    renderIndices(snap.indices || []);
    renderAlloc(snap.allocation || []);
    renderSignals(snap.signals || []);
    renderLog(snap.log || []);
    renderTicker(snap.macro_ticker || []);
    renderPoly(snap.polymarket || []);
    $("topmeta").textContent = `${snap.ts} · ${state.toUpperCase()} · FX ${snap.fx_usdkrw?.toFixed?.(2) ?? "—"}`;
    return state;
  } catch (e) {
    $("topmeta").textContent = "connection error — retrying";
    return "open";
  }
}

async function loop() {
  const state = await refresh();
  setTimeout(loop, state === "open" ? 30000 : 300000);
}

// 시작 버스트: 캐시된 스냅샷을 빠르게 몇 번 샘플해 수익률 곡선을 즉시 띄운 뒤 정상 주기로.
async function bootstrap() {
  for (let i = 0; i < 6; i++) {
    await refresh();
    await new Promise(r => setTimeout(r, 1200));
  }
  loop();
}

startProcessing();
bootstrap();
