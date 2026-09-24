/**
 * Evacuation Simulator — Canvas Renderer + WebSocket Client
 * Features: zoom/pan, time-based smooth interpolation, hazard glows, route arrows
 */

const WS_URL = `ws://${location.host}/ws`;

// ── State ─────────────────────────────────────────────────────────────────────
let building     = null;
let lastState    = null;
let prevState    = null;
let lastStateMs  = 0;       // wall-clock when lastState arrived
let stateIntervalMs = 100;  // running estimate of state update interval

let currentFloor = 1;
let clickedNode  = null;
let hoverNode    = null;
let isDragging   = false;
let dragStartX   = 0, dragStartY = 0;
let didDrag      = false;   // distinguish drag from click

// ── View transform — single source of truth ───────────────────────────────────
let viewScale = 3.0;
let viewPanX  = 0;
let viewPanY  = 0;

function toScreen(wx, wy) {
  return [wx * viewScale + viewPanX, wy * viewScale + viewPanY];
}
function toWorld(sx, sy) {
  return [(sx - viewPanX) / viewScale, (sy - viewPanY) / viewScale];
}

// ── Canvas setup ───────────────────────────────────────────────────────────────
const canvas  = document.getElementById("sim-canvas");
const ctx     = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");

function resizeCanvas() {
  const wrap = canvas.parentElement;
  canvas.width  = wrap.clientWidth  * devicePixelRatio;
  canvas.height = wrap.clientHeight * devicePixelRatio;
  canvas.style.width  = wrap.clientWidth  + "px";
  canvas.style.height = wrap.clientHeight + "px";
  ctx.scale(devicePixelRatio, devicePixelRatio);
  if (building) fitBuilding(false);
}
const logicalW = () => canvas.width  / devicePixelRatio;
const logicalH = () => canvas.height / devicePixelRatio;

// ── Fit building to screen ────────────────────────────────────────────────────
function fitBuilding(reset = true) {
  if (!building) return;
  const nodes = Object.values(building.nodes).filter(n => n.floor === currentFloor);
  if (!nodes.length) return;

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  nodes.forEach(n => {
    if (n.x < minX) minX = n.x; if (n.y < minY) minY = n.y;
    if (n.x > maxX) maxX = n.x; if (n.y > maxY) maxY = n.y;
  });
  const pad  = 70;
  const bw   = maxX - minX || 1, bh = maxY - minY || 1;
  const lw   = logicalW(), lh = logicalH();
  const scaleX = (lw - pad * 2) / bw;
  const scaleY = (lh - pad * 2) / bh;

  if (reset) {
    viewScale = Math.min(scaleX, scaleY, 5.0);
    viewPanX  = (lw - bw * viewScale) / 2 - minX * viewScale;
    viewPanY  = (lh - bh * viewScale) / 2 - minY * viewScale;
  }
}

// ── Zoom & Pan ─────────────────────────────────────────────────────────────────
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const rect   = canvas.getBoundingClientRect();
  const mx     = e.clientX - rect.left;
  const my     = e.clientY - rect.top;
  const factor = e.deltaY < 0 ? 1.18 : 0.85;
  const newScale = Math.max(0.5, Math.min(25, viewScale * factor));
  viewPanX = mx - (mx - viewPanX) * (newScale / viewScale);
  viewPanY = my - (my - viewPanY) * (newScale / viewScale);
  viewScale = newScale;
}, { passive: false });

canvas.addEventListener("mousedown", e => {
  if (e.button !== 0) return;
  isDragging = true;
  didDrag    = false;
  dragStartX = e.clientX - viewPanX;
  dragStartY = e.clientY - viewPanY;
  canvas.style.cursor = "grab";
});

window.addEventListener("mousemove", e => {
  if (isDragging) {
    const newPX = e.clientX - dragStartX;
    const newPY = e.clientY - dragStartY;
    if (Math.abs(newPX - viewPanX) > 2 || Math.abs(newPY - viewPanY) > 2) {
      didDrag = true;
    }
    viewPanX = newPX;
    viewPanY = newPY;
    canvas.style.cursor = "grabbing";
  } else {
    handleHover(e);
  }
});

window.addEventListener("mouseup", e => {
  if (isDragging) {
    isDragging = false;
    canvas.style.cursor = "default";
    if (!didDrag) handleClick(e);  // treat as click only if not dragged
  }
});

// Reset view on double-click
canvas.addEventListener("dblclick", () => fitBuilding(true));

// ── Touch support ──────────────────────────────────────────────────────────────
let lastTouchDist = null;
canvas.addEventListener("touchstart", e => {
  if (e.touches.length === 2) {
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    lastTouchDist = Math.hypot(dx, dy);
  }
}, { passive: true });

canvas.addEventListener("touchmove", e => {
  e.preventDefault();
  if (e.touches.length === 2) {
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    const dist = Math.hypot(dx, dy);
    if (lastTouchDist) {
      const factor = dist / lastTouchDist;
      const mx = (e.touches[0].clientX + e.touches[1].clientX) / 2;
      const my = (e.touches[0].clientY + e.touches[1].clientY) / 2;
      const rect = canvas.getBoundingClientRect();
      const cmx = mx - rect.left, cmy = my - rect.top;
      const newScale = Math.max(0.5, Math.min(25, viewScale * factor));
      viewPanX = cmx - (cmx - viewPanX) * (newScale / viewScale);
      viewPanY = cmy - (cmy - viewPanY) * (newScale / viewScale);
      viewScale = newScale;
    }
    lastTouchDist = dist;
  }
}, { passive: false });

canvas.addEventListener("touchend", () => { lastTouchDist = null; }, { passive: true });

// ── WebSocket ──────────────────────────────────────────────────────────────────
let ws = null;
function connectWS() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    setConn(true);
  };
  ws.onclose = () => { setConn(false); setTimeout(connectWS, 2000); };
  ws.onerror = () => ws.close();
  ws.onmessage = e => handleMessage(JSON.parse(e.data));
}

function setConn(live) {
  document.getElementById("conn-dot").classList.toggle("live", live);
  document.getElementById("conn-text").textContent = live ? "Live" : "Reconnecting…";
}

function handleMessage(msg) {
  if (msg.type === "building") {
    building = msg.data;
    fitBuilding(true);
    populateNodeSelect();
    return;
  }
  if (msg.type === "ack_inject") {
    const ack = document.getElementById("inject-ack");
    ack.textContent = msg.ok ? `✓ ${msg.hazard_type.replace("_"," ")} at ${msg.node}` : "✗ Failed";
    ack.style.color = msg.ok ? "var(--green)" : "var(--red)";
    setTimeout(() => ack.textContent = "", 5000);
    return;
  }
  if (msg.pedestrians !== undefined) {
    const now = performance.now();
    if (lastStateMs > 0) {
      stateIntervalMs = 0.8 * stateIntervalMs + 0.2 * (now - lastStateMs);
    }
    lastStateMs = now;
    prevState   = lastState;
    lastState   = msg;
    updateMetrics(msg.metrics);
    updateSpeedDisplay(msg.speed);
  }
}

function send(obj) {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
}

// ── Rendering — 60 fps RAF ────────────────────────────────────────────────────
function render() {
  requestAnimationFrame(render);

  // Resize check
  const wrap = canvas.parentElement;
  if (Math.abs(canvas.width / devicePixelRatio - wrap.clientWidth) > 2) {
    resizeCanvas();
  }

  const lw = logicalW(), lh = logicalH();
  ctx.clearRect(0, 0, lw, lh);

  if (!building) {
    ctx.fillStyle = "rgba(148,163,184,0.3)";
    ctx.font = "14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Connecting to simulator…", lw / 2, lh / 2);
    return;
  }

  // ── Interpolation alpha (how far between prev and current state) ───────────
  const alpha = lastStateMs > 0
    ? Math.min(1, (performance.now() - lastStateMs) / Math.max(stateIntervalMs, 40))
    : 0;

  const floorNodes = Object.values(building.nodes).filter(n => n.floor === currentFloor);
  const floorEdges = building.edges.filter(e => {
    const s = building.nodes[e.source], t = building.nodes[e.target];
    return s && t && s.floor === currentFloor && t.floor === currentFloor;
  });
  const hazards = lastState ? lastState.hazards : {};

  drawGrid(lw, lh);
  drawEdges(floorEdges, hazards);
  drawHazardGlows(floorNodes, hazards);
  drawNodes(floorNodes, hazards);
  if (lastState) {
    drawRouteLines(lastState.pedestrians, hazards);
    drawAgents(lastState.pedestrians, prevState ? prevState.pedestrians : null, alpha);
  }
}

function drawGrid(lw, lh) {
  const step = Math.max(20, 5 * viewScale);
  const offX = viewPanX % step;
  const offY = viewPanY % step;
  ctx.strokeStyle = "rgba(148,163,184,0.035)";
  ctx.lineWidth = 1;
  for (let x = offX; x < lw; x += step) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, lh); ctx.stroke();
  }
  for (let y = offY; y < lh; y += step) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(lw, y); ctx.stroke();
  }
}

function drawEdges(edges, hazards) {
  edges.forEach(e => {
    const s = building.nodes[e.source], t = building.nodes[e.target];
    if (!s || !t) return;
    const [sx, sy] = toScreen(s.x, s.y);
    const [tx_, ty_] = toScreen(t.x, t.y);
    const hs = (hazards[e.source] || {}).level || 0;
    const ht = (hazards[e.target] || {}).level || 0;
    const h  = Math.max(hs, ht);
    const blk = (hazards[e.source] || {}).blocked || (hazards[e.target] || {}).blocked;
    const w  = Math.max(1.5, (e.width || 2) * viewScale * 0.14);

    ctx.beginPath();
    ctx.moveTo(sx, sy); ctx.lineTo(tx_, ty_);
    if (blk) {
      ctx.strokeStyle = "rgba(239,68,68,0.5)";
      ctx.setLineDash([5, 5]);
    } else if (h > 0.05) {
      const rgb = (hazards[e.source] || hazards[e.target] || {}).rgb || [249,115,22];
      ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${0.2 + h * 0.5})`;
      ctx.setLineDash([]);
    } else {
      ctx.strokeStyle = "rgba(148,163,184,0.16)";
      ctx.setLineDash([]);
    }
    ctx.lineWidth = blk ? w : (h > 0.05 ? w : Math.max(1, w * 0.5));
    ctx.stroke();
    ctx.setLineDash([]);
  });
}

function drawHazardGlows(nodes, hazards) {
  nodes.forEach(n => {
    const hz = hazards[n.id];
    if (!hz || hz.level < 0.05) return;
    const [cx, cy] = toScreen(n.x, n.y);
    const rgb = hz.rgb || [249,115,22];
    const r = Math.max(25, 50 * hz.level) * (viewScale / 3.5);
    const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 2.5);
    grd.addColorStop(0,   `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${Math.min(0.6, hz.level * 0.7)})`);
    grd.addColorStop(0.5, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${hz.level * 0.18})`);
    grd.addColorStop(1,   `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0)`);
    ctx.beginPath();
    ctx.arc(cx, cy, r * 2.5, 0, Math.PI * 2);
    ctx.fillStyle = grd;
    ctx.fill();
  });
}

const NODE_FILL = {
  EXIT:         "#0d2b1a", STAIRWELL: "#0c1e33",
  ELEVATOR:     "#191a38", INTERSECTION: "#1e293b",
  CORRIDOR:     "#162032", ROOM: "#111827",
};
const NODE_STROKE = {
  EXIT: "#22c55e", STAIRWELL: "#38bdf8",
  ELEVATOR: "#818cf8", INTERSECTION: "#475569",
  CORRIDOR: "#334155", ROOM: "#293548",
};

function drawNodes(nodes, hazards) {
  nodes.forEach(n => {
    const hz     = hazards[n.id];
    const hLvl   = hz ? hz.level : 0;
    const blk    = hz && hz.blocked;
    const isHov  = hoverNode === n.id;
    const isSel  = clickedNode === n.id;
    const [cx, cy] = toScreen(n.x, n.y);
    const baseR  = n.is_exit ? 11 : (n.type === "CORRIDOR" || n.type === "INTERSECTION" ? 7 : 9);
    const r      = baseR * Math.max(0.7, Math.min(1.5, viewScale / 3.0));

    // Exit pulse ring
    if (n.is_exit) {
      ctx.beginPath();
      ctx.arc(cx, cy, r + 5, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(34,197,94,0.12)";
      ctx.fill();
    }

    // Hazard-blocked cross-hatch
    if (blk) {
      ctx.beginPath(); ctx.arc(cx, cy, r + 3, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(239,68,68,0.2)"; ctx.fill();
    }

    // Main node circle
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = blk ? "#2d0808"
                  : hLvl > 0.05 && hz ? (() => {
                      const [rr,gg,bb] = hz.rgb;
                      return `rgba(${rr},${gg},${bb},${0.15 + hLvl * 0.2})`;
                    })()
                  : NODE_FILL[n.type] || "#111827";
    ctx.fill();

    // Border
    ctx.strokeStyle = isSel  ? "#38bdf8"
                    : isHov  ? "#94a3b8"
                    : blk    ? "#ef4444"
                    : hLvl > 0.1 && hz ? hz.color
                    : NODE_STROKE[n.type] || "#293548";
    ctx.lineWidth = isSel || isHov ? 2 : (n.is_exit ? 1.5 : 1);
    ctx.stroke();

    // Label
    if (viewScale >= 2.8 || isHov || isSel) {
      const label = n.id.replace(/_f[123]$/i, "").replace(/_/g, " ");
      const fs = Math.max(9, Math.min(12, viewScale * 1.6));
      ctx.font      = `${isSel ? 600 : 400} ${fs}px -apple-system,sans-serif`;
      ctx.fillStyle = n.is_exit ? "#4ade80"
                    : blk ? "#f87171"
                    : isHov || isSel ? "#e2e8f0"
                    : "rgba(148,163,184,0.7)";
      ctx.textAlign = "center";
      ctx.fillText(label, cx, cy + r + 11);
    }
  });
}

// Time-based interpolated agent rendering
function drawAgents(agents, prevAgents, alpha) {
  if (!agents) return;
  const prevMap = {};
  if (prevAgents) prevAgents.forEach(a => prevMap[a.id] = a);

  agents.filter(a => a.floor === currentFloor).forEach(a => {
    if (a.state === "casualty") return;

    // Smooth interpolation based on state interval
    let rx = a.x, ry = a.y;
    const p = prevMap[a.id];
    if (p && p.floor === a.floor) {
      const t = Math.max(0, Math.min(1, alpha));
      rx = p.x + (a.x - p.x) * t;
      ry = p.y + (a.y - p.y) * t;
    }

    const [cx, cy] = toScreen(rx, ry);
    const r = Math.max(3, 4.0 * Math.min(1.4, viewScale / 3.0));

    // Danger pulse ring
    if (a.state === "danger") {
      const pulse = 0.4 + 0.6 * Math.abs(Math.sin(performance.now() / 350));
      ctx.beginPath(); ctx.arc(cx, cy, r + 5 * pulse, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(239,68,68,0.18)"; ctx.fill();
    }

    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = a.color; ctx.fill();

    // Evacuated ring
    if (a.state === "evacuated") {
      ctx.beginPath(); ctx.arc(cx, cy, r + 2, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255,255,255,0.35)"; ctx.lineWidth = 1; ctx.stroke();
    }
  });
}

function drawRouteLines(agents, hazards) {
  if (!agents || !building) return;
  agents.forEach(a => {
    if (a.state === "evacuated" || a.state === "casualty") return;
    if (!a.path || !a.path.length || a.floor !== currentFloor) return;
    const nNode = building.nodes[a.path[0]];
    if (!nNode) return;
    const [ax, ay] = toScreen(a.x, a.y);
    const [bx, by] = toScreen(nNode.x, nNode.y);
    const alpha = a.state === "rerouting" ? 0.55 : 0.12;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
    ctx.strokeStyle = `rgba(56,189,248,${alpha})`;
    ctx.lineWidth = 1; ctx.setLineDash([3, 6]); ctx.stroke(); ctx.setLineDash([]);
  });
}

// ── Hover + Click ──────────────────────────────────────────────────────────────
function getNodeAtScreenPos(mx, my) {
  if (!building) return null;
  const floorNodes = Object.values(building.nodes).filter(n => n.floor === currentFloor);
  for (const n of floorNodes) {
    const [sx, sy] = toScreen(n.x, n.y);
    if (Math.hypot(mx - sx, my - sy) < Math.max(14, 10 * viewScale / 3)) {
      return n;
    }
  }
  return null;
}

function handleHover(e) {
  const rect = canvas.getBoundingClientRect();
  const n = getNodeAtScreenPos(e.clientX - rect.left, e.clientY - rect.top);
  if (n) {
    hoverNode = n.id;
    showTooltip(e.clientX, e.clientY, n);
    canvas.style.cursor = "pointer";
  } else {
    hoverNode = null;
    tooltip.classList.remove("visible");
    canvas.style.cursor = "default";
  }
}

function handleClick(e) {
  const rect = canvas.getBoundingClientRect();
  const n = getNodeAtScreenPos(e.clientX - rect.left, e.clientY - rect.top);
  if (n) {
    clickedNode = n.id;
    const sel = document.getElementById("node-select");
    if (sel) sel.value = n.id;
  } else {
    clickedNode = null;
  }
}

function showTooltip(cx, cy, node) {
  const hz  = lastState ? (lastState.hazards[node.id] || null) : null;
  const occ = lastState
    ? lastState.pedestrians.filter(a => a.current_node === node.id && a.floor === currentFloor).length
    : 0;
  document.getElementById("tt-name").textContent      = node.id.replace(/_/g, " ");
  document.getElementById("tt-type").textContent      = node.type;
  document.getElementById("tt-floor").textContent     = `Floor ${node.floor}`;
  const hzEl = document.getElementById("tt-hazard");
  hzEl.textContent  = hz ? `${(hz.level*100).toFixed(0)}%  ${hz.type.replace(/_/g," ")}` : "Safe";
  hzEl.style.color  = hz && hz.level > 0.3 ? "var(--red)" : hz && hz.level > 0.05 ? "var(--orange)" : "var(--text-muted)";
  document.getElementById("tt-occupancy").textContent = `${occ} persons`;

  const rect = canvas.getBoundingClientRect();
  tooltip.style.left = `${Math.min(cx - rect.left + 14, logicalW() - 180)}px`;
  tooltip.style.top  = `${Math.max(10, cy - rect.top - 80)}px`;
  tooltip.classList.add("visible");
}

// ── Node list ──────────────────────────────────────────────────────────────────
function populateNodeSelect() {
  const sel = document.getElementById("node-select");
  sel.innerHTML = "";
  if (!building) return;
  Object.values(building.nodes)
    .filter(n => !n.is_exit)
    .sort((a, b) => a.id.localeCompare(b.id))
    .forEach(n => {
      const opt = document.createElement("option");
      opt.value = n.id;
      opt.textContent = n.id.replace(/_/g, " ");
      sel.appendChild(opt);
    });
}

// ── Metrics ────────────────────────────────────────────────────────────────────
function updateMetrics(m) {
  if (!m) return;
  setText("m-moving",     m.moving     ?? "—");
  setText("m-rerouting",  m.rerouting  ?? "—");
  setText("m-danger",     m.in_danger  ?? "—");
  setText("m-evacuated",  m.evacuated  ?? "—");
  setText("m-casualties", m.casualties ?? "—");
  setText("m-time",       formatTime(m.sim_time ?? 0));
}
function setText(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }
function formatTime(s) {
  return `${Math.floor(s/60).toString().padStart(2,"0")}:${Math.floor(s%60).toString().padStart(2,"0")}`;
}
function updateSpeedDisplay(spd) {
  const sl = document.getElementById("speed-slider");
  const vl = document.getElementById("speed-val");
  if (sl && Math.abs(parseFloat(sl.value) - spd) > 0.05) sl.value = spd;
  if (vl) vl.textContent = `${parseFloat(spd).toFixed(1)}×`;
}

// ── Floor tabs ─────────────────────────────────────────────────────────────────
document.querySelectorAll(".floor-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    currentFloor = parseInt(btn.dataset.floor);
    document.querySelectorAll(".floor-tab").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    fitBuilding(true);
  });
});

// ── Controls ───────────────────────────────────────────────────────────────────
document.getElementById("btn-play").addEventListener("click",  () => send({action:"play"}));
document.getElementById("btn-pause").addEventListener("click", () => send({action:"pause"}));
document.getElementById("btn-reset").addEventListener("click", () => {
  const n = parseInt(document.getElementById("evacuee-count").value) || 60;
  send({action:"reset", num_evacuees:n});
  fitBuilding(true);
});

const speedSlider = document.getElementById("speed-slider");
speedSlider.addEventListener("input", () => {
  const v = parseFloat(speedSlider.value);
  document.getElementById("speed-val").textContent = `${v.toFixed(1)}×`;
  send({action:"set_speed", speed:v});
});

const intensitySlider = document.getElementById("intensity-slider");
intensitySlider.addEventListener("input", () => {
  document.getElementById("intensity-val").textContent = parseFloat(intensitySlider.value).toFixed(1);
});

document.getElementById("btn-inject").addEventListener("click", () => {
  const node  = document.getElementById("node-select").value;
  const type  = document.getElementById("hazard-type").value;
  const intens = parseFloat(intensitySlider.value);
  if (!node) return;
  send({action:"inject_disaster", node_id:node, hazard_type:type, intensity:intens});
});

document.getElementById("btn-fit").addEventListener("click", () => fitBuilding(true));

// ── Init ───────────────────────────────────────────────────────────────────────
window.addEventListener("resize", resizeCanvas);
resizeCanvas();
requestAnimationFrame(render);
connectWS();
