/**
 * LBP Baseline Simulator — Canvas Renderer + WebSocket Client
 *
 * Responsibilities:
 *  - Receive building layout + simulation state via WebSocket
 *  - Render floor plan (nodes + edges) on HTML5 Canvas
 *  - Render pedestrians as animated circles with smooth interpolation
 *  - Render hazard overlays as radial gradient glows
 *  - Handle user interactions (floor switch, node click, controls)
 */

const WS_URL     = `ws://${location.host}/ws`;
const SIM_DT_MS  = 100;   // expected state interval from server

// ── State ─────────────────────────────────────────────────────────────────────
let building    = null;    // {nodes, edges, exits}
let state       = null;    // latest simulation state
let prevAgents  = {};      // {id: {x,y}} for interpolation
let currentFloor = 1;
let clickedNode  = null;
let hoverNode    = null;

// Canvas transform
let scale = 1.0, offsetX = 0, offsetY = 0;

// ── Canvas setup ──────────────────────────────────────────────────────────────
const canvas  = document.getElementById("sim-canvas");
const ctx     = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");

function resizeCanvas() {
  const wrap = canvas.parentElement;
  canvas.width  = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
  if (building) fitBuilding();
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
let ws = null;
function connectWS() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    document.getElementById("conn-dot").classList.add("live");
    document.getElementById("conn-text").textContent = "Connected";
  };
  ws.onclose = () => {
    document.getElementById("conn-dot").classList.remove("live");
    document.getElementById("conn-text").textContent = "Reconnecting…";
    setTimeout(connectWS, 2000);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = e => handleMessage(JSON.parse(e.data));
}

function handleMessage(msg) {
  if (msg.type === "building") {
    building = msg.data;
    fitBuilding();
    populateNodeSelect();
    return;
  }
  if (msg.type === "ack_inject") {
    const ack = document.getElementById("inject-ack");
    ack.textContent = msg.ok
      ? `✓ ${msg.hazard_type} injected at ${msg.node}`
      : `✗ Injection failed`;
    ack.style.color = msg.ok ? "var(--green)" : "var(--red)";
    setTimeout(() => ack.textContent = "", 4000);
    return;
  }

  // Regular simulation state
  if (msg.pedestrians !== undefined) {
    // Snapshot previous positions for interpolation
    if (state) {
      state.pedestrians.forEach(a => prevAgents[a.id] = {x: a.x, y: a.y});
    }
    state = msg;
    updateMetrics(msg.metrics);
  }
}

function send(obj) {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
}

// ── Building fit-to-screen ────────────────────────────────────────────────────
function fitBuilding() {
  if (!building) return;
  const nodes = Object.values(building.nodes).filter(n => n.floor === currentFloor);
  if (!nodes.length) return;

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  nodes.forEach(n => {
    minX = Math.min(minX, n.x); minY = Math.min(minY, n.y);
    maxX = Math.max(maxX, n.x); maxY = Math.max(maxY, n.y);
  });

  const pad   = 60;
  const bw    = maxX - minX || 1;
  const bh    = maxY - minY || 1;
  const scaleX = (canvas.width  - pad * 2) / bw;
  const scaleY = (canvas.height - pad * 2) / bh;
  scale   = Math.min(scaleX, scaleY, 4.0);
  offsetX = (canvas.width  - bw * scale) / 2 - minX * scale;
  offsetY = (canvas.height - bh * scale) / 2 - minY * scale;
}

function tx(x) { return x * scale + offsetX; }
function ty(y) { return y * scale + offsetY; }

// ── Rendering ─────────────────────────────────────────────────────────────────
function render(timestamp) {
  requestAnimationFrame(render);
  if (!building) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Background grid
  drawGrid();

  const floorNodes = Object.values(building.nodes).filter(n => n.floor === currentFloor);
  const floorEdges = building.edges.filter(e => {
    const s = building.nodes[e.source], t = building.nodes[e.target];
    return s && t && s.floor === currentFloor && t.floor === currentFloor;
  });
  const hazards = state ? state.hazards : {};

  drawEdges(floorEdges, hazards);
  drawHazardGlows(floorNodes, hazards);
  drawNodes(floorNodes, hazards);
  if (state) drawAgents(state.pedestrians, timestamp);
  drawRouteArrows(state);
}

function drawGrid() {
  const step = 40;
  ctx.strokeStyle = "rgba(148,163,184,0.04)";
  ctx.lineWidth = 1;
  for (let x = 0; x < canvas.width; x += step) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
  }
  for (let y = 0; y < canvas.height; y += step) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
  }
}

function drawEdges(edges, hazards) {
  edges.forEach(e => {
    const s = building.nodes[e.source], t = building.nodes[e.target];
    if (!s || !t) return;
    const hs = (hazards[e.source] || {}).level || 0;
    const ht = (hazards[e.target] || {}).level || 0;
    const h  = Math.max(hs, ht);
    const blocked = (hazards[e.source] || {}).blocked || (hazards[e.target] || {}).blocked;

    const w = Math.max(1.5, (e.width || 2) * scale * 0.18);
    ctx.beginPath();
    ctx.moveTo(tx(s.x), ty(s.y));
    ctx.lineTo(tx(t.x), ty(t.y));

    if (blocked) {
      ctx.strokeStyle = "rgba(239,68,68,0.5)";
      ctx.lineWidth   = w;
      ctx.setLineDash([4, 4]);
    } else if (h > 0.1) {
      ctx.strokeStyle = `rgba(249,115,22,${0.3 + h * 0.5})`;
      ctx.lineWidth   = w;
      ctx.setLineDash([]);
    } else {
      ctx.strokeStyle = "rgba(148,163,184,0.18)";
      ctx.lineWidth   = Math.max(1, w * 0.6);
      ctx.setLineDash([]);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  });
}

function drawHazardGlows(nodes, hazards) {
  nodes.forEach(n => {
    const hz = hazards[n.id];
    if (!hz || hz.level < 0.05) return;
    const r = Math.max(18, 32 * hz.level) * scale * 0.35;
    const cx = tx(n.x), cy = ty(n.y);
    const colorMap = {
      GAS_RELEASE:    [249, 115, 22],
      FIRE:           [239, 68, 68],
      EXPLOSION:      [220, 38, 38],
      CHEMICAL_SPILL: [168, 85, 247],
    };
    const [r_, g_, b_] = colorMap[hz.type] || [249, 115, 22];
    const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 2);
    grd.addColorStop(0,   `rgba(${r_},${g_},${b_},${hz.level * 0.55})`);
    grd.addColorStop(0.5, `rgba(${r_},${g_},${b_},${hz.level * 0.20})`);
    grd.addColorStop(1,   `rgba(${r_},${g_},${b_},0)`);
    ctx.beginPath();
    ctx.arc(cx, cy, r * 2, 0, Math.PI * 2);
    ctx.fillStyle = grd;
    ctx.fill();
  });
}

const NODE_COLORS = {
  EXIT:        "#22c55e",
  STAIRWELL:   "#38bdf8",
  ELEVATOR:    "#818cf8",
  INTERSECTION:"#475569",
  CORRIDOR:    "#334155",
  ROOM:        "#1e293b",
};

function drawNodes(nodes, hazards) {
  nodes.forEach(n => {
    const hz    = hazards[n.id];
    const hLvl  = hz ? hz.level : 0;
    const isBlocked = hz && hz.blocked;
    const isHover   = hoverNode === n.id;
    const isClicked = clickedNode === n.id;

    const cx = tx(n.x), cy = ty(n.y);
    const baseR = n.is_exit ? 10 : (n.type === "CORRIDOR" ? 6 : 8);
    const r = baseR * Math.min(1.4, Math.max(0.6, scale * 0.18));

    // Shadow / glow ring for exits
    if (n.is_exit) {
      ctx.beginPath();
      ctx.arc(cx, cy, r + 6, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(34,197,94,0.15)";
      ctx.fill();
    }

    // Node circle
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    if (isBlocked) {
      ctx.fillStyle = "rgba(239,68,68,0.35)";
    } else if (hLvl > 0.1) {
      const alpha = 0.4 + hLvl * 0.4;
      ctx.fillStyle = hz.color
        ? hz.color.replace(")", `,${alpha})`).replace("rgb", "rgba")
        : `rgba(249,115,22,${alpha})`;
    } else {
      ctx.fillStyle = NODE_COLORS[n.type] || "#1e293b";
    }
    ctx.fill();

    // Border
    ctx.strokeStyle = isClicked  ? "#38bdf8"
                    : isHover    ? "rgba(148,163,184,0.7)"
                    : n.is_exit  ? "#22c55e"
                    : isBlocked  ? "#ef4444"
                    : hLvl > 0.1 ? (hz.color || "#f97316")
                    : "rgba(148,163,184,0.22)";
    ctx.lineWidth = isClicked || isHover ? 2 : 1;
    ctx.stroke();

    // Label (only at reasonable scale)
    if (scale > 1.2 || isHover || isClicked) {
      const label = n.id.replace(/_/g, " ").replace(/f[123]$/i, "").trim();
      ctx.font = `${Math.max(9, Math.min(11, scale * 1.8))}px -apple-system, sans-serif`;
      ctx.fillStyle = n.is_exit ? "#22c55e" : "rgba(148,163,184,0.85)";
      ctx.textAlign = "center";
      ctx.fillText(label, cx, cy + r + 11);
    }
  });
}

// Smooth-interpolated agent rendering
let _lastTS = null;
function drawAgents(agents, timestamp) {
  if (!agents) return;
  const dt = _lastTS ? Math.min((timestamp - _lastTS) / 1000, 0.2) : 0;
  _lastTS = timestamp;

  const floorAgents = agents.filter(a => a.floor === currentFloor);

  floorAgents.forEach(a => {
    if (a.state === "casualty") return;

    // Interpolate from previous position
    let rx = a.x, ry = a.y;
    const prev = prevAgents[a.id];
    if (prev && dt < 0.3) {
      const t = Math.min(1, dt * 8);
      rx = prev.x + (a.x - prev.x) * t;
      ry = prev.y + (a.y - prev.y) * t;
    }

    const cx = tx(rx), cy = ty(ry);
    const r  = Math.max(3, 4.5 * Math.min(scale * 0.22, 1.3));

    // Glow for danger state
    if (a.state === "danger") {
      const pulse = 0.5 + 0.5 * Math.sin(timestamp / 300);
      ctx.beginPath();
      ctx.arc(cx, cy, r + 4 * pulse, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(239,68,68,0.2)";
      ctx.fill();
    }

    // Agent circle
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = a.color;
    ctx.fill();

    // Evacuated: fading white ring
    if (a.state === "evacuated") {
      ctx.beginPath();
      ctx.arc(cx, cy, r + 2, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255,255,255,0.4)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  });
}

function drawRouteArrows(simState) {
  if (!simState || !building) return;
  const agents = simState.pedestrians || [];
  agents.forEach(a => {
    if (a.state === "evacuated" || a.state === "casualty") return;
    if (!a.path || !a.path.length) return;
    if (a.floor !== currentFloor) return;
    const next = a.path[0];
    const nNode = building.nodes[next];
    if (!nNode) return;
    const ax = tx(a.x), ay = ty(a.y);
    const bx = tx(nNode.x), by = ty(nNode.y);
    const alpha = a.state === "rerouting" ? 0.6 : 0.15;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.strokeStyle = `rgba(56,189,248,${alpha})`;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 5]);
    ctx.stroke();
    ctx.setLineDash([]);
  });
}

// ── Metrics update ────────────────────────────────────────────────────────────
function updateMetrics(m) {
  if (!m) return;
  setText("m-moving",     m.moving ?? "-");
  setText("m-rerouting",  m.rerouting ?? "-");
  setText("m-danger",     m.in_danger ?? "-");
  setText("m-evacuated",  m.evacuated ?? "-");
  setText("m-casualties", m.casualties ?? "-");
  setText("m-time",       formatTime(m.sim_time ?? 0));
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function formatTime(secs) {
  const m = Math.floor(secs / 60).toString().padStart(2, "0");
  const s = Math.floor(secs % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

// ── Node selector population ──────────────────────────────────────────────────
function populateNodeSelect() {
  const sel = document.getElementById("node-select");
  sel.innerHTML = "";
  if (!building) return;
  const nodes = Object.values(building.nodes)
    .filter(n => !n.is_exit)
    .sort((a, b) => a.id.localeCompare(b.id));
  nodes.forEach(n => {
    const opt = document.createElement("option");
    opt.value = n.id;
    opt.textContent = n.id.replace(/_/g, " ");
    sel.appendChild(opt);
  });
}

// ── Canvas interactions ───────────────────────────────────────────────────────
canvas.addEventListener("mousemove", e => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  hoverNode = null;

  if (!building) return;
  const floorNodes = Object.values(building.nodes).filter(n => n.floor === currentFloor);
  for (const n of floorNodes) {
    const dx = tx(n.x) - mx, dy = ty(n.y) - my;
    if (Math.hypot(dx, dy) < 18) {
      hoverNode = n.id;
      showTooltip(e.clientX, e.clientY, n);
      return;
    }
  }
  tooltip.classList.remove("visible");
});

canvas.addEventListener("click", e => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  if (!building) return;
  const floorNodes = Object.values(building.nodes).filter(n => n.floor === currentFloor);
  for (const n of floorNodes) {
    const dx = tx(n.x) - mx, dy = ty(n.y) - my;
    if (Math.hypot(dx, dy) < 18) {
      clickedNode = n.id;
      document.getElementById("node-select").value = n.id;
      return;
    }
  }
  clickedNode = null;
});

canvas.addEventListener("mouseleave", () => {
  hoverNode = null;
  tooltip.classList.remove("visible");
});

function showTooltip(cx, cy, node) {
  const hz = state ? (state.hazards[node.id] || null) : null;
  const occ = state
    ? state.pedestrians.filter(a => a.current_node === node.id && a.floor === currentFloor).length
    : 0;

  document.getElementById("tt-name").textContent = node.id.replace(/_/g, " ");
  document.getElementById("tt-type").textContent = node.type;
  document.getElementById("tt-floor").textContent = `Floor ${node.floor}`;
  document.getElementById("tt-hazard").textContent = hz
    ? `${(hz.level * 100).toFixed(0)}% — ${hz.type.replace("_", " ")}`
    : "None";
  document.getElementById("tt-hazard").style.color = hz && hz.level > 0.3
    ? "var(--red)" : "var(--text-muted)";
  document.getElementById("tt-occupancy").textContent = `${occ} persons`;

  const rect = canvas.getBoundingClientRect();
  tooltip.style.left = `${cx - rect.left + 14}px`;
  tooltip.style.top  = `${cy - rect.top - 10}px`;
  tooltip.classList.add("visible");
}

// ── Floor tab switching ────────────────────────────────────────────────────────
document.querySelectorAll(".floor-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    currentFloor = parseInt(btn.dataset.floor);
    document.querySelectorAll(".floor-tab").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    fitBuilding();
  });
});

// ── Simulation controls ───────────────────────────────────────────────────────
document.getElementById("btn-play").addEventListener("click",  () => send({action: "play"}));
document.getElementById("btn-pause").addEventListener("click", () => send({action: "pause"}));
document.getElementById("btn-reset").addEventListener("click", () => {
  const n = parseInt(document.getElementById("evacuee-count").value) || 60;
  send({action: "reset", num_evacuees: n});
});

const speedSlider = document.getElementById("speed-slider");
const speedVal    = document.getElementById("speed-val");
speedSlider.addEventListener("input", () => {
  const v = parseFloat(speedSlider.value);
  speedVal.textContent = `${v}×`;
  send({action: "set_speed", speed: v});
});

// ── Disaster injection ────────────────────────────────────────────────────────
const intensitySlider = document.getElementById("intensity-slider");
const intensityVal    = document.getElementById("intensity-val");
intensitySlider.addEventListener("input", () => {
  intensityVal.textContent = parseFloat(intensitySlider.value).toFixed(1);
});

document.getElementById("btn-inject").addEventListener("click", () => {
  const node = document.getElementById("node-select").value;
  const type = document.getElementById("hazard-type").value;
  const intens = parseFloat(intensitySlider.value);
  if (!node) return;
  send({action: "inject_disaster", node_id: node, hazard_type: type, intensity: intens});
});

// ── Init ──────────────────────────────────────────────────────────────────────
window.addEventListener("resize", () => { resizeCanvas(); fitBuilding(); });
resizeCanvas();
requestAnimationFrame(render);
connectWS();
