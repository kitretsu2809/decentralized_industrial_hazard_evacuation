"""
FastAPI + WebSocket server for the Industrial Evacuation Simulator.
Serves the static UI and streams simulation state at ~12 Hz via WebSocket.
"""
from __future__ import annotations
import os, sys, json, asyncio
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

# Ensure repo root on path
_SRV_DIR  = Path(__file__).parent
_SIM_DIR  = _SRV_DIR.parent
_REPO_DIR = _SIM_DIR.parent
sys.path.insert(0, str(_REPO_DIR))

from simulator.engine.simulation import Simulation

app = FastAPI(title="Industrial Evacuation Simulator")

# ── Simulation singleton ──────────────────────────────────────────────────────
sim = Simulation()

# ── WebSocket connection manager ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, data: dict):
        if not self.active:
            return
        msg = json.dumps(data)
        dead = set()
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        self.active -= dead

manager = ConnectionManager()
sim.set_broadcast(manager.broadcast)

# ── Startup: launch simulation loop ──────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    asyncio.create_task(sim.run_loop())

# ── HTTP Endpoints ────────────────────────────────────────────────────────────
_STATIC = _SIM_DIR / "static"

@app.get("/", response_class=HTMLResponse)
async def index():
    return (_STATIC / "index.html").read_text()

@app.get("/building")
async def building():
    return JSONResponse(sim.building_dict())

@app.get("/state")
async def state():
    return JSONResponse(sim.state_dict())

# ── Static files ──────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    # Send initial building layout
    await ws.send_text(json.dumps({"type": "building", "data": sim.building_dict()}))
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            action = msg.get("action")

            if action == "play":
                sim.play()
            elif action == "pause":
                sim.pause()
            elif action == "reset":
                sim.reset(num_evacuees=msg.get("num_evacuees"))
                await ws.send_text(json.dumps({"type": "building",
                                               "data": sim.building_dict()}))
            elif action == "set_speed":
                sim.set_speed(float(msg.get("speed", 1.0)))
            elif action == "set_evacuees":
                sim.num_evacuees = int(msg.get("count", 60))
            elif action == "inject_disaster":
                ok = sim.inject_disaster(
                    msg.get("node_id", ""),
                    msg.get("hazard_type", "GAS_RELEASE"),
                    float(msg.get("intensity", 0.8)),
                )
                await ws.send_text(json.dumps({
                    "type": "ack_inject",
                    "ok": ok,
                    "node": msg.get("node_id"),
                    "hazard_type": msg.get("hazard_type"),
                }))

    except WebSocketDisconnect:
        manager.disconnect(ws)
