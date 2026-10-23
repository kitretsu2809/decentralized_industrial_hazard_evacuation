"""
FastAPI + WebSocket server for the Industrial Evacuation Simulator.
Serves the static UI and streams simulation state at ~12 Hz via WebSocket.
"""
from __future__ import annotations
import os, sys, json, asyncio, tempfile, uuid
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

# Ensure repo root on path
_SRV_DIR  = Path(__file__).parent
_SIM_DIR  = _SRV_DIR.parent
_REPO_DIR = _SIM_DIR.parent
sys.path.insert(0, str(_REPO_DIR))


def _load_project_model_settings() -> None:
    """Load local perception model settings when launched outside Docker.

    Docker passes an absolute container path in compose.  For direct local
    launches, reading these settings keeps ``.env`` useful without adding a
    runtime dependency on python-dotenv. Explicitly exported environment
    variables always take precedence.
    """
    env_file = _REPO_DIR / ".env"
    if not env_file.is_file():
        return
    checkpoint_keys = {"SENSOR_CNN_CHECKPOINT", "YOLO_MODEL_PATH"}
    setting_keys = checkpoint_keys | {"YOLO_DEVICE", "ENABLE_HSV_FIRE_FALLBACK"}
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in setting_keys or os.environ.get(key):
            continue
        value = value.strip().strip("\"'")
        if key in checkpoint_keys:
            checkpoint = Path(value)
            if not checkpoint.is_absolute():
                checkpoint = _REPO_DIR / checkpoint
            value = str(checkpoint)
        os.environ[key] = value


_load_project_model_settings()

from simulator.engine.simulation import Simulation
from simulator.perception import SimulatorPerceptionPipeline

app = FastAPI(title="Industrial Evacuation Simulator")

# ── Simulation singleton ──────────────────────────────────────────────────────
sim = Simulation()
perception = SimulatorPerceptionPipeline(
    on_update=sim.apply_perception_update,
    sensor_provider=sim.virtual_sensor_reading,
)
sim.set_perception_provider(perception.get_state)

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
        for ws in list(self.active):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        if dead:
            self.active.difference_update(dead)

manager = ConnectionManager()
sim.set_broadcast(manager.broadcast)

# ── Startup: launch simulation loop ──────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    asyncio.create_task(sim.run_loop())


@app.on_event("shutdown")
async def _shutdown():
    perception.stop_all()

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


@app.post("/api/disaster")
async def inject_disaster(payload: dict):
    """HTTP fallback for recorded demonstrations before a WebSocket is ready."""
    ok = sim.inject_disaster(
        payload.get("node_id", ""),
        payload.get("hazard_type", "GAS_RELEASE"),
        float(payload.get("intensity", 0.8)),
    )
    return {"ok": ok, "node": payload.get("node_id"), "hazard_type": payload.get("hazard_type")}


@app.get("/api/perception/capabilities")
async def perception_capabilities():
    return perception.capabilities()


@app.get("/api/perception/feeds")
async def perception_feeds():
    return perception.get_state()


@app.post("/api/perception/feeds")
async def assign_perception_feed(
    request: Request,
    node_id: str,
):
    """Assign a recorded hazard/crowd video to a simulated camera node."""
    if node_id not in sim.node_positions:
        raise HTTPException(status_code=404, detail="Unknown simulator node")

    original_name = Path(request.headers.get("x-video-filename", "feed.mp4")).name
    suffix = Path(original_name).suffix.lower()
    allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    if suffix not in allowed:
        raise HTTPException(status_code=415, detail="Use MP4, MOV, AVI, MKV, or WebM video")

    upload_dir = Path(tempfile.gettempdir()) / "lbp_perception_feeds"
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    max_bytes = 250 * 1024 * 1024
    written = 0
    try:
        with destination.open("wb") as output:
            async for chunk in request.stream():
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(status_code=413, detail="Video exceeds the 250 MB demo limit")
                output.write(chunk)
        if written == 0:
            raise HTTPException(status_code=400, detail="The uploaded video is empty")
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    feed = perception.assign_video(
        node_id=node_id,
        floor=sim.node_floors.get(node_id, 1),
        capacity=sim.node_capacities.get(node_id, 10),
        video_path=str(destination),
        original_name=original_name,
    )
    return {"ok": True, "feed": feed, "capabilities": perception.capabilities()}


@app.get("/api/perception/feeds/{feed_id}/media")
async def perception_feed_media(feed_id: str):
    path = perception.media_path(feed_id)
    if not path or not path.is_file():
        raise HTTPException(status_code=404, detail="Feed media not found")
    return FileResponse(path)

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
            elif action == "switch_layout":
                requested_layout = str(msg.get("layout_id", ""))
                if requested_layout in sim.layout_catalog:
                    perception.clear_feeds()
                ok = sim.switch_layout(
                    requested_layout,
                    num_evacuees=msg.get("num_evacuees"),
                )
                if ok:
                    await manager.broadcast({
                        "type": "building",
                        "data": sim.building_dict(),
                    })
                await ws.send_text(json.dumps({
                    "type": "ack_layout",
                    "ok": ok,
                    "layout_id": sim.layout_id,
                    "reason": (
                        f"Loaded {sim.building_name}"
                        if ok
                        else "Unknown facility layout"
                    ),
                }))
            elif action == "set_speed":
                sim.set_speed(float(msg.get("speed", 1.0)))
            elif action == "set_evacuees":
                sim.num_evacuees = int(msg.get("count", 60))
            elif action == "trigger_alarm":
                sim.trigger_alarm()
                await ws.send_text(json.dumps({
                    "type": "ack_alarm",
                    "ok": True,
                    "active": sim.alarm_active,
                    "reason": "Facility alarm activated",
                }))
            elif action == "reset_alarm":
                ok = sim.reset_alarm()
                await ws.send_text(json.dumps({
                    "type": "ack_alarm",
                    "ok": ok,
                    "active": sim.alarm_active,
                    "reason": (
                        "Alarm reset"
                        if ok
                        else "Reset blocked while an active hazard is detected"
                    ),
                }))
            elif action == "toggle_alarm":
                ok = sim.toggle_alarm()
                await ws.send_text(json.dumps({
                    "type": "ack_alarm",
                    "ok": ok,
                    "active": sim.alarm_active,
                    "reason": (
                        "Facility alarm activated"
                        if sim.alarm_active and ok
                        else "Alarm reset"
                        if ok
                        else "Reset blocked while an active hazard is detected"
                    ),
                }))
            elif action == "set_policy_mode":
                mode = msg.get("mode", "dijkstra")
                ok = sim.set_policy_mode(mode)
                await ws.send_text(json.dumps({
                    "type": "ack_policy_mode",
                    "ok": ok,
                    "mode": sim.policy_mode,
                    "policy": sim.state_dict().get("policy", {}),
                }))
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
