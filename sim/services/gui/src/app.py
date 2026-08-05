import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import redis.asyncio as aioredis

# Assumes PYTHONPATH includes the LBP project root
from core.messaging.constants import CHANNEL_ENV_STATE, CHANNEL_GUI_CONTROL, CHANNEL_DISASTER_INJECT
from core.messaging.schemas import DisasterInjectionMsg, SimControlMsg
from src.disaster_injector import DisasterInjector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LBP Interactive GUI Service")

# Setup static files directory
STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

redis_client = None
injector = None

@app.on_event("startup")
async def startup_event():
    global redis_client, injector
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    redis_client = aioredis.from_url(redis_url, decode_responses=True)
    injector = DisasterInjector(redis_client)
    logger.info("Connected to Redis and initialized DisasterInjector.")

@app.on_event("shutdown")
async def shutdown_event():
    if redis_client:
        await redis_client.close()

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = STATIC_DIR / "index.html"
    with open(index_path, "r") as f:
        return f.read()

@app.websocket("/ws/state")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(CHANNEL_ENV_STATE)
    
    try:
        async for message in pubsub.listen():
            if message['type'] == 'message':
                await websocket.send_text(message['data'])
    except WebSocketDisconnect:
        logger.info("Client disconnected from WebSocket")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await pubsub.unsubscribe(CHANNEL_ENV_STATE)

@app.post("/api/inject")
async def inject_disaster(msg: DisasterInjectionMsg):
    try:
        result = await injector.inject(
            threat_type=msg.threat_type,
            target_node=msg.target_node,
            floor=msg.floor,
            intensity=msg.intensity,
            spread_rate=msg.spread_rate,
            metadata=msg.metadata
        )
        return result
    except ValueError as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/control")
async def control_sim(msg: SimControlMsg):
    await redis_client.publish(CHANNEL_GUI_CONTROL, msg.model_dump_json())
    return {"status": "success", "command": msg.command}

@app.get("/api/building")
async def get_building():
    # Reads from Redis key 'lbp:building:graph' (set by environment service on startup)
    data = await redis_client.get("lbp:building:graph")
    if data:
        return json.loads(data)
    else:
        return {"status": "error", "message": "Building graph not found in Redis."}
