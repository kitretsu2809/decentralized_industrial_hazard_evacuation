"""Record the actual simulator UI through Chrome DevTools Protocol.

The resulting video is a screen recording of the running web application, not
an illustrated replacement. The simulator server must be running first.
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

import requests
from PIL import Image
from websocket import create_connection


def cdp(ws, method, params=None):
    cdp.counter += 1
    ws.send(json.dumps({"id": cdp.counter, "method": method, "params": params or {}}))
    while True:
        msg = json.loads(ws.recv())
        if msg.get("id") == cdp.counter:
            return msg


cdp.counter = 0


def evaluate(ws, expression):
    return cdp(ws, "Runtime.evaluate", {"expression": expression, "awaitPromise": True, "returnByValue": True})


def click(ws, selector):
    evaluate(ws, f"document.querySelector({json.dumps(selector)})?.click()")


def set_value(ws, selector, value):
    script = """(() => { const e=document.querySelector(%s); if (!e) return false;
    e.value=%s; e.dispatchEvent(new Event('change',{bubbles:true})); return true; })()""" % (json.dumps(selector), json.dumps(value))
    evaluate(ws, script)


def set_file(ws, selector, path):
    doc = cdp(ws, "DOM.getDocument")
    root = doc["result"]["root"]["nodeId"]
    node = cdp(ws, "DOM.querySelector", {"nodeId": root, "selector": selector})
    node_id = node["result"]["nodeId"]
    cdp(ws, "DOM.setFileInputFiles", {"nodeId": node_id, "files": [str(path)]})


def screenshot(ws, path):
    shot = cdp(ws, "Page.captureScreenshot", {"format": "png", "fromSurface": True})
    path.write_bytes(base64.b64decode(shot["result"]["data"]))


def record(ws, out_dir, seconds, fps, start_index=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = max(1, round(seconds * fps))
    for i in range(frames):
        screenshot(ws, out_dir / f"frame_{start_index + i:05d}.png")
        time.sleep(1 / fps)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def record_scenario(ws, scenario, args, frame_dir):
    """Prepare one narrative and record only that narrative."""
    if scenario == "hazard":
        set_value(ws, "#node-select", "reactor_2")
        set_value(ws, "#hazard-type", "FIRE")
        set_value(ws, "#intensity-slider", "0.85")
        click(ws, "#btn-inject")
        click(ws, "#btn-play")
        record(ws, frame_dir, args.seconds, args.fps)
    elif scenario == "congestion":
        set_value(ws, "#evacuee-count", "300")
        click(ws, "#btn-reset")
        time.sleep(1)
        set_value(ws, "#node-select", "pipe_rack_junc_1")
        set_value(ws, "#hazard-type", "GAS_RELEASE")
        set_value(ws, "#intensity-slider", "0.75")
        click(ws, "#btn-inject")
        click(ws, "#btn-play")
        record(ws, frame_dir, args.seconds, args.fps)
    elif scenario == "perception":
        click(ws, '[data-screen-target="perception"]')
        time.sleep(1)
        set_value(ws, "#camera-node-select", "compressor_shed")
        set_file(ws, "#camera-video-input", args.feed.resolve())
        click(ws, "#btn-camera-upload")
        time.sleep(2)
        record(ws, frame_dir, args.seconds, args.fps)
    elif scenario == "layouts":
        # Keep the map visible while cycling through the three real topologies.
        for layout in ("petrochemical", "hexagonal_cell", "u_shaped_hall"):
            set_value(ws, "#building-layout-select", layout)
            time.sleep(2)
            record(ws, frame_dir, max(1, args.seconds / 3), args.fps,
                   len(list(frame_dir.glob("frame_*.png"))))
    elif scenario == "tour":
        click(ws, "#btn-tour")
        step_seconds = max(1, args.seconds / 5)
        for step in range(5):
            record(ws, frame_dir, step_seconds, args.fps,
                   len(list(frame_dir.glob("frame_*.png"))))
            if step < 4:
                click(ws, ".tour-actions .primary")
                time.sleep(0.8)
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(f"Unknown scenario: {scenario}")


def record_one(scenario, args, output):
    frame_dir = output.parent / f".{output.stem}_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    user_data = Path(f"/tmp/lbp-ui-recording-{scenario}")
    if user_data.exists():
        shutil.rmtree(user_data)
    debug_port = free_port()
    browser = subprocess.Popen([
        "google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu",
        "--disable-dev-shm-usage", f"--remote-debugging-port={debug_port}",
        "--remote-debugging-address=127.0.0.1",
        "--remote-allow-origins=*", "--window-size=1440,900",
        f"--user-data-dir={user_data}", args.url,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        endpoint = None
        last_error = None
        for _ in range(120):
            try:
                tabs = requests.get(f"http://127.0.0.1:{debug_port}/json", timeout=1, proxies={"http": "", "https": ""}).json()
                endpoint = next(t["webSocketDebuggerUrl"] for t in tabs if t.get("type") == "page")
                break
            except Exception as exc:
                last_error = exc
                time.sleep(0.25)
        if not endpoint:
            raise RuntimeError(f"Chrome DevTools endpoint did not start on port {debug_port}: {last_error}")
        ws = create_connection(endpoint, timeout=30)
        cdp(ws, "Page.enable")
        cdp(ws, "Runtime.enable")
        time.sleep(2)
        record_scenario(ws, scenario, args, frame_dir)
        ws.close()
    finally:
        browser.terminate()
        browser.wait(timeout=10)

    frame_pattern = str(frame_dir / "frame_%05d.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
        "-i", frame_pattern, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-movflags", "+faststart", str(output),
    ], check=True)
    shutil.rmtree(frame_dir, ignore_errors=True)
    print(f"Recorded {scenario}: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("hazard", "congestion", "perception", "layouts", "tour", "all"), default="all")
    parser.add_argument("--output", type=Path, help="Output file for one scenario")
    parser.add_argument("--output-dir", type=Path, default=Path("assets/simulator_ui_demos"))
    parser.add_argument("--feed", type=Path, default=Path("demo_assets/pscs_i_crowd_demo_12s.mp4"))
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    if not args.feed.is_file() and args.scenario in ("perception", "all"):
        raise SystemExit(f"Feed not found: {args.feed}")
    scenarios = ["hazard", "congestion", "perception", "layouts", "tour"] if args.scenario == "all" else [args.scenario]
    for scenario in scenarios:
        output = args.output if args.output and len(scenarios) == 1 else args.output_dir / f"{scenario}.mp4"
        record_one(scenario, args, output)


if __name__ == "__main__":
    main()
