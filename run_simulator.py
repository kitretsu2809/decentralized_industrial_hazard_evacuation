#!/usr/bin/env python3
"""
Industrial Evacuation Simulator — Entry Point
==============================================
Launches the FastAPI + WebSocket server serving the evacuation simulation.

Usage:
    python run_simulator.py [--port 8080] [--host 0.0.0.0]

Then open: http://localhost:8080
"""
import sys
import os
import argparse

# Ensure repo root is on path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    parser = argparse.ArgumentParser(description="Industrial Evacuation Simulator")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Industrial Evacuation Simulator")
    print("  Phase 1: Classical Dijkstra Routing + Social Force Model")
    print("=" * 60)
    print(f"\n  ▶  Open browser:  http://{args.host}:{args.port}")
    print(f"     Ctrl+C to stop\n")

    import uvicorn
    uvicorn.run(
        "simulator.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
