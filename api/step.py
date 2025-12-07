"""
Vercel Serverless Function to advance the Python headless simulation and return full state.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Ensure parent is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enforce headless mode
os.environ.setdefault("HEADLESS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from dwight import run_headless_step
from api.state_manager import get_state, reset_state


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


class handler(BaseHTTPRequestHandler):
    def _set_headers(self, status: int = 200):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            steps = int(qs.get("steps", ["120"])[0])
            dt = float(qs.get("dt", ["0.0333"])[0])
            reset = qs.get("reset", ["0"])[0] == "1"

            steps = int(_clamp(steps, 1, 1500))
            dt = _clamp(dt, 1 / 120.0, 0.2)

            state = reset_state() if reset else get_state()
            snapshot = run_headless_step(state, steps=steps, dt=dt)

            self._set_headers(200)
            self.wfile.write(json.dumps(snapshot).encode())

        except Exception as exc:
            self._set_headers(500)
            self.wfile.write(json.dumps({"success": False, "error": str(exc)}).encode())
