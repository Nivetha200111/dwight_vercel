"""
Serverless endpoint to run the Python headless simulation and return stats.
"""

import json
import os
import sys
from urllib.parse import urlparse, parse_qs

# Ensure parent is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enforce headless pygame
os.environ.setdefault("HEADLESS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from http.server import BaseHTTPRequestHandler  # noqa: E402
from dwight import run_headless_simulation  # noqa: E402


def _clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


class handler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
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
            steps = int(qs.get("steps", ["240"])[0])
            dt = float(qs.get("dt", ["0.0333"])[0])

            # Basic safety limits
            steps = _clamp(steps, 1, 1500)
            dt = _clamp(dt, 1 / 120.0, 0.2)

            stats = run_headless_simulation(steps=steps, dt=dt)

            self._set_headers(200)
            self.wfile.write(json.dumps({"success": True, "stats": stats}).encode())
        except Exception as exc:  # pylint: disable=broad-except
            self._set_headers(500)
            self.wfile.write(json.dumps({
                "success": False,
                "error": str(exc)
            }).encode())
