"""
Serverless endpoint to run the Python headless simulation and return stats.
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

from dwight import (
    run_headless_simulation,
    generate_building,
    spawn_people,
    TOTAL_PEOPLE,
    NUM_WARDENS,
)


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


class handler(BaseHTTPRequestHandler):
    """Vercel serverless handler for headless simulation."""

    def _set_headers(self, status: int = 200) -> None:
        """Set response headers."""
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET request - run headless simulation."""
        try:
            # Parse query parameters
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            steps = int(qs.get("steps", ["240"])[0])
            dt = float(qs.get("dt", ["0.0333"])[0])

            # Basic safety limits
            steps = int(_clamp(steps, 1, 1500))
            dt = _clamp(dt, 1 / 120.0, 0.2)

            # Run simulation (returns stats summary)
            stats = run_headless_simulation(steps=steps, dt=dt)

            # Also return a snapshot of maze/exits/people for frontend integration
            maze, exits = generate_building()
            people = spawn_people(maze, TOTAL_PEOPLE, NUM_WARDENS)
            people_payload = [{
                "id": p.id,
                "row": p.row,
                "col": p.col,
                "state": getattr(p, "state", "working"),
                "isWarden": getattr(p, "is_warden", False),
                "health": getattr(p, "health", 100),
                "alive": getattr(p, "alive", True),
                "escaped": getattr(p, "escaped", False),
            } for p in people]
            exits_list = [[r, c] for (r, c) in exits]

            self._set_headers(200)
            self.wfile.write(json.dumps({
                "success": True,
                "stats": stats,
                "maze": maze,
                "exits": exits_list,
                "people": people_payload,
                "sensors": [],
            }).encode())

        except Exception as exc:
            self._set_headers(500)
            self.wfile.write(json.dumps({
                "success": False,
                "error": str(exc),
            }).encode())
