"""
Serverless endpoint to mutate the persistent simulation state (add hazards, trigger alarm, reset).
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

# Ensure parent is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HEADLESS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from api.state_manager import get_state, reset_state
from dwight import run_headless_step


class handler(BaseHTTPRequestHandler):
    def _set_headers(self, status: int = 200):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8') if length > 0 else '{}'
            payload = json.loads(body or '{}')

            action = payload.get('action')
            row = int(payload.get('row', -1))
            col = int(payload.get('col', -1))
            reset = bool(payload.get('reset', False))

            if reset:
                state = reset_state()
                snapshot = run_headless_step(state, steps=1, dt=0.033)
                self._set_headers(200)
                self.wfile.write(json.dumps({'success': True, 'reset': True, 'snapshot': snapshot}).encode())
                return

            state = get_state()
            disasters = state['disasters']
            alarm = state['alarm']

            if action == 'add_fire':
                if 0 <= row < len(state['maze']) and 0 <= col < len(state['maze'][0]):
                    disasters.add_fire(row, col)
                    snapshot = run_headless_step(state, steps=5, dt=0.033)
                    self._set_headers(200)
                    self.wfile.write(json.dumps({'success': True, 'snapshot': snapshot}).encode())
                    return

            if action == 'add_bomb':
                if 0 <= row < len(state['maze']) and 0 <= col < len(state['maze'][0]):
                    disasters.add_bomb(row, col)
                    snapshot = run_headless_step(state, steps=5, dt=0.033)
                    self._set_headers(200)
                    self.wfile.write(json.dumps({'success': True, 'snapshot': snapshot}).encode())
                    return

            if action == 'add_flood':
                if 0 <= row < len(state['maze']) and 0 <= col < len(state['maze'][0]):
                    disasters.add_flood(row, col)
                    snapshot = run_headless_step(state, steps=5, dt=0.033)
                    self._set_headers(200)
                    self.wfile.write(json.dumps({'success': True, 'snapshot': snapshot}).encode())
                    return

            if action == 'add_quake':
                if 0 <= row < len(state['maze']) and 0 <= col < len(state['maze'][0]):
                    disasters.trigger_quake(row, col)
                    snapshot = run_headless_step(state, steps=5, dt=0.033)
                    self._set_headers(200)
                    self.wfile.write(json.dumps({'success': True, 'snapshot': snapshot}).encode())
                    return

            if action == 'trigger_alarm':
                alarm.trigger()
                snapshot = run_headless_step(state, steps=2, dt=0.033)
                self._set_headers(200)
                self.wfile.write(json.dumps({'success': True, 'alarm': True, 'snapshot': snapshot}).encode())
                return

            # Unsupported action
            self._set_headers(400)
            self.wfile.write(json.dumps({'success': False, 'error': 'Unsupported action'}).encode())

        except Exception as exc:
            self._set_headers(500)
            self.wfile.write(json.dumps({'success': False, 'error': str(exc)}).encode())
