"""
Vercel Serverless Function for DWIGHT Neural ACO Emergency Simulation
Returns simulation state data for the Minecraft-themed frontend
"""

import json
import os
import sys

# Ensure we can import from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set headless mode before importing dwight
os.environ["HEADLESS"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from http.server import BaseHTTPRequestHandler
import random
import math
import numpy as np
from collections import defaultdict

# Constants matching dwight.py
ROWS = 45
COLS = 70
TOTAL_PEOPLE = 60
NUM_WARDENS = 4
NUM_SENSORS = 25

# Tile types
FLOOR = 0
WALL = 1
EXIT = 2
DOOR = 3
CORRIDOR = 4
CARPET = 5

def generate_building():
    """Generate building layout."""
    maze = [[FLOOR for _ in range(COLS)] for _ in range(ROWS)]
    exits = []

    for r in range(ROWS):
        maze[r][0] = WALL
        maze[r][COLS-1] = WALL
    for c in range(COLS):
        maze[0][c] = WALL
        maze[ROWS-1][c] = WALL

    h_corr = [ROWS // 3, 2 * ROWS // 3]
    v_corr = [COLS // 4, COLS // 2, 3 * COLS // 4]

    for hr in h_corr:
        for c in range(1, COLS - 1):
            for r in range(hr - 1, hr + 2):
                if 0 < r < ROWS - 1:
                    maze[r][c] = CORRIDOR

    for vc in v_corr:
        for r in range(1, ROWS - 1):
            for c in range(vc - 1, vc + 2):
                if 0 < c < COLS - 1:
                    maze[r][c] = CORRIDOR

    def make_room(r1, r2, c1, c2):
        for r in range(r1, r2 + 1):
            if maze[r][c1] != CORRIDOR: maze[r][c1] = WALL
            if maze[r][c2] != CORRIDOR: maze[r][c2] = WALL
        for c in range(c1, c2 + 1):
            if maze[r1][c] != CORRIDOR: maze[r1][c] = WALL
            if maze[r2][c] != CORRIDOR: maze[r2][c] = WALL

        for r in range(r1 + 1, r2):
            for c in range(c1 + 1, c2):
                if maze[r][c] != CORRIDOR:
                    maze[r][c] = CARPET

        for c in range(c1 + 1, c2):
            if r2 + 1 < ROWS and maze[r2 + 1][c] == CORRIDOR:
                maze[r2][c] = DOOR
                return
            if r1 - 1 > 0 and maze[r1 - 1][c] == CORRIDOR:
                maze[r1][c] = DOOR
                return

    sections = [
        (2, h_corr[0] - 2, 2, v_corr[0] - 2),
        (2, h_corr[0] - 2, v_corr[0] + 2, v_corr[1] - 2),
        (2, h_corr[0] - 2, v_corr[1] + 2, v_corr[2] - 2),
        (2, h_corr[0] - 2, v_corr[2] + 2, COLS - 3),
        (h_corr[0] + 2, h_corr[1] - 2, 2, v_corr[0] - 2),
        (h_corr[0] + 2, h_corr[1] - 2, v_corr[2] + 2, COLS - 3),
        (h_corr[1] + 2, ROWS - 3, 2, v_corr[0] - 2),
        (h_corr[1] + 2, ROWS - 3, v_corr[0] + 2, v_corr[1] - 2),
        (h_corr[1] + 2, ROWS - 3, v_corr[1] + 2, v_corr[2] - 2),
        (h_corr[1] + 2, ROWS - 3, v_corr[2] + 2, COLS - 3),
    ]

    for r1, r2, c1, c2 in sections:
        if r2 - r1 > 3 and c2 - c1 > 3:
            make_room(r1, r2, c1, c2)

    exit_pos = [
        (h_corr[0], 1), (h_corr[1], 1),
        (h_corr[0], COLS - 2), (h_corr[1], COLS - 2),
        (1, v_corr[0]), (1, v_corr[1]), (1, v_corr[2]),
        (ROWS - 2, v_corr[0]), (ROWS - 2, v_corr[1]), (ROWS - 2, v_corr[2]),
    ]

    for er, ec in exit_pos:
        if 0 < er < ROWS - 1 and 0 < ec < COLS - 1:
            maze[er][ec] = EXIT
            exits.append([er, ec])
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = er + dr, ec + dc
                    if 0 < nr < ROWS - 1 and 0 < nc < COLS - 1:
                        if maze[nr][nc] == WALL:
                            maze[nr][nc] = CORRIDOR

    return maze, exits

def generate_people(maze, count, num_wardens):
    """Generate people positions."""
    people = []
    spawns = []

    for r in range(2, ROWS - 2):
        for c in range(2, COLS - 2):
            if maze[r][c] in [CARPET, FLOOR, CORRIDOR]:
                spawns.append((r, c))

    random.shuffle(spawns)
    states = ["working", "headphones", "aware", "evacuating"]

    corridor_spawns = [(r, c) for r, c in spawns if maze[r][c] == CORRIDOR]

    for i in range(min(num_wardens, len(corridor_spawns))):
        r, c = corridor_spawns[i]
        people.append({
            "id": i,
            "row": r,
            "col": c,
            "state": "warden",
            "isWarden": True,
            "health": 100,
            "alive": True,
            "escaped": False
        })

    for i in range(num_wardens, min(count, len(spawns))):
        r, c = spawns[i]
        state = random.choice(states[:2])
        people.append({
            "id": i,
            "row": r,
            "col": c,
            "state": state,
            "isWarden": False,
            "health": 100,
            "alive": True,
            "escaped": False
        })

    return people

def generate_sensors(maze, count):
    """Generate sensor positions."""
    sensors = []
    spawns = []

    for r in range(2, ROWS - 2):
        for c in range(2, COLS - 2):
            if maze[r][c] in [FLOOR, CARPET, CORRIDOR, DOOR]:
                spawns.append((r, c))

    random.shuffle(spawns)
    sensor_types = ['temperature', 'smoke', 'co', 'motion']

    for i, (r, c) in enumerate(spawns[:count]):
        sensors.append({
            "id": i,
            "row": r,
            "col": c,
            "type": sensor_types[i % len(sensor_types)],
            "value": random.uniform(20, 25) if sensor_types[i % len(sensor_types)] == 'temperature' else 0,
            "triggered": False,
            "health": 100
        })

    return sensors

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET request - return initial simulation state."""
        try:
            random.seed()
            maze, exits = generate_building()
            people = generate_people(maze, TOTAL_PEOPLE, NUM_WARDENS)
            sensors = generate_sensors(maze, NUM_SENSORS)

            response_data = {
                "success": True,
                "config": {
                    "rows": ROWS,
                    "cols": COLS,
                    "totalPeople": TOTAL_PEOPLE,
                    "numWardens": NUM_WARDENS,
                    "numSensors": NUM_SENSORS
                },
                "maze": maze,
                "exits": exits,
                "people": people,
                "sensors": sensors,
                "fires": [],
                "smoke": {},
                "stats": {
                    "escaped": 0,
                    "deaths": 0,
                    "total": TOTAL_PEOPLE,
                    "alarmActive": False
                },
                "neural": {
                    "confidence": 0.0,
                    "predictions": [],
                    "safePheromone": 0.1,
                    "dangerPheromone": 0.0
                },
                "rl": {
                    "decisions": 0,
                    "avgReward": 0.0,
                    "epsilon": 0.2
                }
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": False,
                "error": str(e)
            }).encode())

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
