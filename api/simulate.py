"""
Vercel Serverless Function for DWIGHT Neural ACO Emergency Simulation

Returns simulation state data for the Minecraft-themed frontend.
"""

from __future__ import annotations

import json
import os
import sys
import random
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any, Tuple

# Ensure we can import from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set headless mode before importing dwight
os.environ["HEADLESS"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from dwight.config import (
    ROWS, COLS,
    SIMULATION,
    TileType, SensorType,
)
from dwight.core.building import generate_building


def generate_people(
    maze: List[List[int]],
    count: int,
    num_wardens: int
) -> List[Dict[str, Any]]:
    """Generate people positions for the frontend."""
    people: List[Dict[str, Any]] = []
    spawns: List[Tuple[int, int]] = []

    for r in range(2, ROWS - 2):
        for c in range(2, COLS - 2):
            if maze[r][c] in [TileType.CARPET, TileType.FLOOR, TileType.CORRIDOR]:
                spawns.append((r, c))

    random.shuffle(spawns)
    states = ["working", "headphones", "aware", "evacuating"]

    corridor_spawns = [(r, c) for r, c in spawns if maze[r][c] == TileType.CORRIDOR]

    # Wardens
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
            "escaped": False,
        })

    # Civilians
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
            "escaped": False,
        })

    return people


def generate_sensors(maze: List[List[int]], count: int) -> List[Dict[str, Any]]:
    """Generate sensor positions for the frontend."""
    sensors: List[Dict[str, Any]] = []
    spawns: List[Tuple[int, int]] = []

    for r in range(2, ROWS - 2):
        for c in range(2, COLS - 2):
            if maze[r][c] in [TileType.FLOOR, TileType.CARPET, TileType.CORRIDOR, TileType.DOOR]:
                spawns.append((r, c))

    sensor_types = [
        SensorType.TEMPERATURE,
        SensorType.SMOKE,
        SensorType.CO,
        SensorType.MOTION,
    ]

    # Use bucket-based distribution
    buckets_r = 5
    buckets_c = 5
    bucket_h = ROWS // buckets_r
    bucket_w = COLS // buckets_c

    sensor_id = 0
    leftover: List[Tuple[int, int]] = []

    for br in range(buckets_r):
        for bc in range(buckets_c):
            if sensor_id >= count:
                break

            r_min = br * bucket_h
            r_max = ROWS if br == buckets_r - 1 else (br + 1) * bucket_h
            c_min = bc * bucket_w
            c_max = COLS if bc == buckets_c - 1 else (bc + 1) * bucket_w

            candidates = [
                (r, c) for (r, c) in spawns
                if r_min <= r < r_max and c_min <= c < c_max
                and maze[r][c] != TileType.EXIT
            ]

            if not candidates:
                continue

            pick = random.choice(candidates)
            sensor_type = sensor_types[sensor_id % len(sensor_types)]

            sensors.append({
                "id": sensor_id,
                "row": pick[0],
                "col": pick[1],
                "type": sensor_type,
                "value": random.uniform(20, 25) if sensor_type == SensorType.TEMPERATURE else 0,
                "triggered": False,
                "health": 100,
            })
            sensor_id += 1
            leftover.extend(pos for pos in candidates if pos != pick)

    # Fill remaining with leftover positions
    random.shuffle(leftover)
    for r, c in leftover:
        if sensor_id >= count:
            break
        sensor_type = sensor_types[sensor_id % len(sensor_types)]
        sensors.append({
            "id": sensor_id,
            "row": r,
            "col": c,
            "type": sensor_type,
            "value": random.uniform(20, 25) if sensor_type == SensorType.TEMPERATURE else 0,
            "triggered": False,
            "health": 100,
        })
        sensor_id += 1

    return sensors


class handler(BaseHTTPRequestHandler):
    """Vercel serverless handler for initial state generation."""

    def do_GET(self) -> None:
        """Handle GET request - return initial simulation state."""
        try:
            random.seed()
            maze, exits = generate_building()
            people = generate_people(maze, SIMULATION.total_people, SIMULATION.num_wardens)
            sensors = generate_sensors(maze, SIMULATION.num_sensors)

            # Convert exits to list format
            exits_list = [[r, c] for r, c in exits]

            response_data = {
                "success": True,
                "config": {
                    "rows": ROWS,
                    "cols": COLS,
                    "totalPeople": SIMULATION.total_people,
                    "numWardens": SIMULATION.num_wardens,
                    "numSensors": SIMULATION.num_sensors,
                },
                "maze": maze,
                "exits": exits_list,
                "people": people,
                "sensors": sensors,
                "fires": [],
                "smoke": {},
                "stats": {
                    "escaped": 0,
                    "deaths": 0,
                    "total": SIMULATION.total_people,
                    "alarmActive": False,
                },
                "neural": {
                    "confidence": 0.0,
                    "predictions": [],
                    "safePheromone": 0.1,
                    "dangerPheromone": 0.0,
                },
                "rl": {
                    "decisions": 0,
                    "avgReward": 0.0,
                    "epsilon": 0.2,
                },
            }

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": False,
                "error": str(e),
            }).encode())

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
