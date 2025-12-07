"""
Entity Spawner

Functions for spawning people and sensors in the building.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from dwight.config import (
    ROWS, COLS, SIMULATION,
    TileType, PersonState, SensorType,
)
from dwight.core.person import Person
from dwight.sensors.iot_sensor import IoTSensor
from dwight.sensors.sensor_network import IoTSensorNetwork


def spawn_people(
    maze: List[List[int]],
    count: int = SIMULATION.total_people,
    num_wardens: int = SIMULATION.num_wardens
) -> List[Person]:
    """
    Spawn people (civilians and wardens) in the building.

    Args:
        maze: Building layout grid
        count: Total number of people to spawn
        num_wardens: Number of wardens to spawn

    Returns:
        List of Person objects
    """
    people: List[Person] = []
    spawns: List[Tuple[int, int]] = []

    # Find valid spawn positions
    for r in range(2, ROWS - 2):
        for c in range(2, COLS - 2):
            if maze[r][c] in [TileType.CARPET, TileType.FLOOR, TileType.CORRIDOR]:
                spawns.append((r, c))

    random.shuffle(spawns)

    # Prepare states for civilians
    states = [PersonState.WORKING] * 15 + [PersonState.HEADPHONES] * 8
    random.shuffle(states)

    # Spawn wardens in corridors
    corridor_spawns = [
        (r, c) for r, c in spawns if maze[r][c] == TileType.CORRIDOR
    ]

    for i in range(min(num_wardens, len(corridor_spawns))):
        r, c = corridor_spawns[i]
        people.append(Person(i, r, c, PersonState.WARDEN, is_warden=True))

    # Spawn civilians
    for i in range(num_wardens, min(count, len(spawns))):
        r, c = spawns[i]
        state = states[(i - num_wardens) % len(states)]
        people.append(Person(i, r, c, state))

    return people


def spawn_sensors(
    maze: List[List[int]],
    count: int = SIMULATION.num_sensors
) -> IoTSensorNetwork:
    """
    Spawn IoT sensors throughout the building.

    Sensors are distributed evenly across the building using a grid-based
    bucketing approach.

    Args:
        maze: Building layout grid
        count: Number of sensors to spawn

    Returns:
        IoTSensorNetwork with spawned sensors
    """
    network = IoTSensorNetwork()
    spawns: List[Tuple[int, int]] = []

    # Find valid spawn positions (every 4th tile for better distribution)
    for r in range(3, ROWS - 3, 4):
        for c in range(3, COLS - 3, 4):
            if maze[r][c] != TileType.WALL:
                spawns.append((r, c))

    random.shuffle(spawns)

    sensor_types = [
        SensorType.TEMPERATURE,
        SensorType.SMOKE,
        SensorType.CO,
        SensorType.MOTION,
    ]

    for i, (r, c) in enumerate(spawns[:count]):
        sensor_type = sensor_types[i % len(sensor_types)]
        sensor = IoTSensor(
            id=i,
            row=r,
            col=c,
            sensor_type=sensor_type,
        )
        network.add_sensor(sensor)

    return network
