"""
DWIGHT - Neural ACO Emergency Response System

A sophisticated emergency evacuation simulator combining:
- Ant Colony Optimization for distributed pathfinding
- LSTM neural networks for hazard prediction
- IoT sensor fusion for real-time environmental awareness
- Reinforcement learning for dynamic resource allocation

Package Structure:
    dwight/
    ├── config.py          # Configuration and constants
    ├── ai/                # AI/ML components
    │   ├── lstm_predictor.py   # Fire spread prediction
    │   ├── neural_aco.py       # Neural-enhanced ACO
    │   ├── pathfinder.py       # A* with pheromone costs
    │   └── rl_coordinator.py   # Q-learning coordinator
    ├── sensors/           # IoT sensor simulation
    │   ├── iot_sensor.py       # Individual sensor
    │   └── sensor_network.py   # Network with Kalman filtering
    ├── core/              # Core simulation
    │   ├── person.py           # Person agent
    │   ├── disasters.py        # Hazard management
    │   ├── alarm.py            # Alarm system
    │   ├── building.py         # Building generation
    │   └── spawner.py          # Entity spawning
    └── rendering/         # Pygame rendering (optional)

Usage:
    from dwight import run_headless_simulation
    stats = run_headless_simulation(steps=240, dt=1/30)
"""

from dwight.config import (
    # Grid config
    ROWS, COLS, TILE,
    MAP_WIDTH, MAP_HEIGHT,
    PANEL_WIDTH, SCREEN_WIDTH, SCREEN_HEIGHT,

    # Simulation config
    TOTAL_PEOPLE, NUM_WARDENS, NUM_SENSORS,

    # Type constants
    TileType, PersonState, SensorType, HazardType,
    StatsKey, Colors,

    # Config dataclasses
    GRID, SIMULATION, AI, SENSOR, PERSON, DISPLAY,

    # Environment
    HEADLESS,
)

from dwight.ai import (
    SimpleLSTMPredictor,
    NeuralACO,
    NeuralPathfinder,
    RLEvacuationCoordinator,
)

from dwight.sensors import (
    IoTSensor,
    IoTSensorNetwork,
)

from dwight.core import (
    Person,
    Disasters,
    AlarmSystem,
    generate_building,
    spawn_people,
    spawn_sensors,
)

from dwight.simulation import run_headless_simulation

__version__ = "2.0.0"
__author__ = "DWIGHT Team"

__all__ = [
    # Version
    "__version__",

    # Config
    "ROWS", "COLS", "TILE",
    "MAP_WIDTH", "MAP_HEIGHT",
    "PANEL_WIDTH", "SCREEN_WIDTH", "SCREEN_HEIGHT",
    "TOTAL_PEOPLE", "NUM_WARDENS", "NUM_SENSORS",
    "HEADLESS",

    # Type constants
    "TileType", "PersonState", "SensorType", "HazardType",
    "StatsKey", "Colors",

    # Config objects
    "GRID", "SIMULATION", "AI", "SENSOR", "PERSON", "DISPLAY",

    # AI
    "SimpleLSTMPredictor",
    "NeuralACO",
    "NeuralPathfinder",
    "RLEvacuationCoordinator",

    # Sensors
    "IoTSensor",
    "IoTSensorNetwork",

    # Core
    "Person",
    "Disasters",
    "AlarmSystem",
    "generate_building",
    "spawn_people",
    "spawn_sensors",

    # Simulation
    "run_headless_simulation",
]
