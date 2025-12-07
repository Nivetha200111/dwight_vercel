"""
DWIGHT Configuration Module

Centralized configuration for the Neural ACO Emergency Response System.
All magic numbers and constants are defined here for easy tuning.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Tuple, List

# Environment detection
HEADLESS = bool(
    os.environ.get("HEADLESS")
    or os.environ.get("VERCEL")
    or os.environ.get("CI")
)

if HEADLESS:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


@dataclass(frozen=True)
class GridConfig:
    """Grid and display configuration."""
    rows: int = 45
    cols: int = 70
    tile_size: int = 14

    @property
    def map_width(self) -> int:
        return self.cols * self.tile_size

    @property
    def map_height(self) -> int:
        return self.rows * self.tile_size


@dataclass(frozen=True)
class SimulationConfig:
    """Simulation parameters."""
    total_people: int = 60
    num_wardens: int = 4
    num_sensors: int = 25

    # Timing
    neural_update_interval: float = 0.5  # seconds
    rl_update_interval: float = 2.0  # seconds

    # Fire behavior
    fire_spread_delay: float = 5.0  # seconds before fire can spread
    fire_spread_chance: float = 0.01  # probability per frame
    fire_max_age: float = 80.0  # seconds before fire dies

    # Smoke behavior
    smoke_decay_rate: float = 0.97
    smoke_min_threshold: float = 0.02


@dataclass(frozen=True)
class AIConfig:
    """AI/ML parameters."""
    # LSTM
    lstm_hidden_size: int = 32
    lstm_sequence_length: int = 10
    lstm_prediction_threshold: float = 0.3
    lstm_prediction_steps: int = 3

    # ACO
    aco_base_evaporation: float = 0.02
    aco_base_deposit: float = 1.0
    aco_alpha: float = 1.0  # Pheromone importance
    aco_beta: float = 2.0   # Heuristic importance
    aco_modulation_strength: float = 2.0
    aco_safe_pheromone_max: float = 10.0
    aco_danger_pheromone_max: float = 20.0

    # RL
    rl_learning_rate: float = 0.1
    rl_discount: float = 0.95
    rl_epsilon: float = 0.2
    rl_num_actions: int = 8


@dataclass(frozen=True)
class SensorConfig:
    """IoT sensor parameters."""
    # Thresholds
    temperature_threshold: float = 45.0  # Celsius
    smoke_threshold: float = 0.3
    co_threshold: float = 35.0  # ppm
    motion_threshold: float = 0.5

    # Initial values
    room_temperature: float = 22.0

    # Network behavior
    network_latency: float = 0.1  # seconds
    packet_loss_rate: float = 0.02
    noise_level: float = 0.05

    # Battery/health
    battery_drain_rate: float = 0.001
    fire_damage_rate: float = 5.0


@dataclass(frozen=True)
class PersonConfig:
    """Person/agent parameters."""
    base_speed_min: float = 80.0
    base_speed_max: float = 110.0
    warden_speed_multiplier: float = 1.2
    panic_speed_multiplier: float = 1.3

    # Health
    fire_damage_rate: float = 30.0
    smoke_damage_multiplier: float = 10.0

    # Awareness
    awareness_threshold: float = 0.7
    awareness_gain_rate: float = 0.5

    # Stuck timer
    stuck_threshold: float = 3.0


@dataclass(frozen=True)
class DisplayConfig:
    """Display and UI configuration."""
    panel_width: int = 380
    bottom_bar_height: int = 80

    @property
    def screen_width(self) -> int:
        return GRID.map_width + self.panel_width

    @property
    def screen_height(self) -> int:
        return GRID.map_height + self.bottom_bar_height


# Tile type constants
class TileType:
    FLOOR = 0
    WALL = 1
    EXIT = 2
    DOOR = 3
    CORRIDOR = 4
    CARPET = 5


# Person state constants
class PersonState:
    WORKING = "working"
    HEADPHONES = "headphones"  # Legacy
    AWARE = "aware"
    EVACUATING = "evacuating"
    PANICKING = "panicking"
    WARDEN = "warden"


# Sensor type constants
class SensorType:
    TEMPERATURE = "temperature"
    SMOKE = "smoke"
    CO = "co"
    MOTION = "motion"


# Hazard type constants
class HazardType:
    FIRE = "fire"


# Stats keys
class StatsKey:
    ESCAPED = "escaped"
    DEATHS = "deaths"
    TOTAL = "total"
    STEPS_RUN = "steps_run"

    # RL stats
    REWARD = "reward"
    MESSAGE = "message"
    ACTION = "action"
    DECISIONS = "decisions"
    AVG_REWARD = "avg_reward"
    EPSILON = "epsilon"
    EPISODES = "episodes"

    # Sensor stats
    TEMPERATURE_AVG = "temperature_avg"
    TEMPERATURE_MAX = "temperature_max"
    SMOKE_LEVEL = "smoke_level"
    CO_LEVEL = "co_level"
    MOTION_DETECTED = "motion_detected"
    TRIGGERED_SENSORS = "triggered_sensors"
    SENSOR_HEALTH = "sensor_health"
    COVERAGE = "coverage"

    # Hazard info
    TYPE = "type"
    AGE = "age"
    INTENSITY = "intensity"


class Colors:
    """Color palette (Minecraft-inspired)."""
    # Environment
    FLOOR = (180, 175, 165)
    FLOOR_ALT = (170, 165, 155)
    WALL = (45, 48, 55)
    WALL_HIGHLIGHT = (65, 68, 75)
    CORRIDOR = (155, 150, 145)
    CARPET = (95, 65, 65)
    EXIT = (50, 255, 100)
    DOOR = (60, 200, 120)

    # Hazards
    FIRE = (255, 100, 30)
    FIRE_BRIGHT = (255, 220, 80)
    FIRE_CORE = (255, 255, 200)
    SMOKE = (70, 70, 75)
    WATER = (40, 120, 200)

    # Neural/Tech
    NEURAL_GLOW = (0, 255, 200)
    PREDICTION = (255, 50, 200)
    SENSOR_ACTIVE = (0, 200, 255)
    IOT_MESH = (100, 255, 200)

    # Pheromones
    SAFE_PHEROMONE = (0, 255, 150)
    DANGER_PHEROMONE = (255, 80, 80)

    # People states
    NORMAL = (100, 150, 255)
    AWARE = (255, 255, 100)
    EVACUATING = (100, 255, 100)
    PANICKING = (255, 80, 80)
    WARDEN = (255, 215, 0)
    HEADPHONES = (255, 100, 255)

    # UI
    PANEL_BG = (18, 20, 28)
    PANEL_BORDER = (45, 50, 65)
    ACCENT = (0, 230, 180)
    SUCCESS = (60, 255, 120)
    DANGER = (255, 70, 70)
    WARNING = (255, 200, 60)
    TEXT = (255, 255, 255)
    TEXT_DIM = (160, 160, 170)


# Global config instances
GRID = GridConfig()
SIMULATION = SimulationConfig()
AI = AIConfig()
SENSOR = SensorConfig()
PERSON = PersonConfig()
DISPLAY = DisplayConfig()

# Convenience aliases for backward compatibility
ROWS = GRID.rows
COLS = GRID.cols
TILE = GRID.tile_size
TOTAL_PEOPLE = SIMULATION.total_people
NUM_WARDENS = SIMULATION.num_wardens
NUM_SENSORS = SIMULATION.num_sensors
MAP_WIDTH = GRID.map_width
MAP_HEIGHT = GRID.map_height
PANEL_WIDTH = DISPLAY.panel_width
SCREEN_WIDTH = DISPLAY.screen_width
SCREEN_HEIGHT = DISPLAY.screen_height

# Tile type shortcuts
FLOOR = TileType.FLOOR
WALL = TileType.WALL
EXIT = TileType.EXIT
DOOR = TileType.DOOR
CORRIDOR = TileType.CORRIDOR
CARPET = TileType.CARPET
