"""
Disasters Module

Handles fire, smoke, and other hazards in the simulation.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any, TYPE_CHECKING

from dwight.config import (
    ROWS, COLS, TILE, SIMULATION,
    TileType, HazardType, StatsKey,
)

if TYPE_CHECKING:
    from dwight.ai.neural_aco import NeuralACO


class Disasters:
    """
    Manages all disaster events and hazards.

    Handles:
    - Fire spread and lifecycle
    - Smoke generation and decay
    - Camera shake effects
    - Particle systems for visual effects
    """

    def __init__(self) -> None:
        # Active hazards: position -> hazard info
        self.hazards: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # Smoke density map
        self.smoke: Dict[Tuple[int, int], float] = defaultdict(float)

        # Camera shake
        self.shake: float = 0.0
        self.shake_offset: Tuple[float, float] = (0.0, 0.0)

        # Visual particles
        self.particles: List[Dict[str, Any]] = []

    def add_fire(self, row: int, col: int) -> None:
        """Add fire at a position."""
        if (row, col) not in self.hazards:
            self.hazards[(row, col)] = {
                StatsKey.TYPE: HazardType.FIRE,
                StatsKey.AGE: 0.0,
                StatsKey.INTENSITY: 1.0,
            }

    def remove_fire(self, row: int, col: int) -> None:
        """Remove fire at a position."""
        if (row, col) in self.hazards:
            del self.hazards[(row, col)]

    def trigger_shake(self, intensity: float = 1.0) -> None:
        """Trigger camera shake effect."""
        self.shake = max(self.shake, intensity)

    def update(
        self,
        dt: float,
        maze: List[List[int]],
        neural_aco: NeuralACO
    ) -> None:
        """Update all disasters for one frame."""
        self._update_shake(dt)
        self._update_hazards(dt, maze, neural_aco)
        self._update_smoke(dt)
        self._update_particles(dt)

    def _update_shake(self, dt: float) -> None:
        """Update camera shake."""
        self.shake *= 0.9
        if self.shake > 0.01:
            self.shake_offset = (
                random.uniform(-1, 1) * self.shake * 6,
                random.uniform(-1, 1) * self.shake * 6,
            )
        else:
            self.shake_offset = (0.0, 0.0)

    def _update_hazards(
        self,
        dt: float,
        maze: List[List[int]],
        neural_aco: NeuralACO
    ) -> None:
        """Update all hazards."""
        new_hazards: Dict[Tuple[int, int], Dict[str, Any]] = {}
        to_remove: List[Tuple[int, int]] = []

        for (row, col), info in list(self.hazards.items()):
            info[StatsKey.AGE] += dt

            if info[StatsKey.TYPE] == HazardType.FIRE:
                self._update_fire(row, col, info, dt, maze, neural_aco, new_hazards, to_remove)

        # Remove dead fires
        for pos in to_remove:
            if pos in self.hazards:
                del self.hazards[pos]

        # Add new fires
        self.hazards.update(new_hazards)

    def _update_fire(
        self,
        row: int,
        col: int,
        info: Dict[str, Any],
        dt: float,
        maze: List[List[int]],
        neural_aco: NeuralACO,
        new_hazards: Dict[Tuple[int, int], Dict[str, Any]],
        to_remove: List[Tuple[int, int]]
    ) -> None:
        """Update a single fire hazard."""
        # Generate smoke
        for dr in range(-4, 5):
            for dc in range(-4, 5):
                sr, sc = row + dr, col + dc
                if 0 <= sr < ROWS and 0 <= sc < COLS:
                    dist = abs(dr) + abs(dc)
                    self.smoke[(sr, sc)] = min(
                        self.smoke[(sr, sc)] + 0.03 / (dist + 1),
                        1.5
                    )

        # Deposit danger pheromone
        neural_aco.deposit_danger_pheromone((row, col), 2.0 * dt)

        # Spawn fire particles
        if random.random() < 0.4:
            self.particles.append({
                "x": col * TILE + random.randint(2, TILE - 2),
                "y": row * TILE + TILE,
                "vy": -random.uniform(30, 60),
                "vx": random.uniform(-10, 10),
                "life": random.uniform(0.3, 0.7),
                "type": HazardType.FIRE,
            })

        # Fire spread
        if info[StatsKey.AGE] > SIMULATION.fire_spread_delay:
            if random.random() < SIMULATION.fire_spread_chance:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if 0 < nr < ROWS - 1 and 0 < nc < COLS - 1:
                        if maze[nr][nc] not in [TileType.WALL, TileType.EXIT]:
                            if (nr, nc) not in self.hazards:
                                new_hazards[(nr, nc)] = {
                                    StatsKey.TYPE: HazardType.FIRE,
                                    StatsKey.AGE: 0.0,
                                    StatsKey.INTENSITY: 0.8,
                                }
                                break

        # Fire death
        if info[StatsKey.AGE] > SIMULATION.fire_max_age:
            to_remove.append((row, col))

    def _update_smoke(self, dt: float) -> None:
        """Update smoke decay."""
        for key in list(self.smoke.keys()):
            self.smoke[key] *= SIMULATION.smoke_decay_rate
            if self.smoke[key] < SIMULATION.smoke_min_threshold:
                del self.smoke[key]

    def _update_particles(self, dt: float) -> None:
        """Update visual particles."""
        for p in self.particles[:]:
            p["y"] += p["vy"] * dt
            p["x"] += p.get("vx", 0) * dt
            p["life"] -= dt
            if p["life"] <= 0:
                self.particles.remove(p)

    def get_fire_positions(self) -> List[Tuple[int, int]]:
        """Get all current fire positions."""
        return [
            pos for pos, info in self.hazards.items()
            if info[StatsKey.TYPE] == HazardType.FIRE
        ]

    def get_hazard_count(self) -> int:
        """Get total number of active hazards."""
        return len(self.hazards)

    def has_fires(self) -> bool:
        """Check if there are any active fires."""
        return any(
            info[StatsKey.TYPE] == HazardType.FIRE
            for info in self.hazards.values()
        )

    def clear_all(self) -> None:
        """Clear all hazards and effects."""
        self.hazards.clear()
        self.smoke.clear()
        self.particles.clear()
        self.shake = 0.0
        self.shake_offset = (0.0, 0.0)
