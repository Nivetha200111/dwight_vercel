"""
Person Agent

Represents a person (civilian or warden) in the evacuation simulation.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING

from dwight.config import (
    ROWS, COLS, TILE, PERSON,
    TileType, PersonState, Colors,
)

if TYPE_CHECKING:
    from dwight.ai.pathfinder import NeuralPathfinder
    from dwight.ai.neural_aco import NeuralACO


class Person:
    """
    A person in the evacuation simulation.

    Can be either a civilian or a warden (emergency responder).
    Handles awareness, pathfinding, movement, and health.
    """

    # Pre-defined civilian colors
    CIVILIAN_COLORS = [
        (255, 90, 90),    # Red
        (90, 160, 255),   # Blue
        (90, 255, 90),    # Green
        (255, 230, 90),   # Yellow
        (90, 255, 255),   # Cyan
        (255, 160, 90),   # Orange
    ]

    def __init__(
        self,
        pid: int,
        row: int,
        col: int,
        state: str,
        is_warden: bool = False
    ) -> None:
        self.id = pid
        self.row = row
        self.col = col

        # Pixel position for smooth movement
        self.x = col * TILE + TILE // 2
        self.y = row * TILE + TILE // 2
        self.tx = self.x  # Target x
        self.ty = self.y  # Target y

        # State
        self.state = PersonState.WARDEN if is_warden else state
        self.is_warden = is_warden

        # Status
        self.alive = True
        self.escaped = False
        self.health = 100.0
        self.awareness = 1.0 if is_warden else 0.0

        # Pathfinding
        self.path: List[Tuple[int, int]] = []
        self.path_index = 0
        self.target_exit: Optional[Tuple[int, int]] = None

        # Visual
        self.color = Colors.WARDEN if is_warden else random.choice(self.CIVILIAN_COLORS)
        self.walk_frame = 0.0
        self.moving = False

        # Movement
        speed_range = (PERSON.base_speed_min, PERSON.base_speed_max)
        self.speed = random.uniform(*speed_range)
        if is_warden:
            self.speed *= PERSON.warden_speed_multiplier

        # RL coordination target
        self.rl_target: Optional[Tuple[int, int]] = None
        self.stuck_timer = 0.0

    @property
    def position(self) -> Tuple[int, int]:
        """Current grid position."""
        return (self.row, self.col)

    def find_exit(
        self,
        exits: List[Tuple[int, int]],
        pathfinder: NeuralPathfinder,
        maze: List[List[int]],
        hazards: Dict[Tuple[int, int], Dict]
    ) -> None:
        """Find the best exit and calculate path to it."""
        if not exits:
            return

        best_exit = None
        best_score = float("inf")

        for exit_pos in exits:
            er, ec = exit_pos
            dist = abs(self.row - er) + abs(self.col - ec)

            # Check danger near exit
            danger = 0
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    if (er + dr, ec + dc) in hazards:
                        danger += 50

            score = dist + danger
            if score < best_score:
                best_score = score
                best_exit = exit_pos

        if best_exit:
            self.target_exit = best_exit
            self.path = pathfinder.find_path(
                (self.row, self.col), best_exit, maze, hazards
            )
            self.path_index = 0

    def force_open_corridor_door(self, maze: List[List[int]]) -> bool:
        """
        Emergency: carve a door in the nearest wall that borders a corridor.

        Used when person is stuck and needs an escape route.
        """
        best: Optional[Tuple[int, int]] = None
        best_dist = float("inf")

        for r in range(ROWS):
            for c in range(COLS):
                if maze[r][c] != TileType.WALL:
                    continue

                # Check if any neighbor is a corridor
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                        if maze[nr][nc] == TileType.CORRIDOR:
                            dist = abs(self.row - r) + abs(self.col - c)
                            if dist < best_dist:
                                best_dist = dist
                                best = (r, c)
                            break

        if best:
            br, bc = best
            maze[br][bc] = TileType.DOOR
            return True
        return False

    def update(
        self,
        dt: float,
        maze: List[List[int]],
        exits: List[Tuple[int, int]],
        hazards: Dict[Tuple[int, int], Dict],
        pathfinder: NeuralPathfinder,
        alarm_active: bool,
        people: List[Person],
        smoke: Dict[Tuple[int, int], float],
        neural_aco: NeuralACO
    ) -> None:
        """Update person state for one frame."""
        if not self.alive or self.escaped:
            return

        # Apply damage
        self._apply_damage(dt, hazards, smoke, neural_aco)
        if not self.alive:
            return

        # Check escape
        if maze[self.row][self.col] == TileType.EXIT:
            self._escape(neural_aco)
            return

        # Update awareness
        self._update_awareness(dt, hazards, alarm_active)

        # Handle movement
        self._handle_movement(dt, maze, exits, hazards, pathfinder, neural_aco)

    def _apply_damage(
        self,
        dt: float,
        hazards: Dict[Tuple[int, int], Dict],
        smoke: Dict[Tuple[int, int], float],
        neural_aco: NeuralACO
    ) -> None:
        """Apply damage from hazards and smoke."""
        # Fire damage
        if (self.row, self.col) in hazards:
            self.health -= PERSON.fire_damage_rate * dt
            self.state = PersonState.PANICKING

        # Smoke damage
        smoke_level = smoke.get((self.row, self.col), 0)
        if smoke_level > 0.5:
            self.health -= smoke_level * PERSON.smoke_damage_multiplier * dt

        # Check death
        if self.health <= 0:
            self.alive = False
            neural_aco.deposit_danger_pheromone((self.row, self.col), 50)

    def _escape(self, neural_aco: NeuralACO) -> None:
        """Handle successful escape."""
        self.escaped = True
        # Deposit safe pheromone on successful path
        path_taken = [(self.row, self.col)] + self.path[:self.path_index]
        neural_aco.deposit_safe_pheromone(path_taken, True)

    def _update_awareness(
        self,
        dt: float,
        hazards: Dict[Tuple[int, int], Dict],
        alarm_active: bool
    ) -> None:
        """Update awareness state based on environment."""
        if self.state in [
            PersonState.AWARE, PersonState.EVACUATING,
            PersonState.PANICKING, PersonState.WARDEN
        ]:
            return

        # Immediate awareness if alarm is active
        if alarm_active:
            self.awareness = 1.0

        # Gradual awareness from nearby hazards
        for h_pos in hazards:
            dist = abs(self.row - h_pos[0]) + abs(self.col - h_pos[1])
            if dist < 8:
                self.awareness += PERSON.awareness_gain_rate / (dist + 1) * dt

        # State transition
        if self.awareness >= PERSON.awareness_threshold:
            self.state = PersonState.AWARE

    def _handle_movement(
        self,
        dt: float,
        maze: List[List[int]],
        exits: List[Tuple[int, int]],
        hazards: Dict[Tuple[int, int], Dict],
        pathfinder: NeuralPathfinder,
        neural_aco: NeuralACO
    ) -> None:
        """Handle movement logic."""
        # Transition from aware to evacuating
        if self.state == PersonState.AWARE:
            self.state = PersonState.EVACUATING

        if not self.moving:
            if self.state in [PersonState.EVACUATING, PersonState.PANICKING, PersonState.WARDEN]:
                # Need a path
                if not self.path or self.path_index >= len(self.path):
                    self.find_exit(exits, pathfinder, maze, hazards)

                # If still no path, try emergency door
                if not self.path or self.path_index >= len(self.path):
                    if self.force_open_corridor_door(maze):
                        self.find_exit(exits, pathfinder, maze, hazards)
                        self.stuck_timer = 0.0

                # Follow path
                if self.path and self.path_index < len(self.path):
                    next_pos = self.path[self.path_index]
                    nr, nc = next_pos

                    # Check if path blocked by hazard
                    if (nr, nc) in hazards:
                        self.find_exit(exits, pathfinder, maze, hazards)
                        return

                    # Move to next position
                    self.tx = nc * TILE + TILE // 2
                    self.ty = nr * TILE + TILE // 2
                    self.row, self.col = nr, nc
                    self.path_index += 1
                    self.moving = True
                else:
                    self.stuck_timer += dt
            else:
                self.stuck_timer += dt

        # Smooth movement animation
        if self.moving:
            self.walk_frame += dt * 12
            speed = self.speed
            if self.state == PersonState.PANICKING:
                speed *= PERSON.panic_speed_multiplier

            dx = self.tx - self.x
            dy = self.ty - self.y
            dist = math.hypot(dx, dy)

            if dist < 2:
                self.x, self.y = self.tx, self.ty
                self.moving = False
                self.stuck_timer = 0.0
            else:
                self.x += (dx / dist) * speed * dt
                self.y += (dy / dist) * speed * dt

        # Emergency unstuck
        if self.stuck_timer > PERSON.stuck_threshold:
            if self.state in [PersonState.EVACUATING, PersonState.PANICKING, PersonState.WARDEN]:
                if self.force_open_corridor_door(maze):
                    self.find_exit(exits, pathfinder, maze, hazards)
                    self.stuck_timer = 0.0

    def to_dict(self) -> Dict:
        """Convert person to dictionary for serialization."""
        return {
            "id": self.id,
            "row": self.row,
            "col": self.col,
            "state": self.state,
            "isWarden": self.is_warden,
            "health": self.health,
            "alive": self.alive,
            "escaped": self.escaped,
        }
