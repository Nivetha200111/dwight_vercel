"""
Neural-Enhanced A* Pathfinder

A* pathfinding algorithm enhanced with Neural ACO pheromone guidance.
"""

from __future__ import annotations

from heapq import heappush, heappop
from typing import List, Tuple, Dict, Set, TYPE_CHECKING

from dwight.config import TileType

if TYPE_CHECKING:
    from dwight.ai.neural_aco import NeuralACO


class NeuralPathfinder:
    """
    A* pathfinder enhanced with Neural ACO pheromone guidance.

    The pathfinding cost is modified by:
    - Danger pheromone (increases cost significantly)
    - Predicted danger from LSTM (increases cost even more)
    - Safe pheromone (decreases cost)
    - Actual hazards (very high cost)
    """

    # Valid tile types for walking
    WALKABLE_TILES = frozenset([
        TileType.FLOOR,
        TileType.CORRIDOR,
        TileType.CARPET,
        TileType.EXIT,
        TileType.DOOR,
    ])

    def __init__(self, neural_aco: NeuralACO) -> None:
        self.aco = neural_aco
        self._cache: Dict[Tuple[Tuple[int, int], Tuple[int, int], int], List[Tuple[int, int]]] = {}

    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        maze: List[List[int]],
        hazards: Dict[Tuple[int, int], Dict]
    ) -> List[Tuple[int, int]]:
        """
        Find optimal path from start to goal using A* with pheromone costs.

        Args:
            start: Starting position (row, col)
            goal: Goal position (row, col)
            maze: Building layout grid
            hazards: Dictionary of hazard positions and info

        Returns:
            List of positions forming the path, or empty list if no path found
        """
        # Check cache first
        cache_key = (start, goal, len(hazards))
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        rows = len(maze)
        cols = len(maze[0])

        # A* data structures
        # Priority queue: (f_score, g_score, position)
        open_set: List[Tuple[float, float, Tuple[int, int]]] = []
        heappush(open_set, (0.0, 0.0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}

        # Movement directions (4-connected grid)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            _, _, current = heappop(open_set)

            # Goal reached - reconstruct path
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                self._cache[cache_key] = path
                return path

            r, c = current

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)

                # Bounds check
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue

                # Check if walkable
                tile = maze[nr][nc]
                if tile not in self.WALKABLE_TILES:
                    continue

                # Calculate movement cost
                cost = self._calculate_cost(neighbor, hazards)

                tentative_g = g_score[current] + cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    # Heuristic: Manhattan distance
                    h = abs(nr - goal[0]) + abs(nc - goal[1])
                    f = tentative_g + h

                    heappush(open_set, (f, tentative_g, neighbor))

        # No path found
        return []

    def _calculate_cost(
        self,
        position: Tuple[int, int],
        hazards: Dict[Tuple[int, int], Dict]
    ) -> float:
        """Calculate the cost of moving to a position."""
        r, c = position
        cost = 1.0  # Base cost

        # Very high cost for hazards
        if position in hazards:
            cost += 500.0

        # Neural ACO costs
        danger_pheromone = self.aco.danger_pheromone[r, c]
        predicted_danger = self.aco.predicted_danger[r, c]
        safe_pheromone = self.aco.safe_pheromone[r, c]

        # High danger = high cost
        cost += danger_pheromone * 20.0

        # Predicted danger even more costly (trust predictions)
        cost += predicted_danger * 40.0

        # Safe pheromone reduces cost
        cost *= (1.0 / (1.0 + safe_pheromone * 0.2))

        return cost

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from map."""
        path: List[Tuple[int, int]] = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def clear_cache(self) -> None:
        """Clear the path cache (call when hazards change)."""
        self._cache.clear()
