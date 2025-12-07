"""
Building Generation

Procedural building layout generation with rooms, corridors, and exits.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from dwight.config import ROWS, COLS, TileType


def generate_building() -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """
    Generate a building layout with corridors, rooms, and exits.

    Returns:
        Tuple of (maze grid, list of exit positions)
    """
    maze = [[TileType.FLOOR for _ in range(COLS)] for _ in range(ROWS)]
    exits: List[Tuple[int, int]] = []

    # Outer walls
    _add_outer_walls(maze)

    # Main corridors
    h_corr = [ROWS // 3, 2 * ROWS // 3]
    v_corr = [COLS // 4, COLS // 2, 3 * COLS // 4]

    _add_corridors(maze, h_corr, v_corr)

    # Rooms
    _add_rooms(maze, h_corr, v_corr)

    # Exits
    _add_exits(maze, exits, h_corr, v_corr)

    return maze, exits


def _add_outer_walls(maze: List[List[int]]) -> None:
    """Add outer walls to the building."""
    for r in range(ROWS):
        maze[r][0] = TileType.WALL
        maze[r][COLS - 1] = TileType.WALL
    for c in range(COLS):
        maze[0][c] = TileType.WALL
        maze[ROWS - 1][c] = TileType.WALL


def _add_corridors(
    maze: List[List[int]],
    h_corr: List[int],
    v_corr: List[int]
) -> None:
    """Add main corridors."""
    # Horizontal corridors
    for hr in h_corr:
        for c in range(1, COLS - 1):
            for r in range(hr - 1, hr + 2):
                if 0 < r < ROWS - 1:
                    maze[r][c] = TileType.CORRIDOR

    # Vertical corridors
    for vc in v_corr:
        for r in range(1, ROWS - 1):
            for c in range(vc - 1, vc + 2):
                if 0 < c < COLS - 1:
                    maze[r][c] = TileType.CORRIDOR


def _add_rooms(
    maze: List[List[int]],
    h_corr: List[int],
    v_corr: List[int]
) -> None:
    """Add rooms to the building."""
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
            _make_room(maze, r1, r2, c1, c2)


def _make_room(
    maze: List[List[int]],
    r1: int, r2: int,
    c1: int, c2: int
) -> None:
    """Create a room with walls and doors."""
    # Walls
    for r in range(r1, r2 + 1):
        if maze[r][c1] != TileType.CORRIDOR:
            maze[r][c1] = TileType.WALL
        if maze[r][c2] != TileType.CORRIDOR:
            maze[r][c2] = TileType.WALL
    for c in range(c1, c2 + 1):
        if maze[r1][c] != TileType.CORRIDOR:
            maze[r1][c] = TileType.WALL
        if maze[r2][c] != TileType.CORRIDOR:
            maze[r2][c] = TileType.WALL

    # Interior (carpet)
    for r in range(r1 + 1, r2):
        for c in range(c1 + 1, c2):
            if maze[r][c] != TileType.CORRIDOR:
                maze[r][c] = TileType.CARPET

    # Doors facing corridors
    _add_room_doors(maze, r1, r2, c1, c2)

    # Interior doors for large rooms
    if (r2 - r1) > 4 and (c2 - c1) > 4:
        _add_interior_doors(maze, r1, r2, c1, c2)


def _add_room_doors(
    maze: List[List[int]],
    r1: int, r2: int,
    c1: int, c2: int
) -> None:
    """Add doors along corridor-adjacent walls."""
    # Bottom wall
    for c in range(c1 + 1, c2):
        if r2 + 1 < ROWS and maze[r2 + 1][c] == TileType.CORRIDOR:
            maze[r2][c] = TileType.DOOR
    # Top wall
    for c in range(c1 + 1, c2):
        if r1 - 1 > 0 and maze[r1 - 1][c] == TileType.CORRIDOR:
            maze[r1][c] = TileType.DOOR
    # Right wall
    for r in range(r1 + 1, r2):
        if c2 + 1 < COLS and maze[r][c2 + 1] == TileType.CORRIDOR:
            maze[r][c2] = TileType.DOOR
    # Left wall
    for r in range(r1 + 1, r2):
        if c1 - 1 > 0 and maze[r][c1 - 1] == TileType.CORRIDOR:
            maze[r][c1] = TileType.DOOR


def _add_interior_doors(
    maze: List[List[int]],
    r1: int, r2: int,
    c1: int, c2: int
) -> None:
    """Add interior doors for better flow in large rooms."""
    r_mid = (r1 + r2) // 2
    c_mid = (c1 + c2) // 2

    candidates = [
        (r_mid, c_mid),
        (r_mid, c_mid - 2),
        (r_mid, c_mid + 2),
        (r_mid - 2, c_mid),
        (r_mid + 2, c_mid),
        (r1 + (r2 - r1) // 3, c_mid),
        (r1 + 2 * (r2 - r1) // 3, c_mid),
        (r_mid, c1 + (c2 - c1) // 3),
        (r_mid, c1 + 2 * (c2 - c1) // 3),
    ]

    # Filter valid candidates
    valid: List[Tuple[int, int]] = []
    for ir, ic in candidates:
        if r1 + 1 <= ir <= r2 - 1 and c1 + 1 <= ic <= c2 - 1:
            if maze[ir][ic] == TileType.CARPET:
                valid.append((ir, ic))

    # Deduplicate and shuffle
    valid = list(dict.fromkeys(valid))
    random.shuffle(valid)

    # Place up to 4 interior doors
    for ir, ic in valid[:4]:
        maze[ir][ic] = TileType.DOOR


def _add_exits(
    maze: List[List[int]],
    exits: List[Tuple[int, int]],
    h_corr: List[int],
    v_corr: List[int]
) -> None:
    """Add exits to the building."""
    exit_pos = [
        (h_corr[0], 1), (h_corr[1], 1),
        (h_corr[0], COLS - 2), (h_corr[1], COLS - 2),
        (1, v_corr[0]), (1, v_corr[1]), (1, v_corr[2]),
        (ROWS - 2, v_corr[0]), (ROWS - 2, v_corr[1]), (ROWS - 2, v_corr[2]),
    ]

    for er, ec in exit_pos:
        if 0 < er < ROWS - 1 and 0 < ec < COLS - 1:
            maze[er][ec] = TileType.EXIT
            exits.append((er, ec))

            # Clear walls around exit
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = er + dr, ec + dc
                    if 0 < nr < ROWS - 1 and 0 < nc < COLS - 1:
                        if maze[nr][nc] == TileType.WALL:
                            maze[nr][nc] = TileType.CORRIDOR
