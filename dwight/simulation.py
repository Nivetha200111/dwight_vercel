"""
Headless Simulation Runner

Run the evacuation simulation without rendering for serverless environments.
"""

from __future__ import annotations

from typing import Dict, Any

from dwight.config import (
    ROWS, COLS, SIMULATION, StatsKey,
)
from dwight.ai import (
    SimpleLSTMPredictor,
    NeuralACO,
    NeuralPathfinder,
    RLEvacuationCoordinator,
)
from dwight.core import (
    Disasters,
    AlarmSystem,
    generate_building,
    spawn_people,
    spawn_sensors,
)


def run_headless_simulation(
    steps: int = 240,
    dt: float = 1 / 30.0
) -> Dict[str, Any]:
    """
    Run a trimmed-down simulation loop without rendering.

    Designed for serverless environments (e.g., Vercel) where no
    display/audio exists.

    Args:
        steps: Number of simulation steps to run
        dt: Time delta per step (default: ~33ms for 30fps)

    Returns:
        Dictionary with simulation statistics
    """
    # Initialize all systems
    maze, exits = generate_building()
    lstm_predictor = SimpleLSTMPredictor()
    neural_aco = NeuralACO(lstm_predictor)
    pathfinder = NeuralPathfinder(neural_aco)
    disasters = Disasters()
    alarm = AlarmSystem()
    sensor_network = spawn_sensors(maze)
    rl_coordinator = RLEvacuationCoordinator()
    people = spawn_people(maze)

    stats = {
        StatsKey.ESCAPED: 0,
        StatsKey.DEATHS: 0,
        StatsKey.TOTAL: SIMULATION.total_people,
    }

    # Seed an ignition so the loop has meaningful activity
    disasters.add_fire(ROWS // 2, COLS // 2)

    neural_update_timer = 0.0
    rl_update_timer = 0.0
    steps_run = 0

    for _ in range(int(steps)):
        steps_run += 1

        # Update disasters
        disasters.update(dt, maze, neural_aco)
        alarm.update(dt)

        # Auto-trigger alarm on fire
        if disasters.has_fires() and not alarm.active:
            alarm.trigger()

        # Update neural predictions periodically
        neural_update_timer += dt
        if neural_update_timer > SIMULATION.neural_update_interval:
            fire_positions = disasters.get_fire_positions()
            sensor_data = sensor_network.get_sensor_fusion_data()
            neural_aco.update_predictions(fire_positions, sensor_data, maze)
            neural_update_timer = 0.0

        # Update sensors
        people_positions = [
            (p.row, p.col) for p in people
            if p.alive and not p.escaped
        ]
        sensor_network.update(
            dt,
            disasters.get_fire_positions(),
            disasters.smoke,
            people_positions,
            maze
        )

        # Update RL coordinator periodically
        rl_update_timer += dt
        if rl_update_timer > SIMULATION.rl_update_interval and alarm.active:
            wardens = [p for p in people if p.is_warden and p.alive]
            exits_status = {e: False for e in exits}
            rl_coordinator.step(
                disasters.get_fire_positions(),
                people,
                exits_status,
                wardens
            )
            rl_update_timer = 0.0

        # Pheromone evaporation
        neural_aco.evaporate()

        # Update people
        for p in people:
            p.update(
                dt, maze, exits, disasters.hazards, pathfinder,
                alarm.active, people, disasters.smoke, neural_aco
            )

        # Update stats
        stats[StatsKey.ESCAPED] = sum(1 for p in people if p.escaped)
        stats[StatsKey.DEATHS] = sum(1 for p in people if not p.alive)

        # Exit early if everyone is resolved
        if stats[StatsKey.ESCAPED] + stats[StatsKey.DEATHS] >= stats[StatsKey.TOTAL]:
            break

    # Final sensor data for stats
    sensor_snapshot = sensor_network.get_sensor_fusion_data()

    return {
        StatsKey.STEPS_RUN: steps_run,
        StatsKey.ESCAPED: stats[StatsKey.ESCAPED],
        StatsKey.DEATHS: stats[StatsKey.DEATHS],
        "alive": stats[StatsKey.TOTAL] - stats[StatsKey.ESCAPED] - stats[StatsKey.DEATHS],
        "fires_active": len(disasters.get_fire_positions()),
        "neural_confidence": neural_aco.neural_confidence,
        "rl_decisions": rl_coordinator.decisions_made,
        "sensor_coverage": sensor_snapshot.get(StatsKey.COVERAGE, 0),
        "avg_temp": sensor_snapshot.get(StatsKey.TEMPERATURE_AVG, 0),
    }
