"""
IoT Sensor Network

Simulated IoT sensor network with Kalman filtering for noise reduction,
mesh network communication, and sensor fusion.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any

import numpy as np

from dwight.config import SENSOR, SensorType, StatsKey
from dwight.sensors.iot_sensor import IoTSensor


class IoTSensorNetwork:
    """
    Simulated IoT sensor network.

    Features:
    - Multiple sensor types (temperature, smoke, CO, motion)
    - Kalman filtering for noise reduction
    - Battery/health simulation
    - Mesh network communication delay and packet loss
    - Sensor fusion for decision making
    """

    def __init__(self) -> None:
        self.sensors: Dict[int, IoTSensor] = {}
        self.sensor_grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self.alerts: List[Dict[str, Any]] = []

        # Network characteristics
        self.network_latency = SENSOR.network_latency
        self.packet_loss_rate = SENSOR.packet_loss_rate

        # Kalman filter state for each sensor
        self._kalman_state: Dict[int, Dict[str, float]] = {}

    def add_sensor(self, sensor: IoTSensor) -> None:
        """Add a sensor to the network."""
        self.sensors[sensor.id] = sensor
        self.sensor_grid[sensor.position].append(sensor.id)

        # Initialize Kalman filter for this sensor
        self._kalman_state[sensor.id] = {
            "estimate": sensor.value,
            "error": 1.0,
            "process_noise": 0.01,
            "measurement_noise": sensor.noise_level,
        }

    def remove_sensor(self, sensor_id: int) -> None:
        """Remove a sensor from the network."""
        if sensor_id in self.sensors:
            sensor = self.sensors[sensor_id]
            pos = sensor.position
            if pos in self.sensor_grid:
                self.sensor_grid[pos] = [
                    sid for sid in self.sensor_grid[pos] if sid != sensor_id
                ]
            del self.sensors[sensor_id]
            if sensor_id in self._kalman_state:
                del self._kalman_state[sensor_id]

    def kalman_update(self, sensor_id: int, measurement: float) -> float:
        """
        Apply Kalman filter to sensor reading for noise reduction.

        Args:
            sensor_id: ID of the sensor
            measurement: Raw sensor measurement

        Returns:
            Filtered sensor value
        """
        state = self._kalman_state[sensor_id]

        # Prediction step
        predicted_estimate = state["estimate"]
        predicted_error = state["error"] + state["process_noise"]

        # Update step
        kalman_gain = predicted_error / (predicted_error + state["measurement_noise"])
        state["estimate"] = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        state["error"] = (1 - kalman_gain) * predicted_error

        return state["estimate"]

    def update(
        self,
        dt: float,
        fire_positions: List[Tuple[int, int]],
        smoke_map: Dict[Tuple[int, int], float],
        people_positions: List[Tuple[int, int]],
        maze: List[List[int]]
    ) -> None:
        """
        Update all sensors based on environment state.

        Args:
            dt: Time delta
            fire_positions: Current fire cell positions
            smoke_map: Smoke density map
            people_positions: Positions of people (for motion sensors)
            maze: Building layout
        """
        self.alerts.clear()
        fire_set = set(fire_positions)

        for sensor_id, sensor in self.sensors.items():
            # Drain battery
            sensor.drain_battery(dt)
            if not sensor.is_functional:
                continue

            # Apply fire damage
            sensor.take_damage(dt, in_fire=(sensor.position in fire_set))
            if not sensor.is_functional:
                continue

            # Calculate raw reading
            raw_value = self._calculate_raw_reading(
                sensor, fire_positions, smoke_map, people_positions
            )

            # Add sensor noise
            noise = np.random.normal(0, sensor.noise_level)
            noisy_value = raw_value + noise

            # Apply Kalman filter
            filtered_value = self.kalman_update(sensor_id, noisy_value)
            sensor.update_reading(filtered_value)

            # Check for alert condition
            if sensor.check_trigger():
                # Simulate packet loss
                if random.random() > self.packet_loss_rate:
                    self.alerts.append({
                        "sensor_id": sensor_id,
                        "type": sensor.sensor_type,
                        "value": filtered_value,
                        "position": sensor.position,
                        "timestamp": time.time(),
                    })

    def _calculate_raw_reading(
        self,
        sensor: IoTSensor,
        fire_positions: List[Tuple[int, int]],
        smoke_map: Dict[Tuple[int, int], float],
        people_positions: List[Tuple[int, int]]
    ) -> float:
        """Calculate raw sensor reading based on environment."""
        if sensor.sensor_type == SensorType.TEMPERATURE:
            return self._calc_temperature(sensor, fire_positions)
        elif sensor.sensor_type == SensorType.SMOKE:
            return smoke_map.get(sensor.position, 0.0)
        elif sensor.sensor_type == SensorType.CO:
            return self._calc_co(sensor, fire_positions)
        elif sensor.sensor_type == SensorType.MOTION:
            return self._calc_motion(sensor, people_positions)
        return 0.0

    def _calc_temperature(
        self,
        sensor: IoTSensor,
        fire_positions: List[Tuple[int, int]]
    ) -> float:
        """Calculate temperature reading."""
        base_temp = SENSOR.room_temperature
        for fire_pos in fire_positions:
            dist = abs(sensor.row - fire_pos[0]) + abs(sensor.col - fire_pos[1])
            if dist < 10:
                base_temp += 100 / (dist + 1)
        return min(base_temp, 200.0)

    def _calc_co(
        self,
        sensor: IoTSensor,
        fire_positions: List[Tuple[int, int]]
    ) -> float:
        """Calculate CO level reading."""
        co = 0.0
        for fire_pos in fire_positions:
            dist = abs(sensor.row - fire_pos[0]) + abs(sensor.col - fire_pos[1])
            if dist < 8:
                co += 50 / (dist + 1)
        return min(co, 500.0)

    def _calc_motion(
        self,
        sensor: IoTSensor,
        people_positions: List[Tuple[int, int]]
    ) -> float:
        """Calculate motion sensor reading."""
        motion = 0.0
        for px, py in people_positions:
            dist = abs(sensor.row - px) + abs(sensor.col - py)
            if dist < 5:
                motion += 1 / (dist + 1)
        return min(motion, 5.0)

    def get_sensor_fusion_data(self) -> Dict[str, Any]:
        """
        Fuse data from all sensors for decision making.

        Returns:
            Dictionary with aggregated sensor statistics
        """
        data: Dict[str, Any] = {
            StatsKey.TEMPERATURE_AVG: 0.0,
            StatsKey.TEMPERATURE_MAX: 0.0,
            StatsKey.SMOKE_LEVEL: 0.0,
            StatsKey.CO_LEVEL: 0.0,
            StatsKey.MOTION_DETECTED: 0,
            StatsKey.TRIGGERED_SENSORS: [],
            StatsKey.SENSOR_HEALTH: 0.0,
            StatsKey.COVERAGE: 0.0,
        }

        # Filter sensors by type and health
        temp_sensors = [s for s in self.sensors.values()
                       if s.sensor_type == SensorType.TEMPERATURE and s.is_functional]
        smoke_sensors = [s for s in self.sensors.values()
                        if s.sensor_type == SensorType.SMOKE and s.is_functional]
        co_sensors = [s for s in self.sensors.values()
                     if s.sensor_type == SensorType.CO and s.is_functional]
        motion_sensors = [s for s in self.sensors.values()
                         if s.sensor_type == SensorType.MOTION and s.is_functional]

        # Aggregate temperature
        if temp_sensors:
            temps = [s.value for s in temp_sensors]
            data[StatsKey.TEMPERATURE_AVG] = float(np.mean(temps))
            data[StatsKey.TEMPERATURE_MAX] = float(max(temps))

        # Aggregate smoke
        if smoke_sensors:
            data[StatsKey.SMOKE_LEVEL] = float(np.mean([s.value for s in smoke_sensors]))

        # Aggregate CO
        if co_sensors:
            data[StatsKey.CO_LEVEL] = float(np.mean([s.value for s in co_sensors]))

        # Count motion
        if motion_sensors:
            data[StatsKey.MOTION_DETECTED] = sum(1 for s in motion_sensors if s.value > 0.5)

        # Triggered sensors
        data[StatsKey.TRIGGERED_SENSORS] = [
            s.id for s in self.sensors.values() if s.triggered
        ]

        # Health and coverage
        alive_sensors = [s for s in self.sensors.values() if s.is_functional]
        if alive_sensors:
            data[StatsKey.SENSOR_HEALTH] = float(np.mean([s.health for s in alive_sensors]))
            data[StatsKey.COVERAGE] = len(alive_sensors) / len(self.sensors) * 100

        return data

    def get_sensors_by_type(self, sensor_type: str) -> List[IoTSensor]:
        """Get all sensors of a specific type."""
        return [s for s in self.sensors.values() if s.sensor_type == sensor_type]

    def get_triggered_sensors(self) -> List[IoTSensor]:
        """Get all currently triggered sensors."""
        return [s for s in self.sensors.values() if s.triggered and s.is_functional]
