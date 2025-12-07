"""
IoT Sensor Simulation

Simulated IoT sensors with realistic behavior including noise,
battery drain, and health degradation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dwight.config import SENSOR, SensorType


@dataclass
class IoTSensor:
    """
    Simulated IoT sensor with realistic behavior.

    Supports different sensor types:
    - Temperature: Measures ambient temperature
    - Smoke: Detects smoke particles
    - CO: Carbon monoxide detector
    - Motion: PIR motion sensor
    """

    id: int
    row: int
    col: int
    sensor_type: str

    # Current reading
    value: float = 0.0
    threshold: float = 0.0
    triggered: bool = False

    # Health and power
    health: float = 100.0
    battery: float = 100.0

    # Sensor characteristics
    noise_level: float = SENSOR.noise_level
    last_reading: float = 0.0

    def __post_init__(self) -> None:
        """Initialize sensor-type-specific parameters."""
        if self.sensor_type == SensorType.TEMPERATURE:
            self.threshold = SENSOR.temperature_threshold
            self.value = SENSOR.room_temperature
        elif self.sensor_type == SensorType.SMOKE:
            self.threshold = SENSOR.smoke_threshold
            self.value = 0.0
        elif self.sensor_type == SensorType.CO:
            self.threshold = SENSOR.co_threshold
            self.value = 0.0
        elif self.sensor_type == SensorType.MOTION:
            self.threshold = SENSOR.motion_threshold
            self.value = 0.0

    @property
    def is_functional(self) -> bool:
        """Check if sensor is still working."""
        return self.health > 0 and self.battery > 0

    @property
    def position(self) -> tuple[int, int]:
        """Get sensor position as tuple."""
        return (self.row, self.col)

    def drain_battery(self, dt: float) -> None:
        """Simulate battery drain over time."""
        self.battery -= SENSOR.battery_drain_rate * dt
        if self.battery <= 0:
            self.battery = 0
            self.health = 0

    def take_damage(self, dt: float, in_fire: bool = False) -> None:
        """Apply damage from environmental hazards."""
        if in_fire:
            self.health -= SENSOR.fire_damage_rate * dt
            self.health = max(0, self.health)

    def update_reading(self, new_value: float) -> None:
        """Update sensor reading."""
        self.last_reading = self.value
        self.value = new_value

    def check_trigger(self) -> bool:
        """Check if sensor should trigger alert."""
        was_triggered = self.triggered
        self.triggered = self.value > self.threshold
        return self.triggered and not was_triggered  # Return True only on new trigger

    def to_dict(self) -> dict:
        """Convert sensor to dictionary for serialization."""
        return {
            "id": self.id,
            "row": self.row,
            "col": self.col,
            "type": self.sensor_type,
            "value": self.value,
            "triggered": self.triggered,
            "health": self.health,
            "battery": self.battery,
        }
