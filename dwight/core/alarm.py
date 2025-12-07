"""
Alarm System

Manages the building alarm state and visual flash effect.
"""

from __future__ import annotations


class AlarmSystem:
    """
    Building alarm system.

    Handles alarm activation, visual flash timing, and sound integration.
    """

    def __init__(self) -> None:
        self.active: bool = False
        self.flash_timer: float = 0.0
        self.flash_state: bool = False
        self._flash_interval: float = 0.25  # seconds

    def trigger(self) -> bool:
        """
        Activate the alarm.

        Returns:
            True if alarm was newly activated, False if already active
        """
        if not self.active:
            self.active = True
            return True
        return False

    def update(self, dt: float) -> None:
        """Update alarm flash state."""
        if self.active:
            self.flash_timer += dt
            if self.flash_timer > self._flash_interval:
                self.flash_state = not self.flash_state
                self.flash_timer = 0.0

    def reset(self) -> None:
        """Reset alarm to inactive state."""
        self.active = False
        self.flash_state = False
        self.flash_timer = 0.0

    @property
    def is_flashing(self) -> bool:
        """Check if currently in flash-on state."""
        return self.active and self.flash_state
