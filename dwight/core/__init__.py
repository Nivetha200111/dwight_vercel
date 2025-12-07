"""
DWIGHT Core Module

Core simulation components: Person, Disasters, Alarm, Building generation.
"""

from dwight.core.person import Person
from dwight.core.disasters import Disasters
from dwight.core.alarm import AlarmSystem
from dwight.core.building import generate_building
from dwight.core.spawner import spawn_people, spawn_sensors

__all__ = [
    "Person",
    "Disasters",
    "AlarmSystem",
    "generate_building",
    "spawn_people",
    "spawn_sensors",
]
