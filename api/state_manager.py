"""Shared persistent simulation state for serverless endpoints."""

from __future__ import annotations

import os
import pickle
import tempfile
from typing import Any

from dwight import init_headless_state

SIM_STATE = None
STATE_PATH = os.path.join(tempfile.gettempdir(), "dwight_state.pkl")


def _load_from_disk() -> Any:
    if not os.path.exists(STATE_PATH):
        return None
    try:
        with open(STATE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_to_disk(state: Any) -> None:
    try:
        with open(STATE_PATH, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        # Ignore persistence failures silently; in-memory state still works on warm instances.
        pass


def get_state():
    """Return the persistent state, loading from disk if needed."""
    global SIM_STATE
    if SIM_STATE is None:
        SIM_STATE = _load_from_disk()
    if SIM_STATE is None:
        SIM_STATE = init_headless_state()
        _save_to_disk(SIM_STATE)
    return SIM_STATE


def reset_state():
    """Create a fresh state and persist it."""
    global SIM_STATE
    SIM_STATE = init_headless_state()
    _save_to_disk(SIM_STATE)
    return SIM_STATE


def save_state(state: Any) -> None:
    """Persist the given state to disk."""
    global SIM_STATE
    SIM_STATE = state
    _save_to_disk(SIM_STATE)
