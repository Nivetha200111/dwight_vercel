"""Shared persistent simulation state for serverless endpoints."""

from __future__ import annotations

from dwight import init_headless_state

SIM_STATE = None


def get_state():
    global SIM_STATE
    if SIM_STATE is None:
        SIM_STATE = init_headless_state()
    return SIM_STATE


def reset_state():
    global SIM_STATE
    SIM_STATE = init_headless_state()
    return SIM_STATE
