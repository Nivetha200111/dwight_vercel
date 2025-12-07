"""
Neural-Enhanced Ant Colony Optimization

The patentable core innovation: pheromone deposit/evaporation rates are
dynamically modulated by neural network prediction confidence.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple, Dict, Any, TYPE_CHECKING

import numpy as np

from dwight.config import ROWS, COLS, AI

if TYPE_CHECKING:
    from dwight.ai.lstm_predictor import SimpleLSTMPredictor


class NeuralACO:
    """
    Neural-Enhanced Ant Colony Optimization.

    INNOVATION: Pheromone deposit/evaporation rates are DYNAMICALLY
    modulated by neural network prediction confidence.

    When the LSTM is confident about fire spread direction:
    - Increase danger pheromone deposit in predicted areas
    - Decrease safe pheromone evaporation on confirmed safe paths
    - Adjust heuristic weights in pathfinding

    This creates a feedback loop between:
    Neural Prediction -> Pheromone Modulation -> Agent Behavior -> New Data -> Neural Learning
    """

    def __init__(self, lstm_predictor: SimpleLSTMPredictor) -> None:
        self.lstm = lstm_predictor

        # Pheromone matrices
        self.safe_pheromone = np.ones((ROWS, COLS)) * 0.1
        self.danger_pheromone = np.zeros((ROWS, COLS))
        self.predicted_danger = np.zeros((ROWS, COLS))

        # ACO parameters (base values from config)
        self.base_evaporation = AI.aco_base_evaporation
        self.base_deposit = AI.aco_base_deposit
        self.alpha = AI.aco_alpha  # Pheromone importance
        self.beta = AI.aco_beta    # Heuristic importance
        self.modulation_strength = AI.aco_modulation_strength

        # Neural modulation state
        self.neural_confidence: float = 0.0

        # Path edge tracking for visualization/analysis
        self.edge_usage: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = defaultdict(int)

    def update_predictions(
        self,
        fire_positions: List[Tuple[int, int]],
        sensor_data: Dict[str, Any],
        maze: List[List[int]]
    ) -> None:
        """Update neural predictions and modulate pheromones accordingly."""
        predictions = self.lstm.predict_spread(fire_positions, sensor_data, maze)
        self.neural_confidence = self.lstm.prediction_confidence

        # Decay predicted danger over time
        self.predicted_danger *= 0.8

        # Apply predictions to danger pheromone
        for r, c, prob in predictions:
            # Neural confidence modulates how much we trust predictions
            modulated_prob = prob * (0.5 + 0.5 * self.neural_confidence)
            self.predicted_danger[r, c] = max(self.predicted_danger[r, c], modulated_prob)
            self.danger_pheromone[r, c] += modulated_prob * self.modulation_strength

    def deposit_safe_pheromone(
        self,
        path: List[Tuple[int, int]],
        success: bool
    ) -> None:
        """Deposit pheromone along successful evacuation path."""
        if not path or not success:
            return

        # Amount modulated by neural confidence
        amount = self.base_deposit * (1.0 + self.neural_confidence * 0.5)

        for i, (r, c) in enumerate(path):
            # More pheromone at start of path (early decisions matter more)
            decay = 1.0 - (i / len(path)) * 0.5
            self.safe_pheromone[r, c] += amount * decay
            self.safe_pheromone[r, c] = min(
                self.safe_pheromone[r, c],
                AI.aco_safe_pheromone_max
            )

            # Track edge usage
            if i > 0:
                edge = (path[i - 1], (r, c))
                self.edge_usage[edge] += 1

    def deposit_danger_pheromone(
        self,
        position: Tuple[int, int],
        severity: float
    ) -> None:
        """Mark dangerous area with danger pheromone."""
        r, c = position
        self.danger_pheromone[r, c] += severity
        self.danger_pheromone[r, c] = min(
            self.danger_pheromone[r, c],
            AI.aco_danger_pheromone_max
        )

        # Spread to neighbors (danger radiates outward)
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    dist = abs(dr) + abs(dc)
                    spread = severity / (dist + 1)
                    self.danger_pheromone[nr, nc] += spread * 0.3

    def evaporate(self) -> None:
        """
        Evaporate pheromones over time.

        Rate is modulated by neural confidence:
        - When confident, preserve safe paths longer
        - Danger always evaporates faster (environment changes quickly)
        """
        # When confident, preserve safe paths longer
        safe_evap = self.base_evaporation * (1.0 - self.neural_confidence * 0.3)
        danger_evap = self.base_evaporation * 1.2  # Danger evaporates faster

        self.safe_pheromone *= (1.0 - safe_evap)
        self.danger_pheromone *= (1.0 - danger_evap)

        # Clamp values
        self.safe_pheromone = np.clip(self.safe_pheromone, 0.1, AI.aco_safe_pheromone_max)
        self.danger_pheromone = np.clip(self.danger_pheromone, 0, AI.aco_danger_pheromone_max)

    def get_path_desirability(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        exit_pos: Tuple[int, int]
    ) -> float:
        """
        Calculate desirability of moving from->to.

        Combines pheromone factor (tau) with heuristic (eta) using
        standard ACO probability formula.

        Args:
            from_pos: Current position
            to_pos: Target position
            exit_pos: Final exit goal

        Returns:
            Desirability score for this move
        """
        r, c = to_pos

        # Pheromone factor
        tau_safe = self.safe_pheromone[r, c]
        tau_danger = self.danger_pheromone[r, c]
        tau_predicted = self.predicted_danger[r, c]

        # Combined pheromone: safe reduced by danger
        tau = tau_safe / (1 + tau_danger + tau_predicted * 2)

        # Heuristic: inverse distance to exit
        dist = abs(r - exit_pos[0]) + abs(c - exit_pos[1]) + 1
        eta = 1.0 / dist

        # Combined probability (ACO formula)
        desirability = (tau ** self.alpha) * (eta ** self.beta)

        return desirability

    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for visualization/debugging."""
        return {
            "safe": self.safe_pheromone.copy(),
            "danger": self.danger_pheromone.copy(),
            "predicted": self.predicted_danger.copy(),
            "confidence": self.neural_confidence,
            "edge_usage": dict(self.edge_usage),
        }

    def reset(self) -> None:
        """Reset ACO state for new simulation."""
        self.safe_pheromone = np.ones((ROWS, COLS)) * 0.1
        self.danger_pheromone = np.zeros((ROWS, COLS))
        self.predicted_danger = np.zeros((ROWS, COLS))
        self.edge_usage.clear()
        self.neural_confidence = 0.0
        self.lstm.reset()
