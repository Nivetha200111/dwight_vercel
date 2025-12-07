"""
LSTM Fire Spread Predictor

Simplified LSTM-like neural network for predicting fire spread patterns.
In production, this would use PyTorch/TensorFlow.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from dwight.config import (
    ROWS, COLS, AI,
    TileType,
)


class SimpleLSTMPredictor:
    """
    Simplified LSTM-like predictor for fire spread.

    Predicts WHERE fire will spread based on:
    - Current fire positions
    - Historical spread patterns
    - Sensor readings (temperature gradients)
    """

    def __init__(self, grid_size: Tuple[int, int] = (ROWS, COLS)) -> None:
        self.grid_size = grid_size
        self.hidden_size = AI.lstm_hidden_size
        self.sequence_length = AI.lstm_sequence_length

        # Simulated learned weights (seeded for reproducibility)
        np.random.seed(42)
        input_size = self.hidden_size + 4  # hidden + features

        self.W_forget = np.random.randn(self.hidden_size, input_size) * 0.1
        self.W_input = np.random.randn(self.hidden_size, input_size) * 0.1
        self.W_output = np.random.randn(self.hidden_size, input_size) * 0.1
        self.W_cell = np.random.randn(self.hidden_size, input_size) * 0.1
        self.W_pred = np.random.randn(4, self.hidden_size) * 0.1  # 4 directions

        # LSTM state
        self.hidden = np.zeros(self.hidden_size)
        self.cell = np.zeros(self.hidden_size)

        # History buffer for temporal patterns
        self.history: deque = deque(maxlen=self.sequence_length)
        self.prediction_confidence: float = 0.0

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation with clipping for numerical stability."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation with clipping for numerical stability."""
        return np.tanh(np.clip(x, -500, 500))

    def extract_features(
        self,
        fire_positions: List[Tuple[int, int]],
        sensor_readings: Dict[str, Any]
    ) -> np.ndarray:
        """Extract features from current state."""
        if not fire_positions:
            return np.zeros(4)

        # Feature: center of fire mass (normalized)
        center_r = np.mean([p[0] for p in fire_positions])
        center_c = np.mean([p[1] for p in fire_positions])

        # Feature: fire spread direction (based on recent history)
        spread_r, spread_c = 0.0, 0.0
        if len(self.history) > 1:
            prev_center = self.history[-1][:2]
            spread_r = center_r - prev_center[0]
            spread_c = center_c - prev_center[1]

        return np.array([
            center_r / self.grid_size[0],
            center_c / self.grid_size[1],
            spread_r,
            spread_c
        ])

    def forward(self, features: np.ndarray) -> np.ndarray:
        """LSTM forward pass - returns direction probabilities."""
        concat = np.concatenate([self.hidden, features])

        # LSTM gates
        forget_gate = self.sigmoid(self.W_forget @ concat)
        input_gate = self.sigmoid(self.W_input @ concat)
        output_gate = self.sigmoid(self.W_output @ concat)
        cell_candidate = self.tanh(self.W_cell @ concat)

        # Update cell and hidden state
        self.cell = forget_gate * self.cell + input_gate * cell_candidate
        self.hidden = output_gate * self.tanh(self.cell)

        # Predict spread probabilities for 4 directions (N, S, W, E)
        direction_probs = self.sigmoid(self.W_pred @ self.hidden)
        return direction_probs

    def predict_spread(
        self,
        fire_positions: List[Tuple[int, int]],
        sensor_readings: Dict[str, Any],
        maze: List[List[int]],
        steps_ahead: int = AI.lstm_prediction_steps
    ) -> List[Tuple[int, int, float]]:
        """
        Predict where fire will spread in the next N steps.

        Args:
            fire_positions: Current fire cell positions
            sensor_readings: Current sensor data
            maze: Building layout grid
            steps_ahead: How many steps ahead to predict

        Returns:
            List of (row, col, probability) tuples for predicted fire spread
        """
        if not fire_positions:
            self.prediction_confidence = 0.0
            return []

        features = self.extract_features(fire_positions, sensor_readings)
        self.history.append(features)

        direction_probs = self.forward(features)
        self.prediction_confidence = float(np.max(direction_probs))

        predictions: List[Tuple[int, int, float]] = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E

        threshold = AI.lstm_prediction_threshold

        for fire_pos in fire_positions:
            for i, (dr, dc) in enumerate(directions):
                prob = direction_probs[i]
                if prob > threshold:
                    for step in range(1, steps_ahead + 1):
                        nr = fire_pos[0] + dr * step
                        nc = fire_pos[1] + dc * step

                        # Check bounds
                        if 0 < nr < self.grid_size[0] - 1 and 0 < nc < self.grid_size[1] - 1:
                            # Check if not a wall
                            if maze[nr][nc] != TileType.WALL:
                                decay = 0.7 ** step  # Probability decreases with distance
                                predictions.append((nr, nc, prob * decay))

        # Aggregate predictions - keep max probability per cell
        pred_dict: Dict[Tuple[int, int], float] = defaultdict(float)
        for r, c, p in predictions:
            pred_dict[(r, c)] = max(pred_dict[(r, c)], p)

        # Filter low-probability predictions
        return [(r, c, p) for (r, c), p in pred_dict.items() if p > 0.25]

    def reset(self) -> None:
        """Reset LSTM state for new simulation."""
        self.hidden = np.zeros(self.hidden_size)
        self.cell = np.zeros(self.hidden_size)
        self.history.clear()
        self.prediction_confidence = 0.0
