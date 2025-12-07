"""
Reinforcement Learning Evacuation Coordinator

Q-learning agent that learns optimal warden deployment and
crowd flow management strategies.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Tuple, Dict, Any, TYPE_CHECKING

import numpy as np

from dwight.config import ROWS, COLS, AI, StatsKey

if TYPE_CHECKING:
    from dwight.core.person import Person


class RLEvacuationCoordinator:
    """
    Q-learning agent for evacuation coordination.

    State space: (fire_quadrant, crowd_density_quadrant, exits_blocked)
    Action space: Deploy warden to quadrant, prioritize exits

    The agent learns to optimally deploy wardens and manage crowd flow
    to minimize evacuation time and deaths.
    """

    # Action definitions
    ACTIONS = [
        "deploy_NW", "deploy_NE", "deploy_SW", "deploy_SE",
        "open_exit_N", "open_exit_S", "open_exit_E", "open_exit_W"
    ]

    def __init__(self) -> None:
        # Q-table: state -> action values
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(AI.rl_num_actions)
        )

        # Learning parameters
        self.learning_rate = AI.rl_learning_rate
        self.discount = AI.rl_discount
        self.epsilon = AI.rl_epsilon

        # Episode tracking
        self.current_state: Tuple | None = None
        self.last_action: int | None = None
        self.episode_reward: float = 0.0
        self.total_episodes: int = 0

        # Performance metrics
        self.avg_evacuation_time: List[float] = []
        self.death_rate: List[float] = []
        self.decisions_made: int = 0

    def get_state(
        self,
        fire_positions: List[Tuple[int, int]],
        people: List[Person],
        exits_status: Dict[Tuple[int, int], bool]
    ) -> Tuple[int, int, int]:
        """
        Convert environment observation to discrete state.

        Args:
            fire_positions: List of fire cell positions
            people: List of all people in simulation
            exits_status: Dictionary mapping exit positions to blocked status

        Returns:
            State tuple (fire_quadrant, crowd_quadrant, blocked_count)
        """
        # Determine fire quadrant (0-3: NW, NE, SW, SE)
        fire_quad = 0
        if fire_positions:
            avg_r = np.mean([f[0] for f in fire_positions])
            avg_c = np.mean([f[1] for f in fire_positions])
            if avg_r < ROWS // 2:
                fire_quad = 0 if avg_c < COLS // 2 else 1
            else:
                fire_quad = 2 if avg_c < COLS // 2 else 3

        # Determine crowd density quadrant
        crowd_counts = [0, 0, 0, 0]
        for p in people:
            if p.alive and not p.escaped:
                if p.row < ROWS // 2:
                    q = 0 if p.col < COLS // 2 else 1
                else:
                    q = 2 if p.col < COLS // 2 else 3
                crowd_counts[q] += 1

        crowd_quad = int(np.argmax(crowd_counts))

        # Count blocked exits
        blocked = sum(1 for status in exits_status.values() if status)

        return (fire_quad, crowd_quad, min(blocked, 3))

    def choose_action(self, state: Tuple) -> int:
        """
        Choose action using epsilon-greedy policy.

        Args:
            state: Current state tuple

        Returns:
            Action index
        """
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, AI.rl_num_actions - 1)
        else:
            # Exploitation: best known action
            return int(np.argmax(self.q_table[state]))

    def update(self, reward: float, new_state: Tuple) -> None:
        """
        Update Q-table using Q-learning update rule.

        Args:
            reward: Reward received
            new_state: New state after taking action
        """
        if self.current_state is not None and self.last_action is not None:
            old_value = self.q_table[self.current_state][self.last_action]
            next_max = np.max(self.q_table[new_state])

            # Q-learning update
            new_value = old_value + self.learning_rate * (
                reward + self.discount * next_max - old_value
            )
            self.q_table[self.current_state][self.last_action] = new_value

        self.episode_reward += reward
        self.current_state = new_state

    def step(
        self,
        fire_positions: List[Tuple[int, int]],
        people: List[Person],
        exits_status: Dict[Tuple[int, int], bool],
        wardens: List[Person]
    ) -> Dict[str, Any]:
        """
        Take one decision step.

        Args:
            fire_positions: Current fire positions
            people: All people in simulation
            exits_status: Exit blocked status
            wardens: List of warden agents

        Returns:
            Result dictionary with reward, message, and action taken
        """
        state = self.get_state(fire_positions, people, exits_status)
        action = self.choose_action(state)
        self.last_action = action
        self.decisions_made += 1

        # Execute action
        action_name = self.ACTIONS[action]
        result = self._execute_action(action_name, wardens, people)

        # Update Q-table
        self.update(result[StatsKey.REWARD], state)

        return result

    def _execute_action(
        self,
        action_name: str,
        wardens: List[Person],
        people: List[Person]
    ) -> Dict[str, Any]:
        """
        Execute the chosen action.

        Args:
            action_name: Name of action to execute
            wardens: List of warden agents
            people: All people (unused but available for future extensions)

        Returns:
            Result dictionary with reward and message
        """
        reward = 0.0
        message = ""

        if action_name.startswith("deploy_"):
            # Deploy warden to quadrant
            quadrant = action_name.split("_")[1]

            # Calculate target position based on quadrant
            target_row = ROWS // 4 if "N" in quadrant else 3 * ROWS // 4
            target_col = COLS // 4 if "W" in quadrant else 3 * COLS // 4

            # Find available warden
            for warden in wardens:
                if warden.alive and not warden.escaped:
                    warden.rl_target = (target_row, target_col)
                    reward = 5.0
                    message = f"Deployed warden to {quadrant}"
                    break

        elif action_name.startswith("open_exit_"):
            # Prioritize exit direction (simplified - could trigger door openings)
            direction = action_name.split("_")[2]
            reward = 2.0
            message = f"Prioritizing {direction} exits"

        return {
            StatsKey.REWARD: reward,
            StatsKey.MESSAGE: message,
            StatsKey.ACTION: action_name
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get current RL statistics."""
        return {
            StatsKey.EPISODES: self.total_episodes,
            StatsKey.AVG_REWARD: (
                self.episode_reward / max(1, self.decisions_made)
            ),
            StatsKey.EPSILON: self.epsilon,
            StatsKey.DECISIONS: self.decisions_made,
        }

    def reset_episode(self) -> None:
        """Reset for new episode."""
        self.current_state = None
        self.last_action = None
        self.episode_reward = 0.0
        self.total_episodes += 1
