# """
# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                                                                                  ║
# ║     ██████╗ ██╗    ██╗██╗ ██████╗ ██╗  ██╗████████╗    ██╗   ██╗██╗  ██╗                         ║
# ║     ██╔══██╗██║    ██║██║██╔════╝ ██║  ██║╚══██╔══╝    ██║   ██║╚██╗██╔╝                         ║
# ║     ██║  ██║██║ █╗ ██║██║██║  ███╗███████║   ██║       ██║   ██║ ╚███╔╝                          ║
# ║     ██║  ██║██║███╗██║██║██║   ██║██╔══██║   ██║       ██║   ██║ ██╔██╗                          ║
# ║     ██████╔╝╚███╔███╔╝██║╚██████╔╝██║  ██║   ██║        ╚████╔╝ ██╔╝ ██╗                         ║
# ║     ╚═════╝  ╚══╝╚══╝ ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝         ╚═══╝  ╚═╝  ╚═╝                         ║
# ║                                                                                                  ║
# ║              🛡️ NEURAL ACO EMERGENCY RESPONSE SYSTEM (NAERS) 🛡️                                  ║
# ║                                                                                                  ║
# ║  PATENTABLE INNOVATIONS:                                                                         ║
# ║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                   ║
# ║  🧠 Neural Predictive ACO (NP-ACO):                                                              ║
# ║     - LSTM network predicts fire spread 30-60 seconds ahead                                      ║
# ║     - ACO pheromones dynamically weighted by neural predictions                                  ║
# ║     - Real-time path recalculation based on predicted danger zones                               ║
# ║                                                                                                  ║
# ║  📡 IoT Sensor Fusion Layer:                                                                     ║
# ║     - Simulated temperature, smoke, CO, motion sensors                                           ║
# ║     - Kalman filtering for noise reduction                                                       ║
# ║     - Sensor health monitoring with redundancy                                                   ║
# ║                                                                                                  ║
# ║  🤖 Reinforcement Learning Evacuation Coordinator:                                               ║
# ║     - Q-learning agent optimizes warden deployment                                               ║
# ║     - Multi-agent coordination for bottleneck prevention                                         ║
# ║     - Adaptive crowd flow management                                                             ║
# ║                                                                                                  ║
# ║  🎯 3D Perspective View (Non-Isometric):                                                         ║
# ║     - True perspective projection with depth                                                     ║
# ║     - Dynamic lighting based on fire/emergency lights                                            ║
# ║     - Particle systems for fire, smoke, water                                                    ║
# ║                                                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

# CORE USP FOR PATENT:
# ═══════════════════
# "Adaptive Neural-Pheromone Emergency Evacuation System (ANPEES)"

# A system that combines:
# 1. Ant Colony Optimization for distributed pathfinding
# 2. LSTM neural networks for hazard prediction
# 3. IoT sensor fusion for real-time environmental awareness
# 4. Reinforcement learning for dynamic resource allocation

# The pheromone weights are DYNAMICALLY MODULATED by neural network confidence scores,
# creating a hybrid bio-inspired + deep learning approach that is novel and patentable.
# """

# import os

# # Headless mode allows this module to be imported by serverless environments (e.g., Vercel)
# # where there is no display or audio device.
# os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
# HEADLESS = bool(os.environ.get("HEADLESS") or os.environ.get("VERCEL") or os.environ.get("CI"))
# if HEADLESS:
#     os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
#     os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# import pygame
# import pygame.gfxdraw
# import random
# import math
# import numpy as np
# from collections import defaultdict, deque
# from heapq import heappush, heappop
# from dataclasses import dataclass
# from typing import List, Tuple, Dict, Optional
# import time

# pygame.init()
# if not HEADLESS:
#     pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# # ═══════════════════════════════════════════════════════════════════════════════
# # CONFIGURATION
# # ═══════════════════════════════════════════════════════════════════════════════

# ROWS = 45 # Keep rows/cols consistent with original for map generation
# COLS = 70
# TILE = 14
# TOTAL_PEOPLE = 60
# NUM_WARDENS = 4
# NUM_SENSORS = 25

# MAP_WIDTH = COLS * TILE
# MAP_HEIGHT = ROWS * TILE
# PANEL_WIDTH = 380
# SCREEN_WIDTH = MAP_WIDTH + PANEL_WIDTH
# SCREEN_HEIGHT = MAP_HEIGHT + 80

# if HEADLESS:
#     screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
# else:
#     screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
#     pygame.display.set_caption("🧠 DWIGHT UX - Neural ACO Emergency Response System")

# # ═══════════════════════════════════════════════════════════════════════════════
# # DEEP LEARNING - LSTM FIRE PREDICTION (Simplified NumPy Implementation)
# # ═══════════════════════════════════════════════════════════════════════════════

# class SimpleLSTMPredictor:
#     """
#     Simplified LSTM-like predictor for fire spread.
#     In production, this would use PyTorch/TensorFlow.

#     This predicts WHERE fire will spread based on:
#     - Current fire positions
#     - Historical spread patterns
#     - Sensor readings (temperature gradients)
#     """
#     def __init__(self, grid_size=(ROWS, COLS)):
#         self.grid_size = grid_size
#         self.hidden_size = 32
#         self.sequence_length = 10

#         # Simulated learned weights
#         np.random.seed(42)
#         self.W_forget = np.random.randn(self.hidden_size, self.hidden_size + 4) * 0.1
#         self.W_input = np.random.randn(self.hidden_size, self.hidden_size + 4) * 0.1
#         self.W_output = np.random.randn(self.hidden_size, self.hidden_size + 4) * 0.1
#         self.W_cell = np.random.randn(self.hidden_size, self.hidden_size + 4) * 0.1
#         self.W_pred = np.random.randn(4, self.hidden_size) * 0.1  # 4 directions

#         self.hidden = np.zeros(self.hidden_size)
#         self.cell = np.zeros(self.hidden_size)

#         # History buffer
#         self.history = deque(maxlen=self.sequence_length)
#         self.prediction_confidence = 0.0

#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

#     def tanh(self, x):
#         return np.tanh(np.clip(x, -500, 500))

#     def extract_features(self, fire_positions, sensor_readings):
#         """Extract features from current state."""
#         if not fire_positions:
#             return np.zeros(4)

#         # Feature: center of fire mass
#         center_r = np.mean([p[0] for p in fire_positions])
#         center_c = np.mean([p[1] for p in fire_positions])

#         # Feature: fire spread direction (based on recent history)
#         spread_r, spread_c = 0, 0
#         if len(self.history) > 1:
#             prev_center = self.history[-1][:2]
#             spread_r = center_r - prev_center[0]
#             spread_c = center_c - prev_center[1]

#         return np.array([center_r / ROWS, center_c / COLS, spread_r, spread_c])

#     def forward(self, features):
#         """LSTM forward pass."""
#         concat = np.concatenate([self.hidden, features])

#         forget_gate = self.sigmoid(self.W_forget @ concat)
#         input_gate = self.sigmoid(self.W_input @ concat)
#         output_gate = self.sigmoid(self.W_output @ concat)
#         cell_candidate = self.tanh(self.W_cell @ concat)

#         self.cell = forget_gate * self.cell + input_gate * cell_candidate
#         self.hidden = output_gate * self.tanh(self.cell)

#         # Predict spread probabilities for 4 directions
#         direction_probs = self.sigmoid(self.W_pred @ self.hidden)
#         return direction_probs

#     def predict_spread(self, fire_positions, sensor_readings, maze, steps_ahead=3):
#         """
#         Predict where fire will spread in the next N steps.
#         Returns list of (row, col, probability) tuples.
#         """
#         if not fire_positions:
#             self.prediction_confidence = 0.0
#             return []

#         features = self.extract_features(fire_positions, sensor_readings)
#         self.history.append(features)

#         direction_probs = self.forward(features)
#         self.prediction_confidence = float(np.max(direction_probs))

#         predictions = []
#         directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E

#         for fire_pos in fire_positions:
#             for i, (dr, dc) in enumerate(directions):
#                 prob = direction_probs[i]
#                 if prob > 0.3:  # Threshold
#                     for step in range(1, steps_ahead + 1):
#                         nr = fire_pos[0] + dr * step
#                         nc = fire_pos[1] + dc * step
#                         if 0 < nr < ROWS - 1 and 0 < nc < COLS - 1:
#                             if maze[nr][nc] != 1:  # Not wall
#                                 decay = 0.7 ** step
#                                 predictions.append((nr, nc, prob * decay))

#         # Aggregate predictions
#         pred_dict = defaultdict(float)
#         for r, c, p in predictions:
#             pred_dict[(r, c)] = max(pred_dict[(r, c)], p)

#         return [(r, c, p) for (r, c), p in pred_dict.items() if p > 0.25]

#     def reset(self):
#         self.hidden = np.zeros(self.hidden_size)
#         self.cell = np.zeros(self.hidden_size)
#         self.history.clear()
#         self.prediction_confidence = 0.0

# # ═══════════════════════════════════════════════════════════════════════════════
# # IOT SENSOR SIMULATION
# # ═══════════════════════════════════════════════════════════════════════════════

# @dataclass
# class IoTSensor:
#     """Simulated IoT sensor with realistic behavior."""
#     id: int
#     row: int
#     col: int
#     sensor_type: str  # 'temperature', 'smoke', 'co', 'motion'
#     value: float = 0.0
#     threshold: float = 0.0
#     triggered: bool = False
#     health: float = 100.0
#     battery: float = 100.0
#     noise_level: float = 0.05
#     last_reading: float = 0.0

#     def __post_init__(self):
#         if self.sensor_type == 'temperature':
#             self.threshold = 45.0  # Celsius
#             self.value = 22.0  # Room temp
#         elif self.sensor_type == 'smoke':
#             self.threshold = 0.3
#             self.value = 0.0
#         elif self.sensor_type == 'co':
#             self.threshold = 35.0  # ppm
#             self.value = 0.0
#         elif self.sensor_type == 'motion':
#             self.threshold = 0.5
#             self.value = 0.0

# class IoTSensorNetwork:
#     """
#     Simulated IoT sensor network with:
#     - Multiple sensor types
#     - Kalman filtering for noise reduction
#     - Battery/health simulation
#     - Mesh network communication delay
#     """
#     def __init__(self):
#         self.sensors: Dict[int, IoTSensor] = {}
#         self.sensor_grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)
#         self.alerts: List[dict] = []
#         self.network_latency = 0.1  # seconds
#         self.packet_loss_rate = 0.02

#         # Kalman filter state for each sensor
#         self.kalman_state: Dict[int, dict] = {}

#     def add_sensor(self, sensor: IoTSensor):
#         self.sensors[sensor.id] = sensor
#         self.sensor_grid[(sensor.row, sensor.col)].append(sensor.id)
#         self.kalman_state[sensor.id] = {
#             'estimate': sensor.value,
#             'error': 1.0,
#             'process_noise': 0.01,
#             'measurement_noise': sensor.noise_level
#         }

#     def kalman_update(self, sensor_id: int, measurement: float) -> float:
#         """Apply Kalman filter to sensor reading."""
#         state = self.kalman_state[sensor_id]

#         # Prediction
#         predicted_estimate = state['estimate']
#         predicted_error = state['error'] + state['process_noise']

#         # Update
#         kalman_gain = predicted_error / (predicted_error + state['measurement_noise'])
#         state['estimate'] = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
#         state['error'] = (1 - kalman_gain) * predicted_error

#         return state['estimate']

#     def update(self, dt, fire_positions, smoke_map, people_positions, maze):
#         """Update all sensors based on environment."""
#         self.alerts.clear()

#         for sensor_id, sensor in self.sensors.items():
#             # Battery drain
#             sensor.battery -= 0.001 * dt
#             if sensor.battery <= 0:
#                 sensor.health = 0
#                 continue

#             # Health degradation in fire
#             if (sensor.row, sensor.col) in fire_positions:
#                 sensor.health -= 5 * dt

#             if sensor.health <= 0:
#                 continue

#             # Calculate raw reading based on environment
#             raw_value = self._calculate_raw_reading(sensor, fire_positions, smoke_map, people_positions)

#             # Add noise
#             noise = np.random.normal(0, sensor.noise_level)
#             noisy_value = raw_value + noise

#             # Apply Kalman filter
#             filtered_value = self.kalman_update(sensor_id, noisy_value)
#             sensor.value = filtered_value
#             sensor.last_reading = filtered_value

#             # Check threshold
#             was_triggered = sensor.triggered
#             sensor.triggered = filtered_value > sensor.threshold

#             # Generate alert on state change
#             if sensor.triggered and not was_triggered:
#                 if random.random() > self.packet_loss_rate:
#                     self.alerts.append({
#                         'sensor_id': sensor_id,
#                         'type': sensor.sensor_type,
#                         'value': filtered_value,
#                         'position': (sensor.row, sensor.col),
#                         'timestamp': time.time()
#                     })

#     def _calculate_raw_reading(self, sensor, fire_positions, smoke_map, people_positions):
#         """Calculate sensor reading based on environment."""
#         if sensor.sensor_type == 'temperature':
#             base_temp = 22.0
#             for fire_pos in fire_positions:
#                 dist = abs(sensor.row - fire_pos[0]) + abs(sensor.col - fire_pos[1])
#                 if dist < 10:
#                     base_temp += 100 / (dist + 1)
#             return min(base_temp, 200)

#         elif sensor.sensor_type == 'smoke':
#             return smoke_map.get((sensor.row, sensor.col), 0)

#         elif sensor.sensor_type == 'co':
#             co = 0
#             for fire_pos in fire_positions:
#                 dist = abs(sensor.row - fire_pos[0]) + abs(sensor.col - fire_pos[1])
#                 if dist < 8:
#                     co += 50 / (dist + 1)
#             return min(co, 500)

#         elif sensor.sensor_type == 'motion':
#             motion = 0
#             for px, py in people_positions:
#                 dist = abs(sensor.row - px) + abs(sensor.col - py)
#                 if dist < 5:
#                     motion += 1 / (dist + 1)
#             return min(motion, 5)

#         return 0

#     def get_sensor_fusion_data(self) -> dict:
#         """Fuse data from all sensors for decision making."""
#         data = {
#             'temperature_avg': 0,
#             'temperature_max': 0,
#             'smoke_level': 0,
#             'co_level': 0,
#             'motion_detected': 0,
#             'triggered_sensors': [],
#             'sensor_health': 0,
#             'coverage': 0
#         }

#         temp_sensors = [s for s in self.sensors.values() if s.sensor_type == 'temperature' and s.health > 0]
#         smoke_sensors = [s for s in self.sensors.values() if s.sensor_type == 'smoke' and s.health > 0]
#         co_sensors = [s for s in self.sensors.values() if s.sensor_type == 'co' and s.health > 0]
#         motion_sensors = [s for s in self.sensors.values() if s.sensor_type == 'motion' and s.health > 0]

#         if temp_sensors:
#             data['temperature_avg'] = np.mean([s.value for s in temp_sensors])
#             data['temperature_max'] = max(s.value for s in temp_sensors)

#         if smoke_sensors:
#             data['smoke_level'] = np.mean([s.value for s in smoke_sensors])

#         if co_sensors:
#             data['co_level'] = np.mean([s.value for s in co_sensors])

#         if motion_sensors:
#             data['motion_detected'] = sum(1 for s in motion_sensors if s.value > 0.5)

#         data['triggered_sensors'] = [s.id for s in self.sensors.values() if s.triggered]

#         alive_sensors = [s for s in self.sensors.values() if s.health > 0]
#         if alive_sensors:
#             data['sensor_health'] = np.mean([s.health for s in alive_sensors])
#             data['coverage'] = len(alive_sensors) / len(self.sensors) * 100

#         return data

#     def draw(self, surface, shake, time_val):
#         """Draw sensors on the map."""
#         for sensor in self.sensors.values():
#             x = int(sensor.col * TILE + TILE // 2 + shake[0])
#             y = int(sensor.row * TILE + TILE // 2 + shake[1])

#             # Color based on type and status
#             if sensor.health <= 0:
#                 color = (60, 60, 60)
#             elif sensor.triggered:
#                 flash = int(abs(math.sin(time_val * 8)) * 255)
#                 if sensor.sensor_type == 'temperature':
#                     color = (255, flash, 0)
#                 elif sensor.sensor_type == 'smoke':
#                     color = (flash, flash, flash)
#                 elif sensor.sensor_type == 'co':
#                     color = (255, 0, flash)
#                 else:
#                     color = (0, flash, 255)
#             else:
#                 if sensor.sensor_type == 'temperature':
#                     color = (200, 100, 50)
#                 elif sensor.sensor_type == 'smoke':
#                     color = (150, 150, 150)
#                 elif sensor.sensor_type == 'co':
#                     color = (200, 50, 50)
#                 else:
#                     color = (50, 50, 200)

#             # Draw sensor icon
#             pygame.draw.circle(surface, color, (x, y), 4)
#             pygame.draw.circle(surface, (255, 255, 255), (x, y), 4, 1)

#             # Signal waves if triggered
#             if sensor.triggered and sensor.health > 0:
#                 wave_radius = int((time_val * 20) % 15) + 5
#                 alpha = 255 - wave_radius * 10
#                 if alpha > 0:
#                     pygame.gfxdraw.aacircle(surface, x, y, wave_radius, (*color[:3], alpha))

# # ═══════════════════════════════════════════════════════════════════════════════
# # REINFORCEMENT LEARNING - Q-LEARNING EVACUATION COORDINATOR
# # ═══════════════════════════════════════════════════════════════════════════════

# class RLEvacuationCoordinator:
#     """
#     Q-learning agent that learns optimal warden deployment and
#     crowd flow management strategies.

#     State: (fire_quadrant, crowd_density_quadrant, exits_blocked)
#     Actions: (deploy_warden_to_quadrant, open_secondary_exit, etc.)
#     """
#     def __init__(self):
#         self.q_table = defaultdict(lambda: np.zeros(8))  # 8 actions
#         self.learning_rate = 0.1
#         self.discount = 0.95
#         self.epsilon = 0.2

#         self.actions = [
#             'deploy_NW', 'deploy_NE', 'deploy_SW', 'deploy_SE',
#             'open_exit_N', 'open_exit_S', 'open_exit_E', 'open_exit_W'
#         ]

#         self.current_state = None
#         self.last_action = None
#         self.episode_reward = 0
#         self.total_episodes = 0

#         # Performance tracking
#         self.avg_evacuation_time = []
#         self.death_rate = []
#         self.decisions_made = 0

#     def get_state(self, fire_positions, people, exits_status):
#         """Convert environment to discrete state."""
#         # Determine fire quadrant
#         fire_quad = 0
#         if fire_positions:
#             avg_r = np.mean([f[0] for f in fire_positions])
#             avg_c = np.mean([f[1] for f in fire_positions])
#             if avg_r < ROWS // 2:
#                 fire_quad = 0 if avg_c < COLS // 2 else 1
#             else:
#                 fire_quad = 2 if avg_c < COLS // 2 else 3

#         # Crowd density
#         crowd_counts = [0, 0, 0, 0]
#         for p in people:
#             if p.alive and not p.escaped:
#                 q = 0
#                 if p.row < ROWS // 2:
#                     q = 0 if p.col < COLS // 2 else 1
#                 else:
#                     q = 2 if p.col < COLS // 2 else 3
#                 crowd_counts[q] += 1

#         crowd_quad = np.argmax(crowd_counts)

#         # Exits blocked (simplified)
#         blocked = sum(1 for e, status in exits_status.items() if status)

#         return (fire_quad, crowd_quad, min(blocked, 3))

#     def choose_action(self, state):
#         """Epsilon-greedy action selection."""
#         if random.random() < self.epsilon:
#             return random.randint(0, 7)
#         return np.argmax(self.q_table[state])

#     def update(self, reward, new_state):
#         """Update Q-table."""
#         if self.current_state is not None and self.last_action is not None:
#             old_value = self.q_table[self.current_state][self.last_action]
#             next_max = np.max(self.q_table[new_state])

#             new_value = old_value + self.learning_rate * (
#                 reward + self.discount * next_max - old_value
#             )
#             self.q_table[self.current_state][self.last_action] = new_value

#         self.episode_reward += reward
#         self.current_state = new_state

#     def step(self, fire_positions, people, exits_status, wardens):
#         """Take a decision step."""
#         state = self.get_state(fire_positions, people, exits_status)
#         action = self.choose_action(state)
#         self.last_action = action
#         self.decisions_made += 1

#         # Execute action
#         action_name = self.actions[action]
#         result = self._execute_action(action_name, wardens, people)

#         self.update(result['reward'], state)
#         return result

#     def _execute_action(self, action_name, wardens, people):
#         """Execute the chosen action."""
#         reward = 0
#         message = ""

#         if action_name.startswith('deploy_'):
#             quadrant = action_name.split('_')[1]
#             target_row = ROWS // 4 if 'N' in quadrant else 3 * ROWS // 4
#             target_col = COLS // 4 if 'W' in quadrant else 3 * COLS // 4

#             # Find available warden
#             for warden in wardens:
#                 if warden.alive and not warden.escaped:
#                     warden.rl_target = (target_row, target_col)
#                     reward = 5
#                     message = f"Deployed warden to {quadrant}"
#                     break

#         elif action_name.startswith('open_exit_'):
#             reward = 2
#             message = f"Prioritizing {action_name.split('_')[2]} exits"

#         return {'reward': reward, 'message': message, 'action': action_name}

#     def get_stats(self):
#         return {
#             'episodes': self.total_episodes,
#             'avg_reward': self.episode_reward / max(1, self.decisions_made),
#             'epsilon': self.epsilon,
#             'decisions': self.decisions_made
#         }

# # ═══════════════════════════════════════════════════════════════════════════════
# # NEURAL ACO (THE PATENTABLE CORE)
# # ═══════════════════════════════════════════════════════════════════════════════

# class NeuralACO:
#     """
#     Neural-Enhanced Ant Colony Optimization.

#     INNOVATION: Pheromone deposit/evaporation rates are DYNAMICALLY
#     modulated by neural network prediction confidence.

#     When the LSTM is confident about fire spread direction:
#     - Increase danger pheromone deposit in predicted areas
#     - Decrease safe pheromone evaporation on confirmed safe paths
#     - Adjust heuristic weights in pathfinding

#     This creates a feedback loop between:
#     Neural Prediction -> Pheromone Modulation -> Agent Behavior -> New Data -> Neural Learning
#     """
#     def __init__(self, lstm_predictor: SimpleLSTMPredictor):
#         self.lstm = lstm_predictor

#         # Pheromone matrices
#         self.safe_pheromone = np.ones((ROWS, COLS)) * 0.1
#         self.danger_pheromone = np.zeros((ROWS, COLS))
#         self.predicted_danger = np.zeros((ROWS, COLS))

#         # ACO parameters (base values)
#         self.base_evaporation = 0.02
#         self.base_deposit = 1.0
#         self.alpha = 1.0  # Pheromone importance
#         self.beta = 2.0   # Heuristic importance

#         # Neural modulation factors
#         self.neural_confidence = 0.0
#         self.modulation_strength = 2.0

#         # Path edge tracking
#         self.edge_usage = defaultdict(int)

#     def update_predictions(self, fire_positions, sensor_data, maze):
#         """Update neural predictions and modulate pheromones."""
#         predictions = self.lstm.predict_spread(fire_positions, sensor_data, maze)
#         self.neural_confidence = self.lstm.prediction_confidence

#         # Reset predicted danger
#         self.predicted_danger *= 0.8

#         # Apply predictions to danger pheromone
#         for r, c, prob in predictions:
#             # Neural confidence modulates how much we trust predictions
#             modulated_prob = prob * (0.5 + 0.5 * self.neural_confidence)
#             self.predicted_danger[r, c] = max(self.predicted_danger[r, c], modulated_prob)
#             self.danger_pheromone[r, c] += modulated_prob * self.modulation_strength

#     def deposit_safe_pheromone(self, path: List[Tuple[int, int]], success: bool):
#         """Deposit pheromone along successful evacuation path."""
#         if not path or not success:
#             return

#         # Amount modulated by neural confidence
#         amount = self.base_deposit * (1.0 + self.neural_confidence * 0.5)

#         for i, (r, c) in enumerate(path):
#             decay = 1.0 - (i / len(path)) * 0.5  # More at start of path
#             self.safe_pheromone[r, c] += amount * decay
#             self.safe_pheromone[r, c] = min(self.safe_pheromone[r, c], 10.0)

#             if i > 0:
#                 edge = (path[i-1], (r, c))
#                 self.edge_usage[edge] += 1

#     def deposit_danger_pheromone(self, position: Tuple[int, int], severity: float):
#         """Mark dangerous area."""
#         r, c = position
#         self.danger_pheromone[r, c] += severity
#         self.danger_pheromone[r, c] = min(self.danger_pheromone[r, c], 20.0)

#         # Spread to neighbors
#         for dr in range(-2, 3):
#             for dc in range(-2, 3):
#                 nr, nc = r + dr, c + dc
#                 if 0 <= nr < ROWS and 0 <= nc < COLS:
#                     dist = abs(dr) + abs(dc)
#                     spread = severity / (dist + 1)
#                     self.danger_pheromone[nr, nc] += spread * 0.3

#     def evaporate(self):
#         """Evaporate pheromones - rate modulated by neural confidence."""
#         # When confident, preserve safe paths longer
#         safe_evap = self.base_evaporation * (1.0 - self.neural_confidence * 0.3)
#         danger_evap = self.base_evaporation * 1.2  # Danger evaporates faster

#         self.safe_pheromone *= (1.0 - safe_evap)
#         self.danger_pheromone *= (1.0 - danger_evap)
#         self.safe_pheromone = np.clip(self.safe_pheromone, 0.1, 10.0)
#         self.danger_pheromone = np.clip(self.danger_pheromone, 0, 20.0)

#     def get_path_desirability(self, from_pos, to_pos, exit_pos) -> float:
#         """
#         Calculate desirability of moving from->to.
#         Combines pheromone (tau) with heuristic (eta).
#         """
#         r, c = to_pos

#         # Pheromone factor
#         tau_safe = self.safe_pheromone[r, c]
#         tau_danger = self.danger_pheromone[r, c]
#         tau_predicted = self.predicted_danger[r, c]

#         tau = tau_safe / (1 + tau_danger + tau_predicted * 2)

#         # Heuristic: inverse distance to exit
#         dist = abs(r - exit_pos[0]) + abs(c - exit_pos[1]) + 1
#         eta = 1.0 / dist

#         # Combined probability
#         desirability = (tau ** self.alpha) * (eta ** self.beta)

#         return desirability

#     def get_visualization_data(self):
#         """Get data for visualization."""
#         return {
#             'safe': self.safe_pheromone.copy(),
#             'danger': self.danger_pheromone.copy(),
#             'predicted': self.predicted_danger.copy(),
#             'confidence': self.neural_confidence,
#             'edge_usage': dict(self.edge_usage)
#         }

#     def reset(self):
#         self.safe_pheromone = np.ones((ROWS, COLS)) * 0.1
#         self.danger_pheromone = np.zeros((ROWS, COLS))
#         self.predicted_danger = np.zeros((ROWS, COLS))
#         self.edge_usage.clear()
#         self.lstm.reset()

# # ═══════════════════════════════════════════════════════════════════════════════
# # SOUND SYSTEM
# # ═══════════════════════════════════════════════════════════════════════════════

# class SoundSystem:
#     def __init__(self):
#         self.sounds = {}
#         self.alarm_playing = False
#         self.generate_sounds()

#     def generate_sounds(self):
#         sample_rate = 22050

#         # Alarm
#         duration = 0.5
#         t = np.linspace(0, duration, int(sample_rate * duration), False)
#         tone1 = np.sin(2 * np.pi * 800 * t) * 0.3
#         tone2 = np.sin(2 * np.pi * 600 * t) * 0.3
#         alarm_wave = np.concatenate([tone1, tone2])
#         alarm_wave = (alarm_wave * 32767).astype(np.int16)
#         alarm_stereo = np.column_stack((alarm_wave, alarm_wave))
#         self.sounds['alarm'] = pygame.sndarray.make_sound(alarm_stereo)

#         # Neural alert (futuristic beep)
#         duration = 0.2
#         t = np.linspace(0, duration, int(sample_rate * duration), False)
#         freq = 1200 + 400 * np.sin(t * 30)
#         beep = np.sin(2 * np.pi * freq * t) * 0.25 * np.exp(-t * 5)
#         beep = (beep * 32767).astype(np.int16)
#         beep_stereo = np.column_stack((beep, beep))
#         self.sounds['neural'] = pygame.sndarray.make_sound(beep_stereo)

#         # Sensor trigger
#         duration = 0.15
#         t = np.linspace(0, duration, int(sample_rate * duration), False)
#         ping = np.sin(2 * np.pi * 2000 * t) * np.exp(-t * 20) * 0.2
#         ping = (ping * 32767).astype(np.int16)
#         ping_stereo = np.column_stack((ping, ping))
#         self.sounds['sensor'] = pygame.sndarray.make_sound(ping_stereo)

#         # Success
#         duration = 0.3
#         t = np.linspace(0, duration, int(sample_rate * duration), False)
#         chime = np.sin(2 * np.pi * 880 * t) * np.exp(-t * 5) * 0.2
#         chime = (chime * 32767).astype(np.int16)
#         chime_stereo = np.column_stack((chime, chime))
#         self.sounds['escape'] = pygame.sndarray.make_sound(chime_stereo)

#     def play(self, name, volume=0.5):
#         if name in self.sounds:
#             self.sounds[name].set_volume(volume)
#             self.sounds[name].play()

#     def start_alarm(self):
#         if not self.alarm_playing:
#             self.sounds['alarm'].play(-1)
#             self.alarm_playing = True

#     def stop_alarm(self):
#         if self.alarm_playing:
#             self.sounds['alarm'].stop()
#             self.alarm_playing = False

# class SilentSoundSystem:
#     """No-op sound system for headless/serverless execution."""
#     def play(self, name, volume=0.5):
#         return None

#     def start_alarm(self):
#         return None

#     def stop_alarm(self):
#         return None

# sound_system = SilentSoundSystem() if HEADLESS else SoundSystem()

# # ═══════════════════════════════════════════════════════════════════════════════
# # COLORS (Enhanced for 3D look)
# # ═══════════════════════════════════════════════════════════════════════════════

# class Colors:
#     # Environment
#     FLOOR = (180, 175, 165)
#     FLOOR_ALT = (170, 165, 155)
#     WALL = (45, 48, 55)
#     WALL_HIGHLIGHT = (65, 68, 75)
#     CORRIDOR = (155, 150, 145)
#     CARPET = (95, 65, 65)
#     EXIT = (50, 255, 100)
#     DOOR = (120, 75, 40)

#     # Hazards
#     FIRE = (255, 100, 30)
#     FIRE_BRIGHT = (255, 220, 80)
#     FIRE_CORE = (255, 255, 200)
#     SMOKE = (70, 70, 75)
#     WATER = (40, 120, 200)

#     # Neural/Tech
#     NEURAL_GLOW = (0, 255, 200)
#     PREDICTION = (255, 50, 200)
#     SENSOR_ACTIVE = (0, 200, 255)
#     IOT_MESH = (100, 255, 200)

#     # Pheromones
#     SAFE_PHEROMONE = (0, 255, 150)
#     DANGER_PHEROMONE = (255, 80, 80)

#     # People states
#     NORMAL = (100, 150, 255)
#     AWARE = (255, 255, 100)
#     EVACUATING = (100, 255, 100)
#     PANICKING = (255, 80, 80)
#     WARDEN = (255, 215, 0)
#     HEADPHONES = (255, 100, 255)

#     # UI
#     PANEL_BG = (18, 20, 28)
#     PANEL_BORDER = (45, 50, 65)
#     ACCENT = (0, 230, 180)
#     SUCCESS = (60, 255, 120)
#     DANGER = (255, 70, 70)
#     WARNING = (255, 200, 60)
#     TEXT = (255, 255, 255)
#     TEXT_DIM = (160, 160, 170)

# # Tile types
# FLOOR = 0
# WALL = 1
# EXIT = 2
# DOOR = 3
# CORRIDOR = 4
# CARPET = 5

# # States
# STATE_WORKING = "working"
# STATE_HEADPHONES = "headphones"
# STATE_AWARE = "aware"
# STATE_EVACUATING = "evacuating"
# STATE_PANICKING = "panicking"
# STATE_WARDEN = "warden"

# # ═══════════════════════════════════════════════════════════════════════════════
# # PATHFINDER (Enhanced with Neural ACO)
# # ═══════════════════════════════════════════════════════════════════════════════

# class NeuralPathfinder:
#     """A* pathfinder enhanced with Neural ACO pheromone guidance."""

#     def __init__(self, neural_aco: NeuralACO):
#         self.aco = neural_aco
#         self.cache = {}

#     def find_path(self, start, goal, maze, hazards) -> List[Tuple[int, int]]:
#         cache_key = (start, goal, len(hazards))
#         if cache_key in self.cache:
#             return self.cache[cache_key].copy()

#         rows, cols = len(maze), len(maze[0])
#         open_set = []
#         heappush(open_set, (0, 0, start))
#         came_from = {}
#         g_score = {start: 0}

#         while open_set:
#             _, _, current = heappop(open_set)

#             if current == goal:
#                 path = []
#                 while current in came_from:
#                     path.append(current)
#                     current = came_from[current]
#                 path.reverse()
#                 self.cache[cache_key] = path
#                 return path

#             r, c = current
#             for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#                 nr, nc = r + dr, c + dc
#                 if 0 <= nr < rows and 0 <= nc < cols:
#                     tile = maze[nr][nc]
#                     if tile in [FLOOR, CORRIDOR, CARPET, EXIT, DOOR]:
#                         neighbor = (nr, nc)

#                         # Base cost
#                         cost = 1

#                         # Add hazard costs
#                         if neighbor in hazards:
#                             cost += 500

#                         # Neural ACO costs
#                         danger_pheromone = self.aco.danger_pheromone[nr, nc]
#                         predicted_danger = self.aco.predicted_danger[nr, nc]
#                         safe_pheromone = self.aco.safe_pheromone[nr, nc]

#                         # High danger/predicted danger = high cost
#                         cost += danger_pheromone * 20
#                         cost += predicted_danger * 40  # Trust predictions more

#                         # Safe pheromone reduces cost
#                         cost *= (1.0 / (1.0 + safe_pheromone * 0.2))

#                         tentative_g = g_score[current] + cost

#                         if neighbor not in g_score or tentative_g < g_score[neighbor]:
#                             came_from[neighbor] = current
#                             g_score[neighbor] = tentative_g
#                             h = abs(nr - goal[0]) + abs(nc - goal[1])
#                             heappush(open_set, (tentative_g + h, tentative_g, neighbor))

#         return []

#     def clear_cache(self):
#         self.cache.clear()

# # ═══════════════════════════════════════════════════════════════════════════════
# # DISASTERS
# # ═══════════════════════════════════════════════════════════════════════════════

# class Disasters:
#     def __init__(self):
#         self.hazards = {}
#         self.smoke = defaultdict(float)
#         self.shake = 0.0
#         self.shake_offset = (0, 0)
#         self.particles = []

#     def add_fire(self, row, col):
#         if (row, col) not in self.hazards:
#             self.hazards[(row, col)] = {'type': 'fire', 'age': 0, 'intensity': 1.0}

#     def update(self, dt, maze, neural_aco):
#         self.shake *= 0.9
#         if self.shake > 0.01:
#             self.shake_offset = (
#                 random.uniform(-1, 1) * self.shake * 6,
#                 random.uniform(-1, 1) * self.shake * 6
#             )
#         else:
#             self.shake_offset = (0, 0)

#         new_hazards = {}
#         to_remove = []

#         for (row, col), info in list(self.hazards.items()):
#             info['age'] += dt

#             if info['type'] == 'fire':
#                 # Smoke
#                 for dr in range(-4, 5):
#                     for dc in range(-4, 5):
#                         sr, sc = row + dr, col + dc
#                         if 0 <= sr < ROWS and 0 <= sc < COLS:
#                             dist = abs(dr) + abs(dc)
#                             self.smoke[(sr, sc)] = min(
#                                 self.smoke[(sr, sc)] + 0.03 / (dist + 1), 1.5
#                             )

#                 # Deposit danger pheromone
#                 neural_aco.deposit_danger_pheromone((row, col), 2.0 * dt)

#                 # Fire particles
#                 if random.random() < 0.4:
#                     self.particles.append({
#                         'x': col * TILE + random.randint(2, TILE - 2),
#                         'y': row * TILE + TILE,
#                         'vy': -random.uniform(30, 60),
#                         'vx': random.uniform(-10, 10),
#                         'life': random.uniform(0.3, 0.7),
#                         'type': 'fire'
#                     })

#                 # Spread
#                 if info['age'] > 5.0 and random.random() < 0.01:
#                     for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#                         nr, nc = row + dr, col + dc
#                         if 0 < nr < ROWS - 1 and 0 < nc < COLS - 1:
#                             if maze[nr][nc] not in [WALL, EXIT]:
#                                 if (nr, nc) not in self.hazards:
#                                     new_hazards[(nr, nc)] = {'type': 'fire', 'age': 0, 'intensity': 0.8}
#                                     break

#                 if info['age'] > 80:
#                     to_remove.append((row, col))

#         for pos in to_remove:
#             if pos in self.hazards:
#                 del self.hazards[pos]
#         self.hazards.update(new_hazards)

#         # Smoke decay
#         for key in list(self.smoke.keys()):
#             self.smoke[key] *= 0.97
#             if self.smoke[key] < 0.02:
#                 del self.smoke[key]

#         # Particles
#         for p in self.particles[:]:
#             p['y'] += p['vy'] * dt
#             p['x'] += p.get('vx', 0) * dt
#             p['life'] -= dt
#             if p['life'] <= 0:
#                 self.particles.remove(p)

#     def get_fire_positions(self):
#         return [pos for pos, info in self.hazards.items() if info['type'] == 'fire']

#     def draw_particles(self, surface, shake):
#         for p in self.particles:
#             x = int(p['x'] + shake[0])
#             y = int(p['y'] + shake[1])
#             alpha = min(255, int(p['life'] * 400))
#             size = max(1, int(4 * p['life']))

#             if p['type'] == 'fire':
#                 t = p['life']
#                 r = int(255)
#                 g = int(150 + 100 * t)
#                 b = int(50 * t)
#                 pygame.draw.circle(surface, (r, g, b), (x, y), size)

# # ═══════════════════════════════════════════════════════════════════════════════
# # ALARM SYSTEM
# # ═══════════════════════════════════════════════════════════════════════════════

# class AlarmSystem:
#     def __init__(self):
#         self.active = False
#         self.flash_timer = 0
#         self.flash_state = False

#     def trigger(self):
#         if not self.active:
#             self.active = True
#             sound_system.start_alarm()

#     def update(self, dt):
#         if self.active:
#             self.flash_timer += dt
#             if self.flash_timer > 0.25:
#                 self.flash_state = not self.flash_state
#                 self.flash_timer = 0

#     def reset(self):
#         self.active = False
#         self.flash_state = False
#         sound_system.stop_alarm()

# # ═══════════════════════════════════════════════════════════════════════════════
# # BUILDING GENERATION
# # ═══════════════════════════════════════════════════════════════════════════════

# def generate_building():
#     maze = [[FLOOR for _ in range(COLS)] for _ in range(ROWS)]
#     exits = []

#     # Outer walls
#     for r in range(ROWS):
#         maze[r][0] = WALL
#         maze[r][COLS-1] = WALL
#     for c in range(COLS):
#         maze[0][c] = WALL
#         maze[ROWS-1][c] = WALL

#     # Corridors
#     h_corr = [ROWS // 3, 2 * ROWS // 3]
#     v_corr = [COLS // 4, COLS // 2, 3 * COLS // 4]

#     for hr in h_corr:
#         for c in range(1, COLS - 1):
#             for r in range(hr - 1, hr + 2):
#                 if 0 < r < ROWS - 1:
#                     maze[r][c] = CORRIDOR

#     for vc in v_corr:
#         for r in range(1, ROWS - 1):
#             for c in range(vc - 1, vc + 2):
#                 if 0 < c < COLS - 1:
#                     maze[r][c] = CORRIDOR

#     # Rooms
#     def make_room(r1, r2, c1, c2):
#         for r in range(r1, r2 + 1):
#             if maze[r][c1] != CORRIDOR: maze[r][c1] = WALL
#             if maze[r][c2] != CORRIDOR: maze[r][c2] = WALL
#         for c in range(c1, c2 + 1):
#             if maze[r1][c] != CORRIDOR: maze[r1][c] = WALL
#             if maze[r2][c] != CORRIDOR: maze[r2][c] = WALL

#         for r in range(r1 + 1, r2):
#             for c in range(c1 + 1, c2):
#                 if maze[r][c] != CORRIDOR:
#                     maze[r][c] = CARPET

#         # Door
#         for c in range(c1 + 1, c2):
#             if r2 + 1 < ROWS and maze[r2 + 1][c] == CORRIDOR:
#                 maze[r2][c] = DOOR
#                 return
#             if r1 - 1 > 0 and maze[r1 - 1][c] == CORRIDOR:
#                 maze[r1][c] = DOOR
#                 return

#     sections = [
#         (2, h_corr[0] - 2, 2, v_corr[0] - 2),
#         (2, h_corr[0] - 2, v_corr[0] + 2, v_corr[1] - 2),
#         (2, h_corr[0] - 2, v_corr[1] + 2, v_corr[2] - 2),
#         (2, h_corr[0] - 2, v_corr[2] + 2, COLS - 3),
#         (h_corr[0] + 2, h_corr[1] - 2, 2, v_corr[0] - 2),
#         (h_corr[0] + 2, h_corr[1] - 2, v_corr[2] + 2, COLS - 3),
#         (h_corr[1] + 2, ROWS - 3, 2, v_corr[0] - 2),
#         (h_corr[1] + 2, ROWS - 3, v_corr[0] + 2, v_corr[1] - 2),
#         (h_corr[1] + 2, ROWS - 3, v_corr[1] + 2, v_corr[2] - 2),
#         (h_corr[1] + 2, ROWS - 3, v_corr[2] + 2, COLS - 3),
#     ]

#     for r1, r2, c1, c2 in sections:
#         if r2 - r1 > 3 and c2 - c1 > 3:
#             make_room(r1, r2, c1, c2)

#     # Exits
#     exit_pos = [
#         (h_corr[0], 1), (h_corr[1], 1),
#         (h_corr[0], COLS - 2), (h_corr[1], COLS - 2),
#         (1, v_corr[0]), (1, v_corr[1]), (1, v_corr[2]),
#         (ROWS - 2, v_corr[0]), (ROWS - 2, v_corr[1]), (ROWS - 2, v_corr[2]),
#     ]

#     for er, ec in exit_pos:
#         if 0 < er < ROWS - 1 and 0 < ec < COLS - 1:
#             maze[er][ec] = EXIT
#             exits.append((er, ec))
#             for dr in range(-1, 2):
#                 for dc in range(-1, 2):
#                     nr, nc = er + dr, ec + dc
#                     if 0 < nr < ROWS - 1 and 0 < nc < COLS - 1:
#                         if maze[nr][nc] == WALL:
#                             maze[nr][nc] = CORRIDOR

#     return maze, exits

# # ═══════════════════════════════════════════════════════════════════════════════
# # PERSON CLASS
# # ═══════════════════════════════════════════════════════════════════════════════

# class Person:
#     def __init__(self, pid, row, col, state, is_warden=False):
#         self.id = pid
#         self.row = row
#         self.col = col
#         self.x = col * TILE + TILE // 2
#         self.y = row * TILE + TILE // 2
#         self.tx = self.x
#         self.ty = self.y

#         self.state = STATE_WARDEN if is_warden else state
#         self.is_warden = is_warden

#         self.alive = True
#         self.escaped = False
#         self.health = 100
#         self.awareness = 1.0 if is_warden else 0.0
#         self.path = []
#         self.path_index = 0
#         self.target_exit = None

#         self.color = Colors.WARDEN if is_warden else random.choice([
#             (255, 90, 90), (90, 160, 255), (90, 255, 90), (255, 230, 90),
#             (255, 90, 255), (90, 255, 255), (255, 160, 90)
#         ])

#         self.walk_frame = 0
#         self.moving = False
#         self.speed = random.uniform(80, 110) * (1.2 if is_warden else 1.0)

#         self.rl_target = None  # For RL coordinator

#     def find_exit(self, exits, pathfinder, maze, hazards):
#         if not exits:
#             return

#         best_exit = None
#         best_score = float('inf')

#         for exit_pos in exits:
#             er, ec = exit_pos
#             dist = abs(self.row - er) + abs(self.col - ec)

#             danger = 0
#             for dr in range(-3, 4):
#                 for dc in range(-3, 4):
#                     if (er + dr, ec + dc) in hazards:
#                         danger += 50

#             score = dist + danger
#             if score < best_score:
#                 best_score = score
#                 best_exit = exit_pos

#         if best_exit:
#             self.target_exit = best_exit
#             self.path = pathfinder.find_path((self.row, self.col), best_exit, maze, hazards)
#             self.path_index = 0

#     def update(self, dt, maze, exits, hazards, pathfinder, alarm_active, people, smoke, neural_aco):
#         if not self.alive or self.escaped:
#             return

#         # Damage
#         if (self.row, self.col) in hazards:
#             self.health -= 30 * dt
#             self.state = STATE_PANICKING

#         smoke_level = smoke.get((self.row, self.col), 0)
#         if smoke_level > 0.5:
#             self.health -= smoke_level * 10 * dt

#         if self.health <= 0:
#             self.alive = False
#             neural_aco.deposit_danger_pheromone((self.row, self.col), 50)
#             return

#         # Check escape
#         if maze[self.row][self.col] == EXIT:
#             self.escaped = True
#             neural_aco.deposit_safe_pheromone(
#                 [(self.row, self.col)] + self.path[:self.path_index], True
#             )
#             sound_system.play('escape', 0.2)
#             return

#         # Awareness
#         if self.state not in [STATE_AWARE, STATE_EVACUATING, STATE_PANICKING, STATE_WARDEN]:
#             if alarm_active:
#                 self.awareness += 0.15 * dt if self.state != STATE_HEADPHONES else 0.02 * dt

#             for h_pos in hazards:
#                 dist = abs(self.row - h_pos[0]) + abs(self.col - h_pos[1])
#                 if dist < 8:
#                     self.awareness += 0.5 / (dist + 1) * dt

#             if self.awareness >= 0.7:
#                 self.state = STATE_AWARE
#                 return

#         if self.state == STATE_AWARE:
#             self.state = STATE_EVACUATING

#         # Movement
#         if not self.moving:
#             if self.state in [STATE_EVACUATING, STATE_PANICKING, STATE_WARDEN]:
#                 if not self.path or self.path_index >= len(self.path):
#                     self.find_exit(exits, pathfinder, maze, hazards)

#                 if self.path and self.path_index < len(self.path):
#                     next_pos = self.path[self.path_index]
#                     nr, nc = next_pos

#                     # Check if path blocked
#                     if (nr, nc) in hazards:
#                         self.find_exit(exits, pathfinder, maze, hazards)
#                         return

#                     self.tx = nc * TILE + TILE // 2
#                     self.ty = nr * TILE + TILE // 2
#                     self.row, self.col = nr, nc
#                     self.path_index += 1
#                     self.moving = True

#         if self.moving:
#             self.walk_frame += dt * 12
#             speed = self.speed * (1.3 if self.state == STATE_PANICKING else 1.0)

#             dx = self.tx - self.x
#             dy = self.ty - self.y
#             dist = math.hypot(dx, dy)

#             if dist < 2:
#                 self.x, self.y = self.tx, self.ty
#                 self.moving = False
#             else:
#                 self.x += (dx / dist) * speed * dt
#                 self.y += (dy / dist) * speed * dt

#     def draw(self, surface, shake, time_val):
#         if not self.alive or self.escaped:
#             return

#         x = int(self.x + shake[0])
#         y = int(self.y + shake[1])

#         # Shadow
#         pygame.draw.ellipse(surface, (30, 30, 35), (x - 5, y + 4, 10, 5))

#         # Legs
#         if self.moving:
#             offset = math.sin(self.walk_frame) * 2
#             pygame.draw.rect(surface, (40, 40, 50), (x - 3 + int(offset), y, 2, 6))
#             pygame.draw.rect(surface, (40, 40, 50), (x + 1 - int(offset), y, 2, 6))
#         else:
#             pygame.draw.rect(surface, (40, 40, 50), (x - 3, y, 2, 6))
#             pygame.draw.rect(surface, (40, 40, 50), (x + 1, y, 2, 6))

#         # Body outline based on state
#         if self.is_warden:
#             outline = Colors.WARDEN
#         elif self.state == STATE_PANICKING:
#             outline = Colors.DANGER
#         elif self.state == STATE_EVACUATING:
#             outline = Colors.EVACUATING
#         elif self.state == STATE_AWARE:
#             outline = Colors.AWARE
#         elif self.state == STATE_HEADPHONES:
#             outline = Colors.HEADPHONES
#         else:
#             outline = (0, 0, 0)

#         # Body
#         pygame.draw.rect(surface, outline, (x - 5, y - 8, 10, 9))
#         pygame.draw.rect(surface, self.color, (x - 4, y - 7, 8, 7))

#         # Head
#         pygame.draw.rect(surface, outline, (x - 4, y - 14, 8, 7))
#         pygame.draw.rect(surface, (230, 190, 160), (x - 3, y - 13, 6, 5))

#         # Warden hat
#         if self.is_warden:
#             pygame.draw.rect(surface, Colors.WARDEN, (x - 4, y - 16, 8, 2))

#         # Health bar
#         if self.health < 90:
#             bar_w = int(8 * self.health / 100)
#             pygame.draw.rect(surface, (150, 0, 0), (x - 4, y - 20, 8, 2))
#             pygame.draw.rect(surface, (0, 200, 0), (x - 4, y - 20, bar_w, 2))

# # ═══════════════════════════════════════════════════════════════════════════════
# # DRAWING FUNCTIONS
# # ═══════════════════════════════════════════════════════════════════════════════

# def draw_tile(surface, row, col, maze, hazards, time_val, shake, alarm_flash):
#     x = int(col * TILE + shake[0])
#     y = int(row * TILE + shake[1])
#     tile = maze[row][col]

#     # Fire rendering
#     if (row, col) in hazards and hazards[(row, col)]['type'] == 'fire':
#         # Fire base
#         pygame.draw.rect(surface, (60, 30, 20), (x, y, TILE, TILE))

#         # Animated flames
#         for i in range(3):
#             fx = x + 2 + i * 4
#             fh = 8 + math.sin(time_val * 12 + i + col * 0.5) * 4

#             # Multi-color flame
#             colors = [Colors.FIRE_CORE, Colors.FIRE_BRIGHT, Colors.FIRE]
#             color = colors[i % 3]

#             points = [
#                 (fx, y + TILE),
#                 (fx + 3, y + TILE),
#                 (fx + 1.5, y + TILE - fh)
#             ]
#             pygame.draw.polygon(surface, color, points)
#         return

#     # Normal tiles
#     if tile == FLOOR:
#         color = Colors.FLOOR if (row + col) % 2 == 0 else Colors.FLOOR_ALT
#         pygame.draw.rect(surface, color, (x, y, TILE, TILE))
#     elif tile == WALL:
#         pygame.draw.rect(surface, Colors.WALL, (x, y, TILE, TILE))
#         # 3D effect
#         pygame.draw.line(surface, Colors.WALL_HIGHLIGHT, (x, y), (x + TILE, y), 1)
#         pygame.draw.line(surface, Colors.WALL_HIGHLIGHT, (x, y), (x, y + TILE), 1)
#     elif tile == CORRIDOR:
#         pygame.draw.rect(surface, Colors.CORRIDOR, (x, y, TILE, TILE))
#     elif tile == CARPET:
#         pygame.draw.rect(surface, Colors.CARPET, (x, y, TILE, TILE))
#     elif tile == EXIT:
#         glow = int(abs(math.sin(time_val * 3)) * 40)
#         color = (50 + glow, 255, 100 + glow)
#         pygame.draw.rect(surface, color, (x, y, TILE, TILE))
#         pygame.draw.rect(surface, (255, 255, 255), (x + 1, y + 1, TILE - 2, TILE - 2), 1)
#     elif tile == DOOR:
#         pygame.draw.rect(surface, Colors.DOOR, (x, y, TILE, TILE))

#     # Alarm flash
#     if alarm_flash and tile != WALL:
#         overlay = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
#         overlay.fill((255, 0, 0, 20))
#         surface.blit(overlay, (x, y))

# def draw_smoke(surface, smoke, shake):
#     for (r, c), level in smoke.items():
#         if level > 0.1:
#             x = int(c * TILE + shake[0])
#             y = int(r * TILE + shake[1])
#             alpha = min(int(level * 120), 180)
#             overlay = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
#             overlay.fill((65, 65, 70, alpha))
#             surface.blit(overlay, (x, y))

# def draw_neural_predictions(surface, predictions, shake, time_val):
#     """Draw neural network predictions as glowing areas."""
#     for r, c, prob in predictions:
#         x = int(c * TILE + shake[0])
#         y = int(r * TILE + shake[1])

#         pulse = abs(math.sin(time_val * 4)) * 0.3 + 0.7
#         alpha = int(60 * prob * pulse)

#         overlay = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
#         overlay.fill((255, 50, 200, alpha))
#         surface.blit(overlay, (x, y))

# def draw_pheromones(surface, neural_aco, shake):
#     """Visualize pheromone trails."""
#     safe = neural_aco.safe_pheromone
#     danger = neural_aco.danger_pheromone

#     for r in range(ROWS):
#         for c in range(COLS):
#             x = int(c * TILE + shake[0])
#             y = int(r * TILE + shake[1])

#             # Safe pheromone (green)
#             if safe[r, c] > 0.3:
#                 alpha = min(int((safe[r, c] - 0.3) * 30), 80)
#                 overlay = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
#                 overlay.fill((0, 255, 150, alpha))
#                 surface.blit(overlay, (x, y))

#             # Danger pheromone (red)
#             if danger[r, c] > 0.5:
#                 alpha = min(int(danger[r, c] * 15), 100)
#                 overlay = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
#                 overlay.fill((255, 80, 80, alpha))
#                 surface.blit(overlay, (x, y))

# def draw_panel(surface, stats, neural_aco, sensor_network, rl_coordinator, time_val, paused, speed):
#     """Draw the enhanced side panel with ML metrics."""
#     px = MAP_WIDTH
#     pygame.draw.rect(surface, Colors.PANEL_BG, (px, 0, PANEL_WIDTH, SCREEN_HEIGHT))
#     pygame.draw.line(surface, Colors.PANEL_BORDER, (px, 0), (px, SCREEN_HEIGHT), 2)

#     font_big = pygame.font.Font(None, 22)
#     font_med = pygame.font.Font(None, 17)
#     font_small = pygame.font.Font(None, 14)

#     y = 8

#     # Title
#     title = font_big.render("NEURAL ACO EMERGENCY SYSTEM", True, Colors.NEURAL_GLOW)
#     surface.blit(title, (px + 8, y))
#     y += 22

#     # Escape counter
#     pygame.draw.rect(surface, Colors.PANEL_BORDER, (px + 8, y, PANEL_WIDTH - 16, 36), 1)
#     escaped_text = font_big.render(f"ESCAPED: {stats['escaped']}/{stats['total']}", True, Colors.SUCCESS)
#     surface.blit(escaped_text, (px + 14, y + 4))
#     progress = stats['escaped'] / max(1, stats['total'])
#     pygame.draw.rect(surface, (40, 60, 40), (px + 12, y + 22, PANEL_WIDTH - 24, 6))
#     pygame.draw.rect(surface, Colors.SUCCESS, (px + 12, y + 22, int((PANEL_WIDTH - 24) * progress), 6))
#     y += 42

#     # Deaths
#     if stats['deaths'] == 0:
#         surface.blit(font_med.render("ZERO DEATHS!", True, Colors.SUCCESS), (px + 10, y))
#     else:
#         surface.blit(font_med.render(f"DEATHS: {stats['deaths']}", True, Colors.DANGER), (px + 10, y))
#     y += 18

#     pygame.draw.line(surface, Colors.PANEL_BORDER, (px + 8, y), (px + PANEL_WIDTH - 8, y), 1)
#     y += 6

#     # Neural Network Section
#     surface.blit(font_med.render("LSTM FIRE PREDICTION", True, Colors.NEURAL_GLOW), (px + 10, y))
#     y += 16

#     confidence = neural_aco.neural_confidence
#     conf_color = Colors.SUCCESS if confidence > 0.7 else Colors.WARNING if confidence > 0.4 else Colors.TEXT_DIM
#     surface.blit(font_small.render(f"Prediction Confidence: {confidence:.1%}", True, conf_color), (px + 12, y))
#     y += 13

#     # Confidence bar
#     pygame.draw.rect(surface, (40, 40, 50), (px + 12, y, 150, 8))
#     pygame.draw.rect(surface, conf_color, (px + 12, y, int(150 * confidence), 8))
#     y += 14

#     predicted_zones = np.count_nonzero(neural_aco.predicted_danger > 0.25)
#     surface.blit(font_small.render(f"Predicted Danger Zones: {predicted_zones}", True, Colors.PREDICTION), (px + 12, y))
#     y += 16

#     pygame.draw.line(surface, Colors.PANEL_BORDER, (px + 8, y), (px + PANEL_WIDTH - 8, y), 1)
#     y += 6

#     # IoT Sensors Section
#     surface.blit(font_med.render("IOT SENSOR NETWORK", True, Colors.SENSOR_ACTIVE), (px + 10, y))
#     y += 16

#     sensor_data = sensor_network.get_sensor_fusion_data()
#     surface.blit(font_small.render(f"Network Coverage: {sensor_data['coverage']:.0f}%", True, Colors.TEXT), (px + 12, y))
#     y += 13
#     surface.blit(font_small.render(f"Avg Temp: {sensor_data['temperature_avg']:.1f}C", True,
#                                    Colors.DANGER if sensor_data['temperature_avg'] > 40 else Colors.TEXT), (px + 12, y))
#     y += 13
#     surface.blit(font_small.render(f"Max Temp: {sensor_data['temperature_max']:.1f}C", True,
#                                    Colors.DANGER if sensor_data['temperature_max'] > 50 else Colors.TEXT), (px + 12, y))
#     y += 13
#     surface.blit(font_small.render(f"Smoke Level: {sensor_data['smoke_level']:.2f}", True,
#                                    Colors.WARNING if sensor_data['smoke_level'] > 0.2 else Colors.TEXT), (px + 12, y))
#     y += 13
#     surface.blit(font_small.render(f"CO Level: {sensor_data['co_level']:.1f} ppm", True,
#                                    Colors.DANGER if sensor_data['co_level'] > 30 else Colors.TEXT), (px + 12, y))
#     y += 13
#     surface.blit(font_small.render(f"Triggered Sensors: {len(sensor_data['triggered_sensors'])}", True,
#                                    Colors.DANGER if sensor_data['triggered_sensors'] else Colors.TEXT), (px + 12, y))
#     y += 16

#     pygame.draw.line(surface, Colors.PANEL_BORDER, (px + 8, y), (px + PANEL_WIDTH - 8, y), 1)
#     y += 6

#     # RL Section
#     surface.blit(font_med.render("RL COORDINATOR", True, Colors.ACCENT), (px + 10, y))
#     y += 16

#     rl_stats = rl_coordinator.get_stats()
#     surface.blit(font_small.render(f"Decisions Made: {rl_stats['decisions']}", True, Colors.TEXT), (px + 12, y))
#     y += 13
#     surface.blit(font_small.render(f"Avg Reward: {rl_stats['avg_reward']:.2f}", True, Colors.TEXT), (px + 12, y))
#     y += 13
#     surface.blit(font_small.render(f"Exploration: {rl_stats['epsilon']:.0%}", True, Colors.TEXT), (px + 12, y))
#     y += 16

#     pygame.draw.line(surface, Colors.PANEL_BORDER, (px + 8, y), (px + PANEL_WIDTH - 8, y), 1)
#     y += 6

#     # ACO Metrics
#     surface.blit(font_med.render("ANT COLONY OPTIMIZATION", True, Colors.SAFE_PHEROMONE), (px + 10, y))
#     y += 16

#     safe_avg = np.mean(neural_aco.safe_pheromone)
#     danger_avg = np.mean(neural_aco.danger_pheromone)
#     surface.blit(font_small.render(f"Safe Pheromone Avg: {safe_avg:.2f}", True, Colors.SAFE_PHEROMONE), (px + 12, y))
#     y += 13
#     surface.blit(font_small.render(f"Danger Pheromone Avg: {danger_avg:.2f}", True, Colors.DANGER_PHEROMONE), (px + 12, y))
#     y += 13
#     surface.blit(font_small.render(f"Path Edges Tracked: {len(neural_aco.edge_usage)}", True, Colors.TEXT), (px + 12, y))
#     y += 16

#     pygame.draw.line(surface, Colors.PANEL_BORDER, (px + 8, y), (px + PANEL_WIDTH - 8, y), 1)
#     y += 6

#     # Controls
#     surface.blit(font_med.render("CONTROLS", True, Colors.TEXT), (px + 10, y))
#     y += 14

#     controls = [
#         "Click: Add fire",
#         "A: Trigger alarm",
#         "P: Toggle predictions",
#         "T: Toggle pheromones",
#         "S: Toggle sensors",
#         "Space: Pause | R: Reset",
#         "+/-: Speed"
#     ]

#     for ctrl in controls:
#         surface.blit(font_small.render(ctrl, True, Colors.TEXT_DIM), (px + 12, y))
#         y += 12

#     # Status bar at bottom
#     y = SCREEN_HEIGHT - 25
#     status = f"Speed: {speed:.1f}x"
#     if paused:
#         status = "PAUSED | " + status
#     surface.blit(font_small.render(status, True, Colors.TEXT_DIM), (px + 10, y))

# def draw_bottom_bar(surface, stats, alarm_active, alarm_flash, time_val):
#     """Draw bottom status bar."""
#     by = MAP_HEIGHT

#     bg_color = (40, 20, 20) if alarm_active and alarm_flash else Colors.PANEL_BG
#     pygame.draw.rect(surface, bg_color, (0, by, MAP_WIDTH, 80))
#     pygame.draw.line(surface, Colors.PANEL_BORDER, (0, by), (MAP_WIDTH, by), 2)

#     font = pygame.font.Font(None, 20)
#     font_big = pygame.font.Font(None, 28)

#     color = Colors.SUCCESS if stats['deaths'] == 0 else Colors.DANGER
#     text = f"ESCAPED: {stats['escaped']}/{stats['total']}  |  DEATHS: {stats['deaths']}"
#     surface.blit(font_big.render(text, True, color), (15, by + 10))

#     alive = stats['total'] - stats['escaped'] - stats['deaths']
#     surface.blit(font.render(f"Inside Building: {alive}", True, Colors.TEXT_DIM), (15, by + 40))

#     if alarm_active:
#         alarm_color = Colors.DANGER if alarm_flash else Colors.WARNING
#         surface.blit(font.render("ALARM ACTIVE - EVACUATE!", True, alarm_color), (15, by + 58))

# # ═══════════════════════════════════════════════════════════════════════════════
# # SPAWN FUNCTIONS
# # ═══════════════════════════════════════════════════════════════════════════════

# def spawn_people(maze, count, num_wardens):
#     people = []
#     spawns = []

#     for r in range(2, ROWS - 2):
#         for c in range(2, COLS - 2):
#             if maze[r][c] in [CARPET, FLOOR, CORRIDOR]:
#                 spawns.append((r, c))

#     random.shuffle(spawns)

#     states = [STATE_WORKING] * 15 + [STATE_HEADPHONES] * 8
#     random.shuffle(states)

#     # Wardens
#     corridor_spawns = [(r, c) for r, c in spawns if maze[r][c] == CORRIDOR]
#     for i in range(min(num_wardens, len(corridor_spawns))):
#         r, c = corridor_spawns[i]
#         people.append(Person(i, r, c, STATE_WARDEN, is_warden=True))

#     # Regular people
#     for i in range(num_wardens, min(count, len(spawns))):
#         r, c = spawns[i]
#         state = states[(i - num_wardens) % len(states)]
#         people.append(Person(i, r, c, state))

#     return people

# def spawn_sensors(maze, count):
#     """Spawn IoT sensors throughout the building."""
#     network = IoTSensorNetwork()
#     spawns = []

#     for r in range(3, ROWS - 3, 4):
#         for c in range(3, COLS - 3, 4):
#             if maze[r][c] != WALL:
#                 spawns.append((r, c))

#     random.shuffle(spawns)
#     sensor_types = ['temperature', 'smoke', 'co', 'motion']

#     for i, (r, c) in enumerate(spawns[:count]):
#         sensor_type = sensor_types[i % len(sensor_types)]
#         sensor = IoTSensor(
#             id=i,
#             row=r,
#             col=c,
#             sensor_type=sensor_type
#         )
#         network.add_sensor(sensor)

#     return network

# # ═══════════════════════════════════════════════════════════════════════════════
# # MAIN
# # ═══════════════════════════════════════════════════════════════════════════════

# def main():
#     if HEADLESS:
#         print("Headless mode enabled; use run_headless_simulation() instead of the interactive loop.")
#         return

#     # Initialize systems
#     maze, exits = generate_building()
#     lstm_predictor = SimpleLSTMPredictor()
#     neural_aco = NeuralACO(lstm_predictor)
#     pathfinder = NeuralPathfinder(neural_aco)
#     disasters = Disasters()
#     alarm = AlarmSystem()
#     sensor_network = spawn_sensors(maze, NUM_SENSORS)
#     rl_coordinator = RLEvacuationCoordinator()
#     people = spawn_people(maze, TOTAL_PEOPLE, NUM_WARDENS)

#     stats = {'escaped': 0, 'deaths': 0, 'total': TOTAL_PEOPLE}

#     clock = pygame.time.Clock()
#     running = True
#     paused = False
#     speed = 1.0

#     show_predictions = True
#     show_pheromones = True
#     show_sensors = True

#     neural_update_timer = 0
#     rl_update_timer = 0

#     print("=" * 70)
#     print("NEURAL ACO EMERGENCY RESPONSE SYSTEM (NAERS)")
#     print("=" * 70)
#     print("\nPATENTABLE INNOVATIONS:")
#     print("  - Neural Predictive ACO (LSTM + Pheromones)")
#     print("  - IoT Sensor Fusion with Kalman Filtering")
#     print("  - RL-based Evacuation Coordination")
#     print("\nCONTROLS: Click=Fire, A=Alarm, P=Predictions, T=Pheromones, S=Sensors")
#     print("=" * 70)

#     while running:
#         dt = clock.tick(60) / 1000.0
#         dt *= speed
#         time_val = pygame.time.get_ticks() / 1000.0

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False

#             elif event.type == pygame.MOUSEBUTTONDOWN:
#                 mx, my = event.pos
#                 if mx < MAP_WIDTH and my < MAP_HEIGHT:
#                     col = int((mx - disasters.shake_offset[0]) // TILE)
#                     row = int((my - disasters.shake_offset[1]) // TILE)
#                     if 0 < row < ROWS - 1 and 0 < col < COLS - 1:
#                         if event.button == 1:
#                             disasters.add_fire(row, col)
#                             pathfinder.clear_cache()

#             elif event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_SPACE:
#                     paused = not paused
#                 elif event.key == pygame.K_a:
#                     alarm.trigger()
#                 elif event.key == pygame.K_p:
#                     show_predictions = not show_predictions
#                 elif event.key == pygame.K_t:
#                     show_pheromones = not show_pheromones
#                 elif event.key == pygame.K_s:
#                     show_sensors = not show_sensors
#                 elif event.key == pygame.K_r:
#                     # Reset
#                     sound_system.stop_alarm()
#                     maze, exits = generate_building()
#                     lstm_predictor.reset()
#                     neural_aco.reset()
#                     pathfinder.clear_cache()
#                     disasters = Disasters()
#                     alarm = AlarmSystem()
#                     sensor_network = spawn_sensors(maze, NUM_SENSORS)
#                     rl_coordinator = RLEvacuationCoordinator()
#                     people = spawn_people(maze, TOTAL_PEOPLE, NUM_WARDENS)
#                     stats = {'escaped': 0, 'deaths': 0, 'total': TOTAL_PEOPLE}
#                 elif event.key in [pygame.K_EQUALS, pygame.K_PLUS]:
#                     speed = min(speed + 0.5, 5.0)
#                 elif event.key == pygame.K_MINUS:
#                     speed = max(speed - 0.5, 0.5)

#         if not paused:
#             # Update disasters
#             disasters.update(dt, maze, neural_aco)
#             alarm.update(dt)

#             # Auto-trigger alarm on fire
#             if disasters.hazards and not alarm.active:
#                 alarm.trigger()

#             # Update neural predictions
#             neural_update_timer += dt
#             if neural_update_timer > 0.5:
#                 fire_positions = disasters.get_fire_positions()
#                 sensor_data = sensor_network.get_sensor_fusion_data()
#                 neural_aco.update_predictions(fire_positions, sensor_data, maze)
#                 neural_update_timer = 0

#             # Update sensors
#             people_positions = [(p.row, p.col) for p in people if p.alive and not p.escaped]
#             sensor_network.update(dt, disasters.get_fire_positions(), disasters.smoke, people_positions, maze)

#             # Update RL coordinator
#             rl_update_timer += dt
#             if rl_update_timer > 2.0 and alarm.active:
#                 wardens = [p for p in people if p.is_warden and p.alive]
#                 exits_status = {e: False for e in exits}  # Simplified
#                 rl_coordinator.step(disasters.get_fire_positions(), people, exits_status, wardens)
#                 rl_update_timer = 0

#             # Pheromone evaporation
#             neural_aco.evaporate()

#             # Update people
#             prev_escaped = stats['escaped']
#             prev_deaths = stats['deaths']

#             for p in people:
#                 p.update(dt, maze, exits, disasters.hazards, pathfinder,
#                         alarm.active, people, disasters.smoke, neural_aco)

#             # Update stats
#             stats['escaped'] = sum(1 for p in people if p.escaped)
#             stats['deaths'] = sum(1 for p in people if not p.alive)

#             if stats['escaped'] > prev_escaped:
#                 sound_system.play('escape', 0.2)

#         # RENDER
#         screen.fill((30, 32, 38))
#         shake = disasters.shake_offset

#         # Draw map
#         for r in range(ROWS):
#             for c in range(COLS):
#                 draw_tile(screen, r, c, maze, disasters.hazards, time_val, shake,
#                          alarm.active and alarm.flash_state)

#         # Draw smoke
#         draw_smoke(screen, disasters.smoke, shake)

#         # Draw pheromones
#         if show_pheromones:
#             draw_pheromones(screen, neural_aco, shake)

#         # Draw neural predictions
#         if show_predictions:
#             fire_positions = disasters.get_fire_positions()
#             predictions = lstm_predictor.predict_spread(fire_positions, {}, maze)
#             draw_neural_predictions(screen, predictions, shake, time_val)

#         # Draw sensors
#         if show_sensors:
#             sensor_network.draw(screen, shake, time_val)

#         # Draw particles
#         disasters.draw_particles(screen, shake)

#         # Draw people
#         for p in sorted(people, key=lambda x: x.y):
#             p.draw(screen, shake, time_val)

#         # Draw UI
#         draw_panel(screen, stats, neural_aco, sensor_network, rl_coordinator, time_val, paused, speed)
#         draw_bottom_bar(screen, stats, alarm.active, alarm.flash_state, time_val)

#         pygame.display.flip()

#     pygame.quit()
#     sound_system.stop_alarm()

#     print(f"\n{'='*60}")
#     print(f"FINAL: Escaped {stats['escaped']}/{stats['total']}, Deaths: {stats['deaths']}")
#     if stats['deaths'] == 0:
#         print("PERFECT! ZERO DEATHS ACHIEVED!")
#     print(f"Neural Confidence Final: {neural_aco.neural_confidence:.1%}")
#     print(f"RL Decisions Made: {rl_coordinator.decisions_made}")
#     print(f"{'='*60}")

# def run_headless_simulation(steps=240, dt=1 / 30.0):
#     """
#     Run a trimmed-down simulation loop without rendering.
#     Designed for serverless environments (e.g., Vercel) where no display/audio exists.
#     """
#     maze, exits = generate_building()
#     lstm_predictor = SimpleLSTMPredictor()
#     neural_aco = NeuralACO(lstm_predictor)
#     pathfinder = NeuralPathfinder(neural_aco)
#     disasters = Disasters()
#     alarm = AlarmSystem()
#     sensor_network = spawn_sensors(maze, NUM_SENSORS)
#     rl_coordinator = RLEvacuationCoordinator()
#     people = spawn_people(maze, TOTAL_PEOPLE, NUM_WARDENS)

#     stats = {'escaped': 0, 'deaths': 0, 'total': TOTAL_PEOPLE}

#     # Seed an ignition so the loop has meaningful activity.
#     disasters.add_fire(ROWS // 2, COLS // 2)

#     neural_update_timer = 0.0
#     rl_update_timer = 0.0
#     steps_run = 0

#     for _ in range(int(steps)):
#         steps_run += 1
#         disasters.update(dt, maze, neural_aco)
#         alarm.update(dt)

#         if disasters.hazards and not alarm.active:
#             alarm.trigger()

#         neural_update_timer += dt
#         if neural_update_timer > 0.5:
#             fire_positions = disasters.get_fire_positions()
#             sensor_data = sensor_network.get_sensor_fusion_data()
#             neural_aco.update_predictions(fire_positions, sensor_data, maze)
#             neural_update_timer = 0.0

#         people_positions = [(p.row, p.col) for p in people if p.alive and not p.escaped]
#         sensor_network.update(dt, disasters.get_fire_positions(), disasters.smoke, people_positions, maze)

#         rl_update_timer += dt
#         if rl_update_timer > 2.0 and alarm.active:
#             wardens = [p for p in people if p.is_warden and p.alive]
#             exits_status = {e: False for e in exits}
#             rl_coordinator.step(disasters.get_fire_positions(), people, exits_status, wardens)
#             rl_update_timer = 0.0

#         neural_aco.evaporate()

#         for p in people:
#             p.update(dt, maze, exits, disasters.hazards, pathfinder,
#                      alarm.active, people, disasters.smoke, neural_aco)

#         stats['escaped'] = sum(1 for p in people if p.escaped)
#         stats['deaths'] = sum(1 for p in people if not p.alive)

#         # Exit early if everyone is resolved
#         if stats['escaped'] + stats['deaths'] >= stats['total']:
#             break

#     sensor_snapshot = sensor_network.get_sensor_fusion_data()
#     return {
#         'steps_run': steps_run,
#         'escaped': stats['escaped'],
#         'deaths': stats['deaths'],
#         'alive': stats['total'] - stats['escaped'] - stats['deaths'],
#         'fires_active': len(disasters.get_fire_positions()),
#         'neural_confidence': neural_aco.neural_confidence,
#         'rl_decisions': rl_coordinator.decisions_made,
#         'sensor_coverage': sensor_snapshot.get('coverage', 0),
#         'avg_temp': sensor_snapshot.get('temperature_avg', 0),
#     }

# if __name__ == "__main__":
#     main()



"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                  ║
║     ██████╗ ██╗    ██╗██╗ ██████╗ ██╗  ██╗████████╗    ██╗   ██╗██╗  ██╗                         ║
║     ██╔══██╗██║    ██║██║██╔════╝ ██║  ██║╚══██╔══╝    ██║   ██║╚██╗██╔╝                         ║
║     ██║  ██║██║ █╗ ██║██║██║  ███╗███████║   ██║       ██║   ██║ ╚███╔╝                          ║
║     ██║  ██║██║███╗██║██║██║   ██║██╔══██║   ██║       ██║   ██║ ██╔██╗                          ║
║     ██████╔╝╚███╔███╔╝██║╚██████╔╝██║  ██║   ██║        ╚████╔╝ ██╔╝ ██╗                         ║
║     ╚═════╝  ╚══╝╚══╝ ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝         ╚═══╝  ╚═╝  ╚═╝                         ║
║                                                                                                  ║
║              🛡️ NEURAL ACO EMERGENCY RESPONSE SYSTEM (NAERS) 🛡️                                  ║
║                                                                                                  ║
║  PATENTABLE INNOVATIONS:                                                                         ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                   ║
║  🧠 Neural Predictive ACO (NP-ACO):                                                              ║
║     - LSTM network predicts fire spread 30-60 seconds ahead                                      ║
║     - ACO pheromones dynamically weighted by neural predictions                                  ║
║     - Real-time path recalculation based on predicted danger zones                               ║
║                                                                                                  ║
║  📡 IoT Sensor Fusion Layer:                                                                     ║
║     - Simulated temperature, smoke, CO, motion sensors                                           ║
║     - Kalman filtering for noise reduction                                                       ║
║     - Sensor health monitoring with redundancy                                                   ║
║                                                                                                  ║
║  🤖 Reinforcement Learning Evacuation Coordinator:                                               ║
║     - Q-learning agent optimizes warden deployment                                               ║
║     - Multi-agent coordination for bottleneck prevention                                         ║
║     - Adaptive crowd flow management                                                             ║
║                                                                                                  ║
║  🎯 3D Perspective View (Non-Isometric):                                                         ║
║     - True perspective projection with depth                                                     ║
║     - Dynamic lighting based on fire/emergency lights                                            ║
║     - Particle systems for fire, smoke, water                                                    ║
║                                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

CORE USP FOR PATENT:
═══════════════════
"Adaptive Neural-Pheromone Emergency Evacuation System (ANPEES)"

A system that combines:
1. Ant Colony Optimization for distributed pathfinding
2. LSTM neural networks for hazard prediction
3. IoT sensor fusion for real-time environmental awareness
4. Reinforcement learning for dynamic resource allocation

The pheromone weights are DYNAMICALLY MODULATED by neural network confidence scores,
creating a hybrid bio-inspired + deep learning approach that is novel and patentable.

TACTICAL BATTLEFIELD VARIANT (INDIAN ARMY OFFLINE MESH)
═══════════════════════════════════════════════════════
- Soldier-worn handheld devices form a mobile ad-hoc mesh (no towers, no internet)
- Each soldier node = sensor hub + router + mini-planner feeding the Neural ACO core
- Squad-level or fully distributed brains compute blast-safe / ambush-safe routes
- Works in infrastructure-denied urban combat and border scenarios
"""

import os

# Headless mode allows this module to be imported by serverless environments (e.g., Vercel)
# where there is no display or audio device.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
HEADLESS = bool(os.environ.get("HEADLESS") or os.environ.get("VERCEL") or os.environ.get("CI"))
if HEADLESS:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pygame.gfxdraw
import random
import math
import numpy as np
from collections import defaultdict, deque
from heapq import heappush, heappop
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import time

pygame.init()
audio_ready = False
if not HEADLESS:
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        audio_ready = True
    except Exception as e:
        # Keep running without audio if mixer fails (e.g., no device)
        print(f"[audio] mixer init failed: {e}. Running in silent mode.")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

ROWS = 45  # Keep rows/cols consistent with original for map generation
COLS = 70
TILE = 24  # Larger tiles for detailed 3D rendering
TOTAL_PEOPLE = 60
NUM_WARDENS = 4
NUM_SENSORS = 25

MAP_WIDTH = COLS * TILE
MAP_HEIGHT = ROWS * TILE
PANEL_WIDTH = 380
SCREEN_WIDTH = MAP_WIDTH + PANEL_WIDTH
SCREEN_HEIGHT = MAP_HEIGHT + 80

if HEADLESS:
    screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
else:
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("🧠 DWIGHT UX - Neural ACO Emergency Response System")

# ═══════════════════════════════════════════════════════════════════════════════
# DEEP LEARNING - LSTM FIRE PREDICTION (Simplified NumPy Implementation)
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleLSTMPredictor:
    """
    Simplified LSTM-like predictor for fire spread.
    In production, this would use PyTorch/TensorFlow.

    This predicts WHERE fire will spread based on:
    - Current fire positions
    - Historical spread patterns
    - Sensor readings (temperature gradients)
    """
    def __init__(self, grid_size=(ROWS, COLS)):
        self.grid_size = grid_size
        self.hidden_size = 32
        self.sequence_length = 10

        # Simulated learned weights
        np.random.seed(42)
        self.W_forget = np.random.randn(self.hidden_size, self.hidden_size + 4) * 0.1
        self.W_input = np.random.randn(self.hidden_size, self.hidden_size + 4) * 0.1
        self.W_output = np.random.randn(self.hidden_size, self.hidden_size + 4) * 0.1
        self.W_cell = np.random.randn(self.hidden_size, self.hidden_size + 4) * 0.1
        self.W_pred = np.random.randn(4, self.hidden_size) * 0.1  # 4 directions

        self.hidden = np.zeros(self.hidden_size)
        self.cell = np.zeros(self.hidden_size)

        # History buffer
        self.history = deque(maxlen=self.sequence_length)
        self.prediction_confidence = 0.0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))

    def extract_features(self, fire_positions, sensor_readings):
        """Extract features from current state."""
        if not fire_positions:
            return np.zeros(4)

        # Feature: center of fire mass
        center_r = np.mean([p[0] for p in fire_positions])
        center_c = np.mean([p[1] for p in fire_positions])

        # Feature: fire spread direction (based on recent history)
        spread_r, spread_c = 0, 0
        if len(self.history) > 1:
            prev_center = self.history[-1][:2]
            spread_r = center_r - prev_center[0]
            spread_c = center_c - prev_center[1]

        return np.array([center_r / ROWS, center_c / COLS, spread_r, spread_c])

    def forward(self, features):
        """LSTM forward pass."""
        concat = np.concatenate([self.hidden, features])

        forget_gate = self.sigmoid(self.W_forget @ concat)
        input_gate = self.sigmoid(self.W_input @ concat)
        output_gate = self.sigmoid(self.W_output @ concat)
        cell_candidate = self.tanh(self.W_cell @ concat)

        self.cell = forget_gate * self.cell + input_gate * cell_candidate
        self.hidden = output_gate * self.tanh(self.cell)

        # Predict spread probabilities for 4 directions
        direction_probs = self.sigmoid(self.W_pred @ self.hidden)
        return direction_probs

    def predict_spread(self, fire_positions, sensor_readings, maze, steps_ahead=3):
        """
        Predict where fire will spread in the next N steps.
        Returns list of (row, col, probability) tuples.
        """
        if not fire_positions:
            self.prediction_confidence = 0.0
            return []

        features = self.extract_features(fire_positions, sensor_readings)
        self.history.append(features)

        direction_probs = self.forward(features)
        self.prediction_confidence = float(np.max(direction_probs))

        predictions = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E

        for fire_pos in fire_positions:
            for i, (dr, dc) in enumerate(directions):
                prob = direction_probs[i]
                if prob > 0.3:  # Threshold
                    for step in range(1, steps_ahead + 1):
                        nr = fire_pos[0] + dr * step
                        nc = fire_pos[1] + dc * step
                        if 0 < nr < ROWS - 1 and 0 < nc < COLS - 1:
                            if maze[nr][nc] != 1:  # Not wall
                                decay = 0.7 ** step
                                predictions.append((nr, nc, prob * decay))

        # Aggregate predictions
        pred_dict = defaultdict(float)
        for r, c, p in predictions:
            pred_dict[(r, c)] = max(pred_dict[(r, c)], p)

        return [(r, c, p) for (r, c), p in pred_dict.items() if p > 0.25]

    def reset(self):
        self.hidden = np.zeros(self.hidden_size)
        self.cell = np.zeros(self.hidden_size)
        self.history.clear()
        self.prediction_confidence = 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# IOT SENSOR SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IoTSensor:
    """Simulated IoT sensor with realistic behavior."""
    id: int
    row: int
    col: int
    sensor_type: str  # 'temperature', 'smoke', 'co', 'motion'
    value: float = 0.0
    threshold: float = 0.0
    triggered: bool = False
    health: float = 100.0
    battery: float = 100.0
    noise_level: float = 0.05
    last_reading: float = 0.0

    def __post_init__(self):
        if self.sensor_type == 'temperature':
            self.threshold = 45.0  # Celsius
            self.value = 22.0  # Room temp
        elif self.sensor_type == 'smoke':
            self.threshold = 0.3
            self.value = 0.0
        elif self.sensor_type == 'co':
            self.threshold = 35.0  # ppm
            self.value = 0.0
        elif self.sensor_type == 'motion':
            self.threshold = 0.5
            self.value = 0.0

class IoTSensorNetwork:
    """
    Simulated IoT sensor network with:
    - Multiple sensor types
    - Kalman filtering for noise reduction
    - Battery/health simulation
    - Mesh network communication with inter-sensor alerting
    - Alert propagation across the entire office
    """
    def __init__(self):
        self.sensors: Dict[int, IoTSensor] = {}
        self.sensor_grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self.alerts: List[dict] = []
        self.network_latency = 0.1  # seconds
        self.packet_loss_rate = 0.02

        # Kalman filter state for each sensor
        self.kalman_state: Dict[int, dict] = {}

        # === MESH NETWORK COMMUNICATION ===
        self.communication_range = 12  # Tiles - sensors can talk to neighbors within this range
        self.sensor_links: Dict[int, List[int]] = {}  # Adjacency list for sensor mesh
        self.alert_propagation: Dict[int, float] = {}  # sensor_id -> alert level (0-1)
        self.propagation_speed = 2.0  # How fast alerts spread through network
        self.global_alert_level = 0.0  # Overall office alert level (0-1)
        self.alert_source_positions: List[Tuple[int, int]] = []  # Where alerts originated
        self.communication_pulses: List[dict] = []  # Visual pulses traveling between sensors
        self.network_alarm_triggered = False  # Has the network triggered a full alarm?

    def add_sensor(self, sensor: IoTSensor):
        self.sensors[sensor.id] = sensor
        self.sensor_grid[(sensor.row, sensor.col)].append(sensor.id)
        self.kalman_state[sensor.id] = {
            'estimate': sensor.value,
            'error': 1.0,
            'process_noise': 0.01,
            'measurement_noise': sensor.noise_level
        }
        # Initialize alert propagation state
        self.alert_propagation[sensor.id] = 0.0
        self.sensor_links[sensor.id] = []

    def build_mesh_network(self):
        """Build the sensor mesh network by connecting sensors within communication range."""
        self.sensor_links = {sid: [] for sid in self.sensors}

        sensor_list = list(self.sensors.values())
        for i, s1 in enumerate(sensor_list):
            for s2 in sensor_list[i+1:]:
                dist = abs(s1.row - s2.row) + abs(s1.col - s2.col)
                if dist <= self.communication_range:
                    self.sensor_links[s1.id].append(s2.id)
                    self.sensor_links[s2.id].append(s1.id)

    def propagate_alerts(self, dt):
        """Propagate alerts through the sensor mesh network."""
        new_propagation = dict(self.alert_propagation)

        for sensor_id, alert_level in self.alert_propagation.items():
            if alert_level > 0.1:  # Only propagate if there's significant alert
                sensor = self.sensors.get(sensor_id)
                if not sensor or sensor.health <= 0:
                    continue

                # Spread to connected sensors
                for neighbor_id in self.sensor_links.get(sensor_id, []):
                    neighbor = self.sensors.get(neighbor_id)
                    if not neighbor or neighbor.health <= 0:
                        continue

                    # Calculate propagation based on distance
                    dist = abs(sensor.row - neighbor.row) + abs(sensor.col - neighbor.col)
                    propagation_strength = alert_level * 0.8 * (1 - dist / self.communication_range)

                    # Update neighbor's alert level if our signal is stronger
                    if propagation_strength > new_propagation[neighbor_id]:
                        new_propagation[neighbor_id] = max(new_propagation[neighbor_id],
                                                          propagation_strength - 0.1)

                        # Create visual pulse for communication
                        if random.random() < 0.3:  # Don't create too many pulses
                            self.communication_pulses.append({
                                'from': (sensor.row, sensor.col),
                                'to': (neighbor.row, neighbor.col),
                                'progress': 0.0,
                                'speed': 3.0,
                                'color': (255, 200, 50) if alert_level > 0.7 else (100, 200, 255)
                            })

        # Apply propagation updates
        self.alert_propagation = new_propagation

        # Update communication pulses
        updated_pulses = []
        for pulse in self.communication_pulses:
            pulse['progress'] += pulse['speed'] * dt
            if pulse['progress'] < 1.0:
                updated_pulses.append(pulse)
        self.communication_pulses = updated_pulses

        # Calculate global alert level (average of all sensor alerts)
        active_alerts = [a for sid, a in self.alert_propagation.items()
                        if self.sensors.get(sid) and self.sensors[sid].health > 0]
        if active_alerts:
            self.global_alert_level = sum(active_alerts) / len(active_alerts)

            # Trigger network-wide alarm if enough sensors are alerted
            alerted_count = sum(1 for a in active_alerts if a > 0.5)
            if alerted_count >= 3 and not self.network_alarm_triggered:
                self.network_alarm_triggered = True
                sound_system.play('alarm', 0.5)

    def kalman_update(self, sensor_id: int, measurement: float) -> float:
        """Apply Kalman filter to sensor reading."""
        state = self.kalman_state[sensor_id]

        # Prediction
        predicted_estimate = state['estimate']
        predicted_error = state['error'] + state['process_noise']

        # Update
        kalman_gain = predicted_error / (predicted_error + state['measurement_noise'])
        state['estimate'] = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        state['error'] = (1 - kalman_gain) * predicted_error

        return state['estimate']

    def update(self, dt, fire_positions, smoke_map, people_positions, maze):
        """Update all sensors based on environment and propagate alerts through network."""
        self.alerts.clear()

        for sensor_id, sensor in self.sensors.items():
            # Battery drain
            sensor.battery -= 0.001 * dt
            if sensor.battery <= 0:
                sensor.health = 0
                continue

            # Health degradation in fire
            if (sensor.row, sensor.col) in fire_positions:
                sensor.health -= 5 * dt

            if sensor.health <= 0:
                # Dead sensors can't propagate alerts
                self.alert_propagation[sensor_id] = 0
                continue

            # Calculate raw reading based on environment
            raw_value = self._calculate_raw_reading(sensor, fire_positions, smoke_map, people_positions)

            # Add noise
            noise = np.random.normal(0, sensor.noise_level)
            noisy_value = raw_value + noise

            # Apply Kalman filter
            filtered_value = self.kalman_update(sensor_id, noisy_value)
            sensor.value = filtered_value
            sensor.last_reading = filtered_value

            # Check threshold
            was_triggered = sensor.triggered
            sensor.triggered = filtered_value > sensor.threshold

            # Generate alert on state change and initiate network propagation
            if sensor.triggered and not was_triggered:
                if random.random() > self.packet_loss_rate:
                    self.alerts.append({
                        'sensor_id': sensor_id,
                        'type': sensor.sensor_type,
                        'value': filtered_value,
                        'position': (sensor.row, sensor.col),
                        'timestamp': time.time()
                    })

                    # === INITIATE ALERT PROPAGATION ===
                    # This sensor becomes an alert source - full alert level
                    self.alert_propagation[sensor_id] = 1.0
                    self.alert_source_positions.append((sensor.row, sensor.col))

                    # Play sensor alert sound
                    sound_system.play('sensor_alert', 0.4)

            # Maintain alert level for triggered sensors
            if sensor.triggered:
                self.alert_propagation[sensor_id] = max(self.alert_propagation.get(sensor_id, 0), 0.9)

        # === PROPAGATE ALERTS THROUGH MESH NETWORK ===
        self.propagate_alerts(dt)

        # Decay alert propagation over time (alerts fade if source is gone)
        for sensor_id in self.alert_propagation:
            sensor = self.sensors.get(sensor_id)
            if sensor and not sensor.triggered:
                self.alert_propagation[sensor_id] *= 0.98  # Slow decay

    def _calculate_raw_reading(self, sensor, fire_positions, smoke_map, people_positions):
        """Calculate sensor reading based on environment."""
        if sensor.sensor_type == 'temperature':
            base_temp = 22.0
            for fire_pos in fire_positions:
                dist = abs(sensor.row - fire_pos[0]) + abs(sensor.col - fire_pos[1])
                if dist < 10:
                    base_temp += 100 / (dist + 1)
            return min(base_temp, 200)

        elif sensor.sensor_type == 'smoke':
            return smoke_map.get((sensor.row, sensor.col), 0)

        elif sensor.sensor_type == 'co':
            co = 0
            for fire_pos in fire_positions:
                dist = abs(sensor.row - fire_pos[0]) + abs(sensor.col - fire_pos[1])
                if dist < 8:
                    co += 50 / (dist + 1)
            return min(co, 500)

        elif sensor.sensor_type == 'motion':
            motion = 0
            for px, py in people_positions:
                dist = abs(sensor.row - px) + abs(sensor.col - py)
                if dist < 5:
                    motion += 1 / (dist + 1)
            return min(motion, 5)

        return 0

    def get_sensor_fusion_data(self) -> dict:
        """Fuse data from all sensors for decision making."""
        data = {
            'temperature_avg': 0,
            'temperature_max': 0,
            'smoke_level': 0,
            'co_level': 0,
            'motion_detected': 0,
            'triggered_sensors': [],
            'sensor_health': 0,
            'coverage': 0
        }

        temp_sensors = [s for s in self.sensors.values() if s.sensor_type == 'temperature' and s.health > 0]
        smoke_sensors = [s for s in self.sensors.values() if s.sensor_type == 'smoke' and s.health > 0]
        co_sensors = [s for s in self.sensors.values() if s.sensor_type == 'co' and s.health > 0]
        motion_sensors = [s for s in self.sensors.values() if s.sensor_type == 'motion' and s.health > 0]

        if temp_sensors:
            data['temperature_avg'] = np.mean([s.value for s in temp_sensors])
            data['temperature_max'] = max(s.value for s in temp_sensors)

        if smoke_sensors:
            data['smoke_level'] = np.mean([s.value for s in smoke_sensors])

        if co_sensors:
            data['co_level'] = np.mean([s.value for s in co_sensors])

        if motion_sensors:
            data['motion_detected'] = sum(1 for s in motion_sensors if s.value > 0.5)

        data['triggered_sensors'] = [s.id for s in self.sensors.values() if s.triggered]

        alive_sensors = [s for s in self.sensors.values() if s.health > 0]
        if alive_sensors:
            data['sensor_health'] = np.mean([s.health for s in alive_sensors])
            data['coverage'] = len(alive_sensors) / len(self.sensors) * 100

        return data

    def draw(self, surface, shake, time_val):
        """Draw 3D sensors with mesh network links and communication visualization."""

        # === DRAW MESH NETWORK LINKS FIRST (underneath sensors) ===
        for sensor_id, neighbors in self.sensor_links.items():
            sensor = self.sensors.get(sensor_id)
            if not sensor or sensor.health <= 0:
                continue

            x1 = int(sensor.col * TILE + TILE // 2 + shake[0])
            y1 = int(sensor.row * TILE + TILE // 2 + shake[1])

            for neighbor_id in neighbors:
                if neighbor_id <= sensor_id:  # Avoid drawing links twice
                    continue
                neighbor = self.sensors.get(neighbor_id)
                if not neighbor or neighbor.health <= 0:
                    continue

                x2 = int(neighbor.col * TILE + TILE // 2 + shake[0])
                y2 = int(neighbor.row * TILE + TILE // 2 + shake[1])

                # Link color based on alert propagation
                alert1 = self.alert_propagation.get(sensor_id, 0)
                alert2 = self.alert_propagation.get(neighbor_id, 0)
                max_alert = max(alert1, alert2)

                if max_alert > 0.5:
                    # Active alert - orange/red pulsing line
                    pulse = abs(math.sin(time_val * 6))
                    link_color = (255, int(100 + 100 * pulse), 50, int(100 + 100 * max_alert))
                    link_width = 2
                elif max_alert > 0.1:
                    # Propagating alert - yellow line
                    link_color = (200, 200, 100, int(60 + 80 * max_alert))
                    link_width = 1
                else:
                    # Idle link - faint blue
                    link_color = (80, 120, 180, 40)
                    link_width = 1

                # Draw link line
                link_surf = pygame.Surface((abs(x2 - x1) + 10, abs(y2 - y1) + 10), pygame.SRCALPHA)
                lx1, ly1 = 5 if x1 <= x2 else abs(x2 - x1) + 5, 5 if y1 <= y2 else abs(y2 - y1) + 5
                lx2, ly2 = abs(x2 - x1) + 5 if x1 <= x2 else 5, abs(y2 - y1) + 5 if y1 <= y2 else 5
                pygame.draw.line(link_surf, link_color, (lx1, ly1), (lx2, ly2), link_width)
                surface.blit(link_surf, (min(x1, x2) - 5, min(y1, y2) - 5))

        # === DRAW COMMUNICATION PULSES ===
        for pulse in self.communication_pulses:
            fr, fc = pulse['from']
            tr, tc = pulse['to']
            progress = pulse['progress']

            x1 = int(fc * TILE + TILE // 2 + shake[0])
            y1 = int(fr * TILE + TILE // 2 + shake[1])
            x2 = int(tc * TILE + TILE // 2 + shake[0])
            y2 = int(tr * TILE + TILE // 2 + shake[1])

            # Interpolate position
            px = int(x1 + (x2 - x1) * progress)
            py = int(y1 + (y2 - y1) * progress)

            # Draw pulse as glowing circle
            pulse_radius = 4
            pulse_color = pulse['color']
            pygame.gfxdraw.filled_circle(surface, px, py, pulse_radius + 2, (pulse_color[0], pulse_color[1], pulse_color[2], 100))
            pygame.gfxdraw.filled_circle(surface, px, py, pulse_radius, pulse_color)
            pygame.gfxdraw.aacircle(surface, px, py, pulse_radius, (255, 255, 255))

        # === DRAW SENSORS ===
        for sensor in self.sensors.values():
            x = int(sensor.col * TILE + TILE // 2 + shake[0])
            y = int(sensor.row * TILE + TILE // 2 + shake[1])

            # Get alert propagation level for this sensor
            alert_level = self.alert_propagation.get(sensor.id, 0)

            # Color based on type and status with enhanced palette
            if sensor.health <= 0:
                color = (60, 60, 60)
                glow_color = (40, 40, 40)
            elif sensor.triggered:
                flash = abs(math.sin(time_val * 8))
                flash_int = int(flash * 255)
                if sensor.sensor_type == 'temperature':
                    color = (255, flash_int, 0)
                    glow_color = (255, 100, 0, int(flash * 100))
                elif sensor.sensor_type == 'smoke':
                    color = (flash_int, flash_int, flash_int)
                    glow_color = (200, 200, 200, int(flash * 80))
                elif sensor.sensor_type == 'co':
                    color = (255, 0, flash_int)
                    glow_color = (255, 50, 100, int(flash * 100))
                else:
                    color = (0, flash_int, 255)
                    glow_color = (50, 100, 255, int(flash * 100))
            elif alert_level > 0.3:
                # Sensor received alert from network - show warning color
                alert_flash = abs(math.sin(time_val * 5))
                color = (255, int(150 + 100 * alert_flash), 50)
                glow_color = (255, 180, 50, int(80 * alert_level))
            else:
                if sensor.sensor_type == 'temperature':
                    color = (220, 120, 60)
                    glow_color = (200, 100, 50, 60)
                elif sensor.sensor_type == 'smoke':
                    color = (170, 170, 175)
                    glow_color = (150, 150, 150, 50)
                elif sensor.sensor_type == 'co':
                    color = (220, 70, 70)
                    glow_color = (200, 50, 50, 60)
                else:
                    color = (70, 70, 220)
                    glow_color = (50, 50, 200, 60)

            sensor_radius = 6

            # Outer glow for active sensors - enhanced for alerted sensors
            if sensor.health > 0:
                glow_surf = pygame.Surface((sensor_radius * 4 + 8, sensor_radius * 4 + 8), pygame.SRCALPHA)
                if len(glow_color) == 4:
                    glow_radius = sensor_radius * 2 + int(alert_level * 4)
                    pygame.gfxdraw.filled_circle(glow_surf, sensor_radius * 2 + 4, sensor_radius * 2 + 4,
                                                glow_radius, glow_color)
                surface.blit(glow_surf, (x - sensor_radius * 2 - 4, y - sensor_radius * 2 - 4))

            # 3D sensor body with gradient effect
            # Shadow
            pygame.gfxdraw.filled_circle(surface, x + 1, y + 1, sensor_radius,
                                        (max(0, color[0] - 80), max(0, color[1] - 80), max(0, color[2] - 80)))

            # Main body
            pygame.gfxdraw.filled_circle(surface, x, y, sensor_radius, color)
            pygame.gfxdraw.aacircle(surface, x, y, sensor_radius, color)

            # Highlight
            pygame.gfxdraw.filled_circle(surface, x - 2, y - 2, sensor_radius // 2,
                                        (min(255, color[0] + 60), min(255, color[1] + 60), min(255, color[2] + 60)))

            # White border ring
            pygame.gfxdraw.aacircle(surface, x, y, sensor_radius + 1, (255, 255, 255))

            # Center indicator dot - changes color based on alert level
            if alert_level > 0.5:
                center_color = (255, 200, 50)  # Alert received
            elif sensor.health > 50:
                center_color = (255, 255, 255)
            else:
                center_color = (255, 100, 100)
            pygame.gfxdraw.filled_circle(surface, x, y, 2, center_color)

            # Signal waves if triggered - enhanced ripple effect
            if sensor.triggered and sensor.health > 0:
                for wave in range(3):
                    wave_phase = (time_val * 25 + wave * 5) % 20
                    wave_radius = int(wave_phase) + sensor_radius + 3
                    wave_alpha = max(0, 180 - wave_radius * 8)
                    if wave_alpha > 0:
                        pygame.gfxdraw.aacircle(surface, x, y, wave_radius, (*color[:3], wave_alpha))

            # Alert propagation waves for sensors receiving network alerts
            elif alert_level > 0.3 and sensor.health > 0:
                for wave in range(2):
                    wave_phase = (time_val * 15 + wave * 8) % 15
                    wave_radius = int(wave_phase) + sensor_radius + 2
                    wave_alpha = int(max(0, 120 * alert_level - wave_radius * 5))
                    if wave_alpha > 0:
                        pygame.gfxdraw.aacircle(surface, x, y, wave_radius, (255, 200, 100, wave_alpha))

        # === DRAW GLOBAL ALERT INDICATOR ===
        if self.global_alert_level > 0.3:
            # Show "NETWORK ALERT" text at top of screen
            try:
                font = pygame.font.Font(None, 24)
                flash = abs(math.sin(time_val * 4))
                alert_text = f"SENSOR NETWORK ALERT: {self.global_alert_level:.0%}"
                text_surf = font.render(alert_text, True, (255, int(150 + 100 * flash), 50))
                text_rect = text_surf.get_rect(center=(MAP_WIDTH // 2 + shake[0], 15 + shake[1]))
                # Background
                bg_rect = (text_rect.x - 5, text_rect.y - 2, text_rect.width + 10, text_rect.height + 4)
                pygame.draw.rect(surface, (80, 30, 20), bg_rect)
                pygame.draw.rect(surface, (200, 100, 50), bg_rect, 2)
                surface.blit(text_surf, text_rect)
            except:
                pass

# ═══════════════════════════════════════════════════════════════════════════════
# SOLDIER MESH NETWORK (INDIAN ARMY HANDHELD MESH LAYER)
# ═══════════════════════════════════════════════════════════════════════════════

class SoldierMeshNetwork:
    """
    Simulated soldier-worn handheld mesh network.

    - Each alive, non-escaped Person is a mesh node
    - Links exist when two nodes are within MANET range (in tiles)
    - Tracks connectivity and hazard broadcast events for the panel

    This models the offline Indian-Army-style mesh: no towers, no internet,
    just peer-to-peer adjacency.
    """
    def __init__(self, range_tiles: int = 8):
        self.range_tiles = range_tiles
        self.links = set()
        self.node_count = 0
        self.broadcast_events = 0

    def update(self, dt, people, fire_positions, maze):
        # Active nodes = soldiers still in the field
        active = [p for p in people if p.alive and not p.escaped]
        self.node_count = len(active)
        self.links.clear()

        # Build undirected adjacency based on grid distance
        for i in range(len(active)):
            p1 = active[i]
            for j in range(i + 1, len(active)):
                p2 = active[j]
                dist = abs(p1.row - p2.row) + abs(p1.col - p2.col)
                if dist <= self.range_tiles:
                    self.links.add((min(p1.id, p2.id), max(p1.id, p2.id)))

        # If there is at least one fire and at least one node,
        # count this step as a hazard broadcast over the mesh.
        if fire_positions and active:
            self.broadcast_events += 1

    def get_stats(self):
        if self.node_count > 0:
            avg_degree = 2.0 * len(self.links) / self.node_count
        else:
            avg_degree = 0.0
        return {
            'nodes': self.node_count,
            'links': len(self.links),
            'avg_degree': avg_degree,
            'broadcasts': self.broadcast_events
        }

# ═══════════════════════════════════════════════════════════════════════════════
# REINFORCEMENT LEARNING - Q-LEARNING EVACUATION COORDINATOR
# ═══════════════════════════════════════════════════════════════════════════════

class RLEvacuationCoordinator:
    """
    Q-learning agent that learns optimal warden deployment and
    crowd flow management strategies.

    State: (fire_quadrant, crowd_density_quadrant, exits_blocked)
    Actions: (deploy_warden_to_quadrant, open_secondary_exit, etc.)
    """
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(8))  # 8 actions
        self.learning_rate = 0.1
        self.discount = 0.95
        self.epsilon = 0.2

        self.actions = [
            'deploy_NW', 'deploy_NE', 'deploy_SW', 'deploy_SE',
            'open_exit_N', 'open_exit_S', 'open_exit_E', 'open_exit_W'
        ]

        self.current_state = None
        self.last_action = None
        self.episode_reward = 0
        self.total_episodes = 0

        # Performance tracking
        self.avg_evacuation_time = []
        self.death_rate = []
        self.decisions_made = 0

    def get_state(self, fire_positions, people, exits_status):
        """Convert environment to discrete state."""
        # Determine fire quadrant
        fire_quad = 0
        if fire_positions:
            avg_r = np.mean([f[0] for f in fire_positions])
            avg_c = np.mean([f[1] for f in fire_positions])
            if avg_r < ROWS // 2:
                fire_quad = 0 if avg_c < COLS // 2 else 1
            else:
                fire_quad = 2 if avg_c < COLS // 2 else 3

        # Crowd density
        crowd_counts = [0, 0, 0, 0]
        for p in people:
            if p.alive and not p.escaped:
                q = 0
                if p.row < ROWS // 2:
                    q = 0 if p.col < COLS // 2 else 1
                else:
                    q = 2 if p.col < COLS // 2 else 3
                crowd_counts[q] += 1

        crowd_quad = int(np.argmax(crowd_counts))

        # Exits blocked (simplified)
        blocked = sum(1 for e, status in exits_status.items() if status)

        return (int(fire_quad), int(crowd_quad), min(blocked, 3))

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, 7)
        return int(np.argmax(self.q_table[state]))

    def update(self, reward, new_state):
        """Update Q-table."""
        if self.current_state is not None and self.last_action is not None:
            old_value = self.q_table[self.current_state][self.last_action]
            next_max = float(np.max(self.q_table[new_state]))

            new_value = old_value + self.learning_rate * (
                reward + self.discount * next_max - old_value
            )
            self.q_table[self.current_state][self.last_action] = new_value

        self.episode_reward += reward
        self.current_state = new_state

    def step(self, fire_positions, people, exits_status, wardens):
        """Take a decision step."""
        state = self.get_state(fire_positions, people, exits_status)
        action = self.choose_action(state)
        self.last_action = action
        self.decisions_made += 1

        # Execute action
        action_name = self.actions[action]
        result = self._execute_action(action_name, wardens, people)

        self.update(result['reward'], state)
        return result

    def _execute_action(self, action_name, wardens, people):
        """Execute the chosen action."""
        reward = 0
        message = ""

        if action_name.startswith('deploy_'):
            quadrant = action_name.split('_')[1]
            target_row = ROWS // 4 if 'N' in quadrant else 3 * ROWS // 4
            target_col = COLS // 4 if 'W' in quadrant else 3 * COLS // 4

            # Find available warden
            for warden in wardens:
                if warden.alive and not warden.escaped:
                    warden.rl_target = (target_row, target_col)
                    reward = 5
                    message = f"Deployed warden to {quadrant}"
                    break

        elif action_name.startswith('open_exit_'):
            reward = 2
            message = f"Prioritizing {action_name.split('_')[2]} exits"

        return {'reward': reward, 'message': message, 'action': action_name}

    def get_stats(self):
        return {
            'episodes': self.total_episodes,
            'avg_reward': self.episode_reward / max(1, self.decisions_made),
            'epsilon': self.epsilon,
            'decisions': self.decisions_made
        }

# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL ACO (THE PATENTABLE CORE)
# ═══════════════════════════════════════════════════════════════════════════════

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
    def __init__(self, lstm_predictor: SimpleLSTMPredictor):
        self.lstm = lstm_predictor

        # Pheromone matrices
        self.safe_pheromone = np.ones((ROWS, COLS)) * 0.1
        self.danger_pheromone = np.zeros((ROWS, COLS))
        self.predicted_danger = np.zeros((ROWS, COLS))

        # ACO parameters (base values)
        self.base_evaporation = 0.02
        self.base_deposit = 1.0
        self.alpha = 1.0  # Pheromone importance
        self.beta = 2.0   # Heuristic importance

        # Neural modulation factors
        self.neural_confidence = 0.0
        self.modulation_strength = 2.0

        # Path edge tracking
        self.edge_usage = defaultdict(int)

    def update_predictions(self, fire_positions, sensor_data, maze):
        """Update neural predictions and modulate pheromones."""
        predictions = self.lstm.predict_spread(fire_positions, sensor_data, maze)
        self.neural_confidence = self.lstm.prediction_confidence

        # Reset predicted danger
        self.predicted_danger *= 0.8

        # Apply predictions to danger pheromone
        for r, c, prob in predictions:
            # Neural confidence modulates how much we trust predictions
            modulated_prob = prob * (0.5 + 0.5 * self.neural_confidence)
            self.predicted_danger[r, c] = max(self.predicted_danger[r, c], modulated_prob)
            self.danger_pheromone[r, c] += modulated_prob * self.modulation_strength

    def deposit_safe_pheromone(self, path: List[Tuple[int, int]], success: bool):
        """Deposit pheromone along successful evacuation path."""
        if not path or not success:
            return

        # Amount modulated by neural confidence
        amount = self.base_deposit * (1.0 + self.neural_confidence * 0.5)

        for i, (r, c) in enumerate(path):
            decay = 1.0 - (i / len(path)) * 0.5  # More at start of path
            self.safe_pheromone[r, c] += amount * decay
            self.safe_pheromone[r, c] = min(self.safe_pheromone[r, c], 10.0)

            if i > 0:
                edge = (path[i - 1], (r, c))
                self.edge_usage[edge] += 1

    def deposit_danger_pheromone(self, position: Tuple[int, int], severity: float):
        """Mark dangerous area."""
        r, c = position
        self.danger_pheromone[r, c] += severity
        self.danger_pheromone[r, c] = min(self.danger_pheromone[r, c], 20.0)

        # Spread to neighbors
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    dist = abs(dr) + abs(dc)
                    spread = severity / (dist + 1)
                    self.danger_pheromone[nr, nc] += spread * 0.3

    def evaporate(self):
        """Evaporate pheromones - rate modulated by neural confidence."""
        # When confident, preserve safe paths longer
        safe_evap = self.base_evaporation * (1.0 - self.neural_confidence * 0.3)
        danger_evap = self.base_evaporation * 1.2  # Danger evaporates faster

        self.safe_pheromone *= (1.0 - safe_evap)
        self.danger_pheromone *= (1.0 - danger_evap)
        self.safe_pheromone = np.clip(self.safe_pheromone, 0.1, 10.0)
        self.danger_pheromone = np.clip(self.danger_pheromone, 0, 20.0)

    def get_path_desirability(self, from_pos, to_pos, exit_pos) -> float:
        """
        Calculate desirability of moving from->to.
        Combines pheromone (tau) with heuristic (eta).
        """
        r, c = to_pos

        # Pheromone factor
        tau_safe = self.safe_pheromone[r, c]
        tau_danger = self.danger_pheromone[r, c]
        tau_predicted = self.predicted_danger[r, c]

        tau = tau_safe / (1 + tau_danger + tau_predicted * 2)

        # Heuristic: inverse distance to exit
        dist = abs(r - exit_pos[0]) + abs(c - exit_pos[1]) + 1
        eta = 1.0 / dist

        # Combined probability
        desirability = (tau ** self.alpha) * (eta ** self.beta)

        return desirability

    def get_visualization_data(self):
        """Get data for visualization."""
        return {
            'safe': self.safe_pheromone.copy(),
            'danger': self.danger_pheromone.copy(),
            'predicted': self.predicted_danger.copy(),
            'confidence': self.neural_confidence,
            'edge_usage': dict(self.edge_usage)
        }

    def reset(self):
        self.safe_pheromone = np.ones((ROWS, COLS)) * 0.1
        self.danger_pheromone = np.zeros((ROWS, COLS))
        self.predicted_danger = np.zeros((ROWS, COLS))
        self.edge_usage.clear()
        self.lstm.reset()

# ═══════════════════════════════════════════════════════════════════════════════
# SOUND SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class SoundSystem:
    def __init__(self):
        self.sounds = {}
        self.alarm_playing = False
        self.help_beep_timers = {}  # Track beep timers for hurt people
        self.generate_sounds()

    def generate_sounds(self):
        sample_rate = 22050

        # Alarm
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone1 = np.sin(2 * np.pi * 800 * t) * 0.3
        tone2 = np.sin(2 * np.pi * 600 * t) * 0.3
        alarm_wave = np.concatenate([tone1, tone2])
        alarm_wave = (alarm_wave * 32767).astype(np.int16)
        alarm_stereo = np.column_stack((alarm_wave, alarm_wave))
        self.sounds['alarm'] = pygame.sndarray.make_sound(alarm_stereo)

        # Neural alert (futuristic beep)
        duration = 0.2
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        freq = 1200 + 400 * np.sin(t * 30)
        beep = np.sin(2 * np.pi * freq * t) * 0.25 * np.exp(-t * 5)
        beep = (beep * 32767).astype(np.int16)
        beep_stereo = np.column_stack((beep, beep))
        self.sounds['neural'] = pygame.sndarray.make_sound(beep_stereo)

        # Sensor trigger
        duration = 0.15
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        ping = np.sin(2 * np.pi * 2000 * t) * np.exp(-t * 20) * 0.2
        ping = (ping * 32767).astype(np.int16)
        ping_stereo = np.column_stack((ping, ping))
        self.sounds['sensor'] = pygame.sndarray.make_sound(ping_stereo)

        # Success/Escape chime
        duration = 0.3
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        chime = np.sin(2 * np.pi * 880 * t) * np.exp(-t * 5) * 0.2
        chime = (chime * 32767).astype(np.int16)
        chime_stereo = np.column_stack((chime, chime))
        self.sounds['escape'] = pygame.sndarray.make_sound(chime_stereo)

        # HELP BEEP - urgent distress signal
        duration = 0.4
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Three rapid beeps pattern
        beep1 = np.sin(2 * np.pi * 1500 * t) * np.where(t < 0.1, 1, 0)
        beep2 = np.sin(2 * np.pi * 1500 * t) * np.where((t > 0.15) & (t < 0.25), 1, 0)
        beep3 = np.sin(2 * np.pi * 1500 * t) * np.where(t > 0.3, 1, 0)
        help_beep = (beep1 + beep2 + beep3) * 0.3
        help_beep = (help_beep * 32767).astype(np.int16)
        help_stereo = np.column_stack((help_beep, help_beep))
        self.sounds['help_beep'] = pygame.sndarray.make_sound(help_stereo)

        # RESCUE ARRIVAL - heroic fanfare
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Rising triumphant tone
        freq_rise = 400 + 600 * (t / duration)
        rescue = np.sin(2 * np.pi * freq_rise * t) * 0.25 * np.exp(-t * 2)
        rescue += np.sin(2 * np.pi * freq_rise * 1.5 * t) * 0.15 * np.exp(-t * 2)
        rescue = (rescue * 32767).astype(np.int16)
        rescue_stereo = np.column_stack((rescue, rescue))
        self.sounds['rescue'] = pygame.sndarray.make_sound(rescue_stereo)

        # FIRE CRACKLE
        duration = 0.3
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        noise = np.random.randn(len(t)) * 0.15
        # Filter to make it crackly
        crackle = noise * np.exp(-((t - 0.15) ** 2) / 0.01)
        crackle = (crackle * 32767).astype(np.int16)
        crackle_stereo = np.column_stack((crackle, crackle))
        self.sounds['fire'] = pygame.sndarray.make_sound(crackle_stereo)

        # SPRINKLER SPRAY
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # White noise for water spray
        spray = np.random.randn(len(t)) * 0.1
        spray = spray * (1 - np.exp(-t * 10))  # Fade in
        spray = (spray * 32767).astype(np.int16)
        spray_stereo = np.column_stack((spray, spray))
        self.sounds['sprinkler'] = pygame.sndarray.make_sound(spray_stereo)

        # FOOTSTEPS (running)
        duration = 0.15
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        step = np.sin(2 * np.pi * 100 * t) * np.exp(-t * 30) * 0.2
        step = (step * 32767).astype(np.int16)
        step_stereo = np.column_stack((step, step))
        self.sounds['footstep'] = pygame.sndarray.make_sound(step_stereo)

        # DEATH/INJURY sound
        duration = 0.4
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        injury = np.sin(2 * np.pi * 200 * t) * np.exp(-t * 5) * 0.3
        injury += np.sin(2 * np.pi * 150 * t) * np.exp(-t * 4) * 0.2
        injury = (injury * 32767).astype(np.int16)
        injury_stereo = np.column_stack((injury, injury))
        self.sounds['injury'] = pygame.sndarray.make_sound(injury_stereo)

        # PANIC scream (stylized)
        duration = 0.3
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        freq_panic = 800 + 400 * np.sin(t * 50)
        panic = np.sin(2 * np.pi * freq_panic * t) * 0.2 * np.exp(-t * 4)
        panic = (panic * 32767).astype(np.int16)
        panic_stereo = np.column_stack((panic, panic))
        self.sounds['panic'] = pygame.sndarray.make_sound(panic_stereo)

        # SENSOR NETWORK ALERT - cascading warning tone
        duration = 0.6
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Dual-tone alert like emergency broadcast
        tone_a = np.sin(2 * np.pi * 950 * t) * 0.25
        tone_b = np.sin(2 * np.pi * 1050 * t) * 0.25
        # Alternating pattern
        pattern = np.where((t * 8).astype(int) % 2 == 0, tone_a, tone_b)
        sensor_alert = pattern * np.exp(-t * 1.5)
        sensor_alert = (sensor_alert * 32767).astype(np.int16)
        alert_stereo = np.column_stack((sensor_alert, sensor_alert))
        self.sounds['sensor_alert'] = pygame.sndarray.make_sound(alert_stereo)

        # NETWORK BROADCAST - propagation sound
        duration = 0.25
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Quick rising ping for network communication
        freq_sweep = 1000 + 500 * (t / duration)
        broadcast = np.sin(2 * np.pi * freq_sweep * t) * 0.15 * np.exp(-t * 8)
        broadcast = (broadcast * 32767).astype(np.int16)
        broadcast_stereo = np.column_stack((broadcast, broadcast))
        self.sounds['network_broadcast'] = pygame.sndarray.make_sound(broadcast_stereo)

    def play(self, name, volume=0.5):
        if name in self.sounds:
            self.sounds[name].set_volume(volume)
            self.sounds[name].play()

    def start_alarm(self):
        if not self.alarm_playing:
            self.sounds['alarm'].play(-1)
            self.alarm_playing = True

    def stop_alarm(self):
        if self.alarm_playing:
            self.sounds['alarm'].stop()
            self.alarm_playing = False

class SilentSoundSystem:
    """No-op sound system for headless/serverless execution."""
    def play(self, name, volume=0.5):
        return None

    def start_alarm(self):
        return None

    def stop_alarm(self):
        return None

sound_system = SoundSystem() if (not HEADLESS and audio_ready) else SilentSoundSystem()

# ═══════════════════════════════════════════════════════════════════════════════
# COLORS (Enhanced for 3D look)
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    # Environment
    FLOOR = (180, 175, 165)
    FLOOR_ALT = (170, 165, 155)
    WALL = (45, 48, 55)
    WALL_HIGHLIGHT = (65, 68, 75)
    CORRIDOR = (155, 150, 145)
    CARPET = (95, 65, 65)
    EXIT = (50, 255, 100)
    DOOR = (120, 75, 40)

    # Hazards
    FIRE = (255, 100, 30)
    FIRE_BRIGHT = (255, 220, 80)
    FIRE_CORE = (255, 255, 200)
    SMOKE = (70, 70, 75)
    WATER = (40, 120, 200)

    # Neural/Tech
    NEURAL_GLOW = (0, 255, 200)
    PREDICTION = (255, 50, 200)
    SENSOR_ACTIVE = (0, 200, 255)
    IOT_MESH = (100, 255, 200)

    # Pheromones
    SAFE_PHEROMONE = (0, 255, 150)
    DANGER_PHEROMONE = (255, 80, 80)

    # People states
    NORMAL = (100, 150, 255)
    AWARE = (255, 255, 100)
    EVACUATING = (100, 255, 100)
    PANICKING = (255, 80, 80)
    WARDEN = (255, 215, 0)
    HEADPHONES = (255, 100, 255)

    # UI
    PANEL_BG = (18, 20, 28)
    PANEL_BORDER = (45, 50, 65)
    ACCENT = (0, 230, 180)
    SUCCESS = (60, 255, 120)
    DANGER = (255, 70, 70)
    WARNING = (255, 200, 60)
    TEXT = (255, 255, 255)
    TEXT_DIM = (160, 160, 170)

# Tile types
FLOOR = 0
WALL = 1
EXIT = 2
DOOR = 3
CORRIDOR = 4
CARPET = 5

# States
STATE_WORKING = "working"
STATE_HEADPHONES = "headphones"
STATE_AWARE = "aware"
STATE_EVACUATING = "evacuating"
STATE_PANICKING = "panicking"
STATE_WARDEN = "warden"

# ═══════════════════════════════════════════════════════════════════════════════
# PATHFINDER (Enhanced with Neural ACO)
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralPathfinder:
    """A* pathfinder enhanced with Neural ACO pheromone guidance."""

    def __init__(self, neural_aco: NeuralACO):
        self.aco = neural_aco
        self.cache = {}

    def find_path(self, start, goal, maze, hazards) -> List[Tuple[int, int]]:
        cache_key = (start, goal, len(hazards))
        if cache_key in self.cache:
            return self.cache[cache_key].copy()

        rows, cols = len(maze), len(maze[0])
        open_set = []
        heappush(open_set, (0, 0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, _, current = heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                self.cache[cache_key] = path
                return path

            r, c = current
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    tile = maze[nr][nc]
                    if tile in [FLOOR, CORRIDOR, CARPET, EXIT, DOOR]:
                        neighbor = (nr, nc)

                        # Base cost
                        cost = 1

                        # Add hazard costs
                        if neighbor in hazards:
                            cost += 500

                        # Neural ACO costs
                        danger_pheromone = self.aco.danger_pheromone[nr, nc]
                        predicted_danger = self.aco.predicted_danger[nr, nc]
                        safe_pheromone = self.aco.safe_pheromone[nr, nc]

                        # High danger/predicted danger = high cost
                        cost += danger_pheromone * 20
                        cost += predicted_danger * 40  # Trust predictions more

                        # Safe pheromone reduces cost
                        cost *= (1.0 / (1.0 + safe_pheromone * 0.2))

                        tentative_g = g_score[current] + cost

                        if neighbor not in g_score or tentative_g < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g
                            h = abs(nr - goal[0]) + abs(nc - goal[1])
                            heappush(open_set, (tentative_g + h, tentative_g, neighbor))

        return []

    def clear_cache(self):
        self.cache.clear()

# ═══════════════════════════════════════════════════════════════════════════════
# ML-ACTIVATED SPRINKLER SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class SprinklerSystem:
    """
    Intelligent sprinkler system that activates based on:
    - LSTM neural network fire spread predictions
    - Sensor network data
    - ACO danger pheromone levels
    """
    def __init__(self):
        self.sprinklers = {}  # (row, col) -> {'active': bool, 'water_level': float}
        self.water_particles = []
        self.activation_threshold = 0.4  # ML prediction confidence to activate
        self.suppression_rate = 0.8  # How fast sprinklers reduce fire intensity
        self.coverage_radius = 2  # Tiles around each sprinkler

    def install_sprinklers(self, maze):
        """Install sprinklers throughout the building."""
        for r in range(2, ROWS - 2, 4):
            for c in range(2, COLS - 2, 4):
                if maze[r][c] not in [WALL, EXIT]:
                    self.sprinklers[(r, c)] = {
                        'active': False,
                        'water_level': 100.0,
                        'activation_time': 0
                    }

    def update(self, dt, neural_aco, predictions, hazards, time_val):
        """Update sprinklers based on ML predictions and sensor data."""
        activated_zones = set()

        # Check predictions from LSTM
        for r, c, prob in predictions:
            if prob > self.activation_threshold:
                activated_zones.add((r, c))
                # Also activate nearby sprinklers
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        activated_zones.add((r + dr, c + dc))

        # Check danger pheromone levels
        for r in range(ROWS):
            for c in range(COLS):
                if neural_aco.danger_pheromone[r, c] > 1.5:
                    activated_zones.add((r, c))

        # Check actual fire positions
        for pos in hazards:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    activated_zones.add((pos[0] + dr, pos[1] + dc))

        # Activate/deactivate sprinklers
        for pos, sprinkler in self.sprinklers.items():
            should_activate = any(
                abs(pos[0] - zone[0]) <= self.coverage_radius and
                abs(pos[1] - zone[1]) <= self.coverage_radius
                for zone in activated_zones
            )

            if should_activate and sprinkler['water_level'] > 0:
                if not sprinkler['active']:
                    sprinkler['active'] = True
                    sprinkler['activation_time'] = time_val

                # Consume water
                sprinkler['water_level'] -= 5 * dt

                # Generate water particles
                if random.random() < 0.7:
                    for _ in range(3):
                        self.water_particles.append({
                            'x': pos[1] * TILE + random.randint(0, TILE),
                            'y': pos[0] * TILE + 4,
                            'vx': random.uniform(-30, 30),
                            'vy': random.uniform(40, 80),
                            'life': random.uniform(0.3, 0.6),
                            'size': random.randint(2, 4)
                        })
            else:
                sprinkler['active'] = False

        # Update water particles
        for p in self.water_particles[:]:
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            p['vy'] += 150 * dt  # Gravity
            p['life'] -= dt
            if p['life'] <= 0 or p['y'] > ROWS * TILE:
                self.water_particles.remove(p)

        return activated_zones

    def suppress_fire(self, hazards, dt):
        """Reduce fire intensity in sprinkler coverage areas."""
        to_remove = []
        for pos, info in hazards.items():
            if info['type'] == 'fire':
                # Check if any active sprinkler covers this position
                for spr_pos, sprinkler in self.sprinklers.items():
                    if sprinkler['active']:
                        dist = abs(pos[0] - spr_pos[0]) + abs(pos[1] - spr_pos[1])
                        if dist <= self.coverage_radius:
                            info['intensity'] -= self.suppression_rate * dt
                            if info['intensity'] <= 0:
                                to_remove.append(pos)
                            break
        return to_remove

    def draw(self, surface, shake, time_val):
        """Draw sprinklers and water effects."""
        for pos, sprinkler in self.sprinklers.items():
            x = int(pos[1] * TILE + TILE // 2 + shake[0])
            y = int(pos[0] * TILE + 4 + shake[1])

            if sprinkler['active']:
                # Active sprinkler head (pulsing blue)
                pulse = abs(math.sin(time_val * 10)) * 0.5 + 0.5
                color = (int(100 * pulse), int(150 + 50 * pulse), 255)

                # Sprinkler head
                pygame.gfxdraw.filled_circle(surface, x, y, 5, color)
                pygame.gfxdraw.aacircle(surface, x, y, 5, (200, 220, 255))

                # Water spray cone effect
                spray_surf = pygame.Surface((TILE * 2, TILE * 2), pygame.SRCALPHA)
                for i in range(8):
                    angle = math.pi / 2 + math.sin(time_val * 15 + i) * 0.3
                    length = TILE + random.randint(-5, 5)
                    ex = TILE + int(math.cos(angle + i * 0.2 - 0.7) * length)
                    ey = 5 + int(math.sin(angle) * length)
                    pygame.draw.line(spray_surf, (100, 180, 255, 100), (TILE, 5), (ex, ey), 2)
                surface.blit(spray_surf, (x - TILE, y - 5))

                # Mist effect
                for _ in range(2):
                    mx = x + random.randint(-TILE, TILE)
                    my = y + random.randint(5, TILE)
                    pygame.gfxdraw.filled_circle(surface, mx, my, random.randint(2, 4),
                                                (200, 220, 255, 40))
            else:
                # Inactive sprinkler head
                pygame.gfxdraw.filled_circle(surface, x, y, 4, (80, 80, 90))
                pygame.gfxdraw.aacircle(surface, x, y, 4, (120, 120, 130))

        # Draw water droplets
        for p in self.water_particles:
            px = int(p['x'] + shake[0])
            py = int(p['y'] + shake[1])
            size = p['size']
            alpha = min(255, int(p['life'] * 500))

            # Water droplet with glow
            pygame.gfxdraw.filled_circle(surface, px, py, size, (100, 180, 255, alpha))
            pygame.gfxdraw.aacircle(surface, px, py, size, (150, 200, 255, alpha))


# ═══════════════════════════════════════════════════════════════════════════════
# DISASTERS
# ═══════════════════════════════════════════════════════════════════════════════

class Disasters:
    def __init__(self):
        self.hazards = {}
        self.smoke = defaultdict(float)
        self.shake = 0.0
        self.shake_offset = (0, 0)
        self.particles = []

    def add_fire(self, row, col):
        if (row, col) not in self.hazards:
            self.hazards[(row, col)] = {'type': 'fire', 'age': 0, 'intensity': 1.0}

    def update(self, dt, maze, neural_aco):
        self.shake *= 0.9
        if self.shake > 0.01:
            self.shake_offset = (
                random.uniform(-1, 1) * self.shake * 6,
                random.uniform(-1, 1) * self.shake * 6
            )
        else:
            self.shake_offset = (0, 0)

        new_hazards = {}
        to_remove = []

        for (row, col), info in list(self.hazards.items()):
            info['age'] += dt

            if info['type'] == 'fire':
                # Smoke
                for dr in range(-4, 5):
                    for dc in range(-4, 5):
                        sr, sc = row + dr, col + dc
                        if 0 <= sr < ROWS and 0 <= sc < COLS:
                            dist = abs(dr) + abs(dc)
                            self.smoke[(sr, sc)] = min(
                                self.smoke[(sr, sc)] + 0.03 / (dist + 1), 1.5
                            )

                # Deposit danger pheromone
                neural_aco.deposit_danger_pheromone((row, col), 2.0 * dt)

                # Fire particles - more particles for detailed 3D effect
                if random.random() < 0.6:
                    # Spawn multiple particles per fire tile for richer effect
                    for _ in range(2):
                        self.particles.append({
                            'x': col * TILE + random.randint(4, TILE - 4),
                            'y': row * TILE + TILE - random.randint(0, 4),
                            'vy': -random.uniform(50, 100),
                            'vx': random.uniform(-15, 15),
                            'life': random.uniform(0.4, 0.9),
                            'type': 'fire'
                        })

                # Spread
                if info['age'] > 5.0 and random.random() < 0.01:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = row + dr, col + dc
                        if 0 < nr < ROWS - 1 and 0 < nc < COLS - 1:
                            if maze[nr][nc] not in [WALL, EXIT]:
                                if (nr, nc) not in self.hazards:
                                    new_hazards[(nr, nc)] = {'type': 'fire', 'age': 0, 'intensity': 0.8}
                                    break

                if info['age'] > 80:
                    to_remove.append((row, col))

        for pos in to_remove:
            if pos in self.hazards:
                del self.hazards[pos]
        self.hazards.update(new_hazards)

        # Smoke decay
        for key in list(self.smoke.keys()):
            self.smoke[key] *= 0.97
            if self.smoke[key] < 0.02:
                del self.smoke[key]

        # Particles
        for p in self.particles[:]:
            p['y'] += p['vy'] * dt
            p['x'] += p.get('vx', 0) * dt
            p['life'] -= dt
            if p['life'] <= 0:
                self.particles.remove(p)

    def get_fire_positions(self):
        return [pos for pos, info in self.hazards.items() if info['type'] == 'fire']

    def draw_particles(self, surface, shake):
        """Draw enhanced 3D particles with glow effects."""
        for p in self.particles:
            x = int(p['x'] + shake[0])
            y = int(p['y'] + shake[1])
            alpha = min(255, int(p['life'] * 400))
            size = max(2, int(6 * p['life']))

            if p['type'] == 'fire':
                t = p['life']
                # Multi-layered fire particle for 3D glow
                layers = [
                    (size + 3, (255, 80, 0, int(alpha * 0.3))),      # Outer glow
                    (size + 1, (255, 150, 50, int(alpha * 0.6))),   # Mid glow
                    (size, (255, 200 + int(55 * t), int(100 * t))),  # Core
                    (max(1, size - 1), (255, 255, int(200 * t))),   # Hot center
                ]

                # Draw glow layers
                for layer_size, color in layers:
                    if len(color) == 4:  # Has alpha
                        glow_surf = pygame.Surface((layer_size * 2 + 4, layer_size * 2 + 4), pygame.SRCALPHA)
                        pygame.gfxdraw.filled_circle(glow_surf, layer_size + 2, layer_size + 2,
                                                    layer_size, color)
                        surface.blit(glow_surf, (x - layer_size - 2, y - layer_size - 2))
                    else:
                        pygame.gfxdraw.filled_circle(surface, x, y, layer_size, color)
                        pygame.gfxdraw.aacircle(surface, x, y, layer_size, color)

                # Spark trail
                if size > 2 and random.random() > 0.7:
                    spark_x = x + random.randint(-3, 3)
                    spark_y = y + random.randint(-3, 3)
                    pygame.gfxdraw.pixel(surface, spark_x, spark_y, (255, 255, 200))

# ═══════════════════════════════════════════════════════════════════════════════
# ALARM SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class AlarmSystem:
    def __init__(self):
        self.active = False
        self.flash_timer = 0
        self.flash_state = False

    def trigger(self):
        if not self.active:
            self.active = True
            sound_system.start_alarm()

    def update(self, dt):
        if self.active:
            self.flash_timer += dt
            if self.flash_timer > 0.25:
                self.flash_state = not self.flash_state
                self.flash_timer = 0

    def reset(self):
        self.active = False
        self.flash_state = False
        sound_system.stop_alarm()

# ═══════════════════════════════════════════════════════════════════════════════
# RESCUE HELPER SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class Rescuer:
    """
    Emergency rescue helper that responds to hurt people.
    Jumps in from exits to help injured civilians.
    """
    def __init__(self, rid, start_row, start_col):
        self.id = rid
        self.row = start_row
        self.col = start_col
        self.x = start_col * TILE + TILE // 2
        self.y = start_row * TILE + TILE // 2
        self.tx = self.x
        self.ty = self.y

        self.target_person = None
        self.state = 'idle'  # 'idle', 'responding', 'rescuing', 'returning'
        self.path = []
        self.path_index = 0
        self.moving = False
        self.speed = 200  # Very fast for zero deaths goal!

        self.walk_frame = 0
        self.rescue_timer = 0
        self.rescue_start_health = None
        self.helped_count = 0
        self.critical_saves = 0
        self.deaths_prevented = 0
        self.last_save_health = None

        # Animation
        self.jump_offset = 0
        self.spawn_time = 0

    def find_hurt_person(self, people):
        """Find the most critically hurt person nearby."""
        best_target = None
        best_priority = float('inf')

        for p in people:
            if p.alive and not p.escaped and p.health < 80 and p.health > 0:
                # Check if not already being rescued
                if getattr(p, 'being_rescued', False):
                    continue

                # Priority: lower health = higher priority
                dist = abs(self.row - p.row) + abs(self.col - p.col)
                priority = p.health + dist * 0.5

                if priority < best_priority:
                    best_priority = priority
                    best_target = p

        return best_target

    def update(self, dt, maze, people, pathfinder, hazards, time_val):
        self.spawn_time += dt

        # Jump animation when spawning
        if self.spawn_time < 0.5:
            self.jump_offset = math.sin(self.spawn_time * math.pi * 2) * 20
        else:
            self.jump_offset = 0

        if self.state == 'idle':
            # Look for hurt people
            target = self.find_hurt_person(people)
            if target:
                self.target_person = target
                target.being_rescued = True
                self.state = 'responding'
                self.path = pathfinder.find_path((self.row, self.col), (target.row, target.col), maze, hazards)
                self.path_index = 0
                sound_system.play('rescue', 0.4)

        elif self.state == 'responding':
            if not self.target_person or not self.target_person.alive or self.target_person.escaped:
                self.state = 'idle'
                if self.target_person:
                    self.target_person.being_rescued = False
                self.target_person = None
                return

            # Move toward target
            if not self.moving:
                # Check if reached target
                dist = abs(self.row - self.target_person.row) + abs(self.col - self.target_person.col)
                if dist <= 1:
                    self.state = 'rescuing'
                    self.rescue_timer = 0
                    return

                # Update path if target moved
                if self.path_index >= len(self.path) or not self.path:
                    self.path = pathfinder.find_path((self.row, self.col),
                                                     (self.target_person.row, self.target_person.col),
                                                     maze, hazards)
                    self.path_index = 0

                if self.path and self.path_index < len(self.path):
                    next_pos = self.path[self.path_index]
                    nr, nc = next_pos

                    # Skip hazards
                    if (nr, nc) in hazards:
                        self.path = pathfinder.find_path((self.row, self.col),
                                                         (self.target_person.row, self.target_person.col),
                                                         maze, hazards)
                        self.path_index = 0
                        return

                    self.tx = nc * TILE + TILE // 2
                    self.ty = nr * TILE + TILE // 2
                    self.row, self.col = nr, nc
                    self.path_index += 1
                    self.moving = True

        elif self.state == 'rescuing':
            self.rescue_timer += dt

            # Heal the person - FASTER healing for zero deaths goal!
            if self.target_person and self.target_person.alive:
                # Track initial health for critical save detection
                if self.rescue_start_health is None:
                    self.rescue_start_health = self.target_person.health

                # Heal at 80 HP/sec (very fast!)
                self.target_person.health = min(100, self.target_person.health + 80 * dt)

                # Healing particles effect
                if random.random() < 0.3:
                    sound_system.play('sensor', 0.1)  # Healing beep

                # Done rescuing when health is restored
                if self.target_person.health >= 95 or self.rescue_timer > 2.0:
                    self.target_person.being_rescued = False
                    self.helped_count += 1

                    start_health = self.rescue_start_health if self.rescue_start_health is not None else self.target_person.health
                    self.last_save_health = start_health

                    # Track if this was a critical save or near-death prevention
                    if start_health < 25:
                        self.critical_saves += 1
                        if start_health < 12:
                            self.deaths_prevented += 1
                        sound_system.play('neural', 0.4)  # Special sound for critical save

                    sound_system.play('escape', 0.3)  # Success sound
                    self.state = 'idle'
                    self.target_person = None
                    self.rescue_start_health = None
            else:
                if self.target_person:
                    self.target_person.being_rescued = False
                self.state = 'idle'
                self.target_person = None
                self.rescue_start_health = None
                self.last_save_health = None

        # Movement animation
        if self.moving:
            self.walk_frame += dt * 15
            dx = self.tx - self.x
            dy = self.ty - self.y
            dist = math.hypot(dx, dy)

            if dist < 2:
                self.x, self.y = self.tx, self.ty
                self.moving = False
            else:
                self.x += (dx / dist) * self.speed * dt
                self.y += (dy / dist) * self.speed * dt

    def draw(self, surface, shake, time_val):
        x = int(self.x + shake[0])
        y = int(self.y + shake[1]) - int(self.jump_offset)

        scale = 2.0

        # Shadow
        shadow_y = int(self.y + shake[1]) + int(8 * scale)
        shadow_w = int(20 * scale)
        pygame.draw.ellipse(surface, (20, 20, 25, 80),
                           (x - shadow_w // 2, shadow_y, shadow_w, int(8 * scale)))

        # Rescuer colors - bright red/white emergency colors
        body_color = (220, 50, 50)  # Red
        body_light = (255, 100, 100)
        body_dark = (180, 30, 30)
        cross_color = (255, 255, 255)

        # Legs
        leg_w, leg_h = int(6 * scale), int(12 * scale)
        walk_offset = math.sin(self.walk_frame) * 4 * scale if self.moving else 0

        for leg_side in [-1, 1]:
            leg_x = x + leg_side * int(4 * scale)
            leg_offset = walk_offset * leg_side
            pygame.draw.ellipse(surface, (40, 40, 50),
                              (leg_x - leg_w // 2, y + int(leg_offset), leg_w, leg_h))

        # Body
        torso_w, torso_h = int(18 * scale), int(20 * scale)
        torso_y = y - int(8 * scale)

        pygame.gfxdraw.filled_ellipse(surface, x, torso_y, torso_w // 2, torso_h // 2, body_dark)
        pygame.gfxdraw.filled_ellipse(surface, x, torso_y, torso_w // 2 - 2, torso_h // 2 - 2, body_color)
        pygame.gfxdraw.aaellipse(surface, x, torso_y, torso_w // 2, torso_h // 2, body_light)

        # Medical cross on chest
        cross_size = int(6 * scale)
        pygame.draw.rect(surface, cross_color, (x - cross_size // 2, torso_y - cross_size, cross_size // 3, cross_size))
        pygame.draw.rect(surface, cross_color, (x - cross_size // 2, torso_y - cross_size // 2 - cross_size // 6, cross_size, cross_size // 3))

        # Arms
        arm_swing = math.sin(self.walk_frame + 0.5) * 4 * scale if self.moving else 0
        for arm_side in [-1, 1]:
            arm_x = x + arm_side * (torso_w // 2)
            arm_y = torso_y + int(arm_swing * arm_side)
            pygame.draw.ellipse(surface, body_color, (arm_x - int(3 * scale), arm_y - int(6 * scale), int(6 * scale), int(14 * scale)))

        # Head with helmet
        head_radius = int(10 * scale)
        head_y = torso_y - torso_h // 2 - head_radius + int(4 * scale)

        # Helmet (white with red cross)
        pygame.gfxdraw.filled_ellipse(surface, x, head_y, head_radius + 2, head_radius + 2, (200, 200, 200))
        pygame.gfxdraw.filled_ellipse(surface, x, head_y, head_radius, head_radius, (240, 240, 240))
        pygame.gfxdraw.aaellipse(surface, x, head_y, head_radius, head_radius, (255, 255, 255))

        # Cross on helmet
        pygame.draw.rect(surface, (220, 50, 50), (x - int(2 * scale), head_y - int(6 * scale), int(4 * scale), int(8 * scale)))
        pygame.draw.rect(surface, (220, 50, 50), (x - int(4 * scale), head_y - int(4 * scale), int(8 * scale), int(4 * scale)))

        # Face visor
        pygame.gfxdraw.filled_ellipse(surface, x, head_y + int(2 * scale), int(6 * scale), int(4 * scale), (50, 50, 60))

        # Status indicator
        if self.state == 'responding':
            # Flashing light
            flash = abs(math.sin(time_val * 8))
            pygame.gfxdraw.filled_circle(surface, x, head_y - head_radius - int(5 * scale), int(4 * scale),
                                        (255, int(50 + 200 * flash), 50))
        elif self.state == 'rescuing':
            # Healing effect
            for i in range(3):
                angle = time_val * 5 + i * 2.1
                hx = x + int(math.cos(angle) * 15)
                hy = torso_y + int(math.sin(angle) * 15)
                pygame.gfxdraw.filled_circle(surface, hx, hy, 3, (100, 255, 100, 150))

        # "MEDIC" text above
        font = pygame.font.Font(None, 16)
        text = font.render("MEDIC", True, (255, 255, 255))
        text_rect = text.get_rect(center=(x, head_y - head_radius - int(15 * scale)))
        # Background
        pygame.draw.rect(surface, (220, 50, 50), (text_rect.x - 2, text_rect.y - 1, text_rect.width + 4, text_rect.height + 2))
        surface.blit(text, text_rect)


class RescueSystem:
    """Manages multiple rescuers responding to emergencies. GOAL: ZERO DEATHS!"""
    def __init__(self):
        self.rescuers = []
        self.spawn_cooldown = 0
        self.max_rescuers = 8  # More rescuers for zero deaths goal!

    def update(self, dt, maze, exits, people, pathfinder, hazards, time_val, alarm_active):
        # Update existing rescuers
        for rescuer in self.rescuers:
            rescuer.update(dt, maze, people, pathfinder, hazards, time_val)

        # Spawn new rescuers if needed - MORE AGGRESSIVE for zero deaths
        if alarm_active:
            self.spawn_cooldown -= dt

            # Count people needing help - trigger earlier at health < 80
            hurt_people = [p for p in people if p.alive and not p.escaped and p.health < 80
                          and not getattr(p, 'being_rescued', False)]
            hurt_count = len(hurt_people)

            # Count critical cases (health < 30) - need immediate response
            critical_count = sum(1 for p in hurt_people if p.health < 30)

            # Spawn rescuer if hurt people and we have capacity
            # Faster spawning for critical cases!
            spawn_delay = 1.0 if critical_count > 0 else 2.0

            if hurt_count > 0 and len(self.rescuers) < self.max_rescuers and self.spawn_cooldown <= 0:
                # Spawn from nearest exit to the most critical person
                if exits and hurt_people:
                    # Find most critical person
                    most_critical = min(hurt_people, key=lambda p: p.health)

                    # Find nearest exit to them
                    nearest_exit = min(exits, key=lambda e: abs(e[0] - most_critical.row) + abs(e[1] - most_critical.col))

                    new_rescuer = Rescuer(len(self.rescuers), nearest_exit[0], nearest_exit[1])
                    self.rescuers.append(new_rescuer)
                    self.spawn_cooldown = spawn_delay
                    sound_system.play('rescue', 0.5)

                    # Emergency siren for critical cases
                    if critical_count > 0:
                        sound_system.play('sensor_alert', 0.4)

    def get_stats(self):
        """Get rescue statistics for UI."""
        active_rescuers = len([r for r in self.rescuers if r.state != 'idle'])
        total_helped = sum(r.helped_count for r in self.rescuers)
        critical_saves = sum(getattr(r, 'critical_saves', 0) for r in self.rescuers)
        prevented = sum(getattr(r, 'deaths_prevented', 0) for r in self.rescuers)
        return {
            'active': active_rescuers,
            'total': len(self.rescuers),
            'helped': total_helped,
            'critical_saves': critical_saves,
            'deaths_prevented': prevented
        }

    def draw(self, surface, shake, time_val):
        for rescuer in self.rescuers:
            rescuer.draw(surface, shake, time_val)

    def reset(self):
        self.rescuers = []
        self.spawn_cooldown = 0


# ═══════════════════════════════════════════════════════════════════════════════
# BUILDING GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_building():
    maze = [[FLOOR for _ in range(COLS)] for _ in range(ROWS)]
    exits = []

    # Outer walls
    for r in range(ROWS):
        maze[r][0] = WALL
        maze[r][COLS - 1] = WALL
    for c in range(COLS):
        maze[0][c] = WALL
        maze[ROWS - 1][c] = WALL

    # Corridors
    h_corr = [ROWS // 3, 2 * ROWS // 3]
    v_corr = [COLS // 4, COLS // 2, 3 * COLS // 4]

    for hr in h_corr:
        for c in range(1, COLS - 1):
            for r in range(hr - 1, hr + 2):
                if 0 < r < ROWS - 1:
                    maze[r][c] = CORRIDOR

    for vc in v_corr:
        for r in range(1, ROWS - 1):
            for c in range(vc - 1, vc + 2):
                if 0 < c < COLS - 1:
                    maze[r][c] = CORRIDOR

    # Rooms
    def make_room(r1, r2, c1, c2):
        for r in range(r1, r2 + 1):
            if maze[r][c1] != CORRIDOR: maze[r][c1] = WALL
            if maze[r][c2] != CORRIDOR: maze[r][c2] = WALL
        for c in range(c1, c2 + 1):
            if maze[r1][c] != CORRIDOR: maze[r1][c] = WALL
            if maze[r2][c] != CORRIDOR: maze[r2][c] = WALL

        for r in range(r1 + 1, r2):
            for c in range(c1 + 1, c2):
                if maze[r][c] != CORRIDOR:
                    maze[r][c] = CARPET

        # Multiple doors per room: try all four sides, prefer corridors
        door_candidates = []
        for c in range(c1 + 1, c2):
            if r2 + 1 < ROWS and maze[r2 + 1][c] == CORRIDOR:
                door_candidates.append((r2, c))
            if r1 - 1 > 0 and maze[r1 - 1][c] == CORRIDOR:
                door_candidates.append((r1, c))
        for r in range(r1 + 1, r2):
            if c1 - 1 > 0 and maze[r][c1 - 1] == CORRIDOR:
                door_candidates.append((r, c1))
            if c2 + 1 < COLS and maze[r][c2 + 1] == CORRIDOR:
                door_candidates.append((r, c2))

        # Place up to three doors to keep flow without over-opening
        random.shuffle(door_candidates)
        for dr, dc in door_candidates[:3]:
            maze[dr][dc] = DOOR

    sections = [
        (2, h_corr[0] - 2, 2, v_corr[0] - 2),
        (2, h_corr[0] - 2, v_corr[0] + 2, v_corr[1] - 2),
        (2, h_corr[0] - 2, v_corr[1] + 2, v_corr[2] - 2),
        (2, h_corr[0] - 2, v_corr[2] + 2, COLS - 3),
        (h_corr[0] + 2, h_corr[1] - 2, 2, v_corr[0] - 2),
        (h_corr[0] + 2, h_corr[1] - 2, v_corr[2] + 2, COLS - 3),
        (h_corr[1] + 2, ROWS - 3, 2, v_corr[0] - 2),
        (h_corr[1] + 2, ROWS - 3, v_corr[0] + 2, v_corr[1] - 2),
        (h_corr[1] + 2, ROWS - 3, v_corr[1] + 2, v_corr[2] - 2),
        (h_corr[1] + 2, ROWS - 3, v_corr[2] + 2, COLS - 3),
    ]

    for r1, r2, c1, c2 in sections:
        if r2 - r1 > 3 and c2 - c1 > 3:
            make_room(r1, r2, c1, c2)

    # Exits (add more mid-wall and interior corridor exits for faster egress)
    exit_pos = [
        (h_corr[0], 1), (h_corr[1], 1),
        (h_corr[0], COLS - 2), (h_corr[1], COLS - 2),
        (1, v_corr[0]), (1, v_corr[1]), (1, v_corr[2]),
        (ROWS - 2, v_corr[0]), (ROWS - 2, v_corr[1]), (ROWS - 2, v_corr[2]),
        # Mid-edge exits
        (ROWS // 2, 1), (ROWS // 2, COLS - 2),
        (1, COLS // 2), (ROWS - 2, COLS // 2),
        # Interior corridor exits at key intersections
        (h_corr[0], v_corr[1]), (h_corr[1], v_corr[1]),
        (ROWS // 2, v_corr[0]), (ROWS // 2, v_corr[2]),
    ]

    # Deduplicate while preserving intent
    seen_exits = set()
    for er, ec in exit_pos:
        if (er, ec) in seen_exits:
            continue
        seen_exits.add((er, ec))
        if 0 < er < ROWS - 1 and 0 < ec < COLS - 1:
            maze[er][ec] = EXIT
            exits.append((er, ec))
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = er + dr, ec + dc
                    if 0 < nr < ROWS - 1 and 0 < nc < COLS - 1:
                        if maze[nr][nc] == WALL:
                            maze[nr][nc] = CORRIDOR

    return maze, exits

# ═══════════════════════════════════════════════════════════════════════════════
# PERSON CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class Person:
    def __init__(self, pid, row, col, state, is_warden=False):
        self.id = pid
        self.row = row
        self.col = col
        self.x = col * TILE + TILE // 2
        self.y = row * TILE + TILE // 2
        self.tx = self.x
        self.ty = self.y

        self.state = STATE_WARDEN if is_warden else state
        self.is_warden = is_warden

        self.alive = True
        self.escaped = False
        self.health = 100
        self.awareness = 1.0 if is_warden else 0.0
        self.path = []
        self.path_index = 0
        self.target_exit = None

        self.color = Colors.WARDEN if is_warden else random.choice([
            (255, 90, 90), (90, 160, 255), (90, 255, 90), (255, 230, 90),
            (255, 90, 255), (90, 255, 255), (255, 160, 90)
        ])

        self.walk_frame = 0
        self.moving = False
        self.speed = random.uniform(80, 110) * (1.2 if is_warden else 1.0)

        self.rl_target = None  # For RL coordinator

    def emergency_step(self, maze, hazards, smoke):
        """Last-ditch move when trapped: step to least dangerous neighbor."""
        candidates = []
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = self.row + dr, self.col + dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if maze[nr][nc] == WALL:
                continue
            # Danger score prefers clear tiles but allows moving through mild hazard
            hazard_penalty = 2.0 if (nr, nc) in hazards else 0.0
            smoke_penalty = smoke.get((nr, nc), 0) * 1.5
            distance_bias = 0.1 * (abs(dr) + abs(dc))
            danger = hazard_penalty + smoke_penalty + distance_bias
            candidates.append((danger, random.random(), nr, nc))

        if not candidates:
            return False

        candidates.sort()
        _, _, nr, nc = candidates[0]
        self.tx = nc * TILE + TILE // 2
        self.ty = nr * TILE + TILE // 2
        self.row, self.col = nr, nc
        self.path = []
        self.path_index = 0
        self.moving = True
        return True

    def find_exit(self, exits, pathfinder, maze, hazards):
        if not exits:
            return

        best_exit = None
        best_score = float('inf')

        for exit_pos in exits:
            er, ec = exit_pos
            dist = abs(self.row - er) + abs(self.col - ec)

            danger = 0
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    if (er + dr, ec + dc) in hazards:
                        danger += 50

            score = dist + danger
            if score < best_score:
                best_score = score
                best_exit = exit_pos

        if best_exit:
            self.target_exit = best_exit
            self.path = pathfinder.find_path((self.row, self.col), best_exit, maze, hazards)
            self.path_index = 0

    def update(self, dt, maze, exits, hazards, pathfinder, alarm_active, people, smoke, neural_aco):
        if not self.alive or self.escaped:
            return

        # Check if in danger zone
        in_hazard = (self.row, self.col) in hazards
        smoke_level = smoke.get((self.row, self.col), 0)
        near_fire = any(abs(self.row - h[0]) + abs(self.col - h[1]) <= 2 for h in hazards)

        # Damage from fire
        if in_hazard:
            self.health -= 25 * dt  # Slightly reduced for survivability
            self.state = STATE_PANICKING
            sound_system.play('injury', 0.2) if random.random() < 0.02 else None

        # Damage from smoke
        if smoke_level > 0.5:
            self.health -= smoke_level * 8 * dt  # Slightly reduced

        # Immediate alarm response: start evacuating as soon as alarm is on
        if alarm_active and self.state not in [STATE_WARDEN, STATE_EVACUATING, STATE_PANICKING]:
            self.awareness = 1.0
            self.state = STATE_AWARE

        # === HEALTH REGENERATION when safe (ZERO DEATHS GOAL) ===
        if not in_hazard and smoke_level < 0.3 and not near_fire:
            if self.health < 100:
                # Slow regeneration when away from danger
                regen_rate = 5 * dt  # 5 HP/sec when safe
                # Faster regen if being rescued
                if getattr(self, 'being_rescued', False):
                    regen_rate = 0  # Rescuer handles healing
                self.health = min(100, self.health + regen_rate)

        # Play panic sound occasionally
        if self.health < 30 and random.random() < 0.01:
            sound_system.play('panic', 0.2)

        if self.health <= 0:
            self.alive = False
            neural_aco.deposit_danger_pheromone((self.row, self.col), 50)
            sound_system.play('injury', 0.5)  # Death sound
            return

        # Check escape
        if maze[self.row][self.col] == EXIT:
            self.escaped = True
            neural_aco.deposit_safe_pheromone(
                [(self.row, self.col)] + self.path[:self.path_index], True
            )
            sound_system.play('escape', 0.2)
            return

        # Awareness - faster reaction to emergencies
        if self.state not in [STATE_AWARE, STATE_EVACUATING, STATE_PANICKING, STATE_WARDEN]:
            # Alarm awareness - much faster now
            if alarm_active:
                self.awareness += 0.8 * dt if self.state != STATE_HEADPHONES else 0.2 * dt

            # Proximity to hazards - larger range and faster detection
            for h_pos in hazards:
                dist = abs(self.row - h_pos[0]) + abs(self.col - h_pos[1])
                if dist < 15:  # Increased detection range
                    # Immediate awareness if very close
                    if dist < 4:
                        self.awareness = 1.0  # Instant awareness when fire is right next to you
                    else:
                        self.awareness += 2.5 / (dist + 1) * dt  # Much faster awareness gain

            # Smoke also triggers awareness (even headphones users can see/smell smoke)
            if smoke_level > 0.2:
                self.awareness += smoke_level * 2.0 * dt

            # Social awareness - see other people running or wardens alerting
            for other in people:
                if other.id == self.id:
                    continue
                dist = abs(self.row - other.row) + abs(self.col - other.col)
                if dist < 6:
                    # Wardens actively alert nearby people (even headphones users)
                    if other.is_warden and other.state == STATE_WARDEN:
                        self.awareness += 1.5 * dt
                    # Seeing others evacuate or panic triggers awareness
                    elif other.state in [STATE_EVACUATING, STATE_PANICKING]:
                        self.awareness += 0.8 * dt

            # Lower threshold for faster reaction
            if self.awareness >= 0.5:
                self.state = STATE_AWARE
                return

        if self.state == STATE_AWARE:
            self.state = STATE_EVACUATING

        # Movement
        if not self.moving:
            if self.state in [STATE_EVACUATING, STATE_PANICKING, STATE_WARDEN]:
                if not self.path or self.path_index >= len(self.path):
                    self.find_exit(exits, pathfinder, maze, hazards)

                if self.path and self.path_index < len(self.path):
                    next_pos = self.path[self.path_index]
                    nr, nc = next_pos

                    # Check if path blocked
                    if (nr, nc) in hazards:
                        self.find_exit(exits, pathfinder, maze, hazards)
                        # If still blocked, attempt emergency sidestep
                        if (not self.path) and self.emergency_step(maze, hazards, smoke):
                            return
                        return

                    self.tx = nc * TILE + TILE // 2
                    self.ty = nr * TILE + TILE // 2
                    self.row, self.col = nr, nc
                    self.path_index += 1
                    self.moving = True
                else:
                    # No path found (likely surrounded) → emergency move to least dangerous neighbor
                    if self.emergency_step(maze, hazards, smoke):
                        self.state = STATE_PANICKING
                        return

        if self.moving:
            self.walk_frame += dt * 12
            speed = self.speed * (1.3 if self.state == STATE_PANICKING else 1.0)

            dx = self.tx - self.x
            dy = self.ty - self.y
            dist = math.hypot(dx, dy)

            if dist < 2:
                self.x, self.y = self.tx, self.ty
                self.moving = False
            else:
                self.x += (dx / dist) * speed * dt
                self.y += (dy / dist) * speed * dt

    def draw(self, surface, shake, time_val):
        if not self.alive or self.escaped:
            return

        x = int(self.x + shake[0])
        y = int(self.y + shake[1])

        # Scale factor for larger, more detailed humans
        scale = 1.8

        # Enhanced shadow with soft gradient
        shadow_w, shadow_h = int(24 * scale), int(10 * scale)
        shadow_surf = pygame.Surface((shadow_w + 8, shadow_h + 4), pygame.SRCALPHA)
        for i in range(6):
            alpha = 70 - i * 12
            offset = i * 1
            pygame.draw.ellipse(shadow_surf, (15, 15, 20, max(0, alpha)),
                              (offset + 4, offset, shadow_w - offset * 2, shadow_h - offset))
        surface.blit(shadow_surf, (x - shadow_w // 2 - 4, y + int(4 * scale)))

        # Color scheme based on state
        if self.is_warden:
            body_color = (255, 200, 50)
            body_light = (255, 230, 120)
            body_dark = (200, 150, 20)
            outline = Colors.WARDEN
        elif self.state == STATE_PANICKING:
            body_color = (255, 80, 80)
            body_light = (255, 140, 140)
            body_dark = (180, 40, 40)
            outline = Colors.DANGER
        elif self.state == STATE_EVACUATING:
            body_color = (80, 220, 100)
            body_light = (140, 255, 160)
            body_dark = (40, 160, 60)
            outline = Colors.EVACUATING
        elif self.state == STATE_AWARE:
            body_color = (255, 240, 80)
            body_light = (255, 255, 160)
            body_dark = (200, 180, 40)
            outline = Colors.AWARE
        elif self.state == STATE_HEADPHONES:
            body_color = (220, 100, 220)
            body_light = (255, 160, 255)
            body_dark = (160, 60, 160)
            outline = Colors.HEADPHONES
        else:
            body_color = self.color
            body_light = (min(255, self.color[0] + 50), min(255, self.color[1] + 50), min(255, self.color[2] + 50))
            body_dark = (max(0, self.color[0] - 50), max(0, self.color[1] - 50), max(0, self.color[2] - 50))
            outline = (50, 50, 60)

        # Skin tones
        skin_base = (235, 195, 165)
        skin_light = (255, 225, 200)
        skin_shadow = (200, 160, 130)

        # === LEGS ===
        leg_w, leg_h = int(6 * scale), int(14 * scale)
        leg_spacing = int(4 * scale)
        pants_color = (50, 60, 80)
        pants_light = (70, 80, 100)
        pants_dark = (30, 40, 60)
        shoe_color = (40, 35, 35)

        walk_offset = math.sin(self.walk_frame) * 4 * scale if self.moving else 0

        for leg_side in [-1, 1]:
            leg_x = x + leg_side * leg_spacing
            leg_offset = walk_offset * leg_side

            # Pants/leg
            leg_rect = (leg_x - leg_w // 2, y - int(2 * scale) + int(leg_offset), leg_w, leg_h)
            pygame.draw.ellipse(surface, pants_dark, leg_rect)
            pygame.draw.ellipse(surface, pants_color, (leg_rect[0] + 1, leg_rect[1] + 1, leg_w - 2, leg_h - 2))

            # Pants highlight
            pygame.draw.ellipse(surface, pants_light, (leg_rect[0] + 2, leg_rect[1] + 2, leg_w // 2, leg_h // 3))

            # Shoe
            shoe_y = y + int(10 * scale) + int(leg_offset)
            pygame.gfxdraw.filled_ellipse(surface, leg_x, shoe_y, int(4 * scale), int(3 * scale), shoe_color)
            pygame.gfxdraw.aaellipse(surface, leg_x, shoe_y, int(4 * scale), int(3 * scale), (60, 55, 55))

        # === TORSO ===
        torso_w, torso_h = int(16 * scale), int(18 * scale)
        torso_y = y - int(10 * scale)

        # Torso shadow/outline
        pygame.gfxdraw.filled_ellipse(surface, x, torso_y, torso_w // 2 + 2, torso_h // 2 + 2, outline)
        pygame.gfxdraw.aaellipse(surface, x, torso_y, torso_w // 2 + 2, torso_h // 2 + 2, outline)

        # Main torso with gradient effect
        for i in range(torso_h // 2, 0, -1):
            ratio = i / (torso_h // 2)
            r = int(body_dark[0] + (body_color[0] - body_dark[0]) * ratio)
            g = int(body_dark[1] + (body_color[1] - body_dark[1]) * ratio)
            b = int(body_dark[2] + (body_color[2] - body_dark[2]) * ratio)
            pygame.gfxdraw.filled_ellipse(surface, x, torso_y, int(torso_w // 2 * ratio) + 1, i, (r, g, b))

        # Torso highlight (3D shine)
        pygame.gfxdraw.filled_ellipse(surface, x - int(3 * scale), torso_y - int(4 * scale),
                                     int(4 * scale), int(5 * scale), body_light)

        # Collar detail
        pygame.gfxdraw.filled_ellipse(surface, x, torso_y - torso_h // 2 + int(2 * scale),
                                     int(5 * scale), int(3 * scale), skin_shadow)

        # === ARMS ===
        arm_w, arm_h = int(5 * scale), int(12 * scale)
        arm_swing = math.sin(self.walk_frame + 0.5) * 3 * scale if self.moving else 0

        for arm_side in [-1, 1]:
            arm_x = x + arm_side * (torso_w // 2 + int(1 * scale))
            arm_y = torso_y + int(arm_swing * arm_side)

            # Upper arm (shirt color)
            pygame.draw.ellipse(surface, body_dark, (arm_x - arm_w // 2, arm_y - arm_h // 2, arm_w, arm_h))
            pygame.draw.ellipse(surface, body_color, (arm_x - arm_w // 2 + 1, arm_y - arm_h // 2 + 1, arm_w - 2, arm_h - 2))

            # Hand
            hand_y = arm_y + arm_h // 2
            pygame.gfxdraw.filled_ellipse(surface, arm_x, hand_y, int(3 * scale), int(4 * scale), skin_base)
            pygame.gfxdraw.aaellipse(surface, arm_x, hand_y, int(3 * scale), int(4 * scale), skin_light)

        # === HEAD ===
        head_radius = int(10 * scale)
        head_y = torso_y - torso_h // 2 - head_radius + int(3 * scale)

        # Neck
        pygame.gfxdraw.filled_ellipse(surface, x, head_y + head_radius - int(2 * scale),
                                     int(4 * scale), int(5 * scale), skin_shadow)

        # Head shadow/outline
        pygame.gfxdraw.filled_ellipse(surface, x, head_y, head_radius + 2, head_radius + 2, outline)
        pygame.gfxdraw.aaellipse(surface, x, head_y, head_radius + 2, head_radius + 2, outline)

        # Head with gradient
        for i in range(head_radius, 0, -1):
            ratio = i / head_radius
            r = int(skin_shadow[0] + (skin_base[0] - skin_shadow[0]) * ratio)
            g = int(skin_shadow[1] + (skin_base[1] - skin_shadow[1]) * ratio)
            b = int(skin_shadow[2] + (skin_base[2] - skin_shadow[2]) * ratio)
            pygame.gfxdraw.filled_ellipse(surface, x, head_y, i, i, (r, g, b))

        # Head highlight
        pygame.gfxdraw.filled_ellipse(surface, x - int(3 * scale), head_y - int(3 * scale),
                                     int(4 * scale), int(4 * scale), skin_light)

        # Hair (simple cap style)
        hair_color = random.Random(self.id).choice([(40, 30, 20), (80, 50, 30), (30, 30, 35), (150, 100, 50)])
        pygame.gfxdraw.filled_ellipse(surface, x, head_y - int(2 * scale),
                                     head_radius - 1, int(head_radius * 0.7), hair_color)

        # === FACE ===
        eye_y = head_y + int(1 * scale)
        eye_spacing = int(4 * scale)

        # Eyes with detail
        for eye_side in [-1, 1]:
            eye_x = x + eye_side * eye_spacing

            # Eye white
            pygame.gfxdraw.filled_ellipse(surface, eye_x, eye_y, int(3 * scale), int(2 * scale), (255, 255, 255))

            # Pupil
            pupil_offset = int(0.5 * scale) if self.moving else 0
            pygame.gfxdraw.filled_circle(surface, eye_x + pupil_offset, eye_y, int(1.5 * scale), (40, 40, 50))

            # Eye shine
            pygame.gfxdraw.filled_circle(surface, eye_x - 1, eye_y - 1, 1, (255, 255, 255))

        # Eyebrows
        brow_y = eye_y - int(3 * scale)
        if self.state == STATE_PANICKING:
            # Worried eyebrows
            pygame.draw.line(surface, hair_color, (x - eye_spacing - 2, brow_y + 1),
                           (x - eye_spacing + 3, brow_y - 1), 2)
            pygame.draw.line(surface, hair_color, (x + eye_spacing - 3, brow_y - 1),
                           (x + eye_spacing + 2, brow_y + 1), 2)
        else:
            # Normal eyebrows
            pygame.draw.line(surface, hair_color, (x - eye_spacing - 2, brow_y),
                           (x - eye_spacing + 3, brow_y), 2)
            pygame.draw.line(surface, hair_color, (x + eye_spacing - 3, brow_y),
                           (x + eye_spacing + 2, brow_y), 2)

        # Mouth
        mouth_y = head_y + int(5 * scale)
        if self.state == STATE_PANICKING:
            # Open mouth (scared)
            pygame.gfxdraw.filled_ellipse(surface, x, mouth_y, int(3 * scale), int(2 * scale), (60, 30, 30))
        elif self.state == STATE_EVACUATING:
            # Determined expression
            pygame.draw.line(surface, (150, 100, 100), (x - int(3 * scale), mouth_y),
                           (x + int(3 * scale), mouth_y), 2)
        else:
            # Neutral/slight smile
            pygame.draw.arc(surface, (180, 120, 120),
                          (x - int(3 * scale), mouth_y - int(2 * scale), int(6 * scale), int(4 * scale)),
                          0.2, math.pi - 0.2, 2)

        # === ACCESSORIES ===

        # Warden hat
        if self.is_warden:
            hat_y = head_y - head_radius - int(2 * scale)
            # Hat brim
            pygame.gfxdraw.filled_ellipse(surface, x, hat_y + int(4 * scale),
                                         int(12 * scale), int(4 * scale), (180, 140, 0))
            pygame.gfxdraw.aaellipse(surface, x, hat_y + int(4 * scale),
                                    int(12 * scale), int(4 * scale), Colors.WARDEN)
            # Hat top
            pygame.draw.rect(surface, Colors.WARDEN,
                           (x - int(8 * scale), hat_y - int(4 * scale), int(16 * scale), int(8 * scale)))
            pygame.draw.rect(surface, (255, 230, 100),
                           (x - int(7 * scale), hat_y - int(3 * scale), int(14 * scale), int(3 * scale)))
            # Badge
            pygame.gfxdraw.filled_circle(surface, x, hat_y, int(2 * scale), (255, 255, 200))

        # Headphones
        if self.state == STATE_HEADPHONES:
            hp_color = (50, 50, 60)
            hp_light = (80, 80, 90)
            # Ear cups
            for side in [-1, 1]:
                cup_x = x + side * (head_radius + int(2 * scale))
                pygame.gfxdraw.filled_ellipse(surface, cup_x, eye_y, int(4 * scale), int(5 * scale), hp_color)
                pygame.gfxdraw.aaellipse(surface, cup_x, eye_y, int(4 * scale), int(5 * scale), hp_light)
                pygame.gfxdraw.filled_ellipse(surface, cup_x, eye_y, int(2 * scale), int(3 * scale), (30, 30, 35))
            # Headband
            pygame.draw.arc(surface, hp_color,
                          (x - head_radius - int(2 * scale), head_y - head_radius - int(4 * scale),
                           (head_radius + int(2 * scale)) * 2, int(12 * scale)),
                          0.1, math.pi - 0.1, 3)

        # === HEALTH BAR ===
        if self.health < 90:
            bar_w = int(20 * scale)
            bar_h = int(4 * scale)
            bar_y = head_y - head_radius - int(12 * scale)
            if self.is_warden:
                bar_y -= int(10 * scale)

            # Background
            pygame.draw.rect(surface, (40, 10, 10), (x - bar_w // 2 - 1, bar_y - 1, bar_w + 2, bar_h + 2))
            pygame.draw.rect(surface, (100, 20, 20), (x - bar_w // 2, bar_y, bar_w, bar_h))

            # Health fill
            fill_w = int(bar_w * self.health / 100)
            if fill_w > 0:
                if self.health > 60:
                    health_color = (50, 200, 50)
                elif self.health > 30:
                    health_color = (200, 200, 50)
                else:
                    health_color = (200, 50, 50)
                pygame.draw.rect(surface, health_color, (x - bar_w // 2, bar_y, fill_w, bar_h))

            # Highlight
            pygame.draw.line(surface, (255, 255, 255), (x - bar_w // 2, bar_y), (x + bar_w // 2, bar_y), 1)

        # === STATE INDICATOR (floating icon) ===
        if self.state == STATE_EVACUATING:
            # Running icon above head
            icon_y = head_y - head_radius - int(8 * scale)
            pygame.gfxdraw.filled_circle(surface, x, icon_y, int(3 * scale), (100, 255, 150, 180))
        elif self.state == STATE_PANICKING:
            # Exclamation mark
            icon_y = head_y - head_radius - int(10 * scale)
            pygame.draw.rect(surface, (255, 100, 100), (x - 2, icon_y, 4, int(6 * scale)))
            pygame.gfxdraw.filled_circle(surface, x, icon_y + int(8 * scale), 2, (255, 100, 100))

        # === "NEEDS HELP" INDICATOR FOR HURT PEOPLE ===
        if self.health < 80 and self.health > 0:
            help_y = head_y - head_radius - int(25 * scale)
            if self.is_warden:
                help_y -= int(10 * scale)

            # Flashing alarm beacon effect
            flash_intensity = abs(math.sin(time_val * 6))
            beacon_color = (255, int(50 + 200 * flash_intensity), 50)
            beacon_size = int(4 * scale + flash_intensity * 2)

            # Draw alarm beacon circle
            pygame.gfxdraw.filled_circle(surface, x, help_y + int(15 * scale), beacon_size, beacon_color)
            pygame.gfxdraw.aacircle(surface, x, help_y + int(15 * scale), beacon_size, (255, 255, 200))

            # Pulsing rings around beacon
            for ring in range(3):
                ring_alpha = int(150 * (1 - ring / 3) * flash_intensity)
                ring_radius = beacon_size + int(ring * 3 + flash_intensity * 5)
                ring_color = (255, 100, 50, ring_alpha)
                ring_surface = pygame.Surface((ring_radius * 2 + 4, ring_radius * 2 + 4), pygame.SRCALPHA)
                pygame.gfxdraw.aacircle(ring_surface, ring_radius + 2, ring_radius + 2, ring_radius, ring_color)
                surface.blit(ring_surface, (x - ring_radius - 2, help_y + int(15 * scale) - ring_radius - 2))

            # Bold "NEEDS HELP" text with background
            try:
                font_bold = pygame.font.Font(None, 18)
            except:
                font_bold = pygame.font.Font(None, 18)
            text = font_bold.render("NEEDS HELP", True, (255, 255, 255))
            text_rect = text.get_rect(center=(x, help_y))

            # Red background box with border
            bg_rect = (text_rect.x - 4, text_rect.y - 2, text_rect.width + 8, text_rect.height + 4)
            pygame.draw.rect(surface, (150, 30, 30), bg_rect)
            pygame.draw.rect(surface, (255, 100, 100), bg_rect, 2)
            surface.blit(text, text_rect)

            # Being rescued indicator
            if getattr(self, 'being_rescued', False):
                rescue_text = font_bold.render("RESCUE INCOMING", True, (150, 255, 150))
                rescue_rect = rescue_text.get_rect(center=(x, help_y + int(30 * scale)))
                pygame.draw.rect(surface, (30, 100, 30), (rescue_rect.x - 3, rescue_rect.y - 1, rescue_rect.width + 6, rescue_rect.height + 2))
                surface.blit(rescue_text, rescue_rect)

            # Play help beep sound periodically
            if not hasattr(self, '_last_help_beep'):
                self._last_help_beep = 0
            if time_val - self._last_help_beep > 2.0:  # Beep every 2 seconds
                sound_system.play('help_beep', 0.3)
                self._last_help_beep = time_val

# ═══════════════════════════════════════════════════════════════════════════════
# DRAWING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def draw_gradient_rect(surface, rect, color_top, color_bottom):
    """Draw a vertical gradient rectangle for 3D effect."""
    x, y, w, h = rect
    for i in range(h):
        ratio = i / max(h - 1, 1)
        r = int(color_top[0] + (color_bottom[0] - color_top[0]) * ratio)
        g = int(color_top[1] + (color_bottom[1] - color_top[1]) * ratio)
        b = int(color_top[2] + (color_bottom[2] - color_top[2]) * ratio)
        pygame.draw.line(surface, (r, g, b), (x, y + i), (x + w - 1, y + i))

def draw_3d_block(surface, x, y, w, h, top_color, front_color, depth=4):
    """Draw a 3D raised block with lighting."""
    # Darker side color
    side_color = (max(0, front_color[0] - 40), max(0, front_color[1] - 40), max(0, front_color[2] - 40))
    # Lighter top highlight
    highlight = (min(255, top_color[0] + 30), min(255, top_color[1] + 30), min(255, top_color[2] + 30))

    # Draw top face with gradient
    draw_gradient_rect(surface, (x, y, w, h - depth), highlight, top_color)

    # Draw front face (bottom part) with gradient
    draw_gradient_rect(surface, (x, y + h - depth, w, depth), front_color, side_color)

    # Right edge highlight
    pygame.draw.line(surface, side_color, (x + w - 1, y), (x + w - 1, y + h - 1), 1)

    # Top edge highlight
    pygame.draw.line(surface, highlight, (x, y), (x + w - 2, y), 1)

    # Left edge shadow
    pygame.draw.line(surface, (max(0, top_color[0] - 20), max(0, top_color[1] - 20), max(0, top_color[2] - 20)),
                     (x, y + 1), (x, y + h - 1), 1)

def draw_tile(surface, row, col, maze, hazards, time_val, shake, alarm_flash):
    x = int(col * TILE + shake[0])
    y = int(row * TILE + shake[1])
    tile = maze[row][col]

    # Fire rendering - enhanced 3D flames
    if (row, col) in hazards and hazards[(row, col)]['type'] == 'fire':
        # Charred ground with gradient
        draw_gradient_rect(surface, (x, y, TILE, TILE), (80, 40, 25), (40, 20, 10))

        # Animated 3D flames with multiple layers
        num_flames = 5
        for i in range(num_flames):
            fx = x + 2 + i * (TILE - 4) // num_flames
            phase = time_val * 15 + i * 1.2 + col * 0.3 + row * 0.2
            fh = TILE * 0.6 + math.sin(phase) * TILE * 0.25
            fw = TILE // num_flames + 2

            # Flame layers - outer to inner for depth
            flame_layers = [
                ((255, 60, 0), fh * 1.0, fw),      # Outer orange
                ((255, 150, 0), fh * 0.85, fw * 0.8),  # Yellow-orange
                ((255, 220, 50), fh * 0.7, fw * 0.6),  # Yellow
                ((255, 255, 200), fh * 0.5, fw * 0.4), # White core
            ]

            for color, height, width in flame_layers:
                cx = fx + fw // 2
                points = [
                    (cx - width / 2, y + TILE),
                    (cx + width / 2, y + TILE),
                    (cx + width / 4 + math.sin(phase * 1.5) * 2, y + TILE - height * 0.6),
                    (cx + math.sin(phase * 2) * 3, y + TILE - height),
                    (cx - width / 4 + math.sin(phase * 1.3) * 2, y + TILE - height * 0.6),
                ]
                pygame.draw.polygon(surface, color, points)
                # Anti-aliased edges
                pygame.gfxdraw.aapolygon(surface, [(int(p[0]), int(p[1])) for p in points], color)

        # Ember particles
        for i in range(3):
            ex = x + random.randint(2, TILE - 2)
            ey = y + TILE - int((time_val * 40 + i * 20) % TILE)
            ember_color = (255, 200 + random.randint(-50, 50), 50)
            pygame.gfxdraw.filled_circle(surface, ex, ey, 1, ember_color)
        return

    # Normal tiles with 3D rendering
    if tile == FLOOR:
        # Alternating floor tiles with subtle 3D depth
        if (row + col) % 2 == 0:
            top_color = (190, 185, 175)
            bottom_color = (165, 160, 150)
        else:
            top_color = (175, 170, 160)
            bottom_color = (155, 150, 140)

        draw_gradient_rect(surface, (x, y, TILE, TILE), top_color, bottom_color)

        # Subtle tile edge grooves
        pygame.draw.line(surface, (140, 135, 125), (x, y + TILE - 1), (x + TILE - 1, y + TILE - 1), 1)
        pygame.draw.line(surface, (140, 135, 125), (x + TILE - 1, y), (x + TILE - 1, y + TILE - 1), 1)
        pygame.draw.line(surface, (210, 205, 195), (x, y), (x + TILE - 1, y), 1)
        pygame.draw.line(surface, (210, 205, 195), (x, y), (x, y + TILE - 1), 1)

    elif tile == WALL:
        # 3D raised wall block
        wall_height = 6

        # Main wall face with gradient
        wall_top = (65, 70, 80)
        wall_bottom = (35, 38, 45)
        draw_gradient_rect(surface, (x, y, TILE, TILE), wall_top, wall_bottom)

        # 3D top edge (lighter)
        pygame.draw.line(surface, (95, 100, 110), (x, y), (x + TILE - 1, y), 2)
        pygame.draw.line(surface, (85, 90, 100), (x, y + 1), (x + TILE - 1, y + 1), 1)

        # Left edge highlight
        pygame.draw.line(surface, (80, 85, 95), (x, y), (x, y + TILE - 1), 2)

        # Right edge shadow
        pygame.draw.line(surface, (25, 28, 35), (x + TILE - 1, y), (x + TILE - 1, y + TILE - 1), 2)
        pygame.draw.line(surface, (30, 33, 40), (x + TILE - 2, y + 1), (x + TILE - 2, y + TILE - 2), 1)

        # Bottom shadow
        pygame.draw.line(surface, (20, 22, 28), (x, y + TILE - 1), (x + TILE - 1, y + TILE - 1), 2)

        # Brick/block texture
        brick_color = (55, 58, 65)
        for by in range(0, TILE, 8):
            offset = 0 if (by // 8) % 2 == 0 else TILE // 2
            for bx in range(0, TILE, TILE // 2):
                pygame.draw.rect(surface, brick_color, (x + (bx + offset) % TILE, y + by, TILE // 2 - 1, 7), 1)

    elif tile == CORRIDOR:
        # Polished corridor floor with reflective gradient
        top_color = (175, 170, 165)
        mid_color = (160, 155, 150)
        bottom_color = (145, 140, 135)

        draw_gradient_rect(surface, (x, y, TILE, TILE // 2), top_color, mid_color)
        draw_gradient_rect(surface, (x, y + TILE // 2, TILE, TILE // 2), mid_color, bottom_color)

        # Subtle reflection line
        pygame.draw.line(surface, (195, 190, 185), (x + 2, y + TILE // 3), (x + TILE - 3, y + TILE // 3), 1)

        # Edge grooves
        pygame.draw.line(surface, (125, 120, 115), (x, y + TILE - 1), (x + TILE - 1, y + TILE - 1), 1)
        pygame.draw.line(surface, (125, 120, 115), (x + TILE - 1, y), (x + TILE - 1, y + TILE - 1), 1)

    elif tile == CARPET:
        # Rich carpet with fabric texture
        base_color = (110, 70, 70)
        light_color = (130, 85, 85)
        dark_color = (85, 55, 55)

        draw_gradient_rect(surface, (x, y, TILE, TILE), light_color, base_color)

        # Carpet texture pattern
        for ty in range(0, TILE, 3):
            for tx in range(0, TILE, 3):
                if (tx + ty) % 6 == 0:
                    pygame.draw.rect(surface, dark_color, (x + tx, y + ty, 2, 2))
                elif (tx + ty) % 6 == 3:
                    pygame.draw.rect(surface, light_color, (x + tx, y + ty, 2, 2))

        # Soft edge shadow
        pygame.draw.line(surface, (75, 45, 45), (x + TILE - 1, y), (x + TILE - 1, y + TILE - 1), 1)
        pygame.draw.line(surface, (75, 45, 45), (x, y + TILE - 1), (x + TILE - 1, y + TILE - 1), 1)

    elif tile == EXIT:
        # Glowing exit with pulsing 3D effect
        glow_intensity = abs(math.sin(time_val * 3))
        glow = int(glow_intensity * 60)

        # Multi-layer glow effect
        for i in range(3, 0, -1):
            glow_alpha = int(40 * glow_intensity / i)
            glow_rect = pygame.Surface((TILE + i * 4, TILE + i * 4), pygame.SRCALPHA)
            glow_rect.fill((50 + glow, 255, 100 + glow, glow_alpha))
            surface.blit(glow_rect, (x - i * 2, y - i * 2))

        # Main exit with gradient
        top_color = (80 + glow, 255, 140 + glow // 2)
        bottom_color = (40 + glow // 2, 200, 80 + glow // 2)
        draw_gradient_rect(surface, (x, y, TILE, TILE), top_color, bottom_color)

        # 3D raised border
        pygame.draw.rect(surface, (255, 255, 255), (x + 1, y + 1, TILE - 2, TILE - 2), 2)
        pygame.draw.rect(surface, (200, 255, 220), (x + 3, y + 3, TILE - 6, TILE - 6), 1)

        # Exit arrow/icon
        arrow_color = (255, 255, 255)
        cx, cy = x + TILE // 2, y + TILE // 2
        pygame.draw.polygon(surface, arrow_color, [
            (cx - 4, cy - 2), (cx + 2, cy - 2), (cx + 2, cy - 5),
            (cx + 6, cy), (cx + 2, cy + 5), (cx + 2, cy + 2), (cx - 4, cy + 2)
        ])

    elif tile == DOOR:
        # Wooden door with 3D panels
        # Door frame
        pygame.draw.rect(surface, (80, 50, 25), (x, y, TILE, TILE))

        # Door surface with wood grain gradient
        door_top = (150, 95, 50)
        door_bottom = (100, 60, 30)
        draw_gradient_rect(surface, (x + 2, y + 2, TILE - 4, TILE - 4), door_top, door_bottom)

        # Door panels (recessed)
        panel_color = (120, 75, 40)
        panel_shadow = (90, 55, 28)
        panel_highlight = (160, 105, 60)

        # Top panel
        pygame.draw.rect(surface, panel_shadow, (x + 4, y + 4, TILE - 8, TILE // 2 - 5))
        pygame.draw.rect(surface, panel_color, (x + 5, y + 5, TILE - 10, TILE // 2 - 7))
        pygame.draw.line(surface, panel_highlight, (x + 5, y + 5), (x + TILE - 6, y + 5), 1)

        # Bottom panel
        pygame.draw.rect(surface, panel_shadow, (x + 4, y + TILE // 2 + 2, TILE - 8, TILE // 2 - 5))
        pygame.draw.rect(surface, panel_color, (x + 5, y + TILE // 2 + 3, TILE - 10, TILE // 2 - 7))
        pygame.draw.line(surface, panel_highlight, (x + 5, y + TILE // 2 + 3), (x + TILE - 6, y + TILE // 2 + 3), 1)

        # Door handle
        pygame.draw.circle(surface, (200, 180, 50), (x + TILE - 7, y + TILE // 2), 3)
        pygame.gfxdraw.aacircle(surface, x + TILE - 7, y + TILE // 2, 3, (220, 200, 70))

    # Alarm flash with softer blend
    if alarm_flash and tile != WALL:
        overlay = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
        # Pulsing red glow
        pulse = abs(math.sin(time_val * 8)) * 0.5 + 0.5
        overlay.fill((255, 30, 30, int(25 * pulse)))
        surface.blit(overlay, (x, y))

def draw_emergency_floor_strobing(surface, neural_aco, shake, time_val, alarm_active):
    """
    Draw pheromone-based emergency floor strobing to guide evacuation.
    Uses safe pheromone trails to create animated light paths toward exits.
    """
    if not alarm_active:
        return

    safe = neural_aco.safe_pheromone
    danger = neural_aco.danger_pheromone

    # Calculate strobe phase (creates wave effect moving toward exits)
    strobe_speed = 8
    wave_length = 0.3

    for r in range(ROWS):
        for c in range(COLS):
            safe_level = safe[r, c]
            danger_level = danger[r, c]

            # Only strobe on safe paths (where safe pheromone is high)
            if safe_level > 0.4:
                x = int(c * TILE + shake[0])
                y = int(r * TILE + shake[1])

                # Calculate directional wave based on pheromone gradient
                max_safe = safe_level
                direction = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                        if safe[nr, nc] > max_safe:
                            max_safe = safe[nr, nc]
                            direction = dr + dc * 0.1

                # Create animated wave pattern
                wave_phase = (time_val * strobe_speed + (r + c) * wave_length + direction) % (2 * math.pi)
                strobe_intensity = (math.sin(wave_phase) + 1) / 2

                # Intensity based on safe pheromone strength
                base_intensity = min((safe_level - 0.4) / 0.6, 1.0)
                final_intensity = base_intensity * strobe_intensity

                if final_intensity > 0.1:
                    overlay = pygame.Surface((TILE, TILE), pygame.SRCALPHA)

                    # Green strobing for safe path
                    green_alpha = int(80 * final_intensity)
                    overlay.fill((50, 255, 100, green_alpha))

                    # Add LED strip effect (lines on edges)
                    led_brightness = int(200 * final_intensity)
                    led_color = (100, 255, 150, led_brightness)

                    # Edge LEDs
                    pygame.draw.line(overlay, led_color, (0, TILE - 2), (TILE, TILE - 2), 2)
                    pygame.draw.line(overlay, led_color, (0, 0), (TILE, 0), 2)

                    # Directional arrows for guidance
                    if (r + c) % 3 == int(time_val * 2) % 3:
                        arrow_color = (150, 255, 180, int(180 * final_intensity))
                        cx, cy = TILE // 2, TILE // 2
                        pygame.draw.polygon(overlay, arrow_color, [
                            (cx - 4, cy + 3), (cx + 4, cy + 3),
                            (cx + 4, cy), (cx + 6, cy),
                            (cx, cy - 5), (cx - 6, cy),
                            (cx - 4, cy)
                        ])

                    surface.blit(overlay, (x, y))

            # Danger zone warning strobing (red pulsing)
            elif danger_level > 1.0:
                x = int(c * TILE + shake[0])
                y = int(r * TILE + shake[1])

                danger_pulse = abs(math.sin(time_val * 12)) * min(danger_level / 3.0, 1.0)
                if danger_pulse > 0.2:
                    overlay = pygame.Surface((TILE, TILE), pygame.SRCALPHA)

                    # Red warning strobe
                    red_alpha = int(60 * danger_pulse)
                    overlay.fill((255, 50, 30, red_alpha))

                    # Warning X pattern
                    warn_color = (255, 100, 50, int(150 * danger_pulse))
                    pygame.draw.line(overlay, warn_color, (2, 2), (TILE - 2, TILE - 2), 2)
                    pygame.draw.line(overlay, warn_color, (TILE - 2, 2), (2, TILE - 2), 2)

                    surface.blit(overlay, (x, y))

def draw_smoke(surface, smoke, shake):
    """Draw volumetric 3D smoke with layered opacity."""
    for (r, c), level in smoke.items():
        if level > 0.1:
            x = int(c * TILE + shake[0])
            y = int(r * TILE + shake[1])

            # Create layered smoke effect for 3D depth
            overlay = pygame.Surface((TILE, TILE), pygame.SRCALPHA)

            # Multiple smoke layers with different colors for depth
            layers = [
                (75, 75, 80, min(int(level * 80), 120)),   # Dark base
                (85, 85, 90, min(int(level * 60), 100)),   # Mid layer
                (100, 100, 105, min(int(level * 40), 80)), # Light wisps
            ]

            for i, (r_col, g_col, b_col, alpha) in enumerate(layers):
                layer_surf = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
                # Offset each layer slightly for 3D effect
                offset = i * 2
                pygame.draw.ellipse(layer_surf, (r_col, g_col, b_col, alpha),
                                   (offset, offset, TILE - offset * 2, TILE - offset * 2))
                overlay.blit(layer_surf, (0, 0))

            # Add some swirling motion indication
            if level > 0.3:
                swirl_x = int(math.sin(c + level * 5) * 3)
                swirl_y = int(math.cos(r + level * 5) * 3)
                pygame.gfxdraw.filled_circle(overlay, TILE // 2 + swirl_x, TILE // 2 + swirl_y,
                                            TILE // 4, (90, 90, 95, min(int(level * 50), 80)))

            surface.blit(overlay, (x, y))

def draw_neural_predictions(surface, predictions, shake, time_val):
    """Draw neural network predictions as glowing holographic areas."""
    for r, c, prob in predictions:
        x = int(c * TILE + shake[0])
        y = int(r * TILE + shake[1])

        pulse = abs(math.sin(time_val * 4)) * 0.3 + 0.7
        base_alpha = int(80 * prob * pulse)

        # Create holographic prediction effect
        overlay = pygame.Surface((TILE + 4, TILE + 4), pygame.SRCALPHA)

        # Outer glow
        glow_color = (255, 50, 200, base_alpha // 3)
        pygame.draw.rect(overlay, glow_color, (0, 0, TILE + 4, TILE + 4))

        # Inner prediction zone with scanline effect
        inner_color = (255, 80, 220, base_alpha)
        pygame.draw.rect(overlay, inner_color, (2, 2, TILE, TILE))

        # Scanlines for tech effect
        for sy in range(0, TILE, 3):
            line_alpha = int(base_alpha * 0.3)
            pygame.draw.line(overlay, (255, 255, 255, line_alpha), (2, 2 + sy), (TILE + 1, 2 + sy), 1)

        # Pulsing border
        border_alpha = int(pulse * 200)
        pygame.draw.rect(overlay, (255, 100, 255, border_alpha), (2, 2, TILE, TILE), 2)

        # Corner accents
        corner_size = 4
        corner_color = (255, 200, 255, int(pulse * 255))
        pygame.draw.line(overlay, corner_color, (2, 2), (2 + corner_size, 2), 2)
        pygame.draw.line(overlay, corner_color, (2, 2), (2, 2 + corner_size), 2)
        pygame.draw.line(overlay, corner_color, (TILE, 2), (TILE - corner_size, 2), 2)
        pygame.draw.line(overlay, corner_color, (TILE, 2), (TILE, 2 + corner_size), 2)

        surface.blit(overlay, (x - 2, y - 2))

def draw_pheromones(surface, neural_aco, shake):
    """Visualize pheromone trails with glowing 3D effect."""
    safe = neural_aco.safe_pheromone
    danger = neural_aco.danger_pheromone

    for r in range(ROWS):
        for c in range(COLS):
            x = int(c * TILE + shake[0])
            y = int(r * TILE + shake[1])
            cx, cy = x + TILE // 2, y + TILE // 2

            # Safe pheromone (glowing green trail)
            if safe[r, c] > 0.3:
                intensity = (safe[r, c] - 0.3) / 0.7  # Normalize to 0-1
                base_alpha = min(int(intensity * 60), 80)

                # Outer glow
                glow_radius = int(TILE * 0.6 * (1 + intensity * 0.3))
                glow_surf = pygame.Surface((TILE + 8, TILE + 8), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(glow_surf, TILE // 2 + 4, TILE // 2 + 4,
                                            glow_radius, (0, 255, 150, base_alpha // 2))
                surface.blit(glow_surf, (x - 4, y - 4))

                # Inner bright core
                core_radius = int(TILE * 0.3 * (1 + intensity * 0.2))
                pygame.gfxdraw.filled_circle(surface, cx, cy, core_radius, (100, 255, 180, base_alpha))
                pygame.gfxdraw.aacircle(surface, cx, cy, core_radius, (150, 255, 200, min(base_alpha + 40, 255)))

            # Danger pheromone (pulsing red warning)
            if danger[r, c] > 0.5:
                intensity = (danger[r, c] - 0.5) / 0.5
                base_alpha = min(int(intensity * 40), 100)

                # Hazard pattern
                overlay = pygame.Surface((TILE, TILE), pygame.SRCALPHA)

                # Radial danger gradient
                for ring in range(3, 0, -1):
                    ring_alpha = int(base_alpha / ring)
                    ring_radius = TILE // 2 - (3 - ring) * 3
                    pygame.gfxdraw.filled_circle(overlay, TILE // 2, TILE // 2,
                                                ring_radius, (255, 60 + ring * 20, 60 + ring * 20, ring_alpha))

                # Warning stripes for high danger
                if intensity > 0.5:
                    for stripe in range(0, TILE, 6):
                        pygame.draw.line(overlay, (255, 200, 0, base_alpha // 2),
                                        (stripe, 0), (stripe + 3, TILE), 2)

                surface.blit(overlay, (x, y))

def draw_panel(surface, stats, neural_aco, sensor_network, rl_coordinator, mesh_network, time_val, paused, speed):
    """Draw the enhanced side panel with ML + mesh metrics."""
    px = MAP_WIDTH
    pygame.draw.rect(surface, Colors.PANEL_BG, (px, 0, PANEL_WIDTH, SCREEN_HEIGHT))
    pygame.draw.line(surface, Colors.PANEL_BORDER, (px, 0), (px, SCREEN_HEIGHT), 2)

    font_big = pygame.font.Font(None, 22)
    font_med = pygame.font.Font(None, 17)
    font_small = pygame.font.Font(None, 14)

    y = 8

    # Title
    title = font_big.render("NEURAL ACO EMERGENCY SYSTEM", True, Colors.NEURAL_GLOW)
    surface.blit(title, (px + 8, y))
    y += 22

    # Escape counter
    pygame.draw.rect(surface, Colors.PANEL_BORDER, (px + 8, y, PANEL_WIDTH - 16, 36), 1)
    escaped_text = font_big.render(f"ESCAPED: {stats['escaped']}/{stats['total']}", True, Colors.SUCCESS)
    surface.blit(escaped_text, (px + 14, y + 4))
    progress = stats['escaped'] / max(1, stats['total'])
    pygame.draw.rect(surface, (40, 60, 40), (px + 12, y + 22, PANEL_WIDTH - 24, 6))
    pygame.draw.rect(surface, Colors.SUCCESS, (px + 12, y + 22, int((PANEL_WIDTH - 24) * progress), 6))
    y += 42

    # Deaths
    if stats['deaths'] == 0:
        surface.blit(font_med.render("ZERO DEATHS!", True, Colors.SUCCESS), (px + 10, y))
    else:
        surface.blit(font_med.render(f"DEATHS: {stats['deaths']}", True, Colors.DANGER), (px + 10, y))
    y += 18

    pygame.draw.line(surface, Colors.PANEL_BORDER, (px + 8, y), (px + PANEL_WIDTH - 8, y), 1)
    y += 6

    # Neural Network Section
    surface.blit(font_med.render("LSTM FIRE PREDICTION", True, Colors.NEURAL_GLOW), (px + 10, y))
    y += 16

    confidence = neural_aco.neural_confidence
    conf_color = Colors.SUCCESS if confidence > 0.7 else Colors.WARNING if confidence > 0.4 else Colors.TEXT_DIM
    surface.blit(font_small.render(f"Prediction Confidence: {confidence:.1%}", True, conf_color), (px + 12, y))
    y += 13

    # Confidence bar
    pygame.draw.rect(surface, (40, 40, 50), (px + 12, y, 150, 8))
    pygame.draw.rect(surface, conf_color, (px + 12, y, int(150 * confidence), 8))
    y += 14

    predicted_zones = np.count_nonzero(neural_aco.predicted_danger > 0.25)
    surface.blit(font_small.render(f"Predicted Danger Zones: {predicted_zones}", True, Colors.PREDICTION), (px + 12, y))
    y += 16

    pygame.draw.line(surface, Colors.PANEL_BORDER, (px + 8, y), (px + PANEL_WIDTH - 8, y), 1)
    y += 6

    # IoT Sensors Section
    surface.blit(font_med.render("IOT SENSOR NETWORK", True, Colors.SENSOR_ACTIVE), (px + 10, y))
    y += 16

    sensor_data = sensor_network.get_sensor_fusion_data()
    surface.blit(font_small.render(f"Network Coverage: {sensor_data['coverage']:.0f}%", True, Colors.TEXT), (px + 12, y))
    y += 13
    surface.blit(font_small.render(f"Avg Temp: {sensor_data['temperature_avg']:.1f}C", True,
                                   Colors.DANGER if sensor_data['temperature_avg'] > 40 else Colors.TEXT), (px + 12, y))
    y += 13
    surface.blit(font_small.render(f"Max Temp: {sensor_data['temperature_max']:.1f}C", True,
                                   Colors.DANGER if sensor_data['temperature_max'] > 50 else Colors.TEXT), (px + 12, y))
    y += 13
    surface.blit(font_small.render(f"Smoke Level: {sensor_data['smoke_level']:.2f}", True,
                                   Colors.WARNING if sensor_data['smoke_level'] > 0.2 else Colors.TEXT), (px + 12, y))
    y += 13
    surface.blit(font_small.render(f"CO Level: {sensor_data['co_level']:.1f} ppm", True,
                                   Colors.DANGER if sensor_data['co_level'] > 30 else Colors.TEXT), (px + 12, y))
    y += 13
    surface.blit(font_small.render(f"Triggered Sensors: {len(sensor_data['triggered_sensors'])}", True,
                                   Colors.DANGER if sensor_data['triggered_sensors'] else Colors.TEXT), (px + 12, y))
    y += 16

    # Soldier Mesh Section (Indian Army handheld mesh)
    surface.blit(font_med.render("SOLDIER MESH NETWORK", True, Colors.IOT_MESH), (px + 10, y))
    y += 16
    mesh_stats = mesh_network.get_stats()
    surface.blit(font_small.render(f"Active Nodes: {mesh_stats['nodes']}", True, Colors.TEXT), (px + 12, y))
    y += 13
    surface.blit(font_small.render(f"Links: {mesh_stats['links']}", True, Colors.TEXT), (px + 12, y))
    y += 13
    surface.blit(font_small.render(f"Avg Degree: {mesh_stats['avg_degree']:.1f}", True, Colors.TEXT), (px + 12, y))
    y += 13
    surface.blit(font_small.render(f"Hazard Broadcast Steps: {mesh_stats['broadcasts']}", True, Colors.TEXT_DIM), (px + 12, y))
    y += 16

    pygame.draw.line(surface, Colors.PANEL_BORDER, (px + 8, y), (px + PANEL_WIDTH - 8, y), 1)
    y += 6

    # RL Section
    surface.blit(font_med.render("RL COORDINATOR", True, Colors.ACCENT), (px + 10, y))
    y += 16

    rl_stats = rl_coordinator.get_stats()
    surface.blit(font_small.render(f"Decisions Made: {rl_stats['decisions']}", True, Colors.TEXT), (px + 12, y))
    y += 13
    surface.blit(font_small.render(f"Avg Reward: {rl_stats['avg_reward']:.2f}", True, Colors.TEXT), (px + 12, y))
    y += 13
    surface.blit(font_small.render(f"Exploration: {rl_stats['epsilon']:.0%}", True, Colors.TEXT), (px + 12, y))
    y += 16

    pygame.draw.line(surface, Colors.PANEL_BORDER, (px + 8, y), (px + PANEL_WIDTH - 8, y), 1)
    y += 6

    # ACO Metrics
    surface.blit(font_med.render("ANT COLONY OPTIMIZATION", True, Colors.SAFE_PHEROMONE), (px + 10, y))
    y += 16

    safe_avg = float(np.mean(neural_aco.safe_pheromone))
    danger_avg = float(np.mean(neural_aco.danger_pheromone))
    surface.blit(font_small.render(f"Safe Pheromone Avg: {safe_avg:.2f}", True, Colors.SAFE_PHEROMONE), (px + 12, y))
    y += 13
    surface.blit(font_small.render(f"Danger Pheromone Avg: {danger_avg:.2f}", True, Colors.DANGER_PHEROMONE), (px + 12, y))
    y += 13
    surface.blit(font_small.render(f"Path Edges Tracked: {len(neural_aco.edge_usage)}", True, Colors.TEXT), (px + 12, y))
    y += 16

    pygame.draw.line(surface, Colors.PANEL_BORDER, (px + 8, y), (px + PANEL_WIDTH - 8, y), 1)
    y += 6

    # Controls
    surface.blit(font_med.render("CONTROLS", True, Colors.TEXT), (px + 10, y))
    y += 14

    controls = [
        "Click: Add fire",
        "A: Trigger alarm",
        "P: Toggle predictions",
        "T: Toggle pheromones",
        "S: Toggle sensors",
        "Space: Pause | R: Reset",
        "+/-: Speed"
    ]

    for ctrl in controls:
        surface.blit(font_small.render(ctrl, True, Colors.TEXT_DIM), (px + 12, y))
        y += 12

    # Status bar at bottom
    y = SCREEN_HEIGHT - 25
    status = f"Speed: {speed:.1f}x"
    if paused:
        status = "PAUSED | " + status
    surface.blit(font_small.render(status, True, Colors.TEXT_DIM), (px + 10, y))

def draw_bottom_bar(surface, stats, alarm_active, alarm_flash, time_val, rescue_stats=None):
    """Draw bottom status bar with ZERO DEATHS goal indicator."""
    by = MAP_HEIGHT

    bg_color = (40, 20, 20) if alarm_active and alarm_flash else Colors.PANEL_BG
    pygame.draw.rect(surface, bg_color, (0, by, MAP_WIDTH, 80))
    pygame.draw.line(surface, Colors.PANEL_BORDER, (0, by), (MAP_WIDTH, by), 2)

    font = pygame.font.Font(None, 20)
    font_big = pygame.font.Font(None, 28)
    font_goal = pygame.font.Font(None, 24)

    # Main stats
    color = Colors.SUCCESS if stats['deaths'] == 0 else Colors.DANGER
    text = f"ESCAPED: {stats['escaped']}/{stats['total']}  |  DEATHS: {stats['deaths']}"
    surface.blit(font_big.render(text, True, color), (15, by + 10))

    alive = stats['total'] - stats['escaped'] - stats['deaths']
    surface.blit(font.render(f"Inside Building: {alive}", True, Colors.TEXT_DIM), (15, by + 40))

    # === ZERO DEATHS GOAL INDICATOR ===
    goal_x = MAP_WIDTH - 280
    flash = abs(math.sin(time_val * 3))

    if stats['deaths'] == 0:
        # Goal achieved or in progress!
        goal_color = (50, 255, 100)
        goal_text = "ZERO DEATHS GOAL"
        status_text = "ON TRACK!" if alive > 0 else "ACHIEVED!"
        status_color = (100, 255, 150) if alive > 0 else (50, 255, 50)

        # Glowing border
        glow_rect = (goal_x - 5, by + 8, 180, 35)
        pygame.draw.rect(surface, (30, 80, 40), glow_rect)
        pygame.draw.rect(surface, goal_color, glow_rect, 2)
    else:
        # Goal failed
        goal_color = (255, 80, 80)
        goal_text = "ZERO DEATHS GOAL"
        status_text = "FAILED"
        status_color = (255, 100, 100)

        glow_rect = (goal_x - 5, by + 8, 180, 35)
        pygame.draw.rect(surface, (60, 20, 20), glow_rect)
        pygame.draw.rect(surface, goal_color, glow_rect, 2)

    surface.blit(font_goal.render(goal_text, True, goal_color), (goal_x, by + 12))
    surface.blit(font.render(status_text, True, status_color), (goal_x, by + 30))

    # === RESCUE STATS ===
    if rescue_stats:
        rescue_x = MAP_WIDTH - 90
        rescue_color = (255, 200, 100)
        surface.blit(font.render(f"MEDICS: {rescue_stats['active']}/{rescue_stats['total']}", True, rescue_color), (rescue_x, by + 12))
        surface.blit(font.render(f"SAVED: {rescue_stats['helped']}", True, (100, 255, 150)), (rescue_x, by + 28))
        crit_saves = rescue_stats.get('critical_saves', 0)
        prevented = rescue_stats.get('deaths_prevented', 0)
        if crit_saves > 0 or prevented > 0:
            crit_flash = int(200 + 55 * flash)
            combo_text = f"CRIT/PREV: {crit_saves}/{prevented}"
            surface.blit(font.render(combo_text, True, (255, crit_flash, 100)), (rescue_x, by + 44))

    if alarm_active:
        alarm_color = Colors.DANGER if alarm_flash else Colors.WARNING
        surface.blit(font.render("ALARM ACTIVE - EVACUATE!", True, alarm_color), (15, by + 58))

    # Hurt people count
    hurt_count = stats.get('hurt', 0)
    if hurt_count > 0:
        hurt_flash = int(200 + 55 * flash)
        surface.blit(font.render(f"NEEDS HELP: {hurt_count}", True, (255, hurt_flash, 50)), (250, by + 58))

# ═══════════════════════════════════════════════════════════════════════════════
# SPAWN FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def spawn_people(maze, count, num_wardens):
    people = []
    spawns = []

    for r in range(2, ROWS - 2):
        for c in range(2, COLS - 2):
            if maze[r][c] in [CARPET, FLOOR, CORRIDOR]:
                spawns.append((r, c))

    random.shuffle(spawns)

    states = [STATE_WORKING] * 15 + [STATE_HEADPHONES] * 8
    random.shuffle(states)

    # Wardens
    corridor_spawns = [(r, c) for r, c in spawns if maze[r][c] == CORRIDOR]
    for i in range(min(num_wardens, len(corridor_spawns))):
        r, c = corridor_spawns[i]
        people.append(Person(i, r, c, STATE_WARDEN, is_warden=True))

    # Regular people
    for i in range(num_wardens, min(count, len(spawns))):
        r, c = spawns[i]
        state = states[(i - num_wardens) % len(states)]
        people.append(Person(i, r, c, state))

    return people

def spawn_sensors(maze, count):
    """Spawn IoT sensors throughout the building with mesh network connectivity."""
    network = IoTSensorNetwork()
    spawns = []

    # Strategic sensor placement - cover key areas
    for r in range(3, ROWS - 3, 4):
        for c in range(3, COLS - 3, 4):
            if maze[r][c] != WALL:
                spawns.append((r, c))

    random.shuffle(spawns)
    sensor_types = ['temperature', 'smoke', 'co', 'motion']

    for i, (r, c) in enumerate(spawns[:count]):
        sensor_type = sensor_types[i % len(sensor_types)]
        sensor = IoTSensor(
            id=i,
            row=r,
            col=c,
            sensor_type=sensor_type
        )
        network.add_sensor(sensor)

    # Build mesh network connections between sensors
    network.build_mesh_network()

    return network

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if HEADLESS:
        print("Headless mode enabled; use run_headless_simulation() instead of the interactive loop.")
        return

    # Initialize systems
    maze, exits = generate_building()
    lstm_predictor = SimpleLSTMPredictor()
    neural_aco = NeuralACO(lstm_predictor)
    pathfinder = NeuralPathfinder(neural_aco)
    disasters = Disasters()
    alarm = AlarmSystem()
    sensor_network = spawn_sensors(maze, NUM_SENSORS)
    rl_coordinator = RLEvacuationCoordinator()
    people = spawn_people(maze, TOTAL_PEOPLE, NUM_WARDENS)
    soldier_mesh = SoldierMeshNetwork()

    # ML-Activated Sprinkler System
    sprinkler_system = SprinklerSystem()
    sprinkler_system.install_sprinklers(maze)

    # Rescue System for injured people
    rescue_system = RescueSystem()

    stats = {'escaped': 0, 'deaths': 0, 'hurt': 0, 'total': TOTAL_PEOPLE}

    clock = pygame.time.Clock()
    running = True
    paused = False
    speed = 1.0

    show_predictions = True
    show_pheromones = True
    show_sensors = True

    neural_update_timer = 0
    rl_update_timer = 0

    print("=" * 70)
    print("NEURAL ACO EMERGENCY RESPONSE SYSTEM (NAERS)")
    print("=" * 70)
    print("\nPATENTABLE INNOVATIONS:")
    print("  - Neural Predictive ACO (LSTM + Pheromones)")
    print("  - IoT Sensor Fusion with Kalman Filtering")
    print("  - RL-based Evacuation Coordination")
    print("  - Soldier-Worn Offline Mesh (Indian Army tactical variant)")
    print("\nCONTROLS: Click=Fire, A=Alarm, P=Predictions, T=Pheromones, S=Sensors")
    print("=" * 70)

    while running:
        dt = clock.tick(60) / 1000.0
        dt *= speed
        time_val = pygame.time.get_ticks() / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if mx < MAP_WIDTH and my < MAP_HEIGHT:
                    col = int((mx - disasters.shake_offset[0]) // TILE)
                    row = int((my - disasters.shake_offset[1]) // TILE)
                    if 0 < row < ROWS - 1 and 0 < col < COLS - 1:
                        if event.button == 1:
                            disasters.add_fire(row, col)
                            pathfinder.clear_cache()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_a:
                    alarm.trigger()
                elif event.key == pygame.K_p:
                    show_predictions = not show_predictions
                elif event.key == pygame.K_t:
                    show_pheromones = not show_pheromones
                elif event.key == pygame.K_s:
                    show_sensors = not show_sensors
                elif event.key == pygame.K_r:
                    # Reset
                    sound_system.stop_alarm()
                    maze, exits = generate_building()
                    lstm_predictor.reset()
                    neural_aco.reset()
                    pathfinder.clear_cache()
                    disasters = Disasters()
                    alarm = AlarmSystem()
                    sensor_network = spawn_sensors(maze, NUM_SENSORS)
                    rl_coordinator = RLEvacuationCoordinator()
                    people = spawn_people(maze, TOTAL_PEOPLE, NUM_WARDENS)
                    soldier_mesh = SoldierMeshNetwork()
                    sprinkler_system = SprinklerSystem()
                    sprinkler_system.install_sprinklers(maze)
                    rescue_system.reset()
                    stats = {'escaped': 0, 'deaths': 0, 'hurt': 0, 'total': TOTAL_PEOPLE}
                elif event.key in [pygame.K_EQUALS, pygame.K_PLUS]:
                    speed = min(speed + 0.5, 5.0)
                elif event.key == pygame.K_MINUS:
                    speed = max(speed - 0.5, 0.5)

        if not paused:
            # Update disasters
            disasters.update(dt, maze, neural_aco)
            alarm.update(dt)

            # Auto-trigger alarm on fire
            if disasters.hazards and not alarm.active:
                alarm.trigger()

            # Auto-trigger alarm from sensor network (sensors communicate and alert whole office)
            if sensor_network.global_alert_level > 0.5 and not alarm.active:
                alarm.trigger()
                sound_system.play('network_broadcast', 0.6)  # Network triggered the alarm

            # Update neural predictions
            neural_update_timer += dt
            if neural_update_timer > 0.5:
                fire_positions = disasters.get_fire_positions()
                sensor_data = sensor_network.get_sensor_fusion_data()
                neural_aco.update_predictions(fire_positions, sensor_data, maze)
                neural_update_timer = 0

            # Update sensors
            people_positions = [(p.row, p.col) for p in people if p.alive and not p.escaped]
            sensor_network.update(dt, disasters.get_fire_positions(), disasters.smoke, people_positions, maze)

            # Update ML-activated sprinkler system
            fire_positions = disasters.get_fire_positions()
            predictions = lstm_predictor.predict_spread(fire_positions, {}, maze)
            sprinkler_system.update(dt, neural_aco, predictions, disasters.hazards, time_val)

            # Sprinklers suppress fire
            extinguished = sprinkler_system.suppress_fire(disasters.hazards, dt)
            for pos in extinguished:
                if pos in disasters.hazards:
                    del disasters.hazards[pos]

            # Update RL coordinator
            rl_update_timer += dt
            if rl_update_timer > 2.0 and alarm.active:
                wardens = [p for p in people if p.is_warden and p.alive]
                exits_status = {e: False for e in exits}  # Simplified
                rl_coordinator.step(disasters.get_fire_positions(), people, exits_status, wardens)
                rl_update_timer = 0

            # Pheromone evaporation
            neural_aco.evaporate()

            # Update people
            prev_escaped = stats['escaped']
            prev_deaths = stats['deaths']

            for p in people:
                p.update(dt, maze, exits, disasters.hazards, pathfinder,
                         alarm.active, people, disasters.smoke, neural_aco)

            # Update stats
            stats['escaped'] = sum(1 for p in people if p.escaped)
            stats['deaths'] = sum(1 for p in people if not p.alive)
            stats['hurt'] = sum(1 for p in people if p.alive and not p.escaped and p.health < 80)

            if stats['escaped'] > prev_escaped:
                sound_system.play('escape', 0.2)

            # Update soldier mesh (tactical MANET layer)
            soldier_mesh.update(dt, people, disasters.get_fire_positions(), maze)

            # Update rescue system for injured people
            rescue_system.update(dt, maze, exits, people, pathfinder, disasters.hazards, time_val, alarm.active)

        # RENDER
        screen.fill((30, 32, 38))
        shake = disasters.shake_offset

        # Draw map
        for r in range(ROWS):
            for c in range(COLS):
                draw_tile(screen, r, c, maze, disasters.hazards, time_val, shake,
                          alarm.active and alarm.flash_state)

        # Draw smoke
        draw_smoke(screen, disasters.smoke, shake)

        # Draw emergency floor strobing (pheromone-based evacuation guidance)
        draw_emergency_floor_strobing(screen, neural_aco, shake, time_val, alarm.active)

        # Draw pheromones
        if show_pheromones:
            draw_pheromones(screen, neural_aco, shake)

        # Draw neural predictions
        if show_predictions:
            fire_positions = disasters.get_fire_positions()
            predictions = lstm_predictor.predict_spread(fire_positions, {}, maze)
            draw_neural_predictions(screen, predictions, shake, time_val)

        # Draw sensors
        if show_sensors:
            sensor_network.draw(screen, shake, time_val)

        # Draw ML-activated sprinkler system
        sprinkler_system.draw(screen, shake, time_val)

        # Draw particles
        disasters.draw_particles(screen, shake)

        # Draw people
        for p in sorted(people, key=lambda x: x.y):
            p.draw(screen, shake, time_val)

        # Draw rescue system (medics responding to injured)
        rescue_system.draw(screen, shake, time_val)

        # Draw UI
        draw_panel(screen, stats, neural_aco, sensor_network, rl_coordinator, soldier_mesh, time_val, paused, speed)
        rescue_stats = rescue_system.get_stats()
        draw_bottom_bar(screen, stats, alarm.active, alarm.flash_state, time_val, rescue_stats)

        pygame.display.flip()

    pygame.quit()
    sound_system.stop_alarm()

    rescue_summary = rescue_system.get_stats()
    print(f"\n{'=' * 60}")
    print(f"FINAL: Escaped {stats['escaped']}/{stats['total']}, Deaths: {stats['deaths']}")
    if stats['deaths'] == 0:
        print("PERFECT! ZERO DEATHS ACHIEVED!")
    print(f"Rescues: helped {rescue_summary['helped']}, critical saves {rescue_summary['critical_saves']}, deaths prevented {rescue_summary['deaths_prevented']}")
    print(f"Neural Confidence Final: {neural_aco.neural_confidence:.1%}")
    print(f"RL Decisions Made: {rl_coordinator.decisions_made}")
    mesh_stats = soldier_mesh.get_stats()
    print(f"Mesh Nodes: {mesh_stats['nodes']}, Links: {mesh_stats['links']}, Avg Degree: {mesh_stats['avg_degree']:.1f}")
    print(f"{'=' * 60}")

def run_headless_simulation(steps=240, dt=1 / 30.0):
    """
    Run a trimmed-down simulation loop without rendering.
    Designed for serverless environments (e.g., Vercel) where no display/audio exists.
    """
    maze, exits = generate_building()
    lstm_predictor = SimpleLSTMPredictor()
    neural_aco = NeuralACO(lstm_predictor)
    pathfinder = NeuralPathfinder(neural_aco)
    disasters = Disasters()
    alarm = AlarmSystem()
    sensor_network = spawn_sensors(maze, NUM_SENSORS)
    rl_coordinator = RLEvacuationCoordinator()
    people = spawn_people(maze, TOTAL_PEOPLE, NUM_WARDENS)
    soldier_mesh = SoldierMeshNetwork()

    stats = {'escaped': 0, 'deaths': 0, 'hurt': 0, 'total': TOTAL_PEOPLE}

    # Seed an ignition so the loop has meaningful activity.
    disasters.add_fire(ROWS // 2, COLS // 2)

    neural_update_timer = 0.0
    rl_update_timer = 0.0
    steps_run = 0

    for _ in range(int(steps)):
        steps_run += 1
        disasters.update(dt, maze, neural_aco)
        alarm.update(dt)

        if disasters.hazards and not alarm.active:
            alarm.trigger()

        neural_update_timer += dt
        if neural_update_timer > 0.5:
            fire_positions = disasters.get_fire_positions()
            sensor_data = sensor_network.get_sensor_fusion_data()
            neural_aco.update_predictions(fire_positions, sensor_data, maze)
            neural_update_timer = 0.0

        people_positions = [(p.row, p.col) for p in people if p.alive and not p.escaped]
        sensor_network.update(dt, disasters.get_fire_positions(), disasters.smoke, people_positions, maze)

        rl_update_timer += dt
        if rl_update_timer > 2.0 and alarm.active:
            wardens = [p for p in people if p.is_warden and p.alive]
            exits_status = {e: False for e in exits}
            rl_coordinator.step(disasters.get_fire_positions(), people, exits_status, wardens)
            rl_update_timer = 0.0

        neural_aco.evaporate()

        for p in people:
            p.update(dt, maze, exits, disasters.hazards, pathfinder,
                     alarm.active, people, disasters.smoke, neural_aco)

        stats['escaped'] = sum(1 for p in people if p.escaped)
        stats['deaths'] = sum(1 for p in people if not p.alive)

        # Mesh update (even in headless, we track tactical connectivity)
        soldier_mesh.update(dt, people, disasters.get_fire_positions(), maze)

        # Exit early if everyone is resolved
        if stats['escaped'] + stats['deaths'] >= stats['total']:
            break

    sensor_snapshot = sensor_network.get_sensor_fusion_data()
    mesh_stats = soldier_mesh.get_stats()
    return {
        'steps_run': steps_run,
        'escaped': stats['escaped'],
        'deaths': stats['deaths'],
        'alive': stats['total'] - stats['escaped'] - stats['deaths'],
        'fires_active': len(disasters.get_fire_positions()),
        'neural_confidence': neural_aco.neural_confidence,
        'rl_decisions': rl_coordinator.decisions_made,
        'sensor_coverage': sensor_snapshot.get('coverage', 0),
        'avg_temp': sensor_snapshot.get('temperature_avg', 0),
        'mesh_nodes': mesh_stats['nodes'],
        'mesh_links': mesh_stats['links'],
        'mesh_avg_degree': mesh_stats['avg_degree'],
        'mesh_broadcast_steps': mesh_stats['broadcasts'],
    }

if __name__ == "__main__":
    main()
