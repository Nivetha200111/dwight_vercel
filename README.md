# Neural ACO Emergency Response Simulator (DWIGHT)

A single-file Pygame simulation (`dwight.py`) of a smart-building evacuation. It combines neural-augmented ant-colony optimization, IoT sensor fusion, LSTM fire prediction, a Q-learning “RL coordinator”, medics, sprinklers, alarms, smoke/fire dynamics, and a 3D-ish renderer.

## Quickstart
1) Install deps: `pip install -r requirements.txt` (needs `pygame`, `numpy`, etc.).  
2) Run: `python dwight.py`  
3) Controls: Click=fire, `A`=alarm, `P`=predictions, `T`=pheromones, `S`=sensors, `R`=reset, `+/-`=speed, Space=pause.  
Headless: `HEADLESS=1 python - <<'PY'\nimport dwight; dwight.run_headless_simulation()\nPY`

## High-level architecture (dwight.py)
- **Config/Constants**: grid sizes, tile types (`FLOOR`, `WALL`, `EXIT`, `DOOR`, `CARPET`, `CORRIDOR`), state strings, and color palette.
- **SoundSystem**: procedural beeps/chimes/sirens. Safe fallback to a silent system if the mixer fails (logged as `[audio] mixer init failed`).
- **IoTSensor / IoTSensorNetwork**: temperature/smoke/CO/motion sensors with Kalman filtering, packet loss simulation, mesh drawing, and fused stats (avg/max temp, smoke, CO, motion, coverage).
- **SimpleLSTMPredictor**: a NumPy LSTM-like predictor that estimates fire spread; exposes a confidence score.
- **NeuralACO**: safe/danger pheromone grids with evaporation/deposit rates modulated by LSTM confidence; also tracks predicted danger.
- **NeuralPathfinder**: A* variant that mixes heuristic cost with pheromone fields and heavy hazard penalties; caching for reuse.
- **Disasters**: manages hazards (fires), smoke spread, fire particles, camera shake, and fire growth/decay.
- **Person**: civilians/wardens with awareness, health, panic, evacuation pathing, regeneration when safe, and an emergency sidestep when trapped.
- **Rescuer / RescueSystem**: medics spawn from exits, sprint to injured, heal fast, track critical saves and deaths prevented, and render as red/white medics.
- **AlarmSystem**: toggles alarm state, flashing indicator, and looping siren.
- **SprinklerSystem**: grid of sprinklers that suppress nearby fires; simple visual.
- **RLEvacuationCoordinator**: Q-learning stub with 8 actions (deploy wardens to quadrants / open exits); keeps `decisions`, `avg_reward`, `epsilon`.
- **SoldierMeshNetwork**: lightweight MANET-like broadcast tracker (used for UI stats only).
- **Rendering/UI**:
  - `draw_tile`: terrain, walls, exits, and luminescent green doors (including interior room doors).
  - `draw_panel`: side panel with escapes/deaths, neural stats, sensors, mesh stats, RL stats, ACO stats, controls.
  - `draw_bottom_bar`: zero-deaths indicator, medic stats, alarm status, and “NEEDS HELP” count.
- **Building generation**: corridors + rooms with multiple corridor doors and interior green doors for flow; many exits on edges and interior junctions.
- **Main loop**: updates disasters, sensors, neural predictions, RL, pheromones, people, rescuers; draws scene/UI each frame.
- **Headless loop**: `run_headless_simulation()` trims rendering/audio for serverless.

## Algorithm highlights
- **Neural-augmented ACO**: Pheromone deposit/evaporation scales with LSTM confidence; predicted danger raises path costs; safe pheromones reduce them.
- **Kalman-filtered sensors**: Each sensor filters noisy readings; packet loss can drop alerts; fusion layer aggregates coverage/health and env. metrics.
- **LSTM fire prediction**: Computes spread direction/center features, keeps hidden/cell state, outputs per-direction spread probabilities and confidence.
- **RL coordinator (Q-learning stub)**: State = (fire quadrant, crowd quadrant, exits blocked); Actions = deploy wardens/open exits; Epsilon-greedy; updates Q-table with simple reward returned from actions.
- **Crowd/health model**: People take fire/smoke damage, panic near hazards, regenerate when safe, and can be rescued; rescuers heal at high rate and count critical saves.
- **Pathfinding**: A* with pheromone-modulated costs, hazard avoidance, and cache; emergency sidestep if no path.

## RL overlay: why it may show “No decisions yet”
- RL steps only when **alarm is active** and the RL timer exceeds ~2s (`rl_update_timer > 2.0`).  
- If the alarm never triggers, or you reset immediately, `decisions_made` stays 0.  
- To see increments: add fire or press `A`, wait a few seconds; the overlay will update once `rl_coordinator.step()` runs.

## Troubleshooting
- **No audio**: ensure `pygame` is installed and an audio device exists; otherwise a silent SoundSystem is used (check for `[audio] mixer init failed`).
- **Import errors**: install deps via `pip install -r requirements.txt` (needs `pygame`).
- **Doors visibility**: exits are bright green/white; doors are pulsating green tiles (also inside large rooms near their centers).
- **Performance**: reduce map size or disable predictions/pheromones/sensors (P/T/S) if FPS drops.
