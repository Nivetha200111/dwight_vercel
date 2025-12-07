/**
 * DWIGHT Main Entry Point
 * @module main
 *
 * Neural ACO Emergency Evacuation System - Frontend
 */

import { CONFIG, TILE, STATE } from './config.js';
import { gameState, resetGameState } from './state.js';
import { fetchInitialState, initializeFromResponse, runPythonSimulation } from './api.js';
import { initCanvas, resizeCanvas, render } from './renderer/index.js';
import { inBounds, hasFire } from './utils.js';

// Re-export for global access if needed
export { gameState, CONFIG, TILE, STATE };

/** @type {number|null} Animation frame ID */
let animationFrameId = null;

/** @type {number} Last frame timestamp */
let lastTime = 0;

/**
 * Initialize the game
 */
async function initGame() {
    console.log('Initializing DWIGHT UX...');

    // Initialize canvas
    initCanvas();

    // Show loading state
    updateLoadingState(true, 'Loading simulation...');

    try {
        // Fetch initial state from backend
        const data = await fetchInitialState();
        initializeFromResponse(data);

        // Set start time
        gameState.startTime = Date.now();

        // Hide loading
        updateLoadingState(false);

        // Setup event listeners
        setupEventListeners();

        // Update UI
        updateUI();

        // Start game loop
        startGameLoop();

        console.log('DWIGHT UX initialized successfully!');
    } catch (error) {
        console.error('Failed to initialize:', error);
        updateLoadingState(false, null, error.message);
    }
}

/**
 * Update loading state in UI
 * @param {boolean} loading - Whether loading
 * @param {string|null} [message] - Loading message
 * @param {string|null} [error] - Error message
 */
function updateLoadingState(loading, message = null, error = null) {
    const loadingEl = document.getElementById('loading-overlay');
    const errorEl = document.getElementById('error-message');

    if (loadingEl) {
        loadingEl.style.display = loading ? 'flex' : 'none';
        if (message) {
            const msgEl = loadingEl.querySelector('.loading-text');
            if (msgEl) msgEl.textContent = message;
        }
    }

    if (errorEl && error) {
        errorEl.textContent = error;
        errorEl.style.display = 'block';
    }
}

/**
 * Main game loop
 * @param {number} timestamp - Current timestamp
 */
function gameLoop(timestamp) {
    if (!lastTime) lastTime = timestamp;
    const dt = Math.min((timestamp - lastTime) / 1000, 0.1) * gameState.speed;
    lastTime = timestamp;

    if (!gameState.paused) {
        gameState.time += dt;
        update(dt);
    }

    render();
    updateUI();

    animationFrameId = requestAnimationFrame(gameLoop);
}

/**
 * Start the game loop
 */
function startGameLoop() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    lastTime = 0;
    animationFrameId = requestAnimationFrame(gameLoop);
}

/**
 * Stop the game loop
 */
function stopGameLoop() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

/**
 * Update game state
 * @param {number} dt - Delta time in seconds
 */
function update(dt) {
    updateShake(dt);
    updateFires(dt);
    updateSmoke(dt);
    updatePeople(dt);
    updateBombs(dt);
    updateEarthquake(dt);
    updateSensors(dt);
    updateStats();
    generatePredictions();

    // Check win/lose conditions
    checkEndConditions();
}

/**
 * Update screen shake
 * @param {number} dt - Delta time
 */
function updateShake(dt) {
    if (gameState.shake.intensity > 0) {
        gameState.shake.intensity *= 0.9;
        gameState.shake.x = (Math.random() - 0.5) * gameState.shake.intensity * 10;
        gameState.shake.y = (Math.random() - 0.5) * gameState.shake.intensity * 10;

        if (gameState.shake.intensity < 0.01) {
            gameState.shake = { x: 0, y: 0, intensity: 0 };
        }
    }
}

/**
 * Update fire spread and decay
 * @param {number} dt - Delta time
 */
function updateFires(dt) {
    gameState.fires.forEach(fire => {
        fire.age = (fire.age || 0) + dt;

        // Fire spread
        if (fire.age > 5 && Math.random() < 0.005 * dt * 30) {
            const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
            const dir = dirs[Math.floor(Math.random() * dirs.length)];
            const nr = fire.row + dir[0];
            const nc = fire.col + dir[1];

            if (inBounds(nr, nc) &&
                gameState.maze[nr][nc] !== TILE.WALL &&
                gameState.maze[nr][nc] !== TILE.EXIT &&
                !hasFire(nr, nc)) {
                gameState.fires.push({
                    row: nr,
                    col: nc,
                    intensity: 0.8,
                    age: 0
                });
            }
        }
    });

    // Trigger alarm on fire
    if (gameState.fires.length > 0 && !gameState.stats.alarmActive) {
        gameState.stats.alarmActive = true;
    }
}

/**
 * Update smoke spread and decay
 * @param {number} dt - Delta time
 */
function updateSmoke(dt) {
    // Generate smoke from fires
    gameState.fires.forEach(fire => {
        for (let dr = -3; dr <= 3; dr++) {
            for (let dc = -3; dc <= 3; dc++) {
                const key = `${fire.row + dr},${fire.col + dc}`;
                const dist = Math.abs(dr) + Math.abs(dc);
                const add = 0.03 / (dist + 1);
                gameState.smoke[key] = Math.min((gameState.smoke[key] || 0) + add, 1.5);
            }
        }
    });

    // Decay smoke
    Object.keys(gameState.smoke).forEach(key => {
        gameState.smoke[key] *= 0.97;
        if (gameState.smoke[key] < 0.02) {
            delete gameState.smoke[key];
        }
    });
}

/**
 * Update people movement and state
 * @param {number} dt - Delta time
 */
function updatePeople(dt) {
    gameState.people.forEach(person => {
        if (!person.alive || person.escaped) return;

        // Damage from fire
        if (hasFire(person.row, person.col)) {
            person.health -= 30 * dt;
            person.state = STATE.PANICKING;
        }

        // Smoke damage
        const smoke = gameState.smoke[`${person.row},${person.col}`] || 0;
        if (smoke > 0.5) {
            person.health -= smoke * 10 * dt;
        }

        // Check death
        if (person.health <= 0) {
            person.alive = false;
            return;
        }

        // Check escape
        if (gameState.maze[person.row]?.[person.col] === TILE.EXIT) {
            person.escaped = true;
            return;
        }

        // Update awareness
        if (gameState.stats.alarmActive && person.state === STATE.WORKING) {
            person.state = STATE.AWARE;
        }
        if (person.state === STATE.AWARE) {
            person.state = STATE.EVACUATING;
        }

        // Simple movement towards nearest exit
        if (person.state === STATE.EVACUATING || person.state === STATE.PANICKING || person.isWarden) {
            moveTowardsExit(person, dt);
        }
    });
}

/**
 * Move person towards nearest exit
 * @param {Object} person - Person to move
 * @param {number} dt - Delta time
 */
function moveTowardsExit(person, dt) {
    if (gameState.exits.length === 0) return;

    // Find nearest exit
    let nearestExit = null;
    let nearestDist = Infinity;

    gameState.exits.forEach(([er, ec]) => {
        const dist = Math.abs(person.row - er) + Math.abs(person.col - ec);
        if (dist < nearestDist) {
            nearestDist = dist;
            nearestExit = [er, ec];
        }
    });

    if (!nearestExit) return;

    // Simple movement (random step towards exit)
    if (Math.random() < 0.3 * dt * 30) {
        const [er, ec] = nearestExit;
        const dr = Math.sign(er - person.row);
        const dc = Math.sign(ec - person.col);

        // Try to move
        const moves = [];
        if (dr !== 0) moves.push([dr, 0]);
        if (dc !== 0) moves.push([0, dc]);
        if (moves.length === 0) return;

        // Shuffle and try
        moves.sort(() => Math.random() - 0.5);

        for (const [mr, mc] of moves) {
            const nr = person.row + mr;
            const nc = person.col + mc;

            if (inBounds(nr, nc) &&
                gameState.maze[nr][nc] !== TILE.WALL &&
                !hasFire(nr, nc)) {
                person.row = nr;
                person.col = nc;
                break;
            }
        }
    }
}

/**
 * Update bombs
 * @param {number} dt - Delta time
 */
function updateBombs(dt) {
    gameState.bombs.forEach((bomb, index) => {
        bomb.timer = (bomb.timer || 3) - dt;

        if (bomb.timer <= 0) {
            // Explode
            gameState.shake.intensity = 2;

            // Create fires
            for (let dr = -2; dr <= 2; dr++) {
                for (let dc = -2; dc <= 2; dc++) {
                    const nr = bomb.row + dr;
                    const nc = bomb.col + dc;
                    if (inBounds(nr, nc) && !hasFire(nr, nc)) {
                        if (Math.random() < 0.5) {
                            gameState.fires.push({ row: nr, col: nc, intensity: 1, age: 0 });
                        }
                    }
                }
            }

            // Remove bomb
            gameState.bombs.splice(index, 1);
        }
    });
}

/**
 * Update earthquake
 * @param {number} dt - Delta time
 */
function updateEarthquake(dt) {
    if (gameState.earthquakeActive) {
        gameState.earthquakeTimer -= dt;
        gameState.shake.intensity = Math.max(gameState.shake.intensity, 0.5);

        if (gameState.earthquakeTimer <= 0) {
            gameState.earthquakeActive = false;
            gameState.earthquakeEpicenter = null;
        }
    }
}

/**
 * Update sensor readings
 * @param {number} dt - Delta time
 */
function updateSensors(dt) {
    gameState.sensors.forEach(sensor => {
        if (sensor.health <= 0) return;

        // Check for fire proximity
        const nearFire = gameState.fires.some(f =>
            Math.abs(f.row - sensor.row) + Math.abs(f.col - sensor.col) < 5
        );

        if (sensor.type === 'temperature' || sensor.type === 'smoke') {
            sensor.triggered = nearFire;
            if (nearFire) {
                sensor.value = sensor.type === 'temperature' ? 80 : 0.8;
            }
        }
    });
}

/**
 * Update game stats
 */
function updateStats() {
    gameState.stats.escaped = gameState.people.filter(p => p.escaped).length;
    gameState.stats.deaths = gameState.people.filter(p => !p.alive).length;
}

/**
 * Generate fire spread predictions
 */
function generatePredictions() {
    gameState.predictions = [];

    const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];

    gameState.fires.forEach(fire => {
        dirs.forEach(([dr, dc]) => {
            for (let step = 1; step <= 4; step++) {
                const nr = fire.row + dr * step;
                const nc = fire.col + dc * step;

                if (inBounds(nr, nc) && gameState.maze[nr][nc] !== TILE.WALL) {
                    gameState.predictions.push({
                        row: nr,
                        col: nc,
                        prob: 0.8 / step
                    });
                }
            }
        });
    });
}

/**
 * Check end conditions
 */
function checkEndConditions() {
    const alive = gameState.people.filter(p => p.alive && !p.escaped).length;
    const escaped = gameState.stats.escaped;
    const deaths = gameState.stats.deaths;
    const total = gameState.stats.total;

    if (alive === 0) {
        // Game over
        if (escaped === total) {
            showVictory();
        } else if (deaths === total) {
            showDefeat();
        } else {
            showVictory(); // Some escaped
        }
    }
}

/**
 * Show victory screen
 */
function showVictory() {
    const el = document.getElementById('victory-screen');
    if (el) {
        el.style.display = 'flex';
        el.querySelector('.escaped-count').textContent = gameState.stats.escaped;
        el.querySelector('.deaths-count').textContent = gameState.stats.deaths;
    }
    gameState.paused = true;
}

/**
 * Show defeat screen
 */
function showDefeat() {
    const el = document.getElementById('defeat-screen');
    if (el) {
        el.style.display = 'flex';
    }
    gameState.paused = true;
}

/**
 * Update UI elements
 */
function updateUI() {
    // Stats
    const escapeCount = document.getElementById('escape-count');
    const deathCount = document.getElementById('death-count');
    const aliveCount = document.getElementById('alive-count');

    if (escapeCount) escapeCount.textContent = gameState.stats.escaped;
    if (deathCount) deathCount.textContent = gameState.stats.deaths;
    if (aliveCount) {
        const alive = gameState.stats.total - gameState.stats.escaped - gameState.stats.deaths;
        aliveCount.textContent = alive;
    }

    // Alarm indicator
    const alarmIndicator = document.getElementById('alarm-indicator');
    if (alarmIndicator) {
        alarmIndicator.classList.toggle('active', gameState.stats.alarmActive);
    }

    // Speed indicator
    const speedIndicator = document.getElementById('speed-indicator');
    if (speedIndicator) {
        speedIndicator.textContent = `${gameState.speed.toFixed(1)}x`;
    }

    // Pause indicator
    const pauseOverlay = document.getElementById('pause-overlay');
    if (pauseOverlay) {
        pauseOverlay.style.display = gameState.paused ? 'flex' : 'none';
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    const canvas = document.getElementById('game-canvas');

    // Canvas click
    if (canvas) {
        canvas.addEventListener('click', handleCanvasClick);
    }

    // Keyboard
    document.addEventListener('keydown', handleKeyDown);

    // Hotbar buttons
    document.querySelectorAll('.hotbar-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tool = btn.dataset.tool;
            if (tool) {
                gameState.selectedTool = tool;
                updateHotbarSelection();
            }
        });
    });

    // Reset button
    const resetBtn = document.getElementById('reset-btn');
    if (resetBtn) {
        resetBtn.addEventListener('click', resetGame);
    }

    // Python simulation button
    const pySimBtn = document.getElementById('py-sim-btn');
    if (pySimBtn) {
        pySimBtn.addEventListener('click', async () => {
            try {
                await runPythonSimulation();
                updatePythonTelemetry();
            } catch (e) {
                console.error('Python sim failed:', e);
            }
        });
    }
}

/**
 * Handle canvas click
 * @param {MouseEvent} event - Click event
 */
function handleCanvasClick(event) {
    const canvas = document.getElementById('game-canvas');
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Calculate tile size
    const ts = canvas.width / CONFIG.COLS;
    const col = Math.floor(x / ts);
    const row = Math.floor(y / ts);

    if (!inBounds(row, col)) return;

    // Apply selected tool
    switch (gameState.selectedTool) {
        case 'fire':
            if (!hasFire(row, col)) {
                gameState.fires.push({ row, col, intensity: 1, age: 0 });
            }
            break;
        case 'bomb':
            gameState.bombs.push({ row, col, timer: 3 });
            break;
        case 'flood':
            gameState.floods.push({ row, col, depth: 0.6 });
            break;
        case 'earthquake':
            gameState.earthquakeActive = true;
            gameState.earthquakeTimer = 3;
            gameState.earthquakeEpicenter = { row, col };
            break;
        case 'sensor':
            gameState.sensors.push({
                id: gameState.sensors.length,
                row, col,
                type: 'temperature',
                value: 22,
                triggered: false,
                health: 100
            });
            break;
        case 'inspect':
            const person = gameState.people.find(p =>
                p.row === row && p.col === col && p.alive && !p.escaped
            );
            if (person) {
                gameState.selectedPersonId = person.id;
            }
            break;
    }
}

/**
 * Handle keyboard input
 * @param {KeyboardEvent} event - Key event
 */
function handleKeyDown(event) {
    switch (event.key.toLowerCase()) {
        case ' ':
            event.preventDefault();
            gameState.paused = !gameState.paused;
            break;
        case 'a':
            gameState.stats.alarmActive = true;
            break;
        case 'p':
            gameState.showPredictions = !gameState.showPredictions;
            break;
        case 't':
            gameState.showPheromones = !gameState.showPheromones;
            break;
        case 's':
            gameState.showSensors = !gameState.showSensors;
            break;
        case 'm':
            gameState.showMesh = !gameState.showMesh;
            break;
        case 'r':
            resetGame();
            break;
        case '+':
        case '=':
            gameState.speed = Math.min(gameState.speed + 0.5, 5);
            break;
        case '-':
            gameState.speed = Math.max(gameState.speed - 0.5, 0.5);
            break;
        case '1': gameState.selectedTool = 'fire'; updateHotbarSelection(); break;
        case '2': gameState.selectedTool = 'bomb'; updateHotbarSelection(); break;
        case '3': gameState.selectedTool = 'earthquake'; updateHotbarSelection(); break;
        case '4': gameState.selectedTool = 'flood'; updateHotbarSelection(); break;
        case '5': gameState.selectedTool = 'sensor'; updateHotbarSelection(); break;
        case '6': gameState.selectedTool = 'inspect'; updateHotbarSelection(); break;
    }
}

/**
 * Update hotbar selection UI
 */
function updateHotbarSelection() {
    document.querySelectorAll('.hotbar-btn').forEach(btn => {
        btn.classList.toggle('selected', btn.dataset.tool === gameState.selectedTool);
    });
}

/**
 * Update Python telemetry UI
 */
function updatePythonTelemetry() {
    const container = document.getElementById('py-telemetry');
    if (!container) return;

    const { stats, loading, error, lastRun } = gameState.pythonTelemetry;

    if (loading) {
        container.innerHTML = '<div class="loading">Running simulation...</div>';
    } else if (error) {
        container.innerHTML = `<div class="error">${error}</div>`;
    } else if (stats) {
        container.innerHTML = `
            <div class="stat">Escaped: ${stats.escaped}</div>
            <div class="stat">Deaths: ${stats.deaths}</div>
            <div class="stat">Neural Conf: ${(stats.neural_confidence * 100).toFixed(1)}%</div>
            <div class="stat">RL Decisions: ${stats.rl_decisions}</div>
            <div class="timestamp">Last: ${lastRun?.toLocaleTimeString()}</div>
        `;
    }
}

/**
 * Reset the game
 */
async function resetGame() {
    stopGameLoop();

    // Hide end screens
    const victory = document.getElementById('victory-screen');
    const defeat = document.getElementById('defeat-screen');
    if (victory) victory.style.display = 'none';
    if (defeat) defeat.style.display = 'none';

    // Reset state
    resetGameState();

    // Reinitialize
    await initGame();
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initGame);
} else {
    initGame();
}
