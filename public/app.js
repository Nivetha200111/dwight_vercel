/**
 * ═══════════════════════════════════════════════════════════════════════════
 * DWIGHT UX - Minecraft-Themed Neural ACO Emergency Evacuation System
 * Frontend JavaScript - Game Logic, Rendering, and Simulation
 * ═══════════════════════════════════════════════════════════════════════════
 */

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

const CONFIG = {
    ROWS: 45,
    COLS: 70,
    TILE_SIZE: 12,
    TOTAL_PEOPLE: 60,
    NUM_WARDENS: 4,
    NUM_SENSORS: 25,
    API_URL: '/api/simulate',
    PY_SIM_API_URL: '/api/headless'
};

// Tile types
const TILE = {
    FLOOR: 0,
    WALL: 1,
    EXIT: 2,
    DOOR: 3,
    CORRIDOR: 4,
    CARPET: 5
};

// Person states
const STATE = {
    WORKING: 'working',
    HEADPHONES: 'headphones',
    AWARE: 'aware',
    EVACUATING: 'evacuating',
    PANICKING: 'panicking',
    WARDEN: 'warden'
};

// Minecraft color palette
const COLORS = {
    // Tiles
    FLOOR: '#b4afa5',
    FLOOR_ALT: '#aaa69c',
    WALL: '#2d3037',
    WALL_HIGHLIGHT: '#414550',
    CORRIDOR: '#9b9691',
    CARPET: '#614141',
    EXIT: '#17dd62',
    DOOR: '#785028',

    // Hazards
    FIRE_CORE: '#ffffc8',
    FIRE_MID: '#ffdc50',
    FIRE_OUTER: '#ff6414',
    SMOKE: '#464650',
    BOMB: '#ffcf5d',
    BOMB_GLOW: '#ff8c00',
    WATER: '#2b84ff',
    WATER_DARK: '#1a4fa3',
    EARTHQUAKE: '#cfd5dd',

    // Pheromones
    SAFE_PHEROMONE: 'rgba(0, 255, 136, 0.3)',
    DANGER_PHEROMONE: 'rgba(255, 68, 68, 0.4)',
    PREDICTION: 'rgba(192, 101, 255, 0.4)',

    // People
    NORMAL: '#6496ff',
    AWARE: '#ffff64',
    EVACUATING: '#64ff64',
    PANICKING: '#ff5050',
    WARDEN: '#fcee4b',
    HEADPHONES: '#ff64ff',

    // Sensors
    TEMP_SENSOR: '#ff6600',
    SMOKE_SENSOR: '#999999',
    CO_SENSOR: '#ff00ff',
    MOTION_SENSOR: '#00ccff'
};

// ═══════════════════════════════════════════════════════════════════════════
// GAME STATE
// ═══════════════════════════════════════════════════════════════════════════

const gameState = {
    maze: [],
    exits: [],
    people: [],
    sensors: [],
    fires: [],
    bombs: [],
    floods: [],
    earthquakeActive: false,
    earthquakeTimer: 0,
    earthquakeEpicenter: null,
    smoke: {},
    predictions: [],

    // Stats
    stats: {
        escaped: 0,
        deaths: 0,
        total: CONFIG.TOTAL_PEOPLE,
        alarmActive: false
    },

    // Neural/RL data
    neural: {
        confidence: 0,
        predictions: [],
        safePheromone: 0.1,
        dangerPheromone: 0
    },
    rl: {
        decisions: 0,
        avgReward: 0,
        epsilon: 0.2
    },

    // Simulation state
    paused: false,
    speed: 1.0,
    time: 0,
    startTime: null,

    // Display toggles
    showPredictions: true,
    showPheromones: true,
    showSensors: true,

    // Selected tool
    selectedTool: 'fire',

    // Screen shake
    shake: { x: 0, y: 0, intensity: 0 },

    // Python headless telemetry
    pythonTelemetry: {
        loading: false,
        error: '',
        stats: null,
        lastRun: null
    }
};

// Utility helpers
function inBounds(row, col) {
    return row > 0 && row < CONFIG.ROWS - 1 && col > 0 && col < CONFIG.COLS - 1;
}

function isFlooded(row, col) {
    return gameState.floods.some(f => f.row === row && f.col === col);
}

function hasBomb(row, col) {
    return gameState.bombs.some(b => b.row === row && b.col === col);
}

// ═══════════════════════════════════════════════════════════════════════════
// CANVAS & RENDERING
// ═══════════════════════════════════════════════════════════════════════════

let canvas, ctx;
let canvasWidth, canvasHeight;
let tileSize = 10; // Default tile size

function initCanvas() {
    canvas = document.getElementById('game-canvas');
    ctx = canvas.getContext('2d');

    // Disable image smoothing for crisp pixels
    ctx.imageSmoothingEnabled = false;

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
}

function resizeCanvas() {
    const container = document.getElementById('canvas-container');
    if (!container) return;

    const rect = container.getBoundingClientRect();

    // Ensure we have valid dimensions
    const containerWidth = rect.width || 800;
    const containerHeight = rect.height || 600;

    // Calculate tile size to fit the maze
    const maxTileWidth = Math.floor(containerWidth / CONFIG.COLS);
    const maxTileHeight = Math.floor(containerHeight / CONFIG.ROWS);
    tileSize = Math.max(4, Math.min(maxTileWidth, maxTileHeight, 14));

    canvasWidth = CONFIG.COLS * tileSize;
    canvasHeight = CONFIG.ROWS * tileSize;

    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    // Center canvas in container
    canvas.style.width = canvasWidth + 'px';
    canvas.style.height = canvasHeight + 'px';
    canvas.style.position = 'absolute';
    canvas.style.left = '50%';
    canvas.style.top = '50%';
    canvas.style.transform = 'translate(-50%, -50%)';

    // Re-disable smoothing after resize
    if (ctx) {
        ctx.imageSmoothingEnabled = false;
    }

    console.log('Canvas resized:', canvasWidth, 'x', canvasHeight, 'tileSize:', tileSize);
}

// ═══════════════════════════════════════════════════════════════════════════
// DRAWING FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

// Simple tile drawing that doesn't depend on gameState.shake
function drawTileSimple(row, col, tile, shakeX, shakeY) {
    const x = col * tileSize + (shakeX || 0);
    const y = row * tileSize + (shakeY || 0);

    switch (tile) {
        case TILE.FLOOR:
            ctx.fillStyle = (row + col) % 2 === 0 ? COLORS.FLOOR : COLORS.FLOOR_ALT;
            ctx.fillRect(x, y, tileSize, tileSize);
            break;

        case TILE.WALL:
            ctx.fillStyle = COLORS.WALL;
            ctx.fillRect(x, y, tileSize, tileSize);
            ctx.fillStyle = COLORS.WALL_HIGHLIGHT;
            ctx.fillRect(x, y, tileSize, 2);
            ctx.fillRect(x, y, 2, tileSize);
            break;

        case TILE.CORRIDOR:
            ctx.fillStyle = COLORS.CORRIDOR;
            ctx.fillRect(x, y, tileSize, tileSize);
            break;

        case TILE.CARPET:
            ctx.fillStyle = COLORS.CARPET;
            ctx.fillRect(x, y, tileSize, tileSize);
            break;

        case TILE.EXIT:
            const glow = Math.sin(gameState.time * 4) * 0.3 + 0.7;
            ctx.fillStyle = COLORS.EXIT;
            ctx.fillRect(x, y, tileSize, tileSize);
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1;
            ctx.strokeRect(x + 1, y + 1, tileSize - 2, tileSize - 2);
            break;

        case TILE.DOOR:
            ctx.fillStyle = COLORS.DOOR;
            ctx.fillRect(x, y, tileSize, tileSize);
            break;

        default:
            ctx.fillStyle = COLORS.FLOOR;
            ctx.fillRect(x, y, tileSize, tileSize);
    }
}

function drawTile(row, col, tile) {
    const x = col * tileSize + (gameState.shake?.x || 0);
    const y = row * tileSize + (gameState.shake?.y || 0);

    ctx.save();

    switch (tile) {
        case TILE.FLOOR:
            ctx.fillStyle = (row + col) % 2 === 0 ? COLORS.FLOOR : COLORS.FLOOR_ALT;
            ctx.fillRect(x, y, tileSize, tileSize);
            break;

        case TILE.WALL:
            // 3D wall effect
            ctx.fillStyle = COLORS.WALL;
            ctx.fillRect(x, y, tileSize, tileSize);

            // Highlight
            ctx.fillStyle = COLORS.WALL_HIGHLIGHT;
            ctx.fillRect(x, y, tileSize, 2);
            ctx.fillRect(x, y, 2, tileSize);

            // Add stone texture
            ctx.fillStyle = 'rgba(0,0,0,0.1)';
            for (let i = 0; i < 3; i++) {
                const tx = x + Math.random() * tileSize;
                const ty = y + Math.random() * tileSize;
                ctx.fillRect(tx, ty, 2, 2);
            }
            break;

        case TILE.CORRIDOR:
            ctx.fillStyle = COLORS.CORRIDOR;
            ctx.fillRect(x, y, tileSize, tileSize);

            // Subtle pattern
            ctx.fillStyle = 'rgba(0,0,0,0.05)';
            if ((row + col) % 3 === 0) {
                ctx.fillRect(x + 2, y + 2, tileSize - 4, tileSize - 4);
            }
            break;

        case TILE.CARPET:
            ctx.fillStyle = COLORS.CARPET;
            ctx.fillRect(x, y, tileSize, tileSize);

            // Carpet texture
            ctx.fillStyle = 'rgba(0,0,0,0.1)';
            for (let i = 0; i < tileSize; i += 3) {
                ctx.fillRect(x + i, y, 1, tileSize);
            }
            break;

        case TILE.EXIT:
            // Glowing exit
            const glow = Math.sin(gameState.time * 4) * 0.3 + 0.7;
            ctx.fillStyle = COLORS.EXIT;
            ctx.fillRect(x, y, tileSize, tileSize);

            // Glow effect
            ctx.shadowColor = COLORS.EXIT;
            ctx.shadowBlur = 10 * glow;
            ctx.fillRect(x + 2, y + 2, tileSize - 4, tileSize - 4);
            ctx.shadowBlur = 0;

            // Arrow
            ctx.fillStyle = 'white';
            ctx.beginPath();
            ctx.moveTo(x + tileSize / 2, y + 2);
            ctx.lineTo(x + tileSize - 2, y + tileSize / 2);
            ctx.lineTo(x + tileSize / 2, y + tileSize - 2);
            ctx.closePath();
            ctx.fill();
            break;

        case TILE.DOOR:
            ctx.fillStyle = COLORS.DOOR;
            ctx.fillRect(x, y, tileSize, tileSize);

            // Door frame
            ctx.fillStyle = 'rgba(0,0,0,0.3)';
            ctx.fillRect(x, y, 2, tileSize);
            ctx.fillRect(x + tileSize - 2, y, 2, tileSize);
            break;
    }

    ctx.restore();
}

function drawFire(row, col, intensity = 1) {
    const x = col * tileSize + (gameState.shake?.x || 0);
    const y = row * tileSize + (gameState.shake?.y || 0);
    const time = gameState.time;

    ctx.save();

    // Base
    ctx.fillStyle = '#3d1a0a';
    ctx.fillRect(x, y, tileSize, tileSize);

    // Animated flames
    for (let i = 0; i < 3; i++) {
        const fx = x + 2 + i * (tileSize / 3);
        const fh = tileSize * 0.6 + Math.sin(time * 12 + i + col * 0.5) * (tileSize * 0.3);

        const gradient = ctx.createLinearGradient(fx, y + tileSize, fx, y + tileSize - fh);
        gradient.addColorStop(0, COLORS.FIRE_OUTER);
        gradient.addColorStop(0.4, COLORS.FIRE_MID);
        gradient.addColorStop(1, COLORS.FIRE_CORE);

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.moveTo(fx, y + tileSize);
        ctx.lineTo(fx + tileSize / 4, y + tileSize);
        ctx.lineTo(fx + tileSize / 8, y + tileSize - fh);
        ctx.closePath();
        ctx.fill();
    }

    // Glow
    ctx.shadowColor = COLORS.FIRE_OUTER;
    ctx.shadowBlur = 15 * intensity;
    ctx.fillStyle = 'transparent';
    ctx.fillRect(x, y, tileSize, tileSize);

    ctx.restore();
}

function drawSmoke(row, col, level) {
    if (level < 0.1) return;

    const x = col * tileSize + (gameState.shake?.x || 0);
    const y = row * tileSize + (gameState.shake?.y || 0);

    ctx.save();
    ctx.fillStyle = `rgba(70, 70, 80, ${Math.min(level * 0.6, 0.8)})`;
    ctx.fillRect(x, y, tileSize, tileSize);
    ctx.restore();
}

function drawFlood(flood) {
    const x = flood.col * tileSize + (gameState.shake?.x || 0);
    const y = flood.row * tileSize + (gameState.shake?.y || 0);
    const depth = Math.min(1.2, flood.depth || 0.6);
    const wave = Math.sin(gameState.time * 5 + flood.col + flood.row) * 0.1;

    ctx.save();
    ctx.fillStyle = `rgba(43, 132, 255, ${0.35 + depth * 0.25})`;
    ctx.fillRect(x, y + tileSize * (0.15 - wave), tileSize, tileSize);
    ctx.fillStyle = `rgba(255, 255, 255, ${0.08 + depth * 0.05})`;
    ctx.fillRect(x, y + tileSize * 0.6, tileSize, tileSize * 0.4);
    ctx.strokeStyle = COLORS.WATER_DARK;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, tileSize, tileSize);
    ctx.restore();
}

function drawBomb(bomb) {
    const x = bomb.col * tileSize + (gameState.shake?.x || 0);
    const y = bomb.row * tileSize + (gameState.shake?.y || 0);
    const pulse = Math.sin(gameState.time * 8 + (bomb.timer || 0)) * 0.5 + 0.5;

    ctx.save();
    ctx.fillStyle = '#1c1c1c';
    ctx.fillRect(x + 3, y + 3, tileSize - 6, tileSize - 6);
    ctx.fillStyle = `rgba(255, 207, 93, ${0.3 + 0.5 * pulse})`;
    ctx.fillRect(x + 4, y + 4, tileSize - 8, tileSize - 8);
    ctx.strokeStyle = COLORS.BOMB_GLOW;
    ctx.lineWidth = 2;
    ctx.strokeRect(x + 2, y + 2, tileSize - 4, tileSize - 4);

    // Fuse
    const fuseHeight = Math.max(4, tileSize * 0.6 * Math.max(0.1, (bomb.timer || 0) / 3));
    ctx.fillStyle = COLORS.BOMB_GLOW;
    ctx.fillRect(x + tileSize / 2 - 1, y - fuseHeight * 0.4, 2, fuseHeight * 0.6);
    ctx.beginPath();
    ctx.arc(x + tileSize / 2, y - fuseHeight * 0.4, 3, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
}

function drawPerson(person) {
    if (!person.alive || person.escaped) return;

    const x = person.col * tileSize + tileSize / 2 + (gameState.shake?.x || 0);
    const y = person.row * tileSize + tileSize / 2 + (gameState.shake?.y || 0);
    const size = tileSize * 0.6;

    ctx.save();

    // Shadow
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.beginPath();
    ctx.ellipse(x, y + size / 2 + 2, size / 2, size / 4, 0, 0, Math.PI * 2);
    ctx.fill();

    // Body color based on state
    let bodyColor = COLORS.NORMAL;
    let outlineColor = '#000';

    switch (person.state) {
        case STATE.WARDEN:
            bodyColor = COLORS.WARDEN;
            outlineColor = '#8b6914';
            break;
        case STATE.PANICKING:
            bodyColor = COLORS.PANICKING;
            break;
        case STATE.EVACUATING:
            bodyColor = COLORS.EVACUATING;
            break;
        case STATE.AWARE:
            bodyColor = COLORS.AWARE;
            break;
        case STATE.HEADPHONES:
            bodyColor = COLORS.HEADPHONES;
            break;
    }

    // Body
    ctx.fillStyle = outlineColor;
    ctx.fillRect(x - size / 2 - 1, y - size / 2 - 1, size + 2, size * 0.6 + 2);
    ctx.fillStyle = bodyColor;
    ctx.fillRect(x - size / 2, y - size / 2, size, size * 0.6);

    // Head
    ctx.fillStyle = '#e6c8a0';
    ctx.fillRect(x - size / 3, y - size / 2 - size * 0.4, size * 0.66, size * 0.4);

    // Warden hat
    if (person.isWarden) {
        ctx.fillStyle = COLORS.WARDEN;
        ctx.fillRect(x - size / 2, y - size / 2 - size * 0.5, size, size * 0.15);
    }

    // Health bar if damaged
    if (person.health < 90) {
        const barWidth = size;
        const barHeight = 3;
        const healthPercent = person.health / 100;

        ctx.fillStyle = '#600';
        ctx.fillRect(x - barWidth / 2, y - size / 2 - size * 0.7, barWidth, barHeight);
        ctx.fillStyle = '#0c0';
        ctx.fillRect(x - barWidth / 2, y - size / 2 - size * 0.7, barWidth * healthPercent, barHeight);
    }

    ctx.restore();
}

function drawSensor(sensor) {
    if (sensor.health <= 0) return;

    const x = sensor.col * tileSize + tileSize / 2 + (gameState.shake?.x || 0);
    const y = sensor.row * tileSize + tileSize / 2 + (gameState.shake?.y || 0);
    const radius = tileSize * 0.25;

    ctx.save();

    let color;
    switch (sensor.type) {
        case 'temperature': color = COLORS.TEMP_SENSOR; break;
        case 'smoke': color = COLORS.SMOKE_SENSOR; break;
        case 'co': color = COLORS.CO_SENSOR; break;
        case 'motion': color = COLORS.MOTION_SENSOR; break;
        default: color = '#888';
    }

    // Sensor body
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    // White outline
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Triggered animation
    if (sensor.triggered) {
        const wave = (gameState.time * 20) % 15 + 5;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 1 - wave / 20;
        ctx.beginPath();
        ctx.arc(x, y, wave, 0, Math.PI * 2);
        ctx.stroke();
    }

    ctx.restore();
}

function drawPheromones() {
    if (!gameState.showPheromones) return;

    ctx.save();

    // Draw safe pheromone trails
    for (let r = 0; r < CONFIG.ROWS; r++) {
        for (let c = 0; c < CONFIG.COLS; c++) {
            const safe = gameState.neural.safePheromone;
            const danger = gameState.neural.dangerPheromone;

            if (safe > 0.3) {
                const x = c * tileSize + (gameState.shake?.x || 0);
                const y = r * tileSize + (gameState.shake?.y || 0);
                ctx.fillStyle = COLORS.SAFE_PHEROMONE;
                ctx.fillRect(x, y, tileSize, tileSize);
            }
        }
    }

    ctx.restore();
}

function drawPredictions() {
    if (!gameState.showPredictions) return;

    ctx.save();

    gameState.predictions.forEach(pred => {
        const x = pred.col * tileSize + (gameState.shake?.x || 0);
        const y = pred.row * tileSize + (gameState.shake?.y || 0);
        const pulse = Math.sin(gameState.time * 4) * 0.3 + 0.7;

        ctx.fillStyle = `rgba(192, 101, 255, ${0.3 * pred.prob * pulse})`;
        ctx.fillRect(x, y, tileSize, tileSize);
    });

    ctx.restore();
}

function drawParticles() {
    ctx.save();

    gameState.fires.forEach(fire => {
        // Emit particles occasionally
        if (Math.random() < 0.3) {
            const x = fire.col * tileSize + tileSize / 2 + (Math.random() - 0.5) * tileSize;
            const y = fire.row * tileSize + (Math.random() * tileSize * 0.5);
            const size = 2 + Math.random() * 3;

            ctx.fillStyle = Math.random() > 0.5 ? COLORS.FIRE_MID : COLORS.FIRE_OUTER;
            ctx.beginPath();
            ctx.arc(x + (gameState.shake?.x || 0), y + (gameState.shake?.y || 0), size, 0, Math.PI * 2);
            ctx.fill();
        }
    });

    ctx.restore();
}

function drawEarthquakeEffect() {
    if (!gameState.earthquakeActive || !gameState.earthquakeEpicenter) return;

    const shakeX = gameState.shake?.x || 0;
    const shakeY = gameState.shake?.y || 0;
    const centerX = gameState.earthquakeEpicenter.col * tileSize + tileSize / 2 + shakeX;
    const centerY = gameState.earthquakeEpicenter.row * tileSize + tileSize / 2 + shakeY;
    const pulse = Math.sin(gameState.time * 6) * 0.5 + 0.5;
    const reach = Math.max(1, gameState.earthquakeTimer || 0);

    ctx.save();
    ctx.strokeStyle = `rgba(207, 213, 221, ${0.3 + 0.3 * pulse})`;
    ctx.lineWidth = 2;

    for (let i = 1; i <= 3; i++) {
        const radius = tileSize * (i * 2 + reach * 0.4 + pulse);
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.stroke();
    }

    ctx.restore();
}

function render() {
    if (!ctx || !canvas) return;

    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // Background
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Ensure shake has default values
    const shakeX = gameState.shake?.x || 0;
    const shakeY = gameState.shake?.y || 0;

    // Draw tiles - always draw something
    for (let r = 0; r < CONFIG.ROWS; r++) {
        for (let c = 0; c < CONFIG.COLS; c++) {
            const tile = (gameState.maze[r] && gameState.maze[r][c] !== undefined)
                ? gameState.maze[r][c]
                : TILE.FLOOR;
            drawTileSimple(r, c, tile, shakeX, shakeY);
        }
    }

    // Draw pheromones
    drawPheromones();

    // Draw predictions
    drawPredictions();

    // Draw floods and bombs
    gameState.floods.forEach(drawFlood);
    gameState.bombs.forEach(drawBomb);

    // Draw smoke
    Object.entries(gameState.smoke).forEach(([key, level]) => {
        const [r, c] = key.split(',').map(Number);
        drawSmoke(r, c, level);
    });

    // Draw fires
    gameState.fires.forEach(fire => {
        drawFire(fire.row, fire.col, fire.intensity || 1);
    });

    // Draw particles
    drawParticles();

    // Draw sensors
    if (gameState.showSensors) {
        gameState.sensors.forEach(drawSensor);
    }

    // Draw people (sorted by Y for proper overlapping)
    [...gameState.people]
        .sort((a, b) => a.row - b.row)
        .forEach(drawPerson);

    // Earthquake rings
    drawEarthquakeEffect();

    // Alarm flash overlay
    if (gameState.stats.alarmActive) {
        const flash = Math.sin(gameState.time * 8) > 0;
        if (flash) {
            ctx.fillStyle = 'rgba(255, 0, 0, 0.05)';
            ctx.fillRect(0, 0, canvasWidth, canvasHeight);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SIMULATION UPDATE
// ═══════════════════════════════════════════════════════════════════════════

function updateSimulation(dt) {
    if (gameState.paused) return;

    gameState.time += dt;

    // Ensure shake object exists
    if (!gameState.shake) {
        gameState.shake = { x: 0, y: 0, intensity: 0 };
    }

    // Update shake
    gameState.shake.intensity *= 0.9;
    if (gameState.shake.intensity > 0.01) {
        gameState.shake.x = (Math.random() - 0.5) * gameState.shake.intensity * 10;
        gameState.shake.y = (Math.random() - 0.5) * gameState.shake.intensity * 10;
    } else {
        gameState.shake.x = 0;
        gameState.shake.y = 0;
    }

    // Update fires
    updateFires(dt);

    // Update other hazards
    updateFloods(dt);
    updateBombs(dt);
    updateEarthquake(dt);

    // Update people
    updatePeople(dt);

    // Update sensors
    updateSensors(dt);

    // Update neural predictions
    updateNeuralPredictions(dt);

    // Update stats
    updateStats();

    // Check victory condition
    checkVictory();
}

function updateFires(dt) {
    // Fire spread logic
    gameState.fires.forEach(fire => {
        fire.age = (fire.age || 0) + dt;

        // Generate smoke
        for (let dr = -4; dr <= 4; dr++) {
            for (let dc = -4; dc <= 4; dc++) {
                const nr = fire.row + dr;
                const nc = fire.col + dc;
                if (nr >= 0 && nr < CONFIG.ROWS && nc >= 0 && nc < CONFIG.COLS) {
                    const dist = Math.abs(dr) + Math.abs(dc);
                    const key = `${nr},${nc}`;
                    gameState.smoke[key] = Math.min(
                        (gameState.smoke[key] || 0) + 0.02 / (dist + 1),
                        1.5
                    );
                }
            }
        }

        // Spread fire
        if (fire.age > 5 && Math.random() < 0.005 * dt) {
            const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
            const [dr, dc] = dirs[Math.floor(Math.random() * 4)];
            const nr = fire.row + dr;
            const nc = fire.col + dc;

            if (nr > 0 && nr < CONFIG.ROWS - 1 && nc > 0 && nc < CONFIG.COLS - 1) {
                const tile = gameState.maze[nr]?.[nc];
                if (tile !== TILE.WALL && tile !== TILE.EXIT) {
                    if (!gameState.fires.some(f => f.row === nr && f.col === nc)) {
                        gameState.fires.push({ row: nr, col: nc, age: 0, intensity: 0.8 });
                        gameState.shake.intensity = Math.min(gameState.shake.intensity + 0.5, 3);
                        addChatMessage('fire', `Fire spreading to (${nr}, ${nc})!`);
                    }
                }
            }
        }
    });

    // Smoke decay
    Object.keys(gameState.smoke).forEach(key => {
        gameState.smoke[key] *= 0.995;
        if (gameState.smoke[key] < 0.02) {
            delete gameState.smoke[key];
        }
    });

    // Trigger alarm if fire exists
    if (gameState.fires.length > 0 && !gameState.stats.alarmActive) {
        triggerAlarm();
    }
}

function updateFloods(dt) {
    const newFloods = [];
    const seen = new Set(gameState.floods.map(f => `${f.row},${f.col}`));
    let extinguished = false;

    gameState.floods.forEach(flood => {
        flood.depth = Math.min(1.5, (flood.depth || 0.6) + 0.25 * dt);

        // Spread gently to neighbors
        if (Math.random() < 0.45 * dt) {
            const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]];
            const [dr, dc] = dirs[Math.floor(Math.random() * dirs.length)];
            const nr = flood.row + dr;
            const nc = flood.col + dc;

            if (inBounds(nr, nc) && gameState.maze[nr]?.[nc] !== TILE.WALL) {
                const key = `${nr},${nc}`;
                if (!seen.has(key)) {
                    seen.add(key);
                    newFloods.push({ row: nr, col: nc, depth: 0.5 });
                }
            }
        }

        // Extinguish fires on water tiles
        const fireIndex = gameState.fires.findIndex(f => f.row === flood.row && f.col === flood.col);
        if (fireIndex !== -1) {
            gameState.fires.splice(fireIndex, 1);
            extinguished = true;
        }

        // Thin smoke on flooded cells
        const floodKey = `${flood.row},${flood.col}`;
        if (gameState.smoke[floodKey]) {
            gameState.smoke[floodKey] = Math.max(0, gameState.smoke[floodKey] - 0.6);
            if (gameState.smoke[floodKey] < 0.05) delete gameState.smoke[floodKey];
        }

        // Damage sensors submerged in water
        gameState.sensors.forEach(sensor => {
            if (sensor.row === flood.row && sensor.col === flood.col) {
                sensor.health -= 8 * dt;
            }
        });
    });

    if (extinguished) {
        addChatMessage('system', 'Flooded tiles doused nearby flames.');
    }

    gameState.floods.push(...newFloods);
}

function updateBombs(dt) {
    const activeBombs = [];

    gameState.bombs.forEach(bomb => {
        bomb.timer = (bomb.timer ?? 3) - dt;
        bomb.radius = bomb.radius || 3;

        if (bomb.timer <= 0) {
            explodeBomb(bomb);
        } else {
            activeBombs.push(bomb);
        }
    });

    gameState.bombs = activeBombs;
}

function explodeBomb(bomb) {
    gameState.shake.intensity = Math.max(gameState.shake.intensity + 3, 4);
    addChatMessage('fire', `Bomb detonated at (${bomb.row}, ${bomb.col})!`);
    triggerAlarm();

    // Damage people
    gameState.people.forEach(person => {
        if (!person.alive || person.escaped) return;

        const dist = Math.abs(person.row - bomb.row) + Math.abs(person.col - bomb.col);
        if (dist <= bomb.radius + 1) {
            const damage = Math.max(10, (bomb.radius - dist + 1) * 35);
            person.health -= damage;
            person.state = STATE.PANICKING;

            if (person.health <= 0) {
                person.alive = false;
                addChatMessage('death', `Person ${person.id} was lost in an explosion!`);
            }
        }
    });

    // Damage sensors
    gameState.sensors.forEach(sensor => {
        const dist = Math.abs(sensor.row - bomb.row) + Math.abs(sensor.col - bomb.col);
        if (dist <= bomb.radius + 1) {
            sensor.health -= 40 / Math.max(1, dist);
            sensor.triggered = true;
        }
    });

    // Scorch area and add smoke
    for (let dr = -bomb.radius; dr <= bomb.radius; dr++) {
        for (let dc = -bomb.radius; dc <= bomb.radius; dc++) {
            const nr = bomb.row + dr;
            const nc = bomb.col + dc;
            if (!inBounds(nr, nc)) continue;

            const dist = Math.abs(dr) + Math.abs(dc);
            if (dist <= bomb.radius && gameState.maze[nr]?.[nc] !== TILE.WALL && gameState.maze[nr]?.[nc] !== TILE.EXIT) {
                if (!gameState.fires.some(f => f.row === nr && f.col === nc) && Math.random() < 0.35) {
                    gameState.fires.push({ row: nr, col: nc, age: 0, intensity: 1 });
                }
                const key = `${nr},${nc}`;
                gameState.smoke[key] = (gameState.smoke[key] || 0) + 0.7;
            }
        }
    }
}

function updateEarthquake(dt) {
    if (!gameState.earthquakeActive || !gameState.earthquakeEpicenter) return;

    gameState.earthquakeTimer -= dt;
    const epic = gameState.earthquakeEpicenter;
    const baseImpact = Math.max(0, gameState.earthquakeTimer);

    // Intensify shake while active
    gameState.shake.intensity = Math.max(
        gameState.shake.intensity,
        2.5 + Math.sin(gameState.time * 12) * 0.5 + baseImpact * 0.6
    );

    // Damage sensors and people based on proximity
    gameState.sensors.forEach(sensor => {
        const dist = Math.abs(sensor.row - epic.row) + Math.abs(sensor.col - epic.col);
        const impact = Math.max(0, 1 - dist / 12);
        if (impact > 0) {
            sensor.health -= impact * 12 * dt;
            sensor.triggered = true;
        }
    });

    gameState.people.forEach(person => {
        if (!person.alive || person.escaped) return;

        const dist = Math.abs(person.row - epic.row) + Math.abs(person.col - epic.col);
        const impact = Math.max(0, 1 - dist / 14);

        if (impact > 0 && Math.random() < 0.8 * dt) {
            person.health -= impact * 18;
            if (person.state !== STATE.WARDEN) {
                person.state = STATE.PANICKING;
            }
        }
    });

    if (gameState.earthquakeTimer <= 0) {
        gameState.earthquakeActive = false;
        gameState.earthquakeEpicenter = null;
        addChatMessage('system', 'Earthquake tremors subside.');
    }
}

function updatePeople(dt) {
    gameState.people.forEach(person => {
        if (!person.alive || person.escaped) return;

        // Check fire damage
        const inFire = gameState.fires.some(f => f.row === person.row && f.col === person.col);
        if (inFire) {
            person.health -= 30 * dt;
            person.state = STATE.PANICKING;
        }

        // Check smoke damage
        const smokeLevel = gameState.smoke[`${person.row},${person.col}`] || 0;
        if (smokeLevel > 0.5) {
            person.health -= smokeLevel * 10 * dt;
        }

        // Check flood effects
        if (isFlooded(person.row, person.col)) {
            person.health -= 6 * dt;
            if (!person.isWarden) {
                person.state = STATE.PANICKING;
            }
        }

        // Panic near ticking bombs
        const nearBomb = gameState.bombs.some(b =>
            Math.abs(b.row - person.row) + Math.abs(b.col - person.col) <= 2
        );
        if (nearBomb && !person.isWarden) {
            person.state = STATE.PANICKING;
        }

        // Earthquake shake damage
        if (gameState.earthquakeActive && gameState.earthquakeEpicenter) {
            const dist = Math.abs(person.row - gameState.earthquakeEpicenter.row) +
                Math.abs(person.col - gameState.earthquakeEpicenter.col);
            const impact = Math.max(0, 1 - dist / 18);
            if (impact > 0 && Math.random() < 0.6 * dt) {
                person.health -= impact * 12;
                if (!person.isWarden) {
                    person.state = STATE.PANICKING;
                }
            }
        }

        // Check death
        if (person.health <= 0) {
            person.alive = false;
            addChatMessage('death', `Person ${person.id} has perished!`);
            return;
        }

        // Check escape
        if (gameState.maze[person.row]?.[person.col] === TILE.EXIT) {
            person.escaped = true;
            addChatMessage('escape', `Person ${person.id} escaped safely!`);
            return;
        }

        // Update awareness
        if (gameState.stats.alarmActive) {
            if (person.state === STATE.WORKING) {
                person.awareness = (person.awareness || 0) + 0.15 * dt;
            } else if (person.state === STATE.HEADPHONES) {
                person.awareness = (person.awareness || 0) + 0.02 * dt;
            }

            if (person.awareness >= 0.7 && person.state !== STATE.EVACUATING && person.state !== STATE.WARDEN) {
                person.state = STATE.EVACUATING;
            }
        }

        // Movement towards exit
        if (person.state === STATE.EVACUATING || person.state === STATE.WARDEN || person.state === STATE.PANICKING) {
            moveTowardsExit(person, dt);
        }
    });
}

function moveTowardsExit(person, dt) {
    // Find nearest exit
    let nearestExit = null;
    let nearestDist = Infinity;

    gameState.exits.forEach(exit => {
        const dist = Math.abs(exit[0] - person.row) + Math.abs(exit[1] - person.col);
        // Check if exit is blocked by fire
        const blocked = gameState.fires.some(f =>
            Math.abs(f.row - exit[0]) < 3 && Math.abs(f.col - exit[1]) < 3
        );
        if (!blocked && dist < nearestDist) {
            nearestDist = dist;
            nearestExit = exit;
        }
    });

    if (!nearestExit) return;

    // Simple pathfinding - move towards exit
    const dr = nearestExit[0] - person.row;
    const dc = nearestExit[1] - person.col;

    // Check for obstacles
    const moves = [];
    if (dr > 0) moves.push([1, 0]);
    if (dr < 0) moves.push([-1, 0]);
    if (dc > 0) moves.push([0, 1]);
    if (dc < 0) moves.push([0, -1]);

    // Add random variation
    if (Math.random() < 0.2) {
        moves.push([0, 0]); // Sometimes stay still
    }

    // Try each move
    for (const [mr, mc] of moves) {
        const newRow = person.row + mr;
        const newCol = person.col + mc;

        // Check bounds
        if (newRow < 0 || newRow >= CONFIG.ROWS || newCol < 0 || newCol >= CONFIG.COLS) continue;

        // Check tile walkability
        const tile = gameState.maze[newRow]?.[newCol];
        if (tile === TILE.WALL) continue;

        // Avoid stepping on bombs
        if (hasBomb(newRow, newCol)) continue;

        // Check for fire
        if (gameState.fires.some(f => f.row === newRow && f.col === newCol)) continue;

        // Move with probability based on speed
        const floodSlow = isFlooded(newRow, newCol) ? 0.5 : 1;
        const speed = (person.isWarden ? 1.2 : (person.state === STATE.PANICKING ? 1.3 : 1.0)) * floodSlow;
        if (Math.random() < 0.1 * speed * dt * 60) {
            person.row = newRow;
            person.col = newCol;
        }
        break;
    }
}

function updateSensors(dt) {
    gameState.sensors.forEach(sensor => {
        if (sensor.health <= 0) return;

        // Check for nearby fire
        let value = 0;
        gameState.fires.forEach(fire => {
            const dist = Math.abs(sensor.row - fire.row) + Math.abs(sensor.col - fire.col);
            if (dist < 10) {
                if (sensor.type === 'temperature') {
                    value = Math.max(value, 100 / (dist + 1) + 22);
                } else if (sensor.type === 'smoke') {
                    value = Math.max(value, gameState.smoke[`${sensor.row},${sensor.col}`] || 0);
                } else if (sensor.type === 'co') {
                    value = Math.max(value, 50 / (dist + 1));
                }
            }
        });

        // Motion detection
        if (sensor.type === 'motion') {
            gameState.people.forEach(p => {
                if (p.alive && !p.escaped) {
                    const dist = Math.abs(sensor.row - p.row) + Math.abs(sensor.col - p.col);
                    if (dist < 5) {
                        value += 1 / (dist + 1);
                    }
                }
            });
        }

        sensor.value = value;

        // Check threshold
        const thresholds = {
            temperature: 45,
            smoke: 0.3,
            co: 35,
            motion: 0.5
        };

        sensor.triggered = value > (thresholds[sensor.type] || 0.5);

        // Damage sensor in fire
        if (gameState.fires.some(f => f.row === sensor.row && f.col === sensor.col)) {
            sensor.health -= 5 * dt;
        }
    });
}

function updateNeuralPredictions(dt) {
    // Simulate neural network predictions
    if (gameState.fires.length > 0) {
        gameState.neural.confidence = Math.min(gameState.neural.confidence + 0.05 * dt, 0.95);

        // Generate predictions based on fire positions
        gameState.predictions = [];
        gameState.fires.forEach(fire => {
            const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
            directions.forEach(([dr, dc]) => {
                for (let step = 1; step <= 3; step++) {
                    const nr = fire.row + dr * step;
                    const nc = fire.col + dc * step;
                    if (nr > 0 && nr < CONFIG.ROWS - 1 && nc > 0 && nc < CONFIG.COLS - 1) {
                        if (gameState.maze[nr]?.[nc] !== TILE.WALL) {
                            gameState.predictions.push({
                                row: nr,
                                col: nc,
                                prob: 0.7 * Math.pow(0.7, step)
                            });
                        }
                    }
                }
            });
        });
    } else {
        gameState.neural.confidence *= 0.95;
    }

    // Update pheromone averages
    gameState.neural.safePheromone = 0.1 + gameState.stats.escaped * 0.05;
    gameState.neural.dangerPheromone = gameState.fires.length * 0.2;

    // RL decisions
    if (gameState.stats.alarmActive && Math.random() < 0.01 * dt) {
        gameState.rl.decisions++;
        gameState.rl.avgReward = (gameState.rl.avgReward * 0.9) + (Math.random() * 5);
    }
}

function updateStats() {
    gameState.stats.escaped = gameState.people.filter(p => p.escaped).length;
    gameState.stats.deaths = gameState.people.filter(p => !p.alive).length;
}

function checkVictory() {
    const alive = gameState.people.filter(p => p.alive && !p.escaped).length;
    if (alive === 0 && gameState.people.length > 0) {
        showVictoryScreen();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// UI UPDATES
// ═══════════════════════════════════════════════════════════════════════════

function updateUI() {
    // Stats
    document.getElementById('alive-count').textContent =
        gameState.stats.total - gameState.stats.escaped - gameState.stats.deaths;
    document.getElementById('escaped-count').textContent = gameState.stats.escaped;
    document.getElementById('death-count').textContent = gameState.stats.deaths;

    // Alarm indicator
    const alarmIndicator = document.getElementById('alarm-indicator');
    if (gameState.stats.alarmActive) {
        alarmIndicator.className = 'alarm-on';
        alarmIndicator.querySelector('.alarm-text').textContent = 'ALARM ACTIVE!';
    } else {
        alarmIndicator.className = 'alarm-off';
        alarmIndicator.querySelector('.alarm-text').textContent = 'ALARM OFF';
    }

    // Neural confidence
    const confidence = gameState.neural.confidence;
    document.getElementById('neural-confidence-bar').style.width = (confidence * 100) + '%';
    document.getElementById('neural-confidence').textContent = Math.round(confidence * 100) + '%';

    // Speed
    document.getElementById('speed-display').textContent = gameState.speed.toFixed(1) + 'x';

    // Sensors
    const tempSensors = gameState.sensors.filter(s => s.type === 'temperature' && s.health > 0);
    const avgTemp = tempSensors.length > 0
        ? tempSensors.reduce((sum, s) => sum + s.value, 0) / tempSensors.length
        : 22;
    document.getElementById('sensor-temp').textContent = avgTemp.toFixed(1) + 'C';

    const smokeSensors = gameState.sensors.filter(s => s.type === 'smoke' && s.health > 0);
    const avgSmoke = smokeSensors.length > 0
        ? smokeSensors.reduce((sum, s) => sum + s.value, 0) / smokeSensors.length
        : 0;
    document.getElementById('sensor-smoke').textContent = avgSmoke.toFixed(2);

    const coSensors = gameState.sensors.filter(s => s.type === 'co' && s.health > 0);
    const avgCo = coSensors.length > 0
        ? coSensors.reduce((sum, s) => sum + s.value, 0) / coSensors.length
        : 0;
    document.getElementById('sensor-co').textContent = Math.round(avgCo) + ' ppm';

    const motionSensors = gameState.sensors.filter(s => s.type === 'motion' && s.triggered);
    document.getElementById('sensor-motion').textContent = motionSensors.length;

    const aliveSensors = gameState.sensors.filter(s => s.health > 0);
    const coverage = (aliveSensors.length / gameState.sensors.length) * 100;
    document.getElementById('sensor-coverage').style.width = coverage + '%';
    document.getElementById('sensor-coverage-text').textContent = Math.round(coverage) + '%';

    // RL stats
    document.getElementById('rl-decisions').textContent = gameState.rl.decisions;
    document.getElementById('rl-reward').textContent = gameState.rl.avgReward.toFixed(2);
    document.getElementById('rl-epsilon').textContent = Math.round(gameState.rl.epsilon * 100) + '%';

    // Fire stats
    document.getElementById('fire-count').textContent = gameState.fires.length;
    document.getElementById('smoke-zones').textContent = Object.keys(gameState.smoke).length;
    document.getElementById('bomb-count').textContent = gameState.bombs.length;
    document.getElementById('flood-tiles').textContent = gameState.floods.length;
    document.getElementById('earthquake-status').textContent =
        gameState.earthquakeActive ? 'Tremors' : 'Stable';

    const dangerLevel = Math.min(
        gameState.fires.length * 10 +
        Object.keys(gameState.smoke).length / 10 +
        gameState.bombs.length * 8 +
        gameState.floods.length * 4 +
        (gameState.earthquakeActive ? 20 : 0),
        100
    );
    document.getElementById('danger-level').style.width = dangerLevel + '%';

    // Pheromones
    document.getElementById('safe-pheromone').style.width = Math.min(gameState.neural.safePheromone * 10, 100) + '%';
    document.getElementById('danger-pheromone').style.width = Math.min(gameState.neural.dangerPheromone * 10, 100) + '%';
    document.getElementById('tracked-paths').textContent = gameState.stats.escaped * 3;

    // Predictions
    document.querySelector('#prediction-zones .prediction-count').textContent = gameState.predictions.length;

    // Confidence ring
    const circumference = 2 * Math.PI * 40;
    const offset = circumference * (1 - confidence);
    document.getElementById('confidence-ring').style.strokeDashoffset = offset;
    document.getElementById('confidence-percent').textContent = Math.round(confidence * 100) + '%';

    updatePythonTelemetryUI();

    // Pause overlay
    document.getElementById('pause-overlay').classList.toggle('hidden', !gameState.paused);
}

function updatePythonTelemetryUI() {
    const statusEl = document.getElementById('py-status');
    if (!statusEl) return;

    const data = gameState.pythonTelemetry.stats;
    const loading = gameState.pythonTelemetry.loading;
    const error = gameState.pythonTelemetry.error;

    statusEl.textContent = loading ? 'Running...' : (error ? 'Error' : (data ? 'Complete' : 'Idle'));

    const fields = [
        ['py-steps', data?.steps_run],
        ['py-escaped', data?.escaped],
        ['py-deaths', data?.deaths],
        ['py-alive', data?.alive],
        ['py-fires', data?.fires_active],
        ['py-confidence', data ? `${Math.round((data.neural_confidence || 0) * 100)}%` : undefined],
        ['py-rl', data?.rl_decisions],
        ['py-coverage', data ? `${Math.round((data.sensor_coverage || 0) * 100)}%` : undefined],
        ['py-temp', data?.avg_temp ? data.avg_temp.toFixed(1) + 'C' : undefined],
        ['py-mesh-nodes', data?.mesh_nodes],
        ['py-mesh-links', data?.mesh_links],
        ['py-mesh-degree', data?.mesh_avg_degree ? data.mesh_avg_degree.toFixed(1) : undefined],
        ['py-mesh-broadcasts', data?.mesh_broadcast_steps]
    ];

    fields.forEach(([id, val]) => {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = val !== undefined && val !== null ? val : '-';
        }
    });

    const btn = document.getElementById('run-python-btn');
    if (btn) {
        btn.disabled = loading;
        btn.textContent = loading ? 'Running...' : 'Run Python Sim';
    }

    const errorEl = document.getElementById('py-error');
    if (errorEl) {
        errorEl.textContent = error || '';
        errorEl.classList.toggle('hidden', !error);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GAME ACTIONS
// ═══════════════════════════════════════════════════════════════════════════

function addFire(row, col) {
    if (row < 1 || row >= CONFIG.ROWS - 1 || col < 1 || col >= CONFIG.COLS - 1) return;
    if (gameState.maze[row]?.[col] === TILE.WALL) return;
    if (gameState.fires.some(f => f.row === row && f.col === col)) return;

    gameState.fires.push({ row, col, age: 0, intensity: 1 });
    gameState.shake.intensity = 2;
    addChatMessage('fire', `Fire started at (${row}, ${col})!`);
}

function addBomb(row, col) {
    if (!inBounds(row, col) || gameState.maze[row]?.[col] === TILE.EXIT) return;
    if (hasBomb(row, col)) return;

    gameState.bombs.push({ row, col, timer: 3, radius: 3 });
    gameState.shake.intensity = Math.max(gameState.shake.intensity, 1.5);
    addChatMessage('fire', `Bomb armed at (${row}, ${col}) - detonation imminent!`);
    if (!gameState.stats.alarmActive) {
        triggerAlarm();
    }
}

function addFlood(row, col) {
    if (!inBounds(row, col)) return;
    if (isFlooded(row, col)) return;

    gameState.floods.push({ row, col, depth: 0.7 });
    addChatMessage('system', `Flood surge released at (${row}, ${col}).`);
    if (!gameState.stats.alarmActive) {
        triggerAlarm();
    }
}

function getNextSensorId() {
    if (!gameState.sensors.length) return 0;
    return Math.max(...gameState.sensors.map(s => s.id)) + 1;
}

function addSensor(row, col) {
    if (!inBounds(row, col)) return;
    if (gameState.maze[row]?.[col] === TILE.WALL || gameState.maze[row]?.[col] === TILE.EXIT) return;
    if (gameState.sensors.some(s => s.row === row && s.col === col && s.health > 0)) return;

    const sensorTypes = ['temperature', 'smoke', 'co', 'motion'];
    const id = getNextSensorId();
    const type = sensorTypes[id % sensorTypes.length];

    gameState.sensors.push({
        id,
        row,
        col,
        type,
        value: type === 'temperature' ? 22 : 0,
        triggered: false,
        health: 100
    });
    addChatMessage('system', `Sensor placed at (${row}, ${col}).`);
}

function addEarthquake(row, col) {
    if (!inBounds(row, col)) return;

    gameState.earthquakeActive = true;
    gameState.earthquakeTimer = 5;
    gameState.earthquakeEpicenter = { row, col };
    gameState.shake.intensity = 4;
    addChatMessage('alarm', `Earthquake epicenter set at (${row}, ${col})!`);
    triggerAlarm();
}

async function runPythonSimulation() {
    if (gameState.pythonTelemetry.loading) return;

    gameState.pythonTelemetry.loading = true;
    gameState.pythonTelemetry.error = '';
    updatePythonTelemetryUI();

    try {
        const response = await fetch(CONFIG.PY_SIM_API_URL);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Simulation failed');
        }

        gameState.pythonTelemetry.stats = data.stats || null;
        gameState.pythonTelemetry.lastRun = Date.now();
        addChatMessage('system', 'Python headless simulation completed.');
    } catch (err) {
        gameState.pythonTelemetry.error = err.message || 'Unknown error';
        addChatMessage('system', `Python sim error: ${gameState.pythonTelemetry.error}`);
    } finally {
        gameState.pythonTelemetry.loading = false;
        updatePythonTelemetryUI();
    }
}

function triggerAlarm() {
    if (gameState.stats.alarmActive) return;

    gameState.stats.alarmActive = true;
    addChatMessage('alarm', 'EMERGENCY! ALARM ACTIVATED!');

    // Wake up all wardens
    gameState.people.forEach(p => {
        if (p.isWarden && p.alive) {
            p.state = STATE.WARDEN;
            p.awareness = 1;
        }
    });
}

function resetGame() {
    // Show loading
    document.getElementById('loading-screen').style.display = 'flex';
    document.getElementById('game-container').classList.add('hidden');
    document.getElementById('victory-screen').classList.add('hidden');

    // Reset loading bar
    document.querySelector('.loading-bar').style.animation = 'none';
    setTimeout(() => {
        document.querySelector('.loading-bar').style.animation = 'loading 2s ease-out forwards';
    }, 10);

    // Reinitialize
    setTimeout(() => {
        initGame();
    }, 100);
}

function togglePause() {
    gameState.paused = !gameState.paused;
    if (gameState.paused) {
        addChatMessage('system', 'Game paused');
    } else {
        addChatMessage('system', 'Game resumed');
    }
}

function changeSpeed(delta) {
    gameState.speed = Math.max(0.5, Math.min(5, gameState.speed + delta));
}

function addChatMessage(type, text) {
    const container = document.getElementById('chat-messages');
    const msg = document.createElement('div');
    msg.className = `chat-message ${type}`;
    msg.textContent = `[${type.charAt(0).toUpperCase() + type.slice(1)}] ${text}`;
    container.appendChild(msg);

    // Keep only last 5 messages
    while (container.children.length > 5) {
        container.removeChild(container.firstChild);
    }

    // Auto-remove after animation
    setTimeout(() => {
        if (msg.parentNode) {
            msg.remove();
        }
    }, 10000);
}

function showVictoryScreen() {
    gameState.paused = true;

    const victoryScreen = document.getElementById('victory-screen');
    victoryScreen.classList.remove('hidden');

    // Update stats
    document.getElementById('victory-escaped').textContent = gameState.stats.escaped;
    document.getElementById('victory-deaths').textContent = gameState.stats.deaths;

    const elapsed = Math.floor(gameState.time);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    document.getElementById('victory-time').textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;

    // Calculate rating (stars)
    const ratingContainer = document.getElementById('victory-rating');
    ratingContainer.innerHTML = '';

    let stars = 0;
    if (gameState.stats.deaths === 0) stars = 3;
    else if (gameState.stats.deaths <= 3) stars = 2;
    else if (gameState.stats.escaped > gameState.stats.deaths) stars = 1;

    for (let i = 0; i < 3; i++) {
        const star = document.createElement('div');
        star.className = 'star' + (i < stars ? ' earned' : '');
        star.style.animationDelay = (i * 0.2) + 's';
        ratingContainer.appendChild(star);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EVENT HANDLERS
// ═══════════════════════════════════════════════════════════════════════════

function setupEventListeners() {
    // Canvas click
    canvas.addEventListener('click', (e) => {
        if (gameState.paused) return;

        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;

        const col = Math.floor(x / tileSize);
        const row = Math.floor(y / tileSize);

        switch (gameState.selectedTool) {
            case 'fire':
                addFire(row, col);
                break;
            case 'bomb':
                addBomb(row, col);
                break;
            case 'earthquake':
                addEarthquake(row, col);
                break;
            case 'flood':
                addFlood(row, col);
                break;
            case 'sensor':
                addSensor(row, col);
                break;
        }
    });

    // Keyboard
    document.addEventListener('keydown', (e) => {
        switch (e.key.toLowerCase()) {
            case ' ':
                e.preventDefault();
                togglePause();
                break;
            case 'a':
                triggerAlarm();
                break;
            case 'p':
                gameState.showPredictions = !gameState.showPredictions;
                addChatMessage('system', `Predictions: ${gameState.showPredictions ? 'ON' : 'OFF'}`);
                break;
            case 't':
                gameState.showPheromones = !gameState.showPheromones;
                addChatMessage('system', `Pheromones: ${gameState.showPheromones ? 'ON' : 'OFF'}`);
                break;
            case 's':
                gameState.showSensors = !gameState.showSensors;
                addChatMessage('system', `Sensors: ${gameState.showSensors ? 'ON' : 'OFF'}`);
                break;
            case 'r':
                resetGame();
                break;
            case '=':
            case '+':
                changeSpeed(0.5);
                break;
            case '-':
                changeSpeed(-0.5);
                break;
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                selectHotbarSlot(parseInt(e.key) - 1);
                break;
        }
    });

    // Hotbar clicks
    document.querySelectorAll('.hotbar-slot[data-tool]').forEach(slot => {
        slot.addEventListener('click', () => {
            const tool = slot.dataset.tool;
            if (tool === 'reset') {
                resetGame();
            } else {
                selectHotbarSlot(Array.from(slot.parentNode.children).indexOf(slot));
            }
        });
    });

    // Restart button
    document.getElementById('restart-btn').addEventListener('click', resetGame);

    // Python simulation trigger
    const pythonBtn = document.getElementById('run-python-btn');
    if (pythonBtn) {
        pythonBtn.addEventListener('click', runPythonSimulation);
    }
}

function selectHotbarSlot(index) {
    const slots = document.querySelectorAll('.hotbar-slot');
    slots.forEach((slot, i) => {
        slot.classList.toggle('selected', i === index);
    });

    const selectedSlot = slots[index];
    if (selectedSlot && selectedSlot.dataset.tool) {
        gameState.selectedTool = selectedSlot.dataset.tool;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

async function fetchInitialState() {
    try {
        const response = await fetch(CONFIG.API_URL);
        const data = await response.json();

        if (data.success) {
            gameState.maze = data.maze;
            gameState.exits = data.exits;
            gameState.people = data.people;
            gameState.sensors = data.sensors;
            gameState.stats.total = data.config.totalPeople;
            return true;
        }
    } catch (error) {
        console.warn('API not available, generating locally:', error);
    }

    // Generate locally if API fails
    generateLocalState();
    return true;
}

function generateLocalState() {
    // Generate maze
    gameState.maze = [];
    for (let r = 0; r < CONFIG.ROWS; r++) {
        gameState.maze[r] = [];
        for (let c = 0; c < CONFIG.COLS; c++) {
            gameState.maze[r][c] = TILE.FLOOR;
        }
    }

    // Outer walls
    for (let r = 0; r < CONFIG.ROWS; r++) {
        gameState.maze[r][0] = TILE.WALL;
        gameState.maze[r][CONFIG.COLS - 1] = TILE.WALL;
    }
    for (let c = 0; c < CONFIG.COLS; c++) {
        gameState.maze[0][c] = TILE.WALL;
        gameState.maze[CONFIG.ROWS - 1][c] = TILE.WALL;
    }

    // Corridors
    const hCorr = [Math.floor(CONFIG.ROWS / 3), Math.floor(2 * CONFIG.ROWS / 3)];
    const vCorr = [Math.floor(CONFIG.COLS / 4), Math.floor(CONFIG.COLS / 2), Math.floor(3 * CONFIG.COLS / 4)];

    hCorr.forEach(hr => {
        for (let c = 1; c < CONFIG.COLS - 1; c++) {
            for (let r = hr - 1; r <= hr + 1; r++) {
                if (r > 0 && r < CONFIG.ROWS - 1) {
                    gameState.maze[r][c] = TILE.CORRIDOR;
                }
            }
        }
    });

    vCorr.forEach(vc => {
        for (let r = 1; r < CONFIG.ROWS - 1; r++) {
            for (let c = vc - 1; c <= vc + 1; c++) {
                if (c > 0 && c < CONFIG.COLS - 1) {
                    gameState.maze[r][c] = TILE.CORRIDOR;
                }
            }
        }
    });

    // Rooms
    const sections = [
        [2, hCorr[0] - 2, 2, vCorr[0] - 2],
        [2, hCorr[0] - 2, vCorr[0] + 2, vCorr[1] - 2],
        [2, hCorr[0] - 2, vCorr[1] + 2, vCorr[2] - 2],
        [hCorr[0] + 2, hCorr[1] - 2, 2, vCorr[0] - 2],
        [hCorr[1] + 2, CONFIG.ROWS - 3, 2, vCorr[0] - 2],
        [hCorr[1] + 2, CONFIG.ROWS - 3, vCorr[1] + 2, vCorr[2] - 2],
    ];

    sections.forEach(([r1, r2, c1, c2]) => {
        if (r2 - r1 > 3 && c2 - c1 > 3) {
            // Walls
            for (let r = r1; r <= r2; r++) {
                if (gameState.maze[r][c1] !== TILE.CORRIDOR) gameState.maze[r][c1] = TILE.WALL;
                if (gameState.maze[r][c2] !== TILE.CORRIDOR) gameState.maze[r][c2] = TILE.WALL;
            }
            for (let c = c1; c <= c2; c++) {
                if (gameState.maze[r1][c] !== TILE.CORRIDOR) gameState.maze[r1][c] = TILE.WALL;
                if (gameState.maze[r2][c] !== TILE.CORRIDOR) gameState.maze[r2][c] = TILE.WALL;
            }
            // Carpet interior
            for (let r = r1 + 1; r < r2; r++) {
                for (let c = c1 + 1; c < c2; c++) {
                    if (gameState.maze[r][c] !== TILE.CORRIDOR) {
                        gameState.maze[r][c] = TILE.CARPET;
                    }
                }
            }
        }
    });

    // Exits
    gameState.exits = [
        [hCorr[0], 1], [hCorr[1], 1],
        [hCorr[0], CONFIG.COLS - 2], [hCorr[1], CONFIG.COLS - 2],
        [1, vCorr[0]], [1, vCorr[1]], [1, vCorr[2]],
        [CONFIG.ROWS - 2, vCorr[0]], [CONFIG.ROWS - 2, vCorr[1]], [CONFIG.ROWS - 2, vCorr[2]]
    ];

    gameState.exits.forEach(([er, ec]) => {
        if (er > 0 && er < CONFIG.ROWS - 1 && ec > 0 && ec < CONFIG.COLS - 1) {
            gameState.maze[er][ec] = TILE.EXIT;
            // Clear around exits
            for (let dr = -1; dr <= 1; dr++) {
                for (let dc = -1; dc <= 1; dc++) {
                    const nr = er + dr;
                    const nc = ec + dc;
                    if (nr > 0 && nr < CONFIG.ROWS - 1 && nc > 0 && nc < CONFIG.COLS - 1) {
                        if (gameState.maze[nr][nc] === TILE.WALL) {
                            gameState.maze[nr][nc] = TILE.CORRIDOR;
                        }
                    }
                }
            }
        }
    });

    // Generate people
    gameState.people = [];
    const spawns = [];

    for (let r = 2; r < CONFIG.ROWS - 2; r++) {
        for (let c = 2; c < CONFIG.COLS - 2; c++) {
            const tile = gameState.maze[r][c];
            if (tile === TILE.CARPET || tile === TILE.FLOOR || tile === TILE.CORRIDOR) {
                spawns.push([r, c]);
            }
        }
    }

    // Shuffle spawns
    for (let i = spawns.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [spawns[i], spawns[j]] = [spawns[j], spawns[i]];
    }

    const states = [STATE.WORKING, STATE.HEADPHONES];
    const corridorSpawns = spawns.filter(([r, c]) => gameState.maze[r][c] === TILE.CORRIDOR);

    // Wardens
    for (let i = 0; i < Math.min(CONFIG.NUM_WARDENS, corridorSpawns.length); i++) {
        const [r, c] = corridorSpawns[i];
        gameState.people.push({
            id: i,
            row: r,
            col: c,
            state: STATE.WARDEN,
            isWarden: true,
            health: 100,
            alive: true,
            escaped: false,
            awareness: 1
        });
    }

    // Regular people
    for (let i = CONFIG.NUM_WARDENS; i < Math.min(CONFIG.TOTAL_PEOPLE, spawns.length); i++) {
        const [r, c] = spawns[i];
        gameState.people.push({
            id: i,
            row: r,
            col: c,
            state: states[Math.floor(Math.random() * states.length)],
            isWarden: false,
            health: 100,
            alive: true,
            escaped: false,
            awareness: 0
        });
    }

    // Generate sensors
    gameState.sensors = [];
    const sensorTypes = ['temperature', 'smoke', 'co', 'motion'];
    const sensorCandidates = [...spawns];

    // Shuffle candidates
    for (let i = sensorCandidates.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [sensorCandidates[i], sensorCandidates[j]] = [sensorCandidates[j], sensorCandidates[i]];
    }

    let sensorId = 0;
    for (const [r, c] of sensorCandidates) {
        if (sensorId >= CONFIG.NUM_SENSORS) break;
        if (gameState.maze[r][c] === TILE.WALL || gameState.maze[r][c] === TILE.EXIT) continue;
        const type = sensorTypes[sensorId % sensorTypes.length];
        gameState.sensors.push({
            id: sensorId,
            row: r,
            col: c,
            type,
            value: type === 'temperature' ? 22 : 0,
            triggered: false,
            health: 100
        });
        sensorId++;
    }
}

async function initGame() {
    // Reset state
    gameState.fires = [];
    gameState.bombs = [];
    gameState.floods = [];
    gameState.earthquakeActive = false;
    gameState.earthquakeEpicenter = null;
    gameState.earthquakeTimer = 0;
    gameState.smoke = {};
    gameState.predictions = [];
    gameState.paused = false;
    gameState.speed = 1.0;
    gameState.time = 0;
    gameState.startTime = Date.now();
    gameState.shake = { x: 0, y: 0, intensity: 0 };
    gameState.pythonTelemetry = { loading: false, error: '', stats: null, lastRun: null };
    gameState.stats.escaped = 0;
    gameState.stats.deaths = 0;
    gameState.stats.alarmActive = false;
    gameState.neural.confidence = 0;
    gameState.rl.decisions = 0;
    gameState.rl.avgReward = 0;

    // Fetch or generate state
    await fetchInitialState();

    // Hide loading, show game
    setTimeout(() => {
        document.getElementById('loading-screen').style.display = 'none';
        document.getElementById('game-container').classList.remove('hidden');
        document.getElementById('victory-screen').classList.add('hidden');

        // Clear chat
        document.getElementById('chat-messages').innerHTML = '';
        addChatMessage('system', 'World loaded successfully!');
        addChatMessage('system', 'Use 1-5 to pick Fire/Bomb/Quake/Flood/Sensor, then click to place');

        gameState.startTime = Date.now();
    }, 2000);
}

let lastTime = 0;

function gameLoop(currentTime) {
    const dt = Math.min((currentTime - lastTime) / 1000, 0.1) * gameState.speed;
    lastTime = currentTime;

    updateSimulation(dt);
    render();
    updateUI();

    requestAnimationFrame(gameLoop);
}

// Start the game
document.addEventListener('DOMContentLoaded', async () => {
    initCanvas();
    setupEventListeners();
    await initGame();

    // Start game loop
    requestAnimationFrame(gameLoop);
});
